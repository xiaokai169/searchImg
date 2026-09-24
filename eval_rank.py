# -*- coding: utf-8 -*-
"""
共识重排评测脚本

对 config.SEARCH_ZOOMS 每一档跑真实检索链路（与 app.py 一致：按原始 top1 分数选档），
再在选中的档上对比不同 λ 的共识重排效果。特征只算一次，各 λ 复用，所以扫参很快。

核心指标是【前 N 条纯度】—— 用户抱怨的"结果里混入不同品类的商品"就是前几条纯度低。
次要指标是正确品类的召回数与排名分布（判断瓶颈到底在召回还是在排序）。

ground truth 由 QUERIES 里的正则在 product_name 上匹配，标签已逐张看图人工确认：
    images/097e…、459b…、e18c…  都是无人机（侧拍 / 正面 / 45 度）
    images/31a5…                是一台移动 CT（Qinxi 白色环形机架）

用法：
    python eval_rank.py                       # 对比关闭(λ=0) 与 config 当前 λ
    python eval_rank.py --weights 0 .1 .2 .3  # 扫一组 λ
    python eval_rank.py --topn 10             # 改纯度统计口径（默认前 5）
"""
import argparse
import copy
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

from config import (SEARCH_ZOOMS, ZOOM_EARLY_STOP_SCORE,
                    CONSENSUS_RERANK_WEIGHT, CONSENSUS_RERANK_TOP_N)
from database import init_database, get_image_by_faiss_id
from engine import get_engine
from extractor import get_extractor
from preprocess import load_work_image, center_zoom_crops

# ==================== ground truth（人工确认） ====================
# 匹配对象是 product_name 的小写形式。
# ⚠️ CT 不能用 \bct\b：库中真实商品名是 'MCT-I Mobile CT'，词边界匹配会漏掉 'MCT-I'。
QUERIES = [
    ('images/097e7740df9a16de02beb3eb6019b2eb.jpg', '无人机·侧拍',
     r'drone|uav|unmanned|无人机'),
    ('images/459b0f3815168f0c14aa06be6b35c0f9.jpg', '无人机·正面',
     r'drone|uav|unmanned|无人机'),
    ('images/e18caca90ff643d6751f21e89605ee9d.jpg', '无人机·45度',
     r'drone|uav|unmanned|无人机'),
    ('images/31a541c9f184ca3b87f56e9092f17d00.jpg', '移动CT',
     r'\bct\b|mct-|mobile ct|tomograph|移动ct|断层'),
]

CANDIDATE_POOL = 100        # 候选池深度，要比 top_n 深得多才能看清排名分布


def is_hit(img: dict, pattern: re.Pattern) -> bool:
    """
    判定一条结果是否属于查询图的正确品类。

    必须同时看中英文产品名：库中相当一部分条目的 product_name 是中文
    （如 'K20农用植保无人机'），只看英文名会把它们误判成"错"，污染纯度指标。
    两个名字都为空时（约 6%）再回退到 keywords_en。
    """
    text = ' '.join(filter(None, (img.get('product_name') or '',
                                  img.get('product_name_cn') or ''))).lower()
    if not text.strip():
        text = (img.get('keywords_en') or '').lower()
    return bool(pattern.search(text))


def collect_features(path: str, extractor) -> list[tuple[float, list[dict]]]:
    """对每一档 zoom 抽特征并检索，返回 [(zoom, results)]"""
    with open(path, 'rb') as f:
        work = load_work_image(f.read())
    crops, zooms = center_zoom_crops(work, SEARCH_ZOOMS)
    if not crops:
        crops, zooms = [work], [1.0]

    engine = get_engine()
    out = []
    for z, crop in zip(zooms, crops):
        r = engine.search(extractor.extract_clip_from_image(crop),
                          extractor.extract_resnet_from_image(crop),
                          top_k=CANDIDATE_POOL, min_top=0.0)
        out.append((z, r))
        if r and ZOOM_EARLY_STOP_SCORE and r[0]['fused_score'] >= ZOOM_EARLY_STOP_SCORE:
            break                       # 与 app.py 一致：已是高置信匹配就停
    return out


def attach_names(results: list[dict]) -> list[dict]:
    """补上 product_name/keywords（重排需要，展示也需要）"""
    enriched = []
    for item in results:
        img = get_image_by_faiss_id(item['faiss_id'])
        if img is None:
            continue
        d = dict(item)
        d['product_name'] = str(img.get('product_name') or '')
        d['product_name_cn'] = str(img.get('product_name_cn') or '')
        d['keywords_en'] = str(img.get('keywords_en') or '')
        d['keywords_cn'] = str(img.get('keywords_cn') or '')
        enriched.append(d)
    return enriched


def evaluate(results: list[dict], pattern: re.Pattern, weight: float, topn: int):
    """对一份候选跑共识重排并算指标"""
    engine = get_engine()
    work = copy.deepcopy(results)
    if weight > 0:
        work, consensus, boosted = engine.rerank_by_consensus(work, weight=weight)
    else:
        consensus, boosted = set(), 0

    purity = sum(1 for r in work[:topn] if is_hit(r, pattern))
    ranks = [i + 1 for i, r in enumerate(work) if is_hit(r, pattern)]
    return {
        'results': work,
        'top1_hit': is_hit(work[0], pattern) if work else False,
        'purity': purity,
        'recall': len(ranks),
        'ranks': ranks,
        'consensus': consensus,
        'boosted': boosted,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=float, nargs='+',
                    default=[0.0, CONSENSUS_RERANK_WEIGHT],
                    help='要对比的 λ 列表，0 表示关闭重排')
    ap.add_argument('--topn', type=int, default=5, help='纯度统计口径，默认前 5')
    ap.add_argument('--verbose', action='store_true', help='打印每条结果的明细')
    args = ap.parse_args()

    init_database()
    engine = get_engine()
    engine.load()
    if engine.total == 0:
        print('索引为空，先入库再跑评测')
        return 1
    extractor = get_extractor()
    print(f'索引 {engine.total} 张 | top_n={CONSENSUS_RERANK_TOP_N} | 纯度口径=前 {args.topn} 条\n')

    summary = {w: {'purity': 0, 'top1': 0, 'queries': 0} for w in args.weights}

    for path, label, pattern in QUERIES:
        if not os.path.exists(path):
            print(f'!! 缺少测试图 {path}，跳过')
            continue
        pat = re.compile(pattern)
        zooms = collect_features(path, extractor)
        if not any(r for _, r in zooms):
            print(f'=== {label}  {os.path.basename(path)}: 无候选 ===\n')
            continue

        # 与 app.py 一致：按【原始】top1 分数选档，选完再重排
        best_zoom, best = max(((z, r) for z, r in zooms if r),
                              key=lambda x: x[1][0]['fused_score'])
        results = attach_names(best)
        print('=' * 78)
        print(f'{label}  {os.path.basename(path)}  zoom={best_zoom}  候选 {len(results)} 条')

        for w in args.weights:
            ev = evaluate(results, pat, w, args.topn)
            tag = '关闭' if w == 0 else f'λ={w:.2f}'
            mark = 'OK ' if ev['top1_hit'] else 'BAD'
            print(f"  [{tag:>7}] top1 {mark} {ev['results'][0]['fused_score']:.3f} "
                  f"{ev['results'][0]['product_name'][:40]:40s} | "
                  f"前{args.topn}纯度 {ev['purity']}/{args.topn} | "
                  f"召回 {ev['recall']:3d} | 共识词 {len(ev['consensus'])}")
            if args.verbose:
                for i, r in enumerate(ev['results'][:args.topn]):
                    hit = '√' if is_hit(r, pat) else '×'
                    name = r['product_name'] or r['product_name_cn'] or '(无名)'
                    print(f"        {i + 1:2d}. {hit} {r['fused_score']:.3f} "
                          f"{name[:44]:44s} cs={r.get('consensus_score', 0):.2f}")
            summary[w]['purity'] += ev['purity']
            summary[w]['top1'] += int(ev['top1_hit'])
            summary[w]['queries'] += 1
        print()

    print('=' * 78)
    print(f"汇总（纯度满分 = 查询数 × {args.topn}）")
    for w in args.weights:
        s = summary[w]
        if not s['queries']:
            continue
        tag = '关闭' if w == 0 else f'λ={w:.2f}'
        print(f"  {tag:>8}  top1 正确 {s['top1']}/{s['queries']}  "
              f"前{args.topn}纯度 {s['purity']}/{s['queries'] * args.topn}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
