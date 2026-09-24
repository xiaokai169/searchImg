"""
图片尺度 → 自查询命中率 基线验证脚本（一次性）

背景
    库中索引向量是基于 OBS 缩略图（config.OBS_IMAGE_PROCESS，长边 800）算出来的。
    计划让前端把手机照片压到长边 1024 再上传，需要确认这不会损失匹配精度。
    做法：拿库图自身的不同尺寸版本当 query 去查索引，统计自命中率。

为什么只抽"长边 >1280"的图
    OBS 的 m_lfit 是 limit-fit，**只缩不放**。若原图长边 800，请求 w_1024 会原样
    返回 800，各档位退化成同一张图、测不出差异。库图原图尺寸分布很散
    （实测 40 张样本里 800/1080 最常见，但也存在 4096、6648 等），故只筛选
    长边 >1280 的图，保证 w_1280 / w_1024 / w_800 三档都真实生效。

档位说明
    orig         原图，不拼 OBS 参数 —— 代表"现状"（用户直接传手机原图）
    obs_1280     服务端缩到长边 1280
    obs_1024     服务端缩到长边 1024 —— 对应本次前端压缩目标尺寸
    obs_800      服务端缩到长边 800 —— 与索引图同尺度，应接近 100%（sanity check）
    local_1024   本地 PIL LANCZOS 缩到 1024 + JPEG q90，最接近前端 canvas 压缩的真实行为

用法
    python bench_scale.py                 # 40 张样本
    python bench_scale.py 20              # 20 张（快速试跑）
    python bench_scale.py 40 orig,obs_1024

判读
    1. 先看 obs_800 —— 应接近 100%。若不是，说明脚本或索引有问题，先查这个。
    2. 对比 orig 与 obs_1024 / local_1024：
       1024 不低于 orig、且与 obs_800 差距 <2% → 压缩方案安全。
    3. 若 1024 明显低于 orig，说明该尺寸损失了细节，需要上调压缩尺寸。

注意
    脚本绕过 engine.search 的分数阈值（min_top/min_fused 等传放宽值）。否则自查询
    一旦低于门槛就返回空列表，会把尺寸差异掩盖成"所有档位都搜不到"。
"""
import io
import random
import sys
import time

import requests
from PIL import Image

from database import get_all_images
from engine import get_engine
from extractor import get_extractor

DEFAULT_SAMPLE = 40
MIN_EDGE = 1281                    # 只收长边 >=1281 的图，保证 w_1280 档生效
ALL_LEVELS = ['orig', 'obs_1280', 'obs_1024', 'obs_800', 'local_1024']
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_RETRY = 2
SEED = 42
LOCAL_Q = 90                       # 与前端 canvas quality 对齐

LEVEL_LABEL = {
    'orig': '原图',
    'obs_1280': 'obs_1280',
    'obs_1024': 'obs_1024',
    'obs_800': 'obs_800',
    'local_1024': 'local_1024',
}


def download(url: str) -> bytes | None:
    for attempt in range(DOWNLOAD_RETRY + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                return resp.content
        except Exception:
            pass
        if attempt < DOWNLOAD_RETRY:
            time.sleep(0.5 * (attempt + 1))
    return None


def obs_bytes(url: str, width: int) -> bytes | None:
    sep = '&' if '?' in url else '?'
    return download(
        f"{url}{sep}x-image-process=image/resize,m_lfit,w_{width}"
        f"/quality,q_85/auto-orient,1"
    )


def local_resize(raw: bytes, max_edge: int, quality: int = LOCAL_Q) -> bytes | None:
    """本地等比缩放到长边 max_edge 并转 JPEG —— 模拟前端 canvas 压缩"""
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGBA')
            bg = Image.new('RGBA', img.size, (255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert('RGB')
        if max(img.size) > max_edge:                 # 只缩不放
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return None


def stratified_pick(images: list[dict], n: int) -> list[dict]:
    """按品类分层打散，得到一个尽量分散的候选顺序（真正的筛选在尺寸探测时做）"""
    by_cat: dict[str, list[dict]] = {}
    for img in images:
        by_cat.setdefault(img.get('category') or '其他', []).append(img)

    rng = random.Random(SEED)
    cats = list(by_cat.keys())
    rng.shuffle(cats)
    for c in cats:
        rng.shuffle(by_cat[c])

    ordered: list[dict] = []
    idx = 0
    while len(ordered) < len(images):                # 轮转各品类，保证分散
        added = False
        for c in cats:
            if idx < len(by_cat[c]):
                ordered.append(by_cat[c][idx])
                added = True
        if not added:
            break
        idx += 1
    return ordered


def main() -> int:
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE
    levels = (sys.argv[2].split(',') if len(sys.argv) > 2 else ALL_LEVELS)
    for lv in levels:
        if lv not in ALL_LEVELS:
            print(f"[错误] 未知档位 {lv}，可选: {', '.join(ALL_LEVELS)}")
            return 1

    print("=" * 76)
    print("  图片尺度 → 自查询命中率 基线验证")
    print("=" * 76)

    all_imgs = get_all_images()
    if not all_imgs:
        print("[错误] 数据库为空")
        return 1

    engine = get_engine()
    if not engine.load() or engine.total == 0:
        print("[错误] 索引加载失败或为空")
        return 1
    extractor = get_extractor()

    # ===== 筛选长边 >1280 的图，复用已下载的原图作为 orig 档 =====
    print(f"库中 {len(all_imgs)} 张图 / 索引 {engine.total} 个向量")
    print(f"正在筛选长边 >{MIN_EDGE - 1} 的样本（需探测原图尺寸，请稍候）...")

    candidates = stratified_pick(all_imgs, sample_n * 4)
    picked: list[tuple[dict, bytes, tuple[int, int]]] = []
    probed = 0
    for img in candidates:
        if len(picked) >= sample_n:
            break
        probed += 1
        raw = download(img['image_url'])
        if raw is None:
            continue
        try:
            size = Image.open(io.BytesIO(raw)).size
        except Exception:
            continue
        if max(size) >= MIN_EDGE:
            picked.append((img, raw, size))
        if probed % 20 == 0:
            print(f"    已探测 {probed} 张，入选 {len(picked)} 张...")

    if not picked:
        print(f"[错误] 探测 {probed} 张后未找到长边 >{MIN_EDGE - 1} 的图，无法验证")
        return 1

    print(f"入选 {len(picked)} 张（探测了 {probed} 张）")
    print(f"测试档位: {', '.join(LEVEL_LABEL[l] for l in levels)}")
    print()

    stats = {lv: {'hit1': 0, 'hit5': 0, 'n': 0, 'ms': [], 'kb': []} for lv in levels}

    for i, (img, orig_raw, orig_size) in enumerate(picked, 1):
        fid = int(img['faiss_id'])
        url = img['image_url']
        cat = img.get('category', '')
        print(f"[{i}/{len(picked)}] faiss_id={fid} <{cat[:26]}> 原图 {orig_size[0]}x{orig_size[1]}")

        for lv in levels:
            if lv == 'orig':
                raw = orig_raw
            elif lv == 'local_1024':
                raw = local_resize(orig_raw, 1024)
            else:
                w = int(lv.split('_')[1])
                if max(orig_size) <= w:      # lfit 只缩不放，该档会退化成原图
                    print(f"    {LEVEL_LABEL[lv]:<11} 跳过（原图未超过 {w}）")
                    continue
                raw = obs_bytes(url, w)
            if raw is None:
                print(f"    {LEVEL_LABEL[lv]:<11} 跳过（获取失败）")
                continue

            t0 = time.time()
            try:
                feats = extractor.extract_both(raw)
                results = engine.search(
                    feats['clip'], feats['resnet'], top_k=20,
                    min_top=0.0, min_fused=-1.0,
                    min_resnet=-1.0, min_relative=-1.0,
                )
            except Exception as e:
                print(f"    {LEVEL_LABEL[lv]:<11} 提取失败 {type(e).__name__}: {str(e)[:50]}")
                continue
            ms = (time.time() - t0) * 1000

            st = stats[lv]
            st['n'] += 1
            st['ms'].append(ms)
            st['kb'].append(len(raw) / 1024)

            ids = [int(r['faiss_id']) for r in results]
            h1 = bool(ids) and ids[0] == fid
            h5 = fid in ids[:5]
            st['hit1'] += int(h1)
            st['hit5'] += int(h5)

            top = results[0]['fused_score'] if results else 0.0
            mark = '✓@1' if h1 else ('✓@5' if h5 else '✗')
            print(f"    {LEVEL_LABEL[lv]:<11} {len(raw)/1024:7.1f}KB  {mark:<4} "
                  f"top1={top:.3f}  {ms:.0f}ms")

    # ===== 汇总 =====
    print()
    print("=" * 76)
    print(f"{'档位':<14}{'样本':<7}{'平均体积':<12}{'hit@1':<11}{'hit@5':<11}{'推理耗时':<10}")
    print("-" * 76)
    for lv in levels:
        st = stats[lv]
        if st['n'] == 0:
            print(f"{LEVEL_LABEL[lv]:<14}{'无有效样本':<7}")
            continue
        n = st['n']
        print(f"{LEVEL_LABEL[lv]:<14}{n:<7}{sum(st['kb'])/n:>8.1f}KB  "
              f"{st['hit1']/n*100:>8.1f}%  {st['hit5']/n*100:>8.1f}%  "
              f"{sum(st['ms'])/n:>7.0f}ms")
    print("=" * 76)
    print("判读: obs_800 应接近 100%（sanity check）。")
    print("      再比 orig 与 obs_1024 / local_1024 —— 不低于 orig、且与 obs_800")
    print('      差距 <2%，则"前端压到 1024"方案安全。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
