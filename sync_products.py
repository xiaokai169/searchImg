"""
从 Arab-Bee 产品接口同步产品到本地检索引擎
生产者-消费者模式：下载和特征提取并行
用法: python sync_products.py
"""
import sys
import time
import queue
import threading
import requests

from config import (
    ARAB_BEE_API_URL, ARAB_BEE_TOKEN, SYNC_PAGE_SIZE,
    OBS_IMAGE_PROCESS, BATCH_SIZE,
)
from database import insert_image, image_url_exists, init_database
from engine import get_engine
from extractor import get_extractor
from preprocess import get_image_info
from text_utils import build_keywords

API_URL = ARAB_BEE_API_URL
TOKEN = ARAB_BEE_TOKEN
PAGE_SIZE = SYNC_PAGE_SIZE

DOWNLOAD_QUEUE_SIZE = 200
API_DELAY = 0.5        # 页间延迟，避免打爆接口
DOWNLOAD_DELAY = 0.05  # 图片下载间延迟
RETRY_TIMES = 3        # 失败重试次数


def fetch_products(page: int) -> dict:
    params = {"page": page, "size": PAGE_SIZE, "state": "approved"}
    auth = TOKEN if TOKEN.lower().startswith('bearer ') else f"Bearer {TOKEN}"
    resp = requests.get(API_URL, params=params,
                        headers={"Authorization": auth}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_category(item: dict) -> str:
    tags = item.get('tags', [])
    if tags:
        name = tags[0].get('nameEn', '').strip()
        if name:
            return name
    cat = item.get('category', {})
    return cat.get('nameEn', '其他').strip() or '其他'


def _resolve_total(data: dict) -> int | None:
    for key in ('totalItems', 'total', 'count', 'totalCount', 'total_count'):
        val = data.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    for wrapper in ('pagination', 'meta', 'pager'):
        inner = data.get(wrapper)
        if isinstance(inner, dict):
            for key in ('totalItems', 'total', 'count', 'totalCount', 'total_count'):
                val = inner.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
    return None


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    pct = min(done / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    return "▰" * filled + "▱" * (width - filled)


def main():
    if not TOKEN:
        print("[ERROR] 未设置 ARAB_BEE_TOKEN 环境变量")
        print("  请执行: export ARAB_BEE_TOKEN=\"your_jwt_token\"")
        sys.exit(1)

    init_database()
    engine = get_engine()
    if not engine.load():
        print("[Init] 创建新索引")
    print(f"当前索引: {engine.total} 个向量")

    # 先拉第一页获取总数
    try:
        first_page = fetch_products(1)
        first_items = first_page.get('items', [])
        total = _resolve_total(first_page)
    except Exception as e:
        print(f"[ERROR] 无法连接 API: {e}")
        sys.exit(1)

    total_pages = -(-total // PAGE_SIZE) if total else None
    if total:
        print(f"API 返回: {total} 条, {total_pages} 页")
    else:
        print(f"[WARN] API 未返回总数，将拉取到空页为止")

    t0 = time.time()
    download_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    failed_items = []   # 失败记录
    failed_lock = threading.Lock()

    # 共享统计
    stats = {'downloaded': 0, 'dl_fail': 0, 'skipped': 0,
             'success': 0, 'fail': 0, 'total': total or 0,
             'queue_size': 0, 'total_bytes': 0, 'dl_done': False,
             'current_page': 0, 'current_img': ''}
    stats_lock = threading.Lock()

    def progress_thread():
        """每秒刷新进度"""
        last = 0
        while True:
            time.sleep(1)
            with stats_lock:
                dl = stats['downloaded']; df = stats['dl_fail']
                sk = stats['skipped']; t = stats['total']
                s = stats['success']; f = stats['fail']
                qs = stats['queue_size']; tb = stats['total_bytes']
                pg = stats['current_page']; img = stats['current_img']
                done_flag = stats['dl_done']
                elapsed = time.time() - t0

            if done_flag and qs == 0:
                break  # 全部完成

            # 下载进度
            if t > 0:
                dl_total = dl + df + sk
                bar = _progress_bar(dl_total, t)
                print(f"  📥 下载 {bar} {dl_total}/{t} | "
                      f"新{dl} 跳过{sk} 失败{df} | {_fmt_size(tb)} | "
                      f"⏳{_fmt_time(elapsed)}")
            else:
                print(f"  📥 下载 | 第{pg}页 | 新{dl} 跳过{sk} 失败{df} | "
                      f"{_fmt_size(tb)} | ⏳{_fmt_time(elapsed)}")

            if img:
                print(f"     ⤷ 正在: {img[:80]}")

            # 转换进度
            if t > 0:
                bar = _progress_bar(s + f, t)
                print(f"  🔄 转换 {bar} {s+f}/{t} | 成功{s} 失败{f}")
            else:
                print(f"  🔄 转换 | 成功{s} 失败{f} 队列{qs}")
            print()

    def download_worker():
        """逐页拉取 + 下载，边拉边下"""
        all_pages = [(1, first_items)]

        # ---- 先拉完所有页（带延迟，不打击接口） ----
        page_num = 1
        while True:
            page_num += 1
            time.sleep(API_DELAY)
            try:
                data = fetch_products(page_num)
                items = data.get('items', [])
                if not items:
                    break
                all_pages.append((page_num, items))
                # 补总数
                if stats['total'] == 0:
                    nt = _resolve_total(data)
                    if nt:
                        with stats_lock:
                            stats['total'] = nt
            except Exception as e:
                print(f"  ⚠ 第{page_num}页请求失败: {e}，跳过")

        actual_pages = len(all_pages)
        if stats['total'] == 0:
            with stats_lock:
                stats['total'] = sum(len(it) for _, it in all_pages)

        # ---- 逐页下载图片 ----
        for page, items in all_pages:
            pg_total = len(items)
            pg_ok = pg_skip = pg_fail = 0
            pg_bytes = 0

            with stats_lock:
                stats['current_page'] = page

            for idx, item in enumerate(items, 1):
                img_url = (item.get('img') or '').split('?')[0]
                name_en = (item.get('nameEn') or '').strip()[:60]

                if not img_url:
                    pg_fail += 1
                    continue

                # 已入库跳过
                if image_url_exists(img_url):
                    pg_skip += 1
                    with stats_lock:
                        stats['skipped'] += 1
                        stats['current_img'] = f"[跳过] {name_en}"
                    continue

                with stats_lock:
                    stats['current_img'] = f"{page}/{actual_pages}[{idx}/{pg_total}] {name_en}"

                category = get_category(item)
                name_cn = (item.get('name') or '').strip()[:200]
                product_id = str(item.get('id', ''))
                keywords_cn, keywords_en = build_keywords(name_cn, name_en)

                success = False
                last_err = ""
                for attempt in range(RETRY_TIMES):
                    try:
                        img_url_dl = img_url + OBS_IMAGE_PROCESS
                        img_resp = requests.get(img_url_dl, timeout=30)
                        img_resp.raise_for_status()
                        img_bytes = img_resp.content

                        download_queue.put({
                            'image_bytes': img_bytes,
                            'image_url': img_url,
                            'category': category[:50],
                            'product_name': name_en,
                            'product_name_cn': name_cn,
                            'product_id': product_id,
                            'keywords_cn': keywords_cn,
                            'keywords_en': keywords_en,
                        })

                        pg_bytes += len(img_bytes)
                        pg_ok += 1
                        with stats_lock:
                            stats['downloaded'] += 1
                            stats['total_bytes'] += len(img_bytes)
                            stats['queue_size'] = download_queue.qsize()
                        success = True
                        break
                    except Exception as e:
                        last_err = str(e)[:100]
                        if attempt < RETRY_TIMES - 1:
                            time.sleep(2)

                if not success:
                    pg_fail += 1
                    with stats_lock:
                        stats['dl_fail'] += 1
                    with failed_lock:
                        failed_items.append({
                            'img_url': img_url, 'name_en': name_en,
                            'name_cn': name_cn, 'product_id': product_id,
                            'category': category, 'error': last_err,
                        })
                    print(f"    ❌ 失败: {name_en} - {last_err}")

                time.sleep(DOWNLOAD_DELAY)

            # 每页汇总
            parts = [f"第{page}/{actual_pages}页完成"]
            if pg_ok: parts.append(f"{pg_ok}张 {_fmt_size(pg_bytes)}")
            if pg_skip: parts.append(f"跳过{pg_skip}")
            if pg_fail: parts.append(f"❌失败{pg_fail}")
            print(f"  {' | '.join(parts)}")

        download_queue.put(None)
        with stats_lock:
            stats['dl_done'] = True
        print(f"\n  下载阶段完成 ({actual_pages} 页)")

    def process_worker():
        """批量提取特征 + 入库"""
        engine_p = get_engine()
        extractor_p = get_extractor()
        batch = []

        def flush():
            nonlocal batch
            if not batch:
                return
            imgs = [b['image_bytes'] for b in batch]
            try:
                feats = extractor_p.extract_both_batch(imgs)
            except Exception as e:
                print(f"  [转换] 批量失败: {e}")
                for item in batch:
                    try:
                        f = extractor_p.extract_both(item['image_bytes'])
                        _add_one(engine_p, item, f)
                    except Exception as e2:
                        print(f"  [转换] ERR: {e2}")
                        with stats_lock:
                            stats['fail'] += 1
                batch.clear()
                return

            for j, item in enumerate(batch):
                try:
                    if image_url_exists(item['image_url']):
                        with stats_lock:
                            stats['skipped'] += 1
                        continue
                    fid = engine_p.allocate_id()
                    engine_p.add_single(feats['clip'][j], feats['resnet'][j], fid)
                    info = get_image_info(item['image_bytes'])
                    insert_image(
                        faiss_id=fid, image_url=item['image_url'],
                        category=item['category'],
                        product_name=item['product_name'],
                        product_name_cn=item['product_name_cn'],
                        product_id=item['product_id'],
                        keywords_cn=item['keywords_cn'],
                        keywords_en=item['keywords_en'],
                        file_size=len(item['image_bytes']),
                        width=info.get('width', 0), height=info.get('height', 0),
                    )
                    with stats_lock:
                        stats['success'] += 1
                except Exception as e:
                    print(f"  [转换] ERR: {item.get('product_name','')[:30]} - {e}")
                    with stats_lock:
                        stats['fail'] += 1
            batch.clear()

        def _add_one(eng, item, feats):
            if image_url_exists(item['image_url']):
                return
            fid = eng.allocate_id()
            eng.add_single(feats['clip'], feats['resnet'], fid)
            info = get_image_info(item['image_bytes'])
            insert_image(
                faiss_id=fid, image_url=item['image_url'],
                category=item['category'], product_name=item['product_name'],
                product_name_cn=item['product_name_cn'],
                product_id=item['product_id'],
                keywords_cn=item['keywords_cn'],
                keywords_en=item['keywords_en'],
                file_size=len(item['image_bytes']),
                width=info.get('width', 0), height=info.get('height', 0),
            )
            with stats_lock:
                stats['success'] += 1

        while True:
            try:
                item = download_queue.get(timeout=2)
            except queue.Empty:
                flush()
                with stats_lock:
                    stats['queue_size'] = download_queue.qsize()
                continue

            if item is None:
                flush()
                break

            batch.append(item)
            if len(batch) >= BATCH_SIZE:
                flush()
            with stats_lock:
                stats['queue_size'] = download_queue.qsize()

    # ---- 启动 ----
    print(f"\n{'='*55}")
    print(f"  并行同步: 下载 ⇄ 转换")
    print(f"  重试: {RETRY_TIMES}次 | API延迟: {API_DELAY}s | 下载延迟: {DOWNLOAD_DELAY}s")
    print(f"{'='*55}\n")

    pt = threading.Thread(target=progress_thread, name='progress', daemon=True)
    dt = threading.Thread(target=download_worker, name='downloader', daemon=True)
    wt = threading.Thread(target=process_worker, name='processor', daemon=True)

    pt.start()
    dt.start()
    wt.start()

    dt.join()
    wt.join()
    # 等进度线程退出
    time.sleep(2)

    elapsed = time.time() - t0

    # ---- 结果 ----
    print(f"\n{'='*55}")
    print(f"  下载: {stats['downloaded']} 新 / {stats['skipped']} 跳过 / {stats['dl_fail']} 失败")
    print(f"  总下载量: {_fmt_size(stats['total_bytes'])}")
    print(f"  入库: {stats['success']} 成功 / {stats['fail']} 失败")
    print(f"  耗时: {_fmt_time(elapsed)}")
    print(f"  索引: {engine.total} 个向量")
    print(f"{'='*55}")

    # ---- 失败列表 + 重试 ----
    if failed_items:
        print(f"\n  ⚠ 下载失败 {len(failed_items)} 张，逐一重试...")
        retry_ok = 0
        still_fail = 0
        for fi in failed_items:
            try:
                img_url = fi['img_url'] + OBS_IMAGE_PROCESS
                resp = requests.get(img_url, timeout=30)
                resp.raise_for_status()
                item = {
                    'image_bytes': resp.content,
                    'image_url': fi['img_url'],
                    'category': fi['category'],
                    'product_name': fi['name_en'],
                    'product_name_cn': fi['name_cn'],
                    'product_id': fi['product_id'],
                    'keywords_cn': '', 'keywords_en': '',
                }
                feats = get_extractor().extract_both(item['image_bytes'])
                fid = engine.allocate_id()
                engine.add_single(feats['clip'], feats['resnet'], fid)
                info = get_image_info(item['image_bytes'])
                insert_image(
                    faiss_id=fid, image_url=fi['img_url'],
                    category=fi['category'],
                    product_name=fi['name_en'],
                    product_name_cn=fi['name_cn'],
                    product_id=fi['product_id'],
                    file_size=len(item['image_bytes']),
                    width=info.get('width', 0), height=info.get('height', 0),
                )
                retry_ok += 1
                print(f"    ✅ 重试成功: {fi['name_en']}")
            except Exception as e:
                still_fail += 1
                print(f"    ❌ 重试仍失败: {fi['name_en']} - {e}")

        if retry_ok > 0:
            stats['downloaded'] += retry_ok
            stats['success'] += retry_ok
            stats['dl_fail'] -= retry_ok
            print(f"  ✅ 重试成功 {retry_ok} 张")
        if still_fail > 0:
            print(f"  ❌ 最终失败 {still_fail} 张，可重新运行本脚本重试")

    # ---- 保存 ----
    if engine.total > 0:
        print(f"\n  正在保存索引...")
        engine.save()
        print(f"  索引已保存")

        print(f"\n  ⚠ 如果 Flask 服务正在运行，需要重启以加载新索引:")
        print(f"    systemctl restart search-img  (或其他启动方式)")


if __name__ == '__main__':
    main()
