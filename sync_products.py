"""
从 Arab-Bee 产品接口同步产品到本地检索引擎
生产者-消费者模式：下载和特征提取并行，下载不等待转换
用法: python sync_products.py
"""
import sys
import time
import queue
import threading
import requests

from config import (
    ARAB_BEE_API_URL, ARAB_BEE_TOKEN, SYNC_PAGE_SIZE, SYNC_MAX_PAGES,
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
MAX_PAGES = SYNC_MAX_PAGES

# 下载队列（生产者→消费者），放 200 个保证下载跑在转换前面
DOWNLOAD_QUEUE_SIZE = 200


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



def progress_reporter(stats: dict, stats_lock: threading.Lock,
                      stop_event: threading.Event, t0: float):
    """每 3 秒打印下载和转换进度"""
    while not stop_event.is_set():
        time.sleep(3)
        with stats_lock:
            d = stats['downloaded']
            df = stats['download_fail']
            s = stats['success']
            f = stats['fail']
            total = stats.get('total_products', 0)
            qs = stats.get('queue_size', 0)
            total_bytes = stats.get('total_bytes', 0)
            elapsed = time.time() - t0
            d_done = stats.get('download_done', False)

        if total > 0:
            d_pct = (d + df) / total * 100 if total else 0
            s_pct = (s + f) / total * 100 if total else 0
            bar_d = _progress_bar(d + df, total, 20)
            bar_s = _progress_bar(s + f, total, 20)
            d_status = "✓完成" if d_done else f"{d_pct:.0f}%"
            print(f"  📥 下载 {bar_d} {d_status} ({d}成功/{df}失败) | "
                  f"队列 {qs} | 已下载 {_fmt_size(total_bytes)} | "
                  f"⏳ {_fmt_time(elapsed)}")
            print(f"  🔄 转换 {bar_s} {s_pct:.0f}% ({s}成功/{f}失败)")
        else:
            d_status = "✓完成" if d_done else "进行中"
            print(f"  📥 下载: {d_status} ({d}成功/{df}失败) | "
                  f"队列 {qs} | 已下载 {_fmt_size(total_bytes)} | "
                  f"⏳ {_fmt_time(elapsed)}")
            print(f"  🔄 转换: {s}成功/{f}失败")


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    """▰▰▰▱▱▱ 进度条"""
    pct = min(done / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    return f"▰" * filled + f"▱" * (width - filled)


def _fmt_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _fmt_time(seconds: float) -> str:
    """格式化时间"""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


def main():
    if not TOKEN:
        print("[ERROR] 未设置 ARAB_BEE_TOKEN 环境变量")
        print("  请执行: export ARAB_BEE_TOKEN=\"your_jwt_token\"")
        sys.exit(1)

    # 初始化数据库和引擎（不需要 Flask）
    init_database()
    engine = get_engine()
    if not engine.load():
        print("[Init] 创建新索引")

    print(f"当前索引: {engine.total} 个向量")

    # 先拉第一页，获取总数
    try:
        first_page = fetch_products(1)
        total_products = first_page.get('totalItems', 0)
        total_pages = min(MAX_PAGES, -(-total_products // PAGE_SIZE))  # 向上取整
    except Exception as e:
        print(f"[ERROR] 无法连接 API: {e}")
        sys.exit(1)

    t0 = time.time()

    # 共享状态
    download_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    stats = {
        'downloaded': 0, 'download_fail': 0,
        'success': 0, 'fail': 0,
        'total_products': min(total_products, MAX_PAGES * PAGE_SIZE),
        'queue_size': 0,
        'total_bytes': 0,
        'download_done': False,
    }
    stats_lock = threading.Lock()
    stop_event = threading.Event()

    # 把第一页数据预填到队列（已经拿到了，不用再请求一次）
    def download_worker_with_first_page(dl_queue, st, st_lock, stop_evt, first_items):
        """从第一页数据开始下载（避免重复请求）"""
        all_pages = [(1, first_items)]  # (page_num, items)
        for page in range(2, MAX_PAGES + 1):
            if stop_evt.is_set():
                break
            try:
                data = fetch_products(page)
                all_pages.append((page, data.get('items', [])))
            except Exception as e:
                print(f"\n  [下载] 第{page}页请求失败: {e}")

        for page, items in all_pages:
            if stop_evt.is_set():
                break
            page_count = len(items)
            page_bytes = 0
            page_ok = 0
            page_fail = 0
            print(f"\n  [下载] 第 {page}/{total_pages} 页 ({page_count} 条)")
            for idx, item in enumerate(items, 1):
                if stop_evt.is_set():
                    break

                img_url_raw = (item.get('img') or '').split('?')[0]
                if not img_url_raw:
                    with st_lock:
                        st['download_fail'] += 1
                    page_fail += 1
                    continue

                category = get_category(item)
                name_en = (item.get('nameEn') or '').strip()[:100]
                name_cn = (item.get('name') or '').strip()[:200]
                product_id = str(item.get('id', ''))
                keywords_cn, keywords_en = build_keywords(name_cn, name_en)

                try:
                    img_url_download = img_url_raw + OBS_IMAGE_PROCESS
                    img_resp = requests.get(img_url_download, timeout=30)
                    img_resp.raise_for_status()

                    dl_queue.put({
                        'image_bytes': img_resp.content,
                        'image_url': img_url_raw,
                        'category': category[:50],
                        'product_name': name_en,
                        'product_name_cn': name_cn,
                        'product_id': product_id,
                        'keywords_cn': keywords_cn,
                        'keywords_en': keywords_en,
                    })

                    img_size = len(img_resp.content)
                    page_bytes += img_size
                    page_ok += 1
                    with st_lock:
                        st['downloaded'] += 1
                        st['total_bytes'] += img_size
                        st['queue_size'] = dl_queue.qsize()

                    # 每10张打印单张大小
                    if page_ok % 10 == 0:
                        avg = page_bytes / page_ok
                        print(f"    [{page_ok}/{page_count}] "
                              f"avg {_fmt_size(int(avg))}/张, "
                              f"本页累计 {_fmt_size(page_bytes)}")

                except Exception as e:
                    page_fail += 1
                    with st_lock:
                        st['download_fail'] += 1
                        if st['download_fail'] <= 3:
                            print(f"    ERR: {e}")

            # 每页结束打汇总
            if page_ok > 0:
                avg = page_bytes / page_ok
                print(f"  [下载] 第{page}页完成: {page_ok}张 {_fmt_size(page_bytes)} "
                      f"(avg {_fmt_size(int(avg))}/张) | 失败 {page_fail}")

        dl_queue.put(None)  # 结束信号
        with st_lock:
            st['download_done'] = True
        print(f"\n  [下载] 全部完成 ✓")

    # 在 process_worker 中更新 queue_size
    def process_worker_with_progress(dl_queue, st, st_lock, stop_evt):
        """消费者 + 更新队列大小统计"""
        engine = get_engine()
        extractor = get_extractor()
        batch_buffer = []

        def flush_batch():
            nonlocal batch_buffer
            if not batch_buffer:
                return
            batch_bytes = [b['image_bytes'] for b in batch_buffer]
            try:
                features = extractor.extract_both_batch(batch_bytes)
            except Exception as e:
                print(f"\n  [转换] 批量失败: {e}，逐张降级")
                for item in batch_buffer:
                    try:
                        f = extractor.extract_both(item['image_bytes'])
                        _add_single_item(engine, item, f, st, st_lock)
                    except Exception:
                        with st_lock:
                            st['fail'] += 1
                batch_buffer.clear()
                return

            for j, item in enumerate(batch_buffer):
                try:
                    if image_url_exists(item['image_url']):
                        with st_lock:
                            st['fail'] += 1
                        continue
                    faiss_id = engine.allocate_id()
                    engine.add_single(features['clip'][j], features['resnet'][j], faiss_id)
                    info = get_image_info(item['image_bytes'])
                    insert_image(
                        faiss_id=faiss_id, image_url=item['image_url'],
                        category=item['category'], product_name=item['product_name'],
                        product_name_cn=item['product_name_cn'],
                        product_id=item['product_id'],
                        keywords_cn=item['keywords_cn'],
                        keywords_en=item['keywords_en'],
                        file_size=len(item['image_bytes']),
                        width=info.get('width', 0), height=info.get('height', 0),
                    )
                    with st_lock:
                        st['success'] += 1
                except Exception:
                    with st_lock:
                        st['fail'] += 1
            batch_buffer.clear()

        def _add_single_item(eng, item, features, st, st_lock):
            if image_url_exists(item['image_url']):
                return
            faiss_id = eng.allocate_id()
            eng.add_single(features['clip'], features['resnet'], faiss_id)
            info = get_image_info(item['image_bytes'])
            insert_image(
                faiss_id=faiss_id, image_url=item['image_url'],
                category=item['category'], product_name=item['product_name'],
                product_name_cn=item['product_name_cn'],
                product_id=item['product_id'],
                keywords_cn=item['keywords_cn'],
                keywords_en=item['keywords_en'],
                file_size=len(item['image_bytes']),
                width=info.get('width', 0), height=info.get('height', 0),
            )
            with st_lock:
                st['success'] += 1

        while True:
            try:
                item = dl_queue.get(timeout=2)
            except queue.Empty:
                flush_batch()
                with st_lock:
                    st['queue_size'] = dl_queue.qsize()
                continue

            if item is None:
                flush_batch()
                break

            batch_buffer.append(item)
            if len(batch_buffer) >= BATCH_SIZE:
                flush_batch()
            with st_lock:
                st['queue_size'] = dl_queue.qsize()

    # 启动：进度线程 + 下载线程 + 处理线程
    print(f"\n{'='*55}")
    print(f"  并行同步: 下载 ⇄ 转换")
    print(f"  预计总数: {total_products} 张, {total_pages} 页")
    print(f"  队列容量: {DOWNLOAD_QUEUE_SIZE}, 批量: {BATCH_SIZE}")
    print(f"{'='*55}\n")

    progress_thread = threading.Thread(
        target=progress_reporter,
        args=(stats, stats_lock, stop_event, t0),
        name='progress', daemon=True,
    )
    download_thread = threading.Thread(
        target=download_worker_with_first_page,
        args=(download_queue, stats, stats_lock, stop_event, first_page.get('items', [])),
        name='downloader', daemon=True,
    )
    process_thread = threading.Thread(
        target=process_worker_with_progress,
        args=(download_queue, stats, stats_lock, stop_event),
        name='processor', daemon=True,
    )

    progress_thread.start()
    download_thread.start()
    process_thread.start()

    download_thread.join()
    process_thread.join()
    stop_event.set()
    progress_thread.join(timeout=4)

    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print(f"  下载: {stats['downloaded']} 成功 / {stats['download_fail']} 失败")
    print(f"  总下载量: {_fmt_size(stats['total_bytes'])}")
    print(f"  入库: {stats['success']} 成功 / {stats['fail']} 失败")
    print(f"  耗时: {_fmt_time(elapsed)}")
    print(f"  索引: {engine.total} 个向量")
    print(f"{'='*55}")

    if engine.total > 0:
        print(f"\n  正在保存索引...")
        engine.save()
        print(f"  索引已保存 ✓")


if __name__ == '__main__':
    main()
