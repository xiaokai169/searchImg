"""
索引入库 — 双模型版，不存本地文件
图片从 OBS URL 下载 → 提取特征 → 入库（只存 URL）
"""
import json
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests

from config import BATCH_SIZE, INDEX_PROGRESS_FILE, ALLOWED_EXTENSIONS
from database import insert_image, image_url_exists, get_all_images
from engine import get_engine
from extractor import get_extractor
from preprocess import validate_image, get_image_info


def _download_image(url: str, timeout: int = 30) -> bytes:
    """从 URL 下载图片"""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def add_single_image(image_bytes: bytes, image_url: str,
                     category: str = '其他', product_name: str = '',
                     product_name_cn: str = '', product_id: str = '',
                     keywords_cn: str = '', keywords_en: str = '') -> dict:
    """入库单张图片，双模型特征提取，不存本地"""
    ok, err = validate_image(image_bytes)
    if not ok:
        raise ValueError(f"图片校验失败: {err}")

    if image_url_exists(image_url):
        raise ValueError(f"图片已存在: {image_url}")

    info = get_image_info(image_bytes)

    extractor = get_extractor()
    features = extractor.extract_both(image_bytes)

    engine = get_engine()
    faiss_id = engine.allocate_id()
    engine.add_single(features['clip'], features['resnet'], faiss_id)

    db_id = insert_image(
        faiss_id=faiss_id, image_url=image_url, category=category,
        product_name=product_name,
        product_name_cn=product_name_cn,
        product_id=product_id,
        keywords_cn=keywords_cn,
        keywords_en=keywords_en,
        file_size=len(image_bytes),
        width=info.get('width', 0), height=info.get('height', 0),
    )

    # 每入库一张就保存索引（防止进程被 kill 丢失）
    engine.save()

    return {
        "db_id": db_id, "faiss_id": faiss_id,
        "image_url": image_url, "category": category,
        "clip_dim": len(features['clip']),
        "resnet_dim": len(features['resnet']),
    }


def add_images_batch(items: list[tuple]) -> dict:
    """
    批量入库
    items: [(image_bytes, image_url, category, product_name,
             product_name_cn, product_id, keywords_cn, keywords_en), ...]
    """
    extractor = get_extractor()
    engine = get_engine()

    results = []
    success_count = 0
    failed_count = 0

    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        batch_bytes = [b for b, _, _, _, _, _, _, _ in batch]

        try:
            features = extractor.extract_both_batch(batch_bytes)
        except Exception as e:
            print(f"[Indexer] 批量提取失败: {e}，逐张降级")
            for img_bytes, url, cat, pname, pname_cn, pid, kw_cn, kw_en in batch:
                try:
                    r = add_single_image(img_bytes, url, cat, pname,
                                         pname_cn, pid, kw_cn, kw_en)
                    results.append(r)
                    success_count += 1
                except Exception as e2:
                    results.append({"image_url": url, "error": str(e2)})
                    failed_count += 1
            continue

        for j, (img_bytes, url, cat, pname, pname_cn, pid, kw_cn, kw_en) in enumerate(batch):
            try:
                if image_url_exists(url):
                    failed_count += 1
                    continue
                faiss_id = engine.allocate_id()
                engine.add_single(features['clip'][j], features['resnet'][j], faiss_id)
                info = get_image_info(img_bytes)
                db_id = insert_image(
                    faiss_id=faiss_id, image_url=url, category=cat,
                    product_name=pname,
                    product_name_cn=pname_cn,
                    product_id=pid,
                    keywords_cn=kw_cn,
                    keywords_en=kw_en,
                    file_size=len(img_bytes),
                    width=info.get('width', 0), height=info.get('height', 0),
                )
                results.append({"db_id": db_id, "faiss_id": faiss_id, "image_url": url, "category": cat})
                success_count += 1
            except Exception as e:
                results.append({"image_url": url, "error": str(e)})
                failed_count += 1

    return {"success": success_count, "failed": failed_count, "results": results}


# ==================== 后台批量索引 ====================

_index_progress = {
    "status": "idle", "total": 0, "processed": 0,
    "success": 0, "failed": 0, "message": "", "start_time": 0,
}
_progress_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='indexer')


def _update_progress(**kwargs):
    with _progress_lock:
        _index_progress.update(kwargs)
    with open(INDEX_PROGRESS_FILE, 'w') as f:
        json.dump(_index_progress, f, ensure_ascii=False)


def get_index_progress() -> dict:
    with _progress_lock:
        return dict(_index_progress)


def rebuild_index():
    """重建索引（从数据库 URL 重新下载图片提取特征）"""
    if _index_progress['status'] == 'running':
        raise RuntimeError("已有索引任务在运行中")

    _update_progress(status="running", total=0, processed=0, success=0, failed=0,
                     message="正在重建索引...", start_time=time.time())

    def _run():
        try:
            extractor = get_extractor()
            from engine import TwoStageEngine
            new_engine = TwoStageEngine()

            images = get_all_images()
            total = len(images)
            _update_progress(total=total, message=f"重建 {total} 条...")

            success = 0
            failed = 0
            for i in range(0, total, BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]
                batch_bytes = []
                batch_ids = []
                for img in batch:
                    try:
                        data = _download_image(img['image_url'])
                        batch_bytes.append(data)
                        batch_ids.append(img['faiss_id'])
                    except Exception:
                        failed += 1
                        continue
                if batch_bytes:
                    try:
                        features = extractor.extract_both_batch(batch_bytes)
                        new_engine.add(features['clip'], features['resnet'],
                                       np.array(batch_ids, dtype=np.int64))
                        success += len(batch_bytes)
                    except Exception:
                        failed += len(batch_bytes)
                _update_progress(processed=min(i + BATCH_SIZE, total), success=success, failed=failed)

            import engine as eng_mod
            eng_mod._engine = new_engine
            if images:
                new_engine._next_id = max(img['faiss_id'] for img in images) + 1
            new_engine.save()
            _update_progress(status="done", message=f"重建完成! {success}成功 {failed}失败")
        except Exception as e:
            _update_progress(status="error", message=str(e))

    _executor.submit(_run)
