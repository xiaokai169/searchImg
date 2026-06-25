"""
从 Arab-Bee 产品接口同步产品到本地检索引擎
只下载图片提取特征，不存本地文件，元数据存 OBS URL
用法: python sync_products.py
"""
import sys
import os
import time
import requests

from config import (
    ARAB_BEE_API_URL, ARAB_BEE_TOKEN, SYNC_PAGE_SIZE, SYNC_MAX_PAGES,
    FLASK_HOST, FLASK_PORT, OBS_IMAGE_PROCESS,
)
from text_utils import build_keywords

API_URL = ARAB_BEE_API_URL
TOKEN = ARAB_BEE_TOKEN
LOCAL_API = "http://127.0.0.1:{0}".format(FLASK_PORT)
PAGE_SIZE = SYNC_PAGE_SIZE
MAX_PAGES = SYNC_MAX_PAGES


def fetch_products(page: int) -> dict:
    params = {"page": page, "size": PAGE_SIZE, "state": "approved"}
    resp = requests.get(API_URL, params=params,
                        headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
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


def sync_page(page: int):
    print(f"\n>>> 第 {page} 页...")
    data = fetch_products(page)
    items = data.get('items', [])
    print(f"    本页 {len(items)} 条")

    ok = fail = 0
    for i, item in enumerate(items, 1):
        img_url_raw = (item.get('img') or '').split('?')[0]  # 去 hash 参数
        category = get_category(item)
        name_en = (item.get('nameEn') or '').strip()[:100]
        name_cn = (item.get('name') or '').strip()[:200]
        product_id = str(item.get('id', ''))

        # 提取中英文搜索关键字
        keywords_cn, keywords_en = build_keywords(name_cn, name_en)

        if not img_url_raw:
            fail += 1
            continue

        # 华为云 OBS: 下载时拼接图片处理参数（加速+省流量），数据库存原始 URL
        img_url_download = img_url_raw + OBS_IMAGE_PROCESS

        try:
            # 1. 下载图片（拼接处理参数，800px + 85%质量，加速下载）
            img_resp = requests.get(img_url_download, timeout=30)
            img_resp.raise_for_status()
            image_bytes = img_resp.content

            # 2. 发送给本地 API：图片 bytes + 原始 OBS URL + 产品名 + 中文名 + 关键字
            resp = requests.post(f"{LOCAL_API}/api/add_image",
                files={'file': (img_url_raw.rsplit('/', 1)[-1], image_bytes, 'image/jpeg')},
                data={'image_url': img_url_raw, 'category': category[:50],
                      'product_name': name_en,
                      'product_name_cn': name_cn,
                      'product_id': product_id,
                      'keywords_cn': keywords_cn,
                      'keywords_en': keywords_en},
                timeout=60)

            if resp.json().get('code') == 0:
                ok += 1
                if ok % 10 == 0:
                    print(f"    [{ok}/{len(items)}] OK")
            else:
                fail += 1
                print(f"    [{i}] FAIL: {resp.json().get('msg')}")

        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"    [{i}] ERR: {e}")

    print(f"  本页: {ok} OK, {fail} FAIL")
    return ok, fail


def main():
    # 检查 Token 是否已配置
    if not TOKEN:
        print("[ERROR] 未设置 ARAB_BEE_TOKEN 环境变量")
        print("  请执行: export ARAB_BEE_TOKEN=\"your_jwt_token\"")
        sys.exit(1)

    try:
        r = requests.get(f"{LOCAL_API}/api/init", timeout=5)
        print(f"服务就绪: {r.json()['data'].get('db_records', 0)} 条")
    except Exception:
        print("[ERROR] 请先启动 python app.py")
        sys.exit(1)

    t0 = time.time()
    total_ok = total_fail = 0
    for page in range(1, MAX_PAGES + 1):
        o, f = sync_page(page)
        total_ok += o
        total_fail += f

    print(f"\n{'='*50}")
    print(f"  完成: {total_ok} OK, {total_fail} FAIL, {time.time()-t0:.0f}秒")
    r = requests.get(f"{LOCAL_API}/api/stats").json()['data']
    print(f"  索引: {r['db_records']}张, 品类{len(r['by_category'])}个")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
