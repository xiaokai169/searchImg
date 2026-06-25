"""
产品名回填脚本 — 只查 API 不下载图片
用法:
  1. export ARAB_BEE_TOKEN="your_token"
  2. python backfill_names.py

从 Arab-Bee 接口拉取产品列表，按 image_url 匹配，更新本地 DB 的
product_name, product_name_cn, product_id, keywords_cn, keywords_en。
"""
import sys
import os
import requests
import sqlite3
from config import ARAB_BEE_API_URL, ARAB_BEE_TOKEN, DATABASE_PATH, SYNC_PAGE_SIZE, SYNC_MAX_PAGES
from text_utils import build_keywords

API_URL = ARAB_BEE_API_URL
TOKEN = ARAB_BEE_TOKEN

if not TOKEN:
    print("[ERROR] 请先设置 ARAB_BEE_TOKEN 环境变量")
    print('  export ARAB_BEE_TOKEN="your_jwt_token"')
    sys.exit(1)

# Token 可能已含 "Bearer " 前缀，统一处理
AUTH_HEADER = TOKEN if TOKEN.lower().startswith('bearer ') else f"Bearer {TOKEN}"


def build_remote_map():
    """从 API 拉取所有产品，构建 {image_url: {完整产品信息}} 映射"""
    url_map = {}
    for page in range(1, SYNC_MAX_PAGES + 1):
        print(f"  拉取第 {page} 页...")
        params = {"page": page, "size": SYNC_PAGE_SIZE, "state": "approved"}
        resp = requests.get(API_URL, params=params,
                            headers={"Authorization": AUTH_HEADER}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get('items', [])
        if not items:
            break
        for item in items:
            img_url = (item.get('img') or '').split('?')[0]
            name_en = (item.get('nameEn') or '').strip()[:100]
            name_cn = (item.get('name') or '').strip()[:200]
            product_id = str(item.get('id', ''))
            if img_url:
                keywords_cn, keywords_en = build_keywords(name_cn, name_en)
                url_map[img_url] = {
                    'product_name': name_en,
                    'product_name_cn': name_cn,
                    'product_id': product_id,
                    'keywords_cn': keywords_cn,
                    'keywords_en': keywords_en,
                }
        print(f"    本页 {len(items)} 条，累计 {len(url_map)} 条")
    return url_map


def main():
    print("=" * 50)
    print("  产品名回填 — 仅更新 product_name，不下载图片")
    print("=" * 50)

    # 1. 从 API 拉取映射
    print("\n[1/3] 从 API 拉取产品列表...")
    try:
        url_map = build_remote_map()
    except Exception as e:
        print(f"[ERROR] API 请求失败: {e}")
        sys.exit(1)

    if not url_map:
        print("[ERROR] 未获取到任何产品")
        sys.exit(1)
    print(f"  共获取 {len(url_map)} 个产品映射")

    # 2. 匹配本地 DB
    print("\n[2/3] 匹配本地数据库...")
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, image_url, product_name, product_name_cn, product_id, "
        "keywords_cn, keywords_en FROM images "
        "WHERE (product_name = '' OR product_name IS NULL) "
        "   OR (product_name_cn = '' OR product_name_cn IS NULL)"
    ).fetchall()
    print(f"  本地 {len(rows)} 条缺少产品信息")

    # 3. 更新
    print("\n[3/3] 更新...")
    updated = 0
    not_found = 0
    for row in rows:
        info = url_map.get(row['image_url'])
        if info:
            conn.execute(
                "UPDATE images SET product_name = ?, product_name_cn = ?, "
                "product_id = ?, keywords_cn = ?, keywords_en = ? WHERE id = ?",
                (info['product_name'], info['product_name_cn'],
                 info['product_id'], info['keywords_cn'],
                 info['keywords_en'], row['id'])
            )
            updated += 1
        else:
            not_found += 1

    conn.commit()
    conn.close()
    print(f"  完成: {updated} 条已更新, {not_found} 条未匹配")
    print(f"\n现在可以重启 python app.py 体验完整搜索效果")


if __name__ == '__main__':
    main()
