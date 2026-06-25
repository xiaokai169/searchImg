"""
产品图批量导入脚本
用法: python import_products.py <图片目录> [品类]

示例:
  python import_products.py D:/产品图/包包  包包
  python import_products.py D:/产品图/鞋子  鞋子
  python import_products.py D:/产品图        其他    # 不分类
"""
import os
import sys
import time
import requests

BASE_URL = "http://localhost:5000"
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def import_directory(directory: str, category: str = "其他"):
    """扫描目录，批量导入图片"""
    # 收集图片文件
    image_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXT:
                image_files.append(os.path.join(root, f))

    total = len(image_files)
    if total == 0:
        print(f"目录中没有图片: {directory}")
        return

    print(f"发现 {total} 张图片")
    print(f"品类: {category}")
    print(f"{'='*50}")

    success = 0
    failed = 0
    t_start = time.time()

    for i, path in enumerate(image_files, 1):
        try:
            with open(path, 'rb') as f:
                resp = requests.post(
                    f"{BASE_URL}/api/add_image",
                    files={'file': (os.path.basename(path), f.read(), 'image/jpeg')},
                    data={'category': category},
                    timeout=30,
                )

            if resp.json().get('code') == 0:
                success += 1
                if i % 10 == 0 or i == total:
                    elapsed = time.time() - t_start
                    speed = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / speed if speed > 0 else 0
                    print(f"  [{i}/{total}] {success}成功 {failed}失败 "
                          f"速度{speed:.1f}张/秒 剩余{eta:.0f}秒")
            else:
                failed += 1
                print(f"  [{i}] 失败: {resp.json().get('msg')}")

        except Exception as e:
            failed += 1
            print(f"  [{i}] 异常: {e}")

    elapsed = time.time() - t_start
    print(f"\n完成! {success}成功 {failed}失败 耗时{elapsed:.1f}秒")
    print(f"平均速度: {total/elapsed:.1f} 张/秒")

    # 查看最终统计
    resp = requests.get(f"{BASE_URL}/api/stats")
    if resp.json().get('code') == 0:
        stats = resp.json()['data']
        print(f"数据库总图片数: {stats['db_records']}")
        print(f"索引向量数: {stats['indexed_vectors']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "其他"

    if not os.path.isdir(directory):
        print(f"错误: 目录不存在 - {directory}")
        sys.exit(1)

    import_directory(directory, category)
