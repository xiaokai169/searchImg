"""
预下载 easyocr 模型，两种方式任选：

方式1（Python）:  python download_ocr_models.py
方式2（Shell）:   bash download_ocr_models.sh

模型存放路径: ~/.EasyOCR/model/
"""
import os
import zipfile
import sys

MODEL_DIR = os.path.expanduser('~/.EasyOCR/model')

# easyocr 需要的模型（名称, 下载URL, 解压后的文件名）
MODELS = [
    ('craft_mlt_25k',
     'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip',
     'craft_mlt_25k.pth'),
    ('english_g2',
     'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/english_g2.zip',
     'english_g2.pth'),
    ('arabic',
     'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/arabic.zip',
     'arabic.pth'),
    ('zh_sim_g2',
     'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/zh_sim_g2.zip',
     'zh_sim_g2.pth'),
]


def download():
    from urllib.request import urlretrieve

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, url, target_file in MODELS:
        target = os.path.join(MODEL_DIR, target_file)
        if os.path.exists(target):
            size_mb = os.path.getsize(target) / 1024 / 1024
            print(f"  ✅ {name} 已存在 ({size_mb:.1f} MB)")
            continue

        zip_path = os.path.join(MODEL_DIR, f'{name}.zip')
        print(f"  ⬇ {name} 下载中...")
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(MODEL_DIR)
        os.remove(zip_path)
        print(f"  ✅ {name} 完成")

    print(f"\n  全部就绪 → {MODEL_DIR}")


if __name__ == '__main__':
    download()
