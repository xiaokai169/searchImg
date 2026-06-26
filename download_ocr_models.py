"""
预下载 easyocr 模型（优先 HF 镜像，国内更快）
用法: python download_ocr_models.py
"""
import os
import zipfile

MODEL_DIR = os.path.expanduser('~/.EasyOCR/model')

# (文件名前缀, 解压后.pth名)
MODELS = [
    ('craft_mlt_25k', 'craft_mlt_25k.pth'),
    ('english_g2',    'english_g2.pth'),
    ('arabic',        'arabic.pth'),
    ('zh_sim_g2',     'zh_sim_g2.pth'),
]

# 三个源：Gitee 优先 → HF → GitHub
URL_TEMPLATES = [
    'https://gitee.com/mirrors/easyocr-models/raw/main/{name}.zip',
    'https://huggingface.co/itextresearch/itext-EasyOCR/resolve/main/{name}.zip',
    'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/{name}.zip',
]


def download_one(name: str, target_file: str):
    from urllib.request import urlretrieve

    target = os.path.join(MODEL_DIR, target_file)
    if os.path.exists(target):
        size = os.path.getsize(target) / 1024 / 1024
        print(f"  ✅ {name} 已存在 ({size:.1f} MB)")
        return

    for tmpl in URL_TEMPLATES:
        url = tmpl.format(name=name)
        zip_path = os.path.join(MODEL_DIR, f'{name}.zip')
        try:
            print(f"  ⬇ {name} 下载中: {url[:60]}...")
            urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(MODEL_DIR)
            os.remove(zip_path)
            print(f"  ✅ {name} 完成")
            return
        except Exception as e:
            print(f"    失败: {e}，换下一个源...")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    raise RuntimeError(f"❌ {name} 所有源均下载失败")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, file in MODELS:
        download_one(name, file)
    print(f"\n  全部就绪 → {MODEL_DIR}")


if __name__ == '__main__':
    main()
