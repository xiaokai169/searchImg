#!/bin/bash
# 下载 easyocr 模型（国内优先走 HF 镜像）
# 用法: bash download_ocr_models.sh
set -e

MODEL_DIR="$HOME/.EasyOCR/model"
mkdir -p "$MODEL_DIR"

# GitHub（原始）和 HF 镜像（国内更快）
gh="https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6"
hf="https://huggingface.co/itextresearch/itext-EasyOCR/resolve/main"

download_and_unzip() {
    local name="$1"
    local file="$2"
    local target="$MODEL_DIR/$file"

    if [ -f "$target" ]; then
        echo "  ✅ $name 已存在 ($(du -h "$target" | cut -f1))"
        return
    fi

    echo "  ⬇ $name 下载中..."

    # 优先 HF 镜像，不行再回退 GitHub
    if wget -q --show-progress --timeout=30 \
        -O "$MODEL_DIR/${name}.zip" "$hf/${name}.zip" 2>/dev/null; then
        :
    else
        echo "     HF 失败，换 GitHub..."
        wget -q --show-progress -O "$MODEL_DIR/${name}.zip" "$gh/${name}.zip"
    fi

    unzip -oq "$MODEL_DIR/${name}.zip" -d "$MODEL_DIR"
    rm "$MODEL_DIR/${name}.zip"
    echo "  ✅ $name 完成"
}

# 这四个文件能从 easyocr 源码的 download_utils.py 里确认是标准文件名
download_and_unzip "craft_mlt_25k"   "craft_mlt_25k.pth"
download_and_unzip "english_g2"      "english_g2.pth"
download_and_unzip "arabic"          "arabic.pth"
download_and_unzip "zh_sim_g2"       "zh_sim_g2.pth"

echo ""
echo "  全部就绪 → $MODEL_DIR"
ls -lh "$MODEL_DIR/"*.pth 2>/dev/null || echo "  (无 .pth 文件，请检查)"
