#!/bin/bash
# 用 wget 直接下载 easyocr 模型（无需安装 easyocr）
# 用法: bash download_ocr_models.sh

MODEL_DIR="$HOME/.EasyOCR/model"
mkdir -p "$MODEL_DIR"

download_and_unzip() {
    local name="$1"
    local url="$2"
    local target="$MODEL_DIR/$3"

    if [ -f "$target" ]; then
        echo "  ✅ $name 已存在"
        return
    fi

    echo "  ⬇ $name 下载中..."
    wget -q --show-progress -O "$MODEL_DIR/${name}.zip" "$url"
    unzip -oq "$MODEL_DIR/${name}.zip" -d "$MODEL_DIR"
    rm "$MODEL_DIR/${name}.zip"
    echo "  ✅ $name 完成"
}

echo "[OCR] 预下载模型到 $MODEL_DIR"

download_and_unzip "craft_mlt_25k" \
    "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip" \
    "craft_mlt_25k.pth"

download_and_unzip "english_g2" \
    "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/english_g2.zip" \
    "english_g2.pth"

download_and_unzip "arabic" \
    "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/arabic.zip" \
    "arabic.pth"

download_and_unzip "zh_sim_g2" \
    "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/zh_sim_g2.zip" \
    "zh_sim_g2.pth"

echo ""
echo "  全部就绪 → $MODEL_DIR"
ls -lh "$MODEL_DIR/"*.pth
