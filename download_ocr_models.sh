#!/bin/bash
# 下载 easyocr 模型（Gitee 优先 → HF → GitHub）
# 用法: bash download_ocr_models.sh
set -e

MODEL_DIR="$HOME/.EasyOCR/model"
mkdir -p "$MODEL_DIR"

# 三个源，按优先级排列
MIRRORS=(
  "gitee|https://gitee.com/mirrors/easyocr-models/raw/main"
  "hf|https://huggingface.co/itextresearch/itext-EasyOCR/resolve/main"
  "gh|https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6"
)

MODELS=(
  "craft_mlt_25k:craft_mlt_25k.pth"
  "english_g2:english_g2.pth"
  "arabic:arabic.pth"
  "zh_sim_g2:zh_sim_g2.pth"
)

download_one() {
    local name="$1"
    local file="$2"
    local target="$MODEL_DIR/$file"

    if [ -f "$target" ]; then
        echo "  ✅ $name 已存在 ($(du -h "$target" | cut -f1))"
        return
    fi

    echo "  ⬇ $name ..."

    for entry in "${MIRRORS[@]}"; do
        local label="${entry%%|*}"
        local base="${entry##*|}"
        local url="${base}/${name}.zip"

        if wget -q --show-progress --timeout=30 \
            -O "$MODEL_DIR/${name}.zip" "$url" 2>/dev/null; then
            echo "    ✓ $label"
            unzip -oq "$MODEL_DIR/${name}.zip" -d "$MODEL_DIR"
            rm "$MODEL_DIR/${name}.zip"
            echo "  ✅ $name 完成"
            return
        fi
        echo "    ✗ $label 不可用"
    done

    echo "  ❌ $name 所有源均失败"
    exit 1
}

echo "[OCR] 预下载模型 → $MODEL_DIR (Gitee → HF → GitHub)"
echo ""

for m in "${MODELS[@]}"; do
    name="${m%%:*}"
    file="${m##*:}"
    download_one "$name" "$file"
done

echo ""
echo "  全部就绪:"
ls -lh "$MODEL_DIR/"*.pth 2>/dev/null || echo "  (未找到 .pth 文件)"
