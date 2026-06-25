#!/bin/bash
# ========================================
# 以图搜图服务 — 启动脚本
# 用法:
#   bash start.sh          # 安装依赖并启动
#   bash start.sh install  # 仅安装依赖
#   bash start.sh run      # 仅启动服务
# ========================================

set -e

cd "$(dirname "$0")"

# Python 检查
PYTHON=python3
if ! command -v $PYTHON &>/dev/null; then
    PYTHON=python
fi

echo "Python: $($PYTHON --version)"

# 安装依赖
install_deps() {
    echo ""
    echo ">>> 安装 Python 依赖..."
    $PYTHON -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo ">>> 依赖安装完成"
}

# 启动服务
run_server() {
    echo ""
    echo "========================================"
    echo "  以图搜图服务启动中..."
    echo "  地址: http://0.0.0.0:5000"
    echo "  按 Ctrl+C 停止"
    echo "========================================"
    echo ""
    $PYTHON app.py
}

case "${1:-}" in
    install)
        install_deps
        ;;
    run)
        run_server
        ;;
    *)
        install_deps
        run_server
        ;;
esac
