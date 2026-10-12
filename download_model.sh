#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$DIR/models"
MODEL_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
TARGET_DIR="$MODELS_DIR/$MODEL_NAME"

echo "============================================================"
echo "📦 Sub-Pilot 模型下载器: SenseVoice-Small ONNX"
echo "============================================================"

if [ -f "$TARGET_DIR/tokens.txt" ] && ([ -f "$TARGET_DIR/model.int8.onnx" ] || [ -f "$TARGET_DIR/model.onnx" ]); then
    echo "✅ 模型已存在于 $TARGET_DIR，无需重复下载。"
    exit 0
fi

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

TAR_FILE="${MODEL_NAME}.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${TAR_FILE}"

echo "⬇️ 正在从官方源下载模型压缩包 (~990MB)..."
echo "源地址: $URL"

if command -v curl >/dev/null 2>&1; then
    curl -L -C - -O "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -c "$URL"
else
    echo "❌ 请先安装 curl 或 wget。"
    exit 1
fi

echo "📦 正在解压模型文件..."
tar -xvf "$TAR_FILE"
rm -f "$TAR_FILE"

echo "============================================================"
echo "🎉 模型已成功下载并解压到: $TARGET_DIR"
echo "👉 现在可以执行 ./start.sh 或 cargo run --release 启动服务！"
echo "============================================================"
