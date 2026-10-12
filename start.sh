#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/target/release/sub-pilot"

# 检查模型是否存在
MODEL_FOUND=0
for cand in "$DIR/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" "$DIR/../models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" "/Users/yutao/llm/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"; do
    if [ -f "$cand/tokens.txt" ]; then
        MODEL_FOUND=1
        break
    fi
done

if [ "$MODEL_FOUND" -eq 0 ]; then
    echo "⚠️ 未在默认目录检测到 SenseVoice 模型，正在自动拉取..."
    "$DIR/download_model.sh"
fi

if [ ! -f "$BIN" ]; then
    echo "🔨 正在编译 sub-pilot release 版本..."
    (cd "$DIR" && cargo build --release)
fi

PORT="${PORT:-8765}"
HOST="${HOST:-0.0.0.0}"
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

echo "=========================================================="
echo "🚀 启动 Sub-Pilot 局域网实时生肉字幕系统"
echo "📡 本机访问: http://127.0.0.1:${PORT}"
echo "📱 局域网设备 (iPad/手机/电脑): http://${LAN_IP:-localhost}:${PORT}"
echo "🔹 WebSocket 流式 API: ws://${LAN_IP:-localhost}:${PORT}/v1/audio/stream"
echo "🔹 OpenAI 转写 API: http://${LAN_IP:-localhost}:${PORT}/v1/audio/transcriptions"
echo "=========================================================="

exec "$BIN"
