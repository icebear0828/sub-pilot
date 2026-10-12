# ⚡ Sub-Pilot (字幕领航员)

> **超低延迟局域网实时生肉字幕 & 多语言 ASR 推理服务端**  
> 基于 **Rust (Tokio + Axum)** 构建，底层直连 **`sherpa-onnx`** 原生 C++ 驱动，搭载阿里巴巴 **`SenseVoice-Small`** 多语言语音识别模型。

---

## 🌟 核心特性

- 🚀 **极致低延迟 (Ultra-low Latency)**：
  - 非自回归 (Non-Autoregressive) 架构，**单句推理仅需 20ms ~ 30ms**（实时率 RTF < 0.03）。
  - 比 Whisper 快 15 倍以上，完全解决看生肉视频音画不同步的痛点。
- 🌏 **原生多语言自动识别 & 混杂支持**：
  - 精通 **中文、英语、日语、韩语、粤语**，支持同一句内中英、中日自然混杂与中英文人名识别。
  - 自动输出精确语种代码（`ja` / `en` / `zh` / `ko`），无需前端手动切语言。
- 🦀 **Rust 高性能原生后端**：
  - **Zero-GC Jitter**：杜绝 Python 解释器周期性垃圾回收引起的字幕卡顿掉帧。
  - **超低资源占用**：编译产物为单个独立二进制（~28MB），常驻内存仅 **20MB ~ 30MB**，空闲 CPU 占用 **0%**。
- 📡 **双协议 API 接口**：
  - **WebSocket 流式推流 API** (`/v1/audio/stream`)：全双工裸 PCM 音频流推流，毫秒级回传双语字幕。
  - **OpenAI 兼容 REST API** (`/v1/audio/transcriptions`)：标准 Multipart 接口，直接兼容各类现有三方字幕软件与播放器插件。
- 🖥️ **内置轻量响应式 Web 悬浮端**：
  - 单文件二进制内嵌纯静态 HTML5/CSS3 字幕页，局域网内的 iPad / 手机 / 电脑浏览器打开即用。

---

## 🏗️ 架构概览

```
  【客户端应用 (桌面悬浮板 / 手机 / 播放器)】
                      │
                      │ 持续发送 16kHz 16-bit PCM 音频帧 (WebSocket)
                      ▼
   ┌──────────────────────────────────────────────┐
   │         Sub-Pilot (Rust 后端服务)            │
   │  ┌────────────────────────────────────────┐  │
   │  │ 异步流式收发通道 (Axum / Tokio)        │  │
   │  └───────────────────┬────────────────────┘  │
   │                      ▼                       │
   │  ┌────────────────────────────────────────┐  │
   │  │ 原生 ASR 引擎 (sherpa-onnx C++ 绑定)    │  │
   │  │ 模型: SenseVoice-Small ONNX (20~30ms)   │  │
   │  └───────────────────┬────────────────────┘  │
   │                      ▼                       │
   │  ┌────────────────────────────────────────┐  │
   │  │ 多语言检测 & 自动路由翻译               │  │
   │  └────────────────────────────────────────┘  │
   └──────────────────────┬───────────────────────┘
                          │ 广播 JSON 双语字幕消息
                          ▼
  【客户端实时渲染双语置顶字幕 (延迟 < 100ms)】
```

---

## 🚀 快速开始

### 1. 环境准备
- 操作系统：macOS (Apple Silicon / Intel) 或 Linux (x86_64 / aarch64)
- 工具链：安装 [Rust](https://rustup.rs/) (>= 1.75) 与 `curl`、`tar`

### 2. 克隆仓库与拉取模型
```bash
git clone https://github.com/icebear0828/sub-pilot.git
cd sub-pilot

# 一键下载解压 SenseVoice-Small ONNX 官方预转换模型包
./download_model.sh
```

### 3. 编译并启动
```bash
# 方式 A：直接运行启动脚本（自动检测局域网 IP 与模型）
./start.sh

# 方式 B：使用 Cargo 运行
cargo run --release
```

服务默认监听在 `http://0.0.0.0:8765`。

---

## 🔌 API 接口规范与客户端对接

### 1. 流式 WebSocket API (推荐用于实时生肉字幕)
- **端点**：`ws://<服务端IP>:8765/v1/audio/stream`
- **上行推流格式**：纯二进制 PCM 裸流（`16000Hz`、`16-bit Signed LE`、单声道 `Mono`），建议每次推送 `0.3s ~ 0.5s`（约 9600 ~ 16000 字节）。
- **下行响应 (JSON 文本)**：
  ```json
  {
    "text": "うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う。",
    "translated": "[JA] うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う。",
    "language": "ja",
    "latency_ms": 28
  }
  ```

#### Python 客户端示例：
```python
import asyncio
import websockets
import json

async def run_client():
    uri = "ws://127.0.0.1:8765/v1/audio/stream"
    async with websockets.connect(uri) as ws:
        async def receive_subtitles():
            async for message in ws:
                data = json.loads(message)
                print(f"[{data['language'].upper()}] {data['text']} (推理耗时: {data['latency_ms']}ms)")

        asyncio.create_task(receive_subtitles())

        # 模拟读取 16kHz PCM 音频推流
        with open("test.pcm", "rb") as f:
            while chunk := f.read(9600):
                await ws.send(chunk)
                await asyncio.sleep(0.3)

asyncio.run(run_client())
```

---

### 2. OpenAI 兼容 REST API
可直接配置给任何支持自定义 OpenAI Whisper 端点的第三方客户端或插件：
- **端点**：`POST http://<服务端IP>:8765/v1/audio/transcriptions`
- **请求格式**：`multipart/form-data`（包含 `file=@audio.wav`）

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/transcriptions \
  -F "file=@test.wav"
```
**返回**：
```json
{
  "text": "开放时间早上9点至下午5点。",
  "language": "zh"
}
```

---

## ⚙️ 环境变量配置

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | 服务端绑定的主机 IP |
| `PORT` | `8765` | 服务端监听的端口号 |
| `MODEL_DIR` | `./models/...` | 自定义 SenseVoice 模型存放目录 |

---

## 📄 开源许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## ❤️ 致谢
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 提供优秀的跨平台原生语音推理框架。
- [FunASR / SenseVoice](https://github.com/modelscope/FunASR) 阿里巴巴通义实验室开源的高性能多语言语音基础模型。
