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

### 1. 流式 WebSocket API (推荐用于实时生肉字幕与桌面客户端)
- **端点**：
  - `ws://<服务端IP>:8765/v1/audio/stream` 或 `ws://<服务端IP>:8765/ws` (HTTP 普通通道)
  - `wss://<服务端IP>:8766/v1/audio/stream` 或 `wss://<服务端IP>:8766/ws` (TLS 加密通道，麦克风权限安全要求)
- **上行音频推流**：
  - **格式**：纯二进制 PCM 裸流（`16000Hz`、`16-bit Signed LE`、单声道 `Mono`）。
  - **发送节奏**：建议客户端每 `100ms ~ 250ms` 推送一个数据包（约 `3200 ~ 8000` 字节）。
- **上行控制指令 (JSON 文本)**：
  - 可随时通过 WebSocket 发送文本指令动态切换或锁定识别语言，避免背景噪音自动识别跑偏：
    ```json
    { "action": "set_language", "language": "en" }
    ```
    > 支持参数：`"auto"` (多语言自动识别), `"en"` (英语), `"ja"` (日语), `"zh"` (中文), `"ko"` (韩语), `"yue"` (粤语)。

#### 下行数据生命周期与协议设计 (打字机流式模型)
为了彻底解决“整句说完才出字导致体感等待2~3秒”的痛点，Sub-Pilot 采用**双轨流式状态机**：
1. **打字机中间增量 (`is_final: false`, `stage: "partial"`)**：
   - 用户说话期间，服务端每 ~250ms 计算一次中间识别结果并立即下发。
   - 客户端此时应该**覆盖更新**当前未定稿的预览文本，获得实时“打字机逐字输出”视觉体验，感知延迟降低到 **200ms**。
2. **原音定稿 (`is_final: true`, `stage: "asr"`)**：
   - 说话停顿（VAD 判定断句），生成该句最终的完整高精度 ASR 识别文本。
3. **翻译补全 (`is_final: true`, `stage: "trans"`)**：
   - 异步翻译完成后下发，带有对应的译文（`translated`），此时客户端将其替换为最终双语字幕。

#### 下行消息 JSON 结构 (SubtitlePayload)：
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | `number` (u64) | 句子唯一序列号。同一句子的 `partial`、`asr` 和 `trans` 具有相同或对应的行标识。 |
| `is_final` | `boolean` | **核心状态位**：`false` 表示正在说话的临时打字机增量；`true` 表示句子已定稿。 |
| `stage` | `string` | 当前生命周期阶段：`"partial"` (实时打字机) / `"asr"` (原文定稿) / `"trans"` (翻译完成)。 |
| `text` | `string` | ASR 原文识别结果。 |
| `translated` | `string` | 中文翻译文本（在 `stage: "trans"` 时填充）。 |
| `language` | `string` | 识别出的语种代码（如 `en`、`ja`、`zh`、`ko`）。 |
| `latency_ms` | `number` | 推理与翻译耗时统计（毫秒）。 |

#### 下行报文示例：
```json
// 1. 正在说话中（实时打字机流式增量，持续刷新当前行）
{
  "id": 1,
  "is_final": false,
  "stage": "partial",
  "text": "Welcome to New York",
  "translated": "",
  "language": "en",
  "latency_ms": 0
}

// 2. 说话停顿定稿（VAD 切片终态确定，原音定稿上屏）
{
  "id": 1,
  "is_final": true,
  "stage": "asr",
  "text": "Welcome to New York, it's been waiting for you.",
  "translated": "",
  "language": "en",
  "latency_ms": 86
}

// 3. 异步翻译完成（毫秒级回调意译，与 id=1 绑定补全）
{
  "id": 1,
  "is_final": true,
  "stage": "trans",
  "text": "Welcome to New York, it's been waiting for you.",
  "translated": "欢迎来到纽约，它一直在等待着你。",
  "language": "en",
  "latency_ms": 389
}
```

#### Python 客户端标准对接示例：
```python
import asyncio
import websockets
import json

async def run_client():
    uri = "ws://127.0.0.1:8765/v1/audio/stream"
    async with websockets.connect(uri) as ws:
        # 可选：锁定识别英语
        await ws.send(json.dumps({"action": "set_language", "language": "en"}))

        # 下行监听任务
        async def receive_subtitles():
            async for message in ws:
                data = json.loads(message)
                msg_id = data["id"]
                is_final = data["is_final"]
                stage = data["stage"]
                text = data["text"]
                translated = data["translated"]

                if not is_final:
                    # 打字机实时刷新（光标不换行）
                    print(f"\r⚡ [正在输入 #{msg_id}] {text} ...", end="", flush=True)
                else:
                    if stage == "asr":
                        print(f"\n✅ [定稿原文 #{msg_id}] {text}")
                    elif stage == "trans":
                        print(f"🎬 [翻译完成 #{msg_id}] {translated} (总耗时: {data['latency_ms']}ms)")

        asyncio.create_task(receive_subtitles())

        # 上行推流：读取 16kHz 16-bit Mono PCM 文件推流
        with open("test.pcm", "rb") as f:
            chunk_size = 4800  # 约 150ms 音频块
            while chunk := f.read(chunk_size):
                await ws.send(chunk)
                await asyncio.sleep(0.15)

        # 等待最后几句识别与翻译返回
        await asyncio.sleep(2.0)

asyncio.run(run_client())
```

---

### 2. OpenAI 兼容 REST API
可直接配置给任何支持自定义 OpenAI Whisper 端点的第三方客户端或插件（如 Pot-Desktop, Bob, 播放器插件等）：
- **端点**：`POST http://<服务端IP>:8765/v1/audio/transcriptions`
- **请求格式**：`multipart/form-data`（包含 `file=@audio.wav` 或 `.pcm`）

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
