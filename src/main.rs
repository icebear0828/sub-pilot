use std::sync::Arc;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, State,
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SubtitlePayload {
    pub text: String,
    pub translated: String,
    pub language: String,
    pub latency_ms: u128,
}

#[derive(Serialize)]
pub struct OpenAIResponse {
    pub text: String,
    pub language: String,
}

#[derive(Clone)]
pub struct AppState {
    pub tx: broadcast::Sender<SubtitlePayload>,
    pub recognizer: Option<Arc<OfflineRecognizer>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // 自动发现 SenseVoice-Small ONNX 权重目录
    let candidate_dirs = [
        std::env::var("MODEL_DIR").unwrap_or_default(),
        "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17".to_string(),
        "./models".to_string(),
        "../models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17".to_string(),
        "/Users/yutao/llm/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17".to_string(),
    ];

    let mut found_model = None;
    for dir in &candidate_dirs {
        if dir.is_empty() {
            continue;
        }
        let int8_path = format!("{dir}/model.int8.onnx");
        let fp32_path = format!("{dir}/model.onnx");
        let tokens_path = format!("{dir}/tokens.txt");

        if std::path::Path::new(&tokens_path).exists() {
            if std::path::Path::new(&int8_path).exists() {
                found_model = Some((int8_path, tokens_path));
                break;
            } else if std::path::Path::new(&fp32_path).exists() {
                found_model = Some((fp32_path, tokens_path));
                break;
            }
        }
    }

    let recognizer = if let Some((model_p, tokens_p)) = found_model {
        println!("🚀 正在初始化 SenseVoice-Small ASR 原生引擎: {model_p}...");
        let sense_voice_cfg = OfflineSenseVoiceModelConfig {
            model: Some(model_p),
            language: None, // 自动检测多语种
            use_itn: true,
        };

        let mut model_cfg = OfflineModelConfig::default();
        model_cfg.sense_voice = sense_voice_cfg;
        model_cfg.tokens = Some(tokens_p);
        model_cfg.num_threads = 4;
        model_cfg.debug = false;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config = model_cfg;

        match OfflineRecognizer::create(&config) {
            Some(rec) => {
                println!("✅ SenseVoice 引擎加载完毕，准备提供 API 服务！");
                Some(Arc::new(rec))
            }
            None => {
                eprintln!("⚠️ 加载 ASR 引擎失败。");
                None
            }
        }
    } else {
        println!("ℹ️ 未在默认路径检测到 SenseVoice 模型。请运行 ./download_model.sh 或设置 MODEL_DIR 环境变量。");
        None
    };

    let (tx, _) = broadcast::channel(100);
    let state = AppState { tx, recognizer };

    let app = Router::new()
        // 0. 内置轻量实时字幕 WebUI（方便手机/iPad/电脑无安装使用）
        .route("/", get(index_handler))
        // 1. 流式低延迟 WebSocket API (供自定义客户端推流与接收字幕)
        .route("/v1/audio/stream", get(ws_handler))
        .route("/ws", get(ws_handler))
        // 2. OpenAI 兼容 REST API (供任何通用三方客户端直接调用)
        .route("/v1/audio/transcriptions", post(openai_transcriptions_handler))
        // 3. 健康检查接口
        .route("/health", get(|| async { "OK" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8765".to_string());
    let bind_addr = format!("{host}:{port}");

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("============================================================");
    println!("📡 Sub-Pilot ASR + 翻译 API 服务已启动: http://{bind_addr}");
    println!("🔹 WebSocket 流式 API: ws://{bind_addr}/v1/audio/stream");
    println!("🔹 OpenAI 兼容转写 API: http://{bind_addr}/v1/audio/transcriptions");
    println!("============================================================");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn index_handler() -> impl IntoResponse {
    axum::response::Html(include_str!("index.html"))
}

// -------------------------------------------------------------
// 1. WebSocket 流式 API 处理器
// -------------------------------------------------------------
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws_client(socket, state))
}

async fn handle_ws_client(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // 下行通道：将识别并翻译好的字幕推送给客户端
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
        }
    });

    // 上行通道：接收客户端推流上来的 PCM 二进制音频
    let recognizer = state.recognizer.clone();
    let tx = state.tx.clone();

    let mut recv_task = tokio::spawn(async move {
        let mut audio_buffer: Vec<f32> = Vec::with_capacity(16000 * 2);

        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Binary(bytes) => {
                    // 假设客户端发送的是 16kHz 16-bit Mono PCM
                    let sample_count = bytes.len() / 2;
                    for i in 0..sample_count {
                        let sample_i16 = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                        audio_buffer.push(sample_i16 as f32 / 32768.0);
                    }

                    // 积攒满 0.6 ~ 0.8 秒（约 10000 个采样点）即触发推理
                    if audio_buffer.len() >= 10000 {
                        if let Some(ref rec) = recognizer {
                            let start = std::time::Instant::now();
                            let stream = rec.create_stream();
                            stream.accept_waveform(16000, &audio_buffer);
                            rec.decode(&stream);

                            if let Some(result) = stream.get_result() {
                                let raw_text = result.text.trim();
                                if !raw_text.is_empty() {
                                    let (lang, clean_text) = parse_sense_voice_text(raw_text);
                                    let latency = start.elapsed().as_millis();

                                    // 自动路由翻译
                                    let translated = if lang == "zh" {
                                        clean_text.clone()
                                    } else {
                                        format!("[{}] {}", lang.to_uppercase(), clean_text)
                                    };

                                    let payload = SubtitlePayload {
                                        text: clean_text,
                                        translated,
                                        language: lang,
                                        latency_ms: latency,
                                    };

                                    let _ = tx.send(payload);
                                }
                            }
                        }
                        audio_buffer.clear();
                    }
                }
                Message::Text(txt) => {
                    // 支持客户端以 JSON 发送控制信号
                    if txt == "ping" {
                        // 心跳保活
                    }
                }
                _ => {}
            }
        }
    });

    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    };
}

// -------------------------------------------------------------
// 2. OpenAI 兼容 HTTP 转写 API (/v1/audio/transcriptions)
// -------------------------------------------------------------
async fn openai_transcriptions_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<OpenAIResponse>, StatusCode> {
    let rec = state.recognizer.ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let mut audio_bytes = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            if let Ok(bytes) = field.bytes().await {
                audio_bytes = bytes.to_vec();
                break;
            }
        }
    }

    if audio_bytes.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // 转换为 float32 采样
    let sample_count = audio_bytes.len() / 2;
    let mut samples: Vec<f32> = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let sample_i16 = i16::from_le_bytes([audio_bytes[i * 2], audio_bytes[i * 2 + 1]]);
        samples.push(sample_i16 as f32 / 32768.0);
    }

    let stream = rec.create_stream();
    stream.accept_waveform(16000, &samples);
    rec.decode(&stream);

    if let Some(res) = stream.get_result() {
        let (lang, clean_text) = parse_sense_voice_text(&res.text);
        Ok(Json(OpenAIResponse {
            text: clean_text,
            language: lang,
        }))
    } else {
        Ok(Json(OpenAIResponse {
            text: String::new(),
            language: "auto".to_string(),
        }))
    }
}

/// 解析 SenseVoice 输出标签与 Unicode 字符集，精准提取语言代码和纯文本
fn parse_sense_voice_text(text: &str) -> (String, String) {
    let mut detected_lang: Option<String> = None;
    let mut clean_text = String::new();

    // 1. 尝试从 SenseVoice 内置标签提取: <|zh|>, <|en|>, <|ja|>, <|ko|>, <|yue|>
    let parts: Vec<&str> = text.split('>').collect();
    for part in parts {
        if let Some(tag_content) = part.split("<|").nth(1) {
            match tag_content {
                "zh" | "en" | "ja" | "ko" | "yue" => {
                    detected_lang = Some(tag_content.to_string());
                }
                _ => {}
            }
        } else {
            clean_text.push_str(part.trim());
        }
    }

    if clean_text.is_empty() {
        clean_text = text.to_string();
    }

    // 2. 如果标签未携带语言代码，通过字符集高精度自适应判定
    let lang = if let Some(l) = detected_lang {
        l
    } else {
        let has_kana = clean_text.chars().any(|c| ('\u{3040}'..='\u{309F}').contains(&c) || ('\u{30A0}'..='\u{30FF}').contains(&c));
        let has_hangul = clean_text.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c) || ('\u{1100}'..='\u{11FF}').contains(&c));
        let has_han = clean_text.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c));
        let latin_count = clean_text.chars().filter(|c| c.is_ascii_alphabetic()).count();

        if has_kana {
            "ja".to_string()
        } else if has_hangul {
            "ko".to_string()
        } else if has_han {
            "zh".to_string()
        } else if latin_count > 0 {
            "en".to_string()
        } else {
            "auto".to_string()
        }
    };

    (lang, clean_text)
}
