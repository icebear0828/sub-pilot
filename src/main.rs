use std::net::SocketAddr;
use std::sync::Arc;
use axum_server::tls_rustls::RustlsConfig;
use rcgen::generate_simple_self_signed;
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
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;

pub fn create_sense_voice_recognizer(
    model_path: &str,
    tokens_path: &str,
    language: Option<&str>,
) -> Option<OfflineRecognizer> {
    let sense_voice_cfg = OfflineSenseVoiceModelConfig {
        model: Some(model_path.to_string()),
        language: language.map(|s| s.to_string()),
        use_itn: true,
    };

    let mut model_cfg = OfflineModelConfig::default();
    model_cfg.sense_voice = sense_voice_cfg;
    model_cfg.tokens = Some(tokens_path.to_string());
    model_cfg.num_threads = 4;
    model_cfg.debug = false;

    let mut config = OfflineRecognizerConfig::default();
    config.model_config = model_cfg;

    OfflineRecognizer::create(&config)
}

pub fn create_vad_detector(vad_path: &str) -> Option<VoiceActivityDetector> {
    let silero_vad = SileroVadModelConfig {
        model: Some(vad_path.to_string()),
        threshold: 0.6,
        min_silence_duration: 0.75, // 0.75 秒完整停顿才视为句尾，绝不在换气或短暂停顿时将句子切碎！
        min_speech_duration: 0.35,  // 0.35 秒以下过滤，彻底消除碎片杂音和残破尾缀
        window_size: 512,
        max_speech_duration: 10.0,  // 单句最长 10 秒强制切片，保障完整语义
    };

    let mut config = VadModelConfig::default();
    config.silero_vad = silero_vad;
    config.sample_rate = 16000;
    config.num_threads = 2;

    VoiceActivityDetector::create(&config, 30.0)
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SubtitlePayload {
    pub id: u64,
    pub is_final: bool, // false: 中间增量打字机识别结果; true: 句子终态定稿
    pub stage: String,  // "partial" (实时打字机增量) | "asr" (句末原音定稿) | "trans" (最终意译完成)
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
pub struct ModelPaths {
    pub model_path: String,
    pub tokens_path: String,
    pub vad_path: Option<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub tx: broadcast::Sender<SubtitlePayload>,
    pub recognizer: Option<Arc<OfflineRecognizer>>,
    pub http_client: reqwest::Client,
    pub model_paths: Option<ModelPaths>,
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

    let mut model_paths = None;
    let recognizer = if let Some((model_p, tokens_p)) = found_model {
        let parent_dir = std::path::Path::new(&model_p).parent().unwrap();
        let vad_candidate = parent_dir.join("silero_vad.onnx");
        let vad_path = if vad_candidate.exists() {
            Some(vad_candidate.to_string_lossy().to_string())
        } else {
            None
        };

        model_paths = Some(ModelPaths {
            model_path: model_p.clone(),
            tokens_path: tokens_p.clone(),
            vad_path,
        });

        println!("🚀 正在初始化 SenseVoice-Small ASR 原生引擎: {model_p}...");
        create_sense_voice_recognizer(&model_p, &tokens_p, None).map(Arc::new)
    } else {
        println!("ℹ️ 未在默认路径检测到 SenseVoice 模型。请运行 ./download_model.sh 或设置 MODEL_DIR 环境变量。");
        None
    };

    let (tx, _) = broadcast::channel(100);
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(2000))
        .build()
        .unwrap_or_default();

    let state = AppState { tx, recognizer, http_client, model_paths };

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
    let http_addr: SocketAddr = format!("{host}:{port}").parse()?;

    // 生成局域网通用的自签名 TLS 证书（使手机/其他设备浏览器能够直接解锁 getUserMedia 录音权限）
    let san = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        host.clone(),
        "sub-pilot.local".to_string(),
    ];
    let cert_key = generate_simple_self_signed(san).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let cert_pem = cert_key.cert.pem().into_bytes();
    let key_pem = cert_key.signing_key.serialize_pem().into_bytes();
    let tls_config = RustlsConfig::from_pem(cert_pem, key_pem).await?;

    let https_port = std::env::var("HTTPS_PORT").unwrap_or_else(|_| "8766".to_string());
    let https_addr: SocketAddr = format!("{host}:{https_port}").parse()?;

    println!("============================================================");
    println!("📡 Sub-Pilot ASR + 翻译服务已就绪！");
    println!("🔒 HTTPS 安全通道 (推荐其他电脑/手机录音访问): https://{https_addr}");
    println!("🌐 HTTP 普通通道 (本机访问): http://{http_addr}");
    println!("🔹 WebSocket 流式 API: wss://{https_addr}/ws 或 ws://{http_addr}/ws");
    println!("============================================================");

    let app_http = app.clone();
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(http_addr).await.unwrap();
        axum::serve(listener, app_http).await.unwrap();
    });

    axum_server::bind_rustls(https_addr, tls_config)
        .serve(app.into_make_service())
        .await?;

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

    // 上行通道：接收客户端推流上来的 PCM 二进制音频，并支持动态语言控制
    let tx = state.tx.clone();
    let http_client = state.http_client.clone();
    let model_paths = state.model_paths.clone();

    // 每个连接维护独立的 Recognizer 与 VAD 状态，允许客户端独立锁定语种
    let mut current_lang: Option<String> = None;
    let mut current_recognizer = state.recognizer.clone();
    let mut vad_detector = model_paths.as_ref().and_then(|p| {
        p.vad_path.as_ref().and_then(|vad_p| create_vad_detector(vad_p))
    });

    let mut subtitle_seq: u64 = 0;

    let mut recv_task = tokio::spawn(async move {
        let mut audio_buffer: Vec<f32> = Vec::with_capacity(16000 * 4);
        let mut live_speech_buffer: Vec<f32> = Vec::with_capacity(16000 * 6);
        let mut last_partial_time = std::time::Instant::now();

        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Binary(bytes) => {
                    // 接收 16kHz 16-bit Mono PCM
                    let sample_count = bytes.len() / 2;
                    let mut chunk: Vec<f32> = Vec::with_capacity(sample_count);
                    for i in 0..sample_count {
                        let sample_i16 = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                        chunk.push(sample_i16 as f32 / 32768.0);
                    }

                    // 【阶段 2：流式语音检测与打字机输出】
                    if let Some(ref mut vad) = vad_detector {
                        vad.accept_waveform(&chunk);

                        // 2.1 说话中：打字机流式增量识别 (is_final: false)
                        if vad.detected() {
                            live_speech_buffer.extend_from_slice(&chunk);

                            // 每隔约 250ms 且累计至少 0.3s (4800采样点) 进行一次打字机快速推断
                            if last_partial_time.elapsed() >= std::time::Duration::from_millis(250)
                                && live_speech_buffer.len() >= 4800
                            {
                                last_partial_time = std::time::Instant::now();
                                if let Some(ref rec) = current_recognizer {
                                    let stream = rec.create_stream();
                                    stream.accept_waveform(16000, &live_speech_buffer);
                                    rec.decode(&stream);

                                    if let Some(result) = stream.get_result() {
                                        let raw_text = result.text.trim();
                                        if !raw_text.is_empty() {
                                            let (lang, clean_text) = parse_sense_voice_text(raw_text, current_lang.as_deref());
                                            if !clean_text.is_empty() {
                                                let payload = SubtitlePayload {
                                                    id: subtitle_seq + 1, // 当前正在说的句号预留位
                                                    is_final: false,
                                                    stage: "partial".to_string(),
                                                    text: clean_text,
                                                    translated: String::new(),
                                                    language: lang,
                                                    latency_ms: 0,
                                                };
                                                let _ = tx.send(payload);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // 2.2 句末停顿触发：VAD 产生完整语音句 (is_final: true)
                        while !vad.is_empty() {
                            if let Some(segment) = vad.front() {
                                let samples = segment.samples();
                                // 过滤短于 0.5 秒或过短杂音段 (8000 个采样点)，绝不吐出 "ed."、"not." 等残破碎词！
                                if samples.len() >= 8000 {
                                    if let Some(ref rec) = current_recognizer {
                                        let start = std::time::Instant::now();
                                        let stream = rec.create_stream();
                                        stream.accept_waveform(16000, samples);
                                        rec.decode(&stream);

                                        if let Some(result) = stream.get_result() {
                                            let raw_text = result.text.trim();
                                            if !raw_text.is_empty() {
                                                let (lang, clean_text) = parse_sense_voice_text(raw_text, current_lang.as_deref());
                                                // 过滤少于两个有效字符的残断词
                                                if clean_text.trim().chars().count() < 2 {
                                                    vad.pop();
                                                    continue;
                                                }
                                                let asr_latency = start.elapsed().as_millis();
                                                subtitle_seq += 1;
                                                let msg_id = subtitle_seq;

                                                println!("⚡ [ASR-Final] id={msg_id} 原始={raw_text:?} -> 解析语种={lang:?}, 识别={clean_text:?} (耗时={asr_latency}ms)");

                                                // 阶段 1：定稿原音直出上屏 (is_final: true)
                                                let live_payload = SubtitlePayload {
                                                    id: msg_id,
                                                    is_final: true,
                                                    stage: "asr".to_string(),
                                                    text: clean_text.clone(),
                                                    translated: String::new(),
                                                    language: lang.clone(),
                                                    latency_ms: asr_latency,
                                                };
                                                let _ = tx.send(live_payload);

                                                // 阶段 2：异步并发调用翻译
                                                let tx_async = tx.clone();
                                                let client_async = http_client.clone();
                                                let text_for_trans = clean_text.clone();
                                                let lang_for_trans = lang.clone();

                                                tokio::spawn(async move {
                                                    let trans_start = std::time::Instant::now();
                                                    let translated = translate_to_chinese(&client_async, &text_for_trans, &lang_for_trans).await;
                                                    let total_latency = asr_latency + trans_start.elapsed().as_millis();

                                                    println!("🎬 [Trans-Complete] id={msg_id} 翻译={translated:?} (总耗时={total_latency}ms)");

                                                    let full_payload = SubtitlePayload {
                                                        id: msg_id,
                                                        is_final: true,
                                                        stage: "trans".to_string(),
                                                        text: text_for_trans,
                                                        translated,
                                                        language: lang_for_trans,
                                                        latency_ms: total_latency,
                                                    };
                                                    let _ = tx_async.send(full_payload);
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            vad.pop();
                            // 句子定稿完成，清空实时流式打字机音频缓存
                            live_speech_buffer.clear();
                        }
                    } else {
                        // 若未安装 VAD 模型，使用动态平滑分块（1.2 秒最小聚合）
                        audio_buffer.extend_from_slice(&chunk);
                        if audio_buffer.len() >= 19200 { // 1.2 秒
                            if let Some(ref rec) = current_recognizer {
                                let start = std::time::Instant::now();
                                let stream = rec.create_stream();
                                stream.accept_waveform(16000, &audio_buffer);
                                rec.decode(&stream);

                                if let Some(result) = stream.get_result() {
                                    let raw_text = result.text.trim();
                                    if !raw_text.is_empty() {
                                        let (lang, clean_text) = parse_sense_voice_text(raw_text, current_lang.as_deref());
                                        let asr_latency = start.elapsed().as_millis();
                                        subtitle_seq += 1;
                                        let msg_id = subtitle_seq;

                                        let live_payload = SubtitlePayload {
                                            id: msg_id,
                                            is_final: true,
                                            stage: "asr".to_string(),
                                            text: clean_text.clone(),
                                            translated: String::new(),
                                            language: lang.clone(),
                                            latency_ms: asr_latency,
                                        };
                                        let _ = tx.send(live_payload);

                                        let tx_async = tx.clone();
                                        let client_async = http_client.clone();
                                        let text_for_trans = clean_text.clone();
                                        let lang_for_trans = lang.clone();

                                        tokio::spawn(async move {
                                            let translated = translate_to_chinese(&client_async, &text_for_trans, &lang_for_trans).await;
                                            let full_payload = SubtitlePayload {
                                                id: msg_id,
                                                is_final: true,
                                                stage: "trans".to_string(),
                                                text: text_for_trans,
                                                translated,
                                                language: lang_for_trans,
                                                latency_ms: asr_latency,
                                            };
                                            let _ = tx_async.send(full_payload);
                                        });
                                    }
                                }
                            }
                            audio_buffer.clear();
                        }
                    }
                }
                Message::Text(txt) => {
                    // 支持客户端以 JSON 发送控制信号（动态切换语种）
                    if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&txt) {
                        if let Some(action) = cmd.get("action").and_then(|v| v.as_str()) {
                            if action == "set_language" {
                                if let Some(lang_code) = cmd.get("language").and_then(|v| v.as_str()) {
                                    println!("🌐 客户端请求切换识别语种: {lang_code}");
                                    if lang_code == "auto" {
                                        current_lang = None;
                                        if let Some(ref paths) = model_paths {
                                            current_recognizer = create_sense_voice_recognizer(&paths.model_path, &paths.tokens_path, None).map(Arc::new);
                                        }
                                    } else {
                                        current_lang = Some(lang_code.to_string());
                                        if let Some(ref paths) = model_paths {
                                            current_recognizer = create_sense_voice_recognizer(&paths.model_path, &paths.tokens_path, Some(lang_code)).map(Arc::new);
                                        }
                                    }
                                }
                            }
                        }
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
        let (lang, clean_text) = parse_sense_voice_text(&res.text, None);
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

/// 解析 SenseVoice 输出标签与 Unicode 字符集，结合目标语言与兜底纠偏，精准提取真实语言代码和纯文本
fn parse_sense_voice_text(text: &str, target_lang: Option<&str>) -> (String, String) {
    let mut detected_tag: Option<String> = None;
    let mut clean_text = String::new();

    // 1. 精确解析 SenseVoice 的特殊标记: <|zh|><|NEUTRAL|><|Speech|><|woitn|> text
    let mut cur = text;
    while let Some(start) = cur.find("<|") {
        clean_text.push_str(&cur[..start]);
        let rem = &cur[start + 2..];
        if let Some(end) = rem.find("|>") {
            let tag = &rem[..end];
            match tag {
                "zh" | "en" | "ja" | "ko" | "yue" => {
                    detected_tag = Some(tag.to_string());
                }
                _ => {}
            }
            cur = &rem[end + 2..];
        } else {
            clean_text.push_str("<|");
            cur = rem;
        }
    }
    clean_text.push_str(cur);
    let mut clean_text = clean_text.trim().to_string();

    if clean_text.is_empty() {
        clean_text = text.to_string();
    }

    // 2. 核心字符集特征提取 (平假名、片假名、韩文字母、汉字、英文拉丁字母)
    let has_kana = clean_text.chars().any(|c| ('\u{3040}'..='\u{309F}').contains(&c) || ('\u{30A0}'..='\u{30FF}').contains(&c));
    let has_hangul = clean_text.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c) || ('\u{1100}'..='\u{11FF}').contains(&c));
    let latin_count = clean_text.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let han_count = clean_text.chars().filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c)).count();

    // 3. 【阶段3：字符集二次校验与强制纠偏兜底】
    let final_lang = if let Some(target) = target_lang {
        // 如果客户端明确锁定了目标语种（例如专心追日漫 / 听英文课）
        match target {
            "ja" | "en" | "ko" | "zh" | "yue" => target.to_string(),
            _ => {
                if has_kana {
                    "ja".to_string()
                } else if has_hangul {
                    "ko".to_string()
                } else if latin_count > 0 && latin_count >= han_count {
                    "en".to_string()
                } else {
                    detected_tag.unwrap_or_else(|| "zh".to_string())
                }
            }
        }
    } else {
        // 自动模式下的强力纠偏：
        // 规则 A：只要文本中包含日文假名，哪怕模型误打了 <|zh|> 标签，也无条件纠正为 "ja"！
        if has_kana {
            "ja".to_string()
        } else if has_hangul {
            "ko".to_string()
        } else if latin_count > 0 && latin_count > han_count * 2 {
            // 规则 B：拉丁字母数量明显占据主导地位，纠正为 "en"
            "en".to_string()
        } else if let Some(tag) = detected_tag {
            // 规则 C：如果打着 zh 标签但完全没有汉字且全是英文字母，纠正为 "en"
            if tag == "zh" && han_count == 0 && latin_count > 0 {
                "en".to_string()
            } else {
                tag
            }
        } else if han_count > 0 {
            "zh".to_string()
        } else if latin_count > 0 {
            "en".to_string()
        } else {
            "auto".to_string()
        }
    };

    (final_lang, clean_text)
}

/// 自动将外语极速翻译为地道简体中文；若是中文则返回空字符串，彻底防止双语中文重复！
async fn translate_to_chinese(client: &reqwest::Client, text: &str, lang: &str) -> String {
    // 1. 如果本身就是中文/粤语，无需翻译，返回空字符串让前端只显示单行中文
    if lang == "zh" || lang == "yue" {
        return String::new();
    }

    // 2. 优先检查用户是否配置了外部 LLM 翻译接口 (如 DeepSeek, OpenAI, 本地翻译模型等)
    if let Ok(api_url) = std::env::var("LLM_TRANSLATE_URL") {
        let api_key = std::env::var("LLM_TRANSLATE_KEY").unwrap_or_else(|_| "none".to_string());
        let model = std::env::var("LLM_TRANSLATE_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());

        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a professional subtitle translator. Translate the input audio text into natural, concise Simplified Chinese subtitles. Output translation only, without explanation, quotes, or notes."},
                {"role": "user", "content": text}
            ],
            "temperature": 0.3,
            "max_tokens": 120
        });

        if let Ok(resp) = client.post(&api_url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .timeout(std::time::Duration::from_millis(1500))
            .send()
            .await 
        {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }

    // 3. 默认开箱即用：自动调用免 Key 极速神经翻译接口 (耗时仅 80ms ~ 150ms)
    let sl = match lang {
        "ja" => "ja",
        "en" => "en",
        "ko" => "ko",
        _ => "auto",
    };

    if let Ok(resp) = client.get("https://translate.googleapis.com/translate_a/single")
        .query(&[
            ("client", "gtx"),
            ("sl", sl),
            ("tl", "zh-CN"),
            ("dt", "t"),
            ("q", text),
        ])
        .header("User-Agent", "Mozilla/5.0")
        .timeout(std::time::Duration::from_millis(1200))
        .send()
        .await
    {
        if let Ok(json) = resp.json::<serde_json::Value>().await {
            if let Some(arr) = json.get(0).and_then(|v| v.as_array()) {
                let mut translated = String::new();
                for item in arr {
                    if let Some(seg) = item.get(0).and_then(|v| v.as_str()) {
                        translated.push_str(seg);
                    }
                }
                let trimmed = translated.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }

    // 4. 若所有翻译通道均因断网超时，回退为前缀标识
    format!("[{}] {}", lang.to_uppercase(), text)
}

