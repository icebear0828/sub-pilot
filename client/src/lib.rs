pub mod asr;
pub mod audio;
pub mod db;
pub mod export;
pub mod state;
pub mod vad;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use tauri::{LogicalSize, State, WebviewWindow};

use crate::audio::{AudioDeviceInfo, AudioEngineHandle};
use crate::db::{SessionRecord, UtteranceRecord};
use crate::export::{ExportOptions, Exporter};
use crate::state::{AppState, AppStatusDto, CaptureMode};

fn find_model_dir(data_dir: &PathBuf) -> PathBuf {
    let mut candidates = vec![
        std::env::var("MODEL_DIR").unwrap_or_default(),
        data_dir.join("models").to_string_lossy().to_string(),
        data_dir.join("models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17").to_string_lossy().to_string(),
        "../models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17".to_string(),
        "/Users/yutao/llm/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17".to_string(),
        "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17".to_string(),
        "./models".to_string(),
    ];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("models").to_string_lossy().to_string());
            candidates.push(parent.join("models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17").to_string_lossy().to_string());
        }
    }

    for path_str in &candidates {
        if path_str.is_empty() {
            continue;
        }
        let p = PathBuf::from(path_str);
        if p.exists() && (p.join("silero_vad.onnx").exists() || p.join("model.int8.onnx").exists()) {
            return p;
        }
    }
    data_dir.join("models")
}

fn get_data_dir() -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".voiceforge_mini")
}

fn dirs_next() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("APPDATA").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".local/share"))
    }
}

// ── Tauri Commands ────────────────────────────────────────────────────────────

#[tauri::command]
fn get_app_status(state: State<Arc<AppState>>) -> AppStatusDto {
    state.get_status_dto()
}

#[tauri::command]
fn set_capture_mode(mode: String, state: State<Arc<AppState>>) -> Result<(), String> {
    let new_mode = match mode.as_str() {
        "push_to_talk" => CaptureMode::PushToTalk,
        "continuous_vad" => CaptureMode::ContinuousVad,
        "silent_realtime" => CaptureMode::SilentRealtime,
        "silent_deferred" => CaptureMode::SilentDeferred,
        _ => return Err(format!("Unknown capture mode: {}", mode)),
    };
    *state.capture_mode.lock().unwrap() = new_mode;
    let is_silent = matches!(new_mode, CaptureMode::SilentRealtime | CaptureMode::SilentDeferred);
    state.is_silent.store(is_silent, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
fn toggle_mute(state: State<Arc<AppState>>) -> bool {
    let current = state.is_muted.load(Ordering::SeqCst);
    let new_val = !current;
    state.is_muted.store(new_val, Ordering::SeqCst);
    new_val
}

#[tauri::command]
fn toggle_silent_mode(
    window: WebviewWindow,
    state: State<Arc<AppState>>,
) -> Result<bool, String> {
    let current_silent = state.is_silent.load(Ordering::SeqCst);
    let new_silent = !current_silent;
    state.is_silent.store(new_silent, Ordering::SeqCst);

    if new_silent {
        *state.capture_mode.lock().unwrap() = CaptureMode::SilentDeferred;
        let _ = window.hide();
    } else {
        *state.capture_mode.lock().unwrap() = CaptureMode::ContinuousVad;
        let _ = window.show();
    }
    Ok(new_silent)
}

#[tauri::command]
fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    AudioEngineHandle::list_input_devices().map_err(|e| e.to_string())
}

#[tauri::command]
async fn start_recording(
    device_name: Option<String>,
    engine: State<'_, AudioEngineHandle>,
) -> Result<(), String> {
    engine.start_capture(device_name).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn stop_recording(
    engine: State<'_, AudioEngineHandle>,
) -> Result<(), String> {
    engine.stop_capture().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn select_input_device(
    device_name: Option<String>,
    engine: State<'_, AudioEngineHandle>,
) -> Result<(), String> {
    engine.start_capture(device_name).await.map_err(|e| e.to_string())
}

#[tauri::command]
fn list_sessions(state: State<Arc<AppState>>) -> Result<Vec<SessionRecord>, String> {
    state.db.list_sessions().map_err(|e| e.to_string())
}

#[tauri::command]
fn create_session(
    name: String,
    mode: String,
    speaker: String,
    state: State<Arc<AppState>>,
) -> Result<SessionRecord, String> {
    let session_id = format!("sess_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let mut rec = state
        .db
        .get_or_create_session(&session_id, &mode, &speaker)
        .map_err(|e| e.to_string())?;
    if !name.is_empty() {
        rec.name = name;
    }
    *state.current_session_id.lock().unwrap() = session_id;
    Ok(rec)
}

#[tauri::command]
fn set_active_session(session_id: String, state: State<Arc<AppState>>) -> Result<(), String> {
    *state.current_session_id.lock().unwrap() = session_id;
    Ok(())
}

#[tauri::command]
fn list_utterances(
    session_id: Option<String>,
    status_filter: Option<String>,
    state: State<Arc<AppState>>,
) -> Result<Vec<UtteranceRecord>, String> {
    state
        .db
        .list_utterances(session_id.as_deref(), status_filter.as_deref())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn update_utterance_text(
    id: String,
    text: String,
    text_normalized: Option<String>,
    verified: bool,
    state: State<Arc<AppState>>,
) -> Result<(), String> {
    state
        .db
        .update_utterance_text(&id, &text, text_normalized.as_deref(), verified)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn mark_utterance_status(
    id: String,
    status: String,
    state: State<Arc<AppState>>,
) -> Result<(), String> {
    state
        .db
        .mark_utterance_status(&id, &status)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_utterance(id: String, state: State<Arc<AppState>>) -> Result<(), String> {
    state.db.delete_utterance(&id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn run_batch_asr(
    session_id: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<usize, String> {
    let pending = state
        .db
        .get_pending_asr_ids(session_id.as_deref())
        .map_err(|e| e.to_string())?;

    let count = pending.len();
    let state_clone = (*state).clone();

    tokio::task::spawn_blocking(move || {
        for (id, filepath) in pending {
            let path = std::path::Path::new(&filepath);
            if let Ok(mut reader) = hound::WavReader::open(path) {
                let spec = reader.spec();
                let samples: Vec<f32> = reader
                    .samples::<i16>()
                    .filter_map(Result::ok)
                    .map(|s| s as f32 / 32768.0)
                    .collect();

                audio::run_asr_for_utterance(&state_clone, &id, &samples, spec.sample_rate as i32);
            }
        }
    });

    Ok(count)
}

#[tauri::command]
fn export_dataset(
    options: ExportOptions,
    state: State<Arc<AppState>>,
) -> Result<String, String> {
    let records = state
        .db
        .list_utterances(options.session_id.as_deref(), None)
        .map_err(|e| e.to_string())?;

    Exporter::export(&records, &options).map_err(|e| e.to_string())
}

#[tauri::command]
fn read_audio_base64(filepath: String) -> Result<String, String> {
    let bytes = fs::read(&filepath).map_err(|e| format!("Failed to read {}: {}", filepath, e))?;
    let encoded = BASE64_STANDARD.encode(&bytes);
    Ok(format!("data:audio/wav;base64,{}", encoded))
}

#[tauri::command]
fn switch_view_mode(mode: String, window: WebviewWindow) -> Result<(), String> {
    if mode == "hud" {
        let _ = window.set_size(LogicalSize::new(760.0, 160.0));
        let _ = window.set_always_on_top(true);
        let _ = window.set_resizable(true);
    } else {
        let _ = window.set_size(LogicalSize::new(1160.0, 760.0));
        let _ = window.set_always_on_top(false);
        let _ = window.set_resizable(true);
    }
    Ok(())
}

#[tauri::command]
fn start_dragging(window: WebviewWindow) {
    let _ = window.start_dragging();
}

#[tauri::command]
fn minimize_window(window: WebviewWindow) {
    let _ = window.minimize();
}

#[tauri::command]
fn close_window(window: WebviewWindow) {
    let _ = window.hide();
}

#[tauri::command]
fn open_data_directory(state: State<Arc<AppState>>) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&state.data_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&state.data_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn open_model_directory(state: State<Arc<AppState>>) -> Result<(), String> {
    let _ = std::fs::create_dir_all(&state.model_dir);
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&state.model_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&state.model_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── Application Entry ─────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let data_dir = get_data_dir();
    let model_dir = find_model_dir(&data_dir);

    println!("📂 VoiceForge Mini Data Dir: {:?}", data_dir);
    println!("🤖 VoiceForge Mini Model Dir: {:?}", model_dir);

    let state = Arc::new(
        AppState::new(data_dir, model_dir)
            .expect("Failed to initialize AppState"),
    );

    let audio_engine = AudioEngineHandle::new((*state).clone());

    tauri::Builder::default()
        .manage(state)
        .manage(audio_engine)
        .invoke_handler(tauri::generate_handler![
            get_app_status,
            start_recording,
            stop_recording,
            set_capture_mode,
            toggle_mute,
            toggle_silent_mode,
            list_input_devices,
            select_input_device,
            list_sessions,
            create_session,
            set_active_session,
            list_utterances,
            update_utterance_text,
            mark_utterance_status,
            delete_utterance,
            run_batch_asr,
            export_dataset,
            read_audio_base64,
            switch_view_mode,
            start_dragging,
            minimize_window,
            close_window,
            open_data_directory,
            open_model_directory,
        ])
        .run(tauri::generate_context!())
        .expect("Error while running VoiceForge Mini");
}
