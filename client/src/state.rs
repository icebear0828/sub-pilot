use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use serde::{Deserialize, Serialize};

use crate::asr::AsrEngine;
use crate::db::Database;
use crate::vad::VadConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    PushToTalk,
    ContinuousVad,
    SilentRealtime,
    SilentDeferred,
}

impl Default for CaptureMode {
    fn default() -> Self {
        Self::ContinuousVad
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MicStatus {
    Idle,
    Listening,
    Recording,
    Processing,
    Muted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppStatusDto {
    pub mode: CaptureMode,
    pub status: MicStatus,
    pub is_capturing: bool,
    pub is_muted: bool,
    pub is_silent: bool,
    pub model_ready: bool,
    pub model_dir: String,
    pub session_id: String,
    pub peak_dbfs: f32,
    pub current_duration_s: f32,
    pub pending_asr_count: usize,
    pub total_utterances: usize,
    pub error_message: Option<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub asr: Arc<Mutex<Option<AsrEngine>>>,
    pub vad_config: Arc<Mutex<VadConfig>>,
    pub capture_mode: Arc<Mutex<CaptureMode>>,
    pub mic_status: Arc<Mutex<MicStatus>>,
    pub is_capturing: Arc<AtomicBool>,
    pub is_muted: Arc<AtomicBool>,
    pub is_silent: Arc<AtomicBool>,
    pub current_session_id: Arc<Mutex<String>>,
    pub last_activity: Arc<Mutex<Instant>>,
    pub current_peak_dbfs: Arc<Mutex<f32>>,
    pub current_recording_duration_s: Arc<Mutex<f32>>,
    pub last_error: Arc<Mutex<Option<String>>>,
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
}

impl AppState {
    pub fn new(data_dir: PathBuf, model_dir: PathBuf) -> anyhow::Result<Self> {
        let db = Arc::new(Database::new(&data_dir)?);
        let default_session = format!("sess_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        let _ = db.get_or_create_session(&default_session, "continuous_vad", "default_user");

        Ok(Self {
            db,
            asr: Arc::new(Mutex::new(None)),
            vad_config: Arc::new(Mutex::new(VadConfig::default())),
            capture_mode: Arc::new(Mutex::new(CaptureMode::ContinuousVad)),
            mic_status: Arc::new(Mutex::new(MicStatus::Idle)),
            is_capturing: Arc::new(AtomicBool::new(false)),
            is_muted: Arc::new(AtomicBool::new(false)),
            is_silent: Arc::new(AtomicBool::new(false)),
            current_session_id: Arc::new(Mutex::new(default_session)),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            current_peak_dbfs: Arc::new(Mutex::new(-100.0)),
            current_recording_duration_s: Arc::new(Mutex::new(0.0)),
            last_error: Arc::new(Mutex::new(None)),
            data_dir,
            model_dir,
        })
    }

    pub fn get_status_dto(&self) -> AppStatusDto {
        let mode = *self.capture_mode.lock().unwrap();
        let status = *self.mic_status.lock().unwrap();
        let is_capturing = self.is_capturing.load(Ordering::SeqCst);
        let is_muted = self.is_muted.load(Ordering::SeqCst);
        let is_silent = self.is_silent.load(Ordering::SeqCst);
        let session_id = self.current_session_id.lock().unwrap().clone();
        let peak_dbfs = *self.current_peak_dbfs.lock().unwrap();
        let current_duration_s = *self.current_recording_duration_s.lock().unwrap();
        let pending_asr_count = self.db.get_pending_asr_ids(Some(&session_id)).unwrap_or_default().len();
        let total_utterances = self.db.get_total_utterances_count(Some(&session_id)).unwrap_or_default();
        let error_message = self.last_error.lock().unwrap().clone();

        let model_ready = self.model_dir.join("silero_vad.onnx").exists()
            || self.model_dir.join("model.int8.onnx").exists();

        AppStatusDto {
            mode,
            status,
            is_capturing,
            is_muted,
            is_silent,
            model_ready,
            model_dir: self.model_dir.to_string_lossy().to_string(),
            session_id,
            peak_dbfs,
            current_duration_s,
            pending_asr_count,
            total_utterances,
            error_message,
        }
    }
}
