use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::asr::AsrEngine;
use crate::db::UtteranceRecord;
use crate::state::{AppState, CaptureMode, MicStatus};
use crate::vad::VadDetector;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

pub enum AudioControlCommand {
    Start(Option<String>),
    Stop,
}

#[derive(Clone)]
pub struct AudioEngineHandle {
    cmd_tx: mpsc::Sender<AudioControlCommand>,
}

impl AudioEngineHandle {
    pub fn new(state: AppState) -> Self {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<AudioControlCommand>(16);

        // Run audio thread on dedicated OS thread to isolate cpal::Stream (!Send on macOS CoreAudio)
        std::thread::spawn(move || {
            #[cfg(windows)]
            unsafe {
                use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            }

            let mut current_stream: Option<Stream> = None;
            let mut stop_pipeline_tx: Option<mpsc::Sender<()>> = None;

            while let Some(cmd) = cmd_rx.blocking_recv() {
                match cmd {
                    AudioControlCommand::Start(device_name) => {
                        // Stop any existing stream
                        if let Some(tx) = stop_pipeline_tx.take() {
                            let _ = tx.blocking_send(());
                        }
                        current_stream = None;

                        match setup_input_stream(&state, device_name.as_deref()) {
                            Ok((stream, tx)) => {
                                let _ = stream.play();
                                current_stream = Some(stream);
                                stop_pipeline_tx = Some(tx);
                                state.is_capturing.store(true, Ordering::SeqCst);
                                *state.last_error.lock().unwrap() = None;
                                *state.mic_status.lock().unwrap() = MicStatus::Listening;
                            }
                            Err(e) => {
                                let err_msg = format!("无法打开麦克风: {}", e);
                                eprintln!("{}", err_msg);
                                *state.last_error.lock().unwrap() = Some(err_msg);
                                state.is_capturing.store(false, Ordering::SeqCst);
                                *state.mic_status.lock().unwrap() = MicStatus::Idle;
                            }
                        }
                    }
                    AudioControlCommand::Stop => {
                        if let Some(tx) = stop_pipeline_tx.take() {
                            let _ = tx.blocking_send(());
                        }
                        if let Some(stream) = current_stream.take() {
                            let _ = stream.pause();
                        }
                        state.is_capturing.store(false, Ordering::SeqCst);
                        *state.mic_status.lock().unwrap() = MicStatus::Idle;
                    }
                }
            }
        });

        Self { cmd_tx }
    }

    pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>> {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        let host = cpal::default_host();
        let default_name = host.default_input_device().and_then(|d| d.name().ok());
        let devices = host.input_devices().context("Failed to get input devices")?;

        let mut list = Vec::new();
        for dev in devices {
            if let Ok(name) = dev.name() {
                let is_default = default_name.as_ref().map(|d| d == &name).unwrap_or(false);
                list.push(AudioDeviceInfo { name, is_default });
            }
        }
        Ok(list)
    }

    pub async fn start_capture(&self, device_name: Option<String>) -> Result<()> {
        self.cmd_tx
            .send(AudioControlCommand::Start(device_name))
            .await
            .map_err(|e| anyhow!("Failed to send start audio command: {}", e))
    }

    pub async fn stop_capture(&self) -> Result<()> {
        self.cmd_tx
            .send(AudioControlCommand::Stop)
            .await
            .map_err(|e| anyhow!("Failed to send stop audio command: {}", e))
    }
}

fn setup_input_stream(
    state: &AppState,
    device_name: Option<&str>,
) -> Result<(Stream, mpsc::Sender<()>)> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }

    let host = cpal::default_host();
    let device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("Input device '{}' not found", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device available"))?
    };

    let default_config = device
        .default_input_config()
        .context("Failed to get default input config")?;
    let sample_rate = default_config.sample_rate().0;
    let channels = default_config.channels();
    let sample_format = default_config.sample_format();

    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(100);
    let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);

    let err_fn = |err| eprintln!("Audio stream error: {:?}", err);
    let stream_config: StreamConfig = default_config.into();

    let stream = match sample_format {
        SampleFormat::F32 => {
            let tx = audio_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    let mono = downmix_to_mono_f32(data, channels as usize);
                    let _ = tx.blocking_send(mono);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let tx = audio_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let mono = downmix_to_mono_i16(data, channels as usize);
                    let _ = tx.blocking_send(mono);
                },
                err_fn,
                None,
            )?
        }
        _ => return Err(anyhow!("Unsupported sample format: {:?}", sample_format)),
    };

    let state_clone = state.clone();
    tokio::spawn(async move {
        run_processing_loop(state_clone, sample_rate, &mut audio_rx, &mut stop_rx).await;
    });

    Ok((stream, stop_tx))
}

async fn run_processing_loop(
    state: AppState,
    native_sample_rate: u32,
    audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    stop_rx: &mut mpsc::Receiver<()>,
) {
    let vad_path = state.model_dir.join("silero_vad.onnx");
    let mut vad = if vad_path.exists() {
        let vad_cfg = state.vad_config.lock().unwrap().clone();
        VadDetector::new(&vad_path, &vad_cfg).ok()
    } else {
        None
    };

    let wavs_dir = state.data_dir.join("wavs");
    let _ = std::fs::create_dir_all(&wavs_dir);

    let mut resample_buffer_16k: Vec<f32> = Vec::new();
    let mut speech_recording_buffer: Vec<f32> = Vec::new();
    let mut recording_start_time: Option<Instant> = None;
    let mut wall_clock_start = String::new();
    let mut session_offset_start_ms = 0i64;
    let session_start_instant = Instant::now();

    *state.mic_status.lock().unwrap() = MicStatus::Listening;

    loop {
        tokio::select! {
            _ = stop_rx.recv() => {
                break;
            }
            Some(native_mono) = audio_rx.recv() => {
                // Panic Mute
                if state.is_muted.load(Ordering::SeqCst) {
                    *state.mic_status.lock().unwrap() = MicStatus::Muted;
                    *state.current_peak_dbfs.lock().unwrap() = -100.0;
                    continue;
                }

                // Peak dBFS calculation
                let peak = native_mono.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
                let peak_dbfs = if peak > 0.00001 {
                    20.0 * peak.log10().max(-100.0)
                } else {
                    -100.0
                };
                *state.current_peak_dbfs.lock().unwrap() = peak_dbfs;

                // Downsample to 16kHz
                let chunk_16k = resample_to_16k(&native_mono, native_sample_rate);
                resample_buffer_16k.extend_from_slice(&chunk_16k);

                // Process in 512-sample (32ms) VAD windows
                while resample_buffer_16k.len() >= 512 {
                    let window: Vec<f32> = resample_buffer_16k.drain(..512).collect();

                    if let Some(ref mut detector) = vad {
                        detector.accept_waveform(&window);

                        if detector.is_speech_detected() {
                            if recording_start_time.is_none() {
                                recording_start_time = Some(Instant::now());
                                wall_clock_start = Utc::now().to_rfc3339();
                                session_offset_start_ms = session_start_instant.elapsed().as_millis() as i64;
                                *state.mic_status.lock().unwrap() = MicStatus::Recording;
                            }
                            speech_recording_buffer.extend_from_slice(&window);
                            let dur_s = recording_start_time.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
                            *state.current_recording_duration_s.lock().unwrap() = dur_s;
                        }

                        while let Some(segment_samples) = detector.pop_segment() {
                            let duration_ms = (segment_samples.len() as f32 / 16.0) as i64;
                            let wall_clock_end = Utc::now().to_rfc3339();
                            let session_offset_end_ms = session_start_instant.elapsed().as_millis() as i64;

                            *state.current_recording_duration_s.lock().unwrap() = 0.0;
                            *state.mic_status.lock().unwrap() = MicStatus::Listening;
                            recording_start_time = None;

                            // Filter fragments < 600ms
                            if segment_samples.len() >= 9600 {
                                save_and_dispatch_utterance(
                                    &state,
                                    segment_samples,
                                    duration_ms,
                                    wall_clock_start.clone(),
                                    wall_clock_end,
                                    session_offset_start_ms,
                                    session_offset_end_ms,
                                    peak_dbfs,
                                );
                            }
                            speech_recording_buffer.clear();
                        }
                    } else {
                        // Energy-based fallback: 1.5s windows
                        speech_recording_buffer.extend_from_slice(&window);
                        if speech_recording_buffer.len() >= 24000 {
                            let samples = std::mem::take(&mut speech_recording_buffer);
                            let now = Utc::now().to_rfc3339();
                            let end_offset = session_start_instant.elapsed().as_millis() as i64;
                            save_and_dispatch_utterance(
                                &state,
                                samples,
                                1500,
                                now.clone(),
                                now,
                                end_offset.saturating_sub(1500),
                                end_offset,
                                peak_dbfs,
                            );
                        }
                    }
                }
            }
        }
    }
}

fn save_and_dispatch_utterance(
    state: &AppState,
    samples: Vec<f32>,
    duration_ms: i64,
    wall_clock_start: String,
    wall_clock_end: String,
    session_offset_start_ms: i64,
    session_offset_end_ms: i64,
    peak_dbfs: f32,
) {
    let utt_id = format!("utt_{}", Utc::now().format("%Y%m%d_%H%M%S_%3f"));
    let session_id = state.current_session_id.lock().unwrap().clone();
    let wav_filename = format!("{}.wav", utt_id);
    let wav_path = state.data_dir.join("wavs").join(&wav_filename);

    if let Err(e) = write_wav_file(&wav_path, &samples, 16000) {
        eprintln!("Failed to write WAV file {:?}: {:?}", wav_path, e);
        return;
    }

    let record = UtteranceRecord {
        id: utt_id.clone(),
        session_id: session_id.clone(),
        audio_filepath: wav_path.to_string_lossy().to_string(),
        master_filepath: None,
        sample_rate: 16000,
        channels: 1,
        bit_depth: 16,
        duration_ms,
        wall_clock_start,
        wall_clock_end,
        session_offset_start_ms,
        session_offset_end_ms,
        speaker: "default_user".to_string(),
        language: "zh".to_string(),
        emotion: None,
        text: String::new(),
        text_normalized: String::new(),
        asr_model: String::new(),
        asr_confidence: 0.0,
        peak_dbfs,
        integrated_lufs: None,
        verified: false,
        split: "train".to_string(),
        words: Vec::new(),
        status: "Pending_ASR".to_string(),
        created_at: Utc::now().to_rfc3339(),
    };

    if let Err(e) = state.db.insert_utterance(&record) {
        eprintln!("Failed to insert utterance into DB: {:?}", e);
        return;
    }

    let current_mode = *state.capture_mode.lock().unwrap();
    if current_mode == CaptureMode::SilentDeferred {
        return;
    }

    let state_clone = state.clone();
    let samples_clone = samples.clone();
    tokio::spawn(async move {
        run_asr_for_utterance(&state_clone, &utt_id, &samples_clone, 16000);
    });
}

pub fn run_asr_for_utterance(state: &AppState, utt_id: &str, samples: &[f32], sample_rate: i32) {
    let mut asr_guard = state.asr.lock().unwrap();

    if asr_guard.is_none() {
        let model_path = state.model_dir.join("model.int8.onnx");
        let tokens_path = state.model_dir.join("tokens.txt");
        if model_path.exists() && tokens_path.exists() {
            if let Ok(engine) = AsrEngine::new(&model_path, &tokens_path, None) {
                *asr_guard = Some(engine);
            }
        }
    }

    if let Some(ref engine) = *asr_guard {
        if let Ok(res) = engine.transcribe(samples, sample_rate) {
            let _ = state.db.update_utterance_asr(
                utt_id,
                &res.text,
                &res.language,
                res.emotion.as_deref(),
                res.confidence,
                engine.model_name(),
                &res.words,
            );
        }
    }
}

fn write_wav_file(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn downmix_to_mono_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    let frames = data.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut sum = 0.0;
        for c in 0..channels {
            sum += data[f * channels + c];
        }
        mono.push(sum / (channels as f32));
    }
    mono
}

fn downmix_to_mono_i16(data: &[i16], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.iter().map(|&s| s as f32 / 32768.0).collect();
    }
    let frames = data.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += data[f * channels + c] as f32 / 32768.0;
        }
        mono.push(sum / (channels as f32));
    }
    mono
}

fn resample_to_16k(input: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == 16000 {
        return input.to_vec();
    }
    let ratio = 16000.0 / (from_rate as f64);
    let target_len = (input.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let src_idx = (i as f64) / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len().saturating_sub(1));
        let frac = (src_idx - (idx0 as f64)) as f32;
        let s = input[idx0] * (1.0 - frac) + input[idx1] * frac;
        output.push(s);
    }
    output
}
