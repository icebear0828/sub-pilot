use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use serde::{Deserialize, Serialize};

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

pub fn log_to_file(data_dir: &Path, msg: &str) {
    let log_file = data_dir.join("voiceforge.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log_file) {
        use std::io::Write;
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let _ = writeln!(f, "[{}] {}", now, msg);
    }
}

pub struct AsrTask {
    pub utt_id: String,
    pub samples: Vec<f32>,
    pub sample_rate: i32,
}

#[derive(Clone)]
pub struct AudioEngineHandle {
    cmd_tx: Sender<AudioControlCommand>,
}

impl AudioEngineHandle {
    pub fn new(state: AppState) -> Self {
        let (cmd_tx, cmd_rx) = channel::<AudioControlCommand>();
        let (asr_tx, asr_rx) = channel::<AsrTask>();

        // Spawn dedicated, non-blocking ASR worker thread
        let state_for_asr = state.clone();
        let data_dir_asr = state.data_dir.clone();
        std::thread::Builder::new()
            .name("voiceforge-asr-worker".into())
            .spawn(move || {
                log_to_file(&data_dir_asr, "ASR worker thread started successfully");
                while let Ok(task) = asr_rx.recv() {
                    let state_ref = &state_for_asr;
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_asr_for_utterance(state_ref, &task.utt_id, &task.samples, task.sample_rate);
                    }));
                    if let Err(e) = res {
                        log_to_file(&data_dir_asr, &format!("ASR worker panicked on task {}: {:?}", task.utt_id, e));
                    }
                }
                log_to_file(&data_dir_asr, "ASR worker thread exiting");
            })
            .expect("Failed to spawn asr worker thread");

        let state_clone = state.clone();
        let asr_tx_for_worker = asr_tx.clone();
        std::thread::Builder::new()
            .name("voiceforge-audio-worker".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
                    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                }

                log_to_file(&state_clone.data_dir, "Audio worker thread started successfully");

                let mut current_stream: Option<Stream> = None;
                let mut stop_pipeline_tx: Option<Sender<()>> = None;

                while let Ok(cmd) = cmd_rx.recv() {
                    match cmd {
                        AudioControlCommand::Start(device_name) => {
                            log_to_file(&state_clone.data_dir, &format!("Received Start command, device: {:?}", device_name));
                            // Stop existing stream if any
                            if let Some(tx) = stop_pipeline_tx.take() {
                                let _ = tx.send(());
                            }
                            current_stream = None;

                            match setup_input_stream(&state_clone, device_name.as_deref(), asr_tx_for_worker.clone()) {
                                Ok((stream, tx)) => {
                                    if let Err(e) = stream.play() {
                                        let err_msg = format!("麦克风启动播放失败: {}", e);
                                        log_to_file(&state_clone.data_dir, &err_msg);
                                        *state_clone.last_error.lock().unwrap() = Some(err_msg);
                                        state_clone.is_capturing.store(false, Ordering::SeqCst);
                                        *state_clone.mic_status.lock().unwrap() = MicStatus::Idle;
                                    } else {
                                        log_to_file(&state_clone.data_dir, "Microphone stream successfully running");
                                        current_stream = Some(stream);
                                        stop_pipeline_tx = Some(tx);
                                        state_clone.is_capturing.store(true, Ordering::SeqCst);
                                        *state_clone.last_error.lock().unwrap() = None;
                                        *state_clone.mic_status.lock().unwrap() = MicStatus::Listening;
                                    }
                                }
                                Err(e) => {
                                    let err_msg = format!("无法打开麦克风设备: {}", e);
                                    log_to_file(&state_clone.data_dir, &err_msg);
                                    eprintln!("{}", err_msg);
                                    *state_clone.last_error.lock().unwrap() = Some(err_msg);
                                    state_clone.is_capturing.store(false, Ordering::SeqCst);
                                    *state_clone.mic_status.lock().unwrap() = MicStatus::Idle;
                                }
                            }
                        }
                        AudioControlCommand::Stop => {
                            log_to_file(&state_clone.data_dir, "Received Stop command");
                            if let Some(tx) = stop_pipeline_tx.take() {
                                let _ = tx.send(());
                            }
                            if let Some(stream) = current_stream.take() {
                                let _ = stream.pause();
                            }
                            state_clone.is_capturing.store(false, Ordering::SeqCst);
                            *state_clone.mic_status.lock().unwrap() = MicStatus::Idle;
                        }
                    }
                }
                log_to_file(&state_clone.data_dir, "Audio worker thread exiting");
            })
            .expect("Failed to spawn audio worker thread");

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

    pub fn start_capture(&self, device_name: Option<String>) -> Result<()> {
        self.cmd_tx
            .send(AudioControlCommand::Start(device_name))
            .map_err(|e| anyhow!("Failed to send start audio command: {}", e))
    }

    pub fn stop_capture(&self) -> Result<()> {
        self.cmd_tx
            .send(AudioControlCommand::Stop)
            .map_err(|e| anyhow!("Failed to send stop audio command: {}", e))
    }
}

fn setup_input_stream(
    state: &AppState,
    device_name: Option<&str>,
    asr_tx: Sender<AsrTask>,
) -> Result<(Stream, Sender<()>)> {
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

    let dev_name = device.name().unwrap_or_else(|_| "Unknown".into());
    log_to_file(&state.data_dir, &format!("Selected audio device: {}", dev_name));

    let default_config = device
        .default_input_config()
        .context("Failed to get default input config")?;
    let sample_rate = default_config.sample_rate().0;
    let channels = default_config.channels();
    let sample_format = default_config.sample_format();

    log_to_file(&state.data_dir, &format!("Device config: rate={}, channels={}, format={:?}", sample_rate, channels, sample_format));

    // Non-blocking sync channel between real-time cpal thread and processing thread
    let (sample_tx, sample_rx) = sync_channel::<Vec<f32>>(64);
    let (stop_tx, stop_rx) = channel::<()>();

    let err_fn = move |err| {
        eprintln!("Audio stream error: {:?}", err);
    };

    let stream_config: StreamConfig = default_config.into();

    let stream = match sample_format {
        SampleFormat::F32 => {
            let tx = sample_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    let mono = downmix_to_mono_f32(data, channels as usize);
                    let _ = tx.try_send(mono);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let tx = sample_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let mono = downmix_to_mono_i16(data, channels as usize);
                    let _ = tx.try_send(mono);
                },
                err_fn,
                None,
            )?
        }
        _ => return Err(anyhow!("Unsupported sample format: {:?}", sample_format)),
    };

    let state_clone = state.clone();
    let data_dir = state.data_dir.clone();
    std::thread::Builder::new()
        .name("voiceforge-dsp-loop".into())
        .spawn(move || {
            log_to_file(&data_dir, "DSP processing loop started");
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_processing_loop(state_clone, sample_rate, sample_rx, stop_rx, asr_tx);
            }));
            if let Err(e) = res {
                log_to_file(&data_dir, &format!("DSP processing loop panicked: {:?}", e));
            } else {
                log_to_file(&data_dir, "DSP processing loop exited normally");
            }
        })
        .expect("Failed to spawn dsp processing thread");

    Ok((stream, stop_tx))
}

fn run_processing_loop(
    state: AppState,
    native_sample_rate: u32,
    sample_rx: Receiver<Vec<f32>>,
    stop_rx: Receiver<()>,
    asr_tx: Sender<AsrTask>,
) {
    let vad_path = state.model_dir.join("silero_vad.onnx");
    let mut vad = if vad_path.exists() {
        let vad_cfg = state.vad_config.lock().unwrap().clone();
        match VadDetector::new(&vad_path, &vad_cfg) {
            Ok(d) => {
                log_to_file(&state.data_dir, "Silero VAD initialized successfully");
                Some(d)
            }
            Err(e) => {
                log_to_file(&state.data_dir, &format!("Failed to load Silero VAD from {:?}: {:?}", vad_path, e));
                None
            }
        }
    } else {
        log_to_file(&state.data_dir, &format!("Silero VAD not found at {:?}, using energy chunk fallback", vad_path));
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

    while let Ok(native_mono) = sample_rx.recv() {
        // Check stop signal
        if stop_rx.try_recv().is_ok() {
            break;
        }

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

                    log_to_file(
                        &state.data_dir,
                        &format!(
                            "VAD popped segment: {} samples ({}ms), peak={:.1}dBFS",
                            segment_samples.len(),
                            duration_ms,
                            peak_dbfs
                        ),
                    );

                    // Filter fragments < 0.25s (4000 samples)
                    if segment_samples.len() >= 4000 {
                        save_and_dispatch_utterance(
                            &state,
                            segment_samples,
                            duration_ms,
                            wall_clock_start.clone(),
                            wall_clock_end,
                            session_offset_start_ms,
                            session_offset_end_ms,
                            peak_dbfs,
                            &asr_tx,
                        );
                    } else {
                        log_to_file(
                            &state.data_dir,
                            &format!("Ignored short segment ({} samples < 4000)", segment_samples.len()),
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
                        &asr_tx,
                    );
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
    asr_tx: &Sender<AsrTask>,
) {
    let utt_id = format!("utt_{}", Utc::now().format("%Y%m%d_%H%M%S_%3f"));
    let session_id = state.current_session_id.lock().unwrap().clone();
    let wav_filename = format!("{}.wav", utt_id);
    let wav_path = state.data_dir.join("wavs").join(&wav_filename);

    log_to_file(
        &state.data_dir,
        &format!("Writing utterance WAV: {} (duration: {}ms, file: {:?})", utt_id, duration_ms, wav_filename),
    );

    if let Err(e) = write_wav_file(&wav_path, &samples, 16000) {
        log_to_file(&state.data_dir, &format!("Failed to write WAV file {:?}: {:?}", wav_path, e));
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
        log_to_file(&state.data_dir, &format!("Failed to insert utterance into DB: {:?}", e));
        eprintln!("Failed to insert utterance into DB: {:?}", e);
        return;
    }

    log_to_file(&state.data_dir, &format!("Successfully saved utterance {} to DB", utt_id));

    let current_mode = *state.capture_mode.lock().unwrap();
    if current_mode == CaptureMode::SilentDeferred {
        return;
    }

    // Safely enqueue ASR task to dedicated background worker (zero-panic guarantee)
    if let Err(e) = asr_tx.send(AsrTask {
        utt_id: utt_id.clone(),
        samples,
        sample_rate: 16000,
    }) {
        log_to_file(&state.data_dir, &format!("Failed to enqueue ASR task for {}: {:?}", utt_id, e));
    }
}

pub fn run_asr_for_utterance(state: &AppState, utt_id: &str, samples: &[f32], sample_rate: i32) {
    let mut asr_guard = state.asr.lock().unwrap();

    if asr_guard.is_none() {
        let model_path = state.model_dir.join("model.int8.onnx");
        let tokens_path = state.model_dir.join("tokens.txt");
        if model_path.exists() && tokens_path.exists() {
            log_to_file(&state.data_dir, &format!("Initializing ASR engine from {:?}", state.model_dir));
            match AsrEngine::new(&model_path, &tokens_path, None) {
                Ok(engine) => {
                    log_to_file(&state.data_dir, "ASR engine initialized successfully");
                    *asr_guard = Some(engine);
                }
                Err(e) => {
                    log_to_file(&state.data_dir, &format!("Failed to initialize ASR engine: {:?}", e));
                }
            }
        } else {
            log_to_file(&state.data_dir, &format!("Model files not found in {:?}, skipping ASR", state.model_dir));
        }
    }

    if let Some(ref engine) = *asr_guard {
        log_to_file(&state.data_dir, &format!("Running ASR transcription for {}...", utt_id));
        match engine.transcribe(samples, sample_rate) {
            Ok(res) => {
                log_to_file(&state.data_dir, &format!("ASR completed for {}: [{}] {}", utt_id, res.language, res.text));
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
            Err(e) => {
                log_to_file(&state.data_dir, &format!("ASR transcription error for {}: {:?}", utt_id, e));
            }
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
