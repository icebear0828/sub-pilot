use std::path::Path;
use anyhow::{anyhow, Result};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

#[derive(Clone, Debug)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_silence_duration_s: f32,
    pub min_speech_duration_s: f32,
    pub max_speech_duration_s: f32,
    pub window_size: i32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.50,
            min_silence_duration_s: 0.50,
            min_speech_duration_s: 0.80,
            max_speech_duration_s: 15.0,
            window_size: 512,
        }
    }
}

pub struct VadDetector {
    detector: VoiceActivityDetector,
    _sample_rate: i32,
}

impl VadDetector {
    pub fn new(model_path: &Path, config: &VadConfig) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!("VAD model not found at {:?}", model_path));
        }

        let silero_vad = SileroVadModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            threshold: config.threshold,
            min_silence_duration: config.min_silence_duration_s,
            min_speech_duration: config.min_speech_duration_s,
            window_size: config.window_size,
            max_speech_duration: config.max_speech_duration_s,
        };

        let mut vad_config = VadModelConfig::default();
        vad_config.silero_vad = silero_vad;
        vad_config.sample_rate = 16000;
        vad_config.num_threads = 2;

        let detector = VoiceActivityDetector::create(&vad_config, 30.0)
            .ok_or_else(|| anyhow!("Failed to create Silero VAD detector"))?;

        Ok(Self {
            detector,
            _sample_rate: 16000,
        })
    }

    pub fn accept_waveform(&mut self, samples: &[f32]) {
        self.detector.accept_waveform(samples);
    }

    pub fn is_speech_detected(&self) -> bool {
        self.detector.detected()
    }

    pub fn is_empty(&self) -> bool {
        self.detector.is_empty()
    }

    /// Retrieve the next finished speech segment samples
    pub fn pop_segment(&mut self) -> Option<Vec<f32>> {
        if self.detector.is_empty() {
            return None;
        }
        let segment = self.detector.front()?;
        let samples = segment.samples().to_vec();
        self.detector.pop();
        Some(samples)
    }

    pub fn reset(&mut self) {
        self.detector.reset();
    }
}
