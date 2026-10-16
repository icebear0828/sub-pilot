use std::path::Path;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
};

use crate::db::WordAlignment;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AsrResult {
    pub text: String,
    pub language: String,
    pub emotion: Option<String>,
    pub confidence: f32,
    pub words: Vec<WordAlignment>,
}

pub struct AsrEngine {
    recognizer: OfflineRecognizer,
    model_name: String,
}

impl AsrEngine {
    pub fn new(model_path: &Path, tokens_path: &Path, language: Option<&str>) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!("ASR model not found at {:?}", model_path));
        }
        if !tokens_path.exists() {
            return Err(anyhow!("Tokens file not found at {:?}", tokens_path));
        }

        let sense_voice_cfg = OfflineSenseVoiceModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            language: language.map(|s| s.to_string()),
            use_itn: true,
        };

        let mut model_cfg = OfflineModelConfig::default();
        model_cfg.sense_voice = sense_voice_cfg;
        model_cfg.tokens = Some(tokens_path.to_string_lossy().to_string());
        model_cfg.num_threads = 4;
        model_cfg.debug = false;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config = model_cfg;

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("Failed to initialize sherpa-onnx OfflineRecognizer"))?;

        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("SenseVoice-Small")
            .to_string();

        Ok(Self {
            recognizer,
            model_name,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn transcribe(&self, samples: &[f32], sample_rate: i32) -> Result<AsrResult> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate, samples);
        self.recognizer.decode(&stream);

        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("No ASR result produced"))?;

        let raw_text = result.text.trim();
        let (language, emotion, clean_text) = parse_sense_voice_tags(raw_text);

        // Approximate word-level timestamp distribution if not provided directly
        let total_duration_ms = (samples.len() as f32 / sample_rate as f32 * 1000.0) as i64;
        let words = generate_char_level_alignments(&clean_text, total_duration_ms);

        Ok(AsrResult {
            text: clean_text,
            language,
            emotion,
            confidence: 0.95, // High default confidence for SenseVoice clean decoded outputs
            words,
        })
    }
}

/// Parses SenseVoice tags like <|zh|><|NEUTRAL|><|Speech|><|woitn|> text
pub fn parse_sense_voice_tags(raw: &str) -> (String, Option<String>, String) {
    let mut detected_lang = "zh".to_string();
    let mut detected_emotion = None;
    let mut clean = String::new();

    let mut cur = raw;
    while let Some(start) = cur.find("<|") {
        clean.push_str(&cur[..start]);
        let rem = &cur[start + 2..];
        if let Some(end) = rem.find("|>") {
            let tag = &rem[..end];
            match tag {
                "zh" | "en" | "ja" | "ko" | "yue" => {
                    detected_lang = tag.to_string();
                }
                "HAPPY" | "SAD" | "ANGRY" | "NEUTRAL" | "FEARFUL" | "DISGUSTED" | "SURPRISED" => {
                    detected_emotion = Some(tag.to_lowercase());
                }
                _ => {}
            }
            cur = &rem[end + 2..];
        } else {
            clean.push_str("<|");
            cur = rem;
        }
    }
    clean.push_str(cur);
    let clean = clean.trim().to_string();

    (detected_lang, detected_emotion, clean)
}

fn generate_char_level_alignments(text: &str, total_duration_ms: i64) -> Vec<WordAlignment> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() || total_duration_ms <= 0 {
        return Vec::new();
    }

    let count = chars.len() as i64;
    let step = total_duration_ms / count.max(1);

    chars
        .into_iter()
        .enumerate()
        .map(|(i, ch)| {
            let s_ms = (i as i64) * step;
            let e_ms = if (i as i64) == count - 1 {
                total_duration_ms
            } else {
                ((i as i64) + 1) * step
            };
            WordAlignment {
                word: ch.to_string(),
                start_ms: s_ms,
                end_ms: e_ms,
            }
        })
        .collect()
}
