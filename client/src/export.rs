use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::db::UtteranceRecord;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportOptions {
    pub format: String, // "jsonl" | "gpt_sovits" | "ljspeech" | "srt_md" | "all"
    pub target_dir: String,
    pub session_id: Option<String>,
    pub only_verified: bool,
    pub val_ratio: f32, // e.g. 0.1 for 10%
    pub copy_audio: bool,
}

pub struct Exporter;

impl Exporter {
    pub fn export(records: &[UtteranceRecord], options: &ExportOptions) -> Result<String> {
        let target_path = PathBuf::from(&options.target_dir);
        fs::create_dir_all(&target_path)
            .context("Failed to create export target directory")?;

        let wavs_dir = target_path.join("wavs");
        if options.copy_audio {
            fs::create_dir_all(&wavs_dir)
                .context("Failed to create export wavs directory")?;
        }

        // Filter records
        let filtered: Vec<&UtteranceRecord> = records
            .iter()
            .filter(|r| {
                if options.only_verified && !r.verified {
                    return false;
                }
                if r.status == "Rejected" {
                    return false;
                }
                true
            })
            .collect();

        if filtered.is_empty() {
            return Ok("No matching records to export".to_string());
        }

        // Copy audio if requested
        if options.copy_audio {
            for rec in &filtered {
                let src = Path::new(&rec.audio_filepath);
                if src.exists() {
                    let dest_filename = format!("{}.wav", rec.id);
                    let dest = wavs_dir.join(dest_filename);
                    let _ = fs::copy(src, dest);
                }
            }
        }

        let fmt = options.format.to_lowercase();
        if fmt == "jsonl" || fmt == "all" {
            Self::export_jsonl(&filtered, &target_path)?;
        }
        if fmt == "gpt_sovits" || fmt == "all" {
            Self::export_gpt_sovits(&filtered, &target_path, options.val_ratio)?;
        }
        if fmt == "ljspeech" || fmt == "all" {
            Self::export_ljspeech(&filtered, &target_path)?;
        }
        if fmt == "srt_md" || fmt == "all" {
            Self::export_srt_and_markdown(&filtered, &target_path)?;
        }

        Ok(format!("Successfully exported {} records to {:?}", filtered.len(), target_path))
    }

    fn export_jsonl(records: &[&UtteranceRecord], target_dir: &Path) -> Result<()> {
        let file = File::create(target_dir.join("metadata.jsonl"))?;
        let mut writer = BufWriter::new(file);

        for rec in records {
            let line = serde_json::to_string(rec)?;
            writeln!(writer, "{}", line)?;
        }
        writer.flush()?;
        Ok(())
    }

    fn export_gpt_sovits(records: &[&UtteranceRecord], target_dir: &Path, val_ratio: f32) -> Result<()> {
        let train_file = File::create(target_dir.join("train.list"))?;
        let mut train_writer = BufWriter::new(train_file);

        let mut val_writer = if val_ratio > 0.0 {
            let val_file = File::create(target_dir.join("val.list"))?;
            Some(BufWriter::new(val_file))
        } else {
            None
        };

        for (i, rec) in records.iter().enumerate() {
            let wav_rel_path = format!("wavs/{}.wav", rec.id);
            let speaker = if rec.speaker.is_empty() { "default_user" } else { &rec.speaker };
            let lang = if rec.language.is_empty() { "zh" } else { &rec.language };
            let text = if !rec.text_normalized.is_empty() {
                &rec.text_normalized
            } else {
                &rec.text
            };

            // Standard GPT-SoVITS 4-column format: wav_path|speaker|language|text
            let line = format!("{}|{}|{}|{}", wav_rel_path, speaker, lang, text);

            let is_val = val_writer.is_some() && (i % 10 < (val_ratio * 10.0) as usize);
            if is_val {
                if let Some(ref mut w) = val_writer {
                    writeln!(w, "{}", line)?;
                }
            } else {
                writeln!(train_writer, "{}", line)?;
            }
        }

        train_writer.flush()?;
        if let Some(ref mut w) = val_writer {
            w.flush()?;
        }
        Ok(())
    }

    fn export_ljspeech(records: &[&UtteranceRecord], target_dir: &Path) -> Result<()> {
        let file = File::create(target_dir.join("metadata.csv"))?;
        let mut writer = BufWriter::new(file);

        for rec in records {
            let raw_text = &rec.text;
            let norm_text = if !rec.text_normalized.is_empty() {
                &rec.text_normalized
            } else {
                &rec.text
            };
            // id|raw_text|normalized_text
            writeln!(writer, "{}|{}|{}", rec.id, raw_text, norm_text)?;
        }
        writer.flush()?;
        Ok(())
    }

    fn export_srt_and_markdown(records: &[&UtteranceRecord], target_dir: &Path) -> Result<()> {
        let srt_file = File::create(target_dir.join("transcript.srt"))?;
        let mut srt_writer = BufWriter::new(srt_file);

        let md_file = File::create(target_dir.join("transcript.md"))?;
        let mut md_writer = BufWriter::new(md_file);

        writeln!(md_writer, "# 语音转写语料导出文档\n")?;

        for (idx, rec) in records.iter().enumerate() {
            let seq = idx + 1;
            let start_time = format_timestamp_srt(rec.session_offset_start_ms);
            let end_time = format_timestamp_srt(rec.session_offset_end_ms);

            // SRT Block
            writeln!(srt_writer, "{}", seq)?;
            writeln!(srt_writer, "{} --> {}", start_time, end_time)?;
            writeln!(srt_writer, "{}\n", rec.text)?;

            // Markdown Line
            writeln!(
                md_writer,
                "- **[{}]** ({}) {}",
                format_timestamp_md(rec.session_offset_start_ms),
                rec.speaker,
                rec.text
            )?;
        }

        srt_writer.flush()?;
        md_writer.flush()?;
        Ok(())
    }
}

fn format_timestamp_srt(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let millis = ms % 1000;
    let seconds = total_seconds % 60;
    let minutes = (total_seconds / 60) % 60;
    let hours = total_seconds / 3600;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

fn format_timestamp_md(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let millis = ms % 1000;
    let seconds = total_seconds % 60;
    let minutes = (total_seconds / 60) % 60;
    let hours = total_seconds / 3600;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}
