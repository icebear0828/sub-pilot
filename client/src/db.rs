use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub mode: String,
    pub speaker: String,
    pub utterance_count: i64,
    pub total_duration_ms: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WordAlignment {
    #[serde(rename = "w")]
    pub word: String,
    #[serde(rename = "s_ms")]
    pub start_ms: i64,
    #[serde(rename = "e_ms")]
    pub end_ms: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtteranceRecord {
    pub id: String,
    pub session_id: String,
    pub audio_filepath: String,
    pub master_filepath: Option<String>,
    pub sample_rate: i32,
    pub channels: i32,
    pub bit_depth: i32,
    pub duration_ms: i64,
    pub wall_clock_start: String,
    pub wall_clock_end: String,
    pub session_offset_start_ms: i64,
    pub session_offset_end_ms: i64,
    pub speaker: String,
    pub language: String,
    pub emotion: Option<String>,
    pub text: String,
    pub text_normalized: String,
    pub asr_model: String,
    pub asr_confidence: f32,
    pub peak_dbfs: f32,
    pub integrated_lufs: Option<f32>,
    pub verified: bool,
    pub split: String,
    pub words: Vec<WordAlignment>,
    pub status: String, // "Pending_ASR", "Transcribed", "Verified", "Rejected"
    pub created_at: String,
}

pub struct Database {
    db_path: PathBuf,
}

impl Database {
    pub fn new(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .context("Failed to create data directory")?;
        let db_path = data_dir.join("voiceforge.db");
        let conn = Connection::open(&db_path)
            .context("Failed to open SQLite database")?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;

             CREATE TABLE IF NOT EXISTS sessions (
                 id TEXT PRIMARY KEY,
                 name TEXT NOT NULL,
                 created_at TEXT NOT NULL,
                 mode TEXT NOT NULL,
                 speaker TEXT NOT NULL DEFAULT 'default_user'
             );

             CREATE TABLE IF NOT EXISTS utterances (
                 id TEXT PRIMARY KEY,
                 session_id TEXT NOT NULL,
                 audio_filepath TEXT NOT NULL,
                 master_filepath TEXT,
                 sample_rate INTEGER NOT NULL,
                 channels INTEGER NOT NULL,
                 bit_depth INTEGER NOT NULL,
                 duration_ms INTEGER NOT NULL,
                 wall_clock_start TEXT NOT NULL,
                 wall_clock_end TEXT NOT NULL,
                 session_offset_start_ms INTEGER NOT NULL,
                 session_offset_end_ms INTEGER NOT NULL,
                 speaker TEXT NOT NULL DEFAULT 'default_user',
                 language TEXT NOT NULL DEFAULT 'zh',
                 emotion TEXT,
                 text TEXT NOT NULL DEFAULT '',
                 text_normalized TEXT NOT NULL DEFAULT '',
                 asr_model TEXT NOT NULL DEFAULT '',
                 asr_confidence REAL NOT NULL DEFAULT 0.0,
                 peak_dbfs REAL NOT NULL DEFAULT -100.0,
                 integrated_lufs REAL,
                 verified INTEGER NOT NULL DEFAULT 0,
                 split TEXT NOT NULL DEFAULT 'train',
                 words_json TEXT NOT NULL DEFAULT '[]',
                 status TEXT NOT NULL DEFAULT 'Pending_ASR',
                 created_at TEXT NOT NULL,
                 FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
             );

             CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id);
             CREATE INDEX IF NOT EXISTS idx_utterances_status ON utterances(status);
             CREATE INDEX IF NOT EXISTS idx_utterances_created ON utterances(created_at);
            "
        ).context("Failed to initialize database tables")?;

        Ok(Self { db_path })
    }

    fn connect(&self) -> Result<Connection> {
        Connection::open(&self.db_path).context("Failed to connect to database")
    }

    pub fn get_or_create_session(&self, session_id: &str, mode: &str, speaker: &str) -> Result<SessionRecord> {
        let conn = self.connect()?;
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO sessions (id, name, created_at, mode, speaker)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO NOTHING",
            params![session_id, format!("Session {}", session_id), now, mode, speaker],
        )?;

        self.get_session(session_id)
            .and_then(|opt| opt.ok_or_else(|| anyhow::anyhow!("Session not found after creation")))
    }

    pub fn get_session(&self, session_id: &str) -> Result<Option<SessionRecord>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.created_at, s.mode, s.speaker,
                    COUNT(u.id) as count,
                    COALESCE(SUM(u.duration_ms), 0) as total_duration
             FROM sessions s
             LEFT JOIN utterances u ON s.id = u.session_id AND u.status != 'Rejected'
             WHERE s.id = ?1
             GROUP BY s.id"
        )?;

        let res = stmt.query_row(params![session_id], |row| {
            Ok(SessionRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                mode: row.get(3)?,
                speaker: row.get(4)?,
                utterance_count: row.get(5)?,
                total_duration_ms: row.get(6)?,
            })
        }).optional()?;

        Ok(res)
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionRecord>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.created_at, s.mode, s.speaker,
                    COUNT(u.id) as count,
                    COALESCE(SUM(u.duration_ms), 0) as total_duration
             FROM sessions s
             LEFT JOIN utterances u ON s.id = u.session_id AND u.status != 'Rejected'
             GROUP BY s.id
             ORDER BY s.created_at DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(SessionRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                mode: row.get(3)?,
                speaker: row.get(4)?,
                utterance_count: row.get(5)?,
                total_duration_ms: row.get(6)?,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn insert_utterance(&self, record: &UtteranceRecord) -> Result<()> {
        let conn = self.connect()?;
        let words_json = serde_json::to_string(&record.words).unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "INSERT INTO utterances (
                id, session_id, audio_filepath, master_filepath,
                sample_rate, channels, bit_depth, duration_ms,
                wall_clock_start, wall_clock_end, session_offset_start_ms, session_offset_end_ms,
                speaker, language, emotion, text, text_normalized,
                asr_model, asr_confidence, peak_dbfs, integrated_lufs,
                verified, split, words_json, status, created_at
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26
             )",
            params![
                record.id,
                record.session_id,
                record.audio_filepath,
                record.master_filepath,
                record.sample_rate,
                record.channels,
                record.bit_depth,
                record.duration_ms,
                record.wall_clock_start,
                record.wall_clock_end,
                record.session_offset_start_ms,
                record.session_offset_end_ms,
                record.speaker,
                record.language,
                record.emotion,
                record.text,
                record.text_normalized,
                record.asr_model,
                record.asr_confidence,
                record.peak_dbfs,
                record.integrated_lufs,
                if record.verified { 1 } else { 0 },
                record.split,
                words_json,
                record.status,
                record.created_at,
            ],
        )?;

        Ok(())
    }

    pub fn update_utterance_text(
        &self,
        id: &str,
        text: &str,
        text_normalized: Option<&str>,
        verified: bool,
    ) -> Result<()> {
        let conn = self.connect()?;
        let norm = text_normalized.unwrap_or(text);
        let status = if verified { "Verified" } else { "Transcribed" };

        conn.execute(
            "UPDATE utterances
             SET text = ?1, text_normalized = ?2, verified = ?3, status = ?4
             WHERE id = ?5",
            params![text, norm, if verified { 1 } else { 0 }, status, id],
        )?;

        Ok(())
    }

    pub fn update_utterance_asr(
        &self,
        id: &str,
        text: &str,
        language: &str,
        emotion: Option<&str>,
        confidence: f32,
        model: &str,
        words: &[WordAlignment],
    ) -> Result<()> {
        let conn = self.connect()?;
        let words_json = serde_json::to_string(words).unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "UPDATE utterances
             SET text = ?1, text_normalized = ?1, language = ?2, emotion = ?3,
                 asr_confidence = ?4, asr_model = ?5, words_json = ?6, status = 'Transcribed'
             WHERE id = ?7 AND status = 'Pending_ASR'",
            params![text, language, emotion, confidence, model, words_json, id],
        )?;

        Ok(())
    }

    pub fn mark_utterance_status(&self, id: &str, status: &str) -> Result<()> {
        let conn = self.connect()?;
        let verified = if status == "Verified" { 1 } else { 0 };
        conn.execute(
            "UPDATE utterances SET status = ?1, verified = ?2 WHERE id = ?3",
            params![status, verified, id],
        )?;
        Ok(())
    }

    pub fn delete_utterance(&self, id: &str) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM utterances WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn list_utterances(
        &self,
        session_id: Option<&str>,
        status_filter: Option<&str>,
    ) -> Result<Vec<UtteranceRecord>> {
        let conn = self.connect()?;
        let mut query = "SELECT id, session_id, audio_filepath, master_filepath,
                                sample_rate, channels, bit_depth, duration_ms,
                                wall_clock_start, wall_clock_end, session_offset_start_ms, session_offset_end_ms,
                                speaker, language, emotion, text, text_normalized,
                                asr_model, asr_confidence, peak_dbfs, integrated_lufs,
                                verified, split, words_json, status, created_at
                         FROM utterances WHERE 1=1".to_string();

        let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(sid) = session_id {
            query.push_str(" AND session_id = ?");
            param_values.push(Box::new(sid.to_string()));
        }

        if let Some(sf) = status_filter {
            if sf != "all" {
                query.push_str(" AND status = ?");
                param_values.push(Box::new(sf.to_string()));
            }
        }

        query.push_str(" ORDER BY created_at ASC");

        let mut stmt = conn.prepare(&query)?;
        let params_slice: Vec<&dyn rusqlite::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(params_slice.as_slice(), |row| {
            let words_json_str: String = row.get(23)?;
            let words: Vec<WordAlignment> = serde_json::from_str(&words_json_str).unwrap_or_default();
            let verified_int: i32 = row.get(21)?;

            Ok(UtteranceRecord {
                id: row.get(0)?,
                session_id: row.get(1)?,
                audio_filepath: row.get(2)?,
                master_filepath: row.get(3)?,
                sample_rate: row.get(4)?,
                channels: row.get(5)?,
                bit_depth: row.get(6)?,
                duration_ms: row.get(7)?,
                wall_clock_start: row.get(8)?,
                wall_clock_end: row.get(9)?,
                session_offset_start_ms: row.get(10)?,
                session_offset_end_ms: row.get(11)?,
                speaker: row.get(12)?,
                language: row.get(13)?,
                emotion: row.get(14)?,
                text: row.get(15)?,
                text_normalized: row.get(16)?,
                asr_model: row.get(17)?,
                asr_confidence: row.get(18)?,
                peak_dbfs: row.get(19)?,
                integrated_lufs: row.get(20)?,
                verified: verified_int != 0,
                split: row.get(22)?,
                words,
                status: row.get(24)?,
                created_at: row.get(25)?,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn get_pending_asr_ids(&self, session_id: Option<&str>) -> Result<Vec<(String, String)>> {
        let conn = self.connect()?;
        let mut query = "SELECT id, audio_filepath FROM utterances WHERE status = 'Pending_ASR'".to_string();
        let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(sid) = session_id {
            query.push_str(" AND session_id = ?");
            param_values.push(Box::new(sid.to_string()));
        }

        let mut stmt = conn.prepare(&query)?;
        let params_slice: Vec<&dyn rusqlite::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(params_slice.as_slice(), |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn get_total_utterances_count(&self, session_id: Option<&str>) -> Result<usize> {
        let conn = self.connect()?;
        let count: i64 = if let Some(sid) = session_id {
            conn.query_row(
                "SELECT COUNT(*) FROM utterances WHERE session_id = ?1 AND status != 'Rejected'",
                params![sid],
                |r| r.get(0),
            )?
        } else {
            conn.query_row(
                "SELECT COUNT(*) FROM utterances WHERE status != 'Rejected'",
                [],
                |r| r.get(0),
            )?
        };
        Ok(count as usize)
    }
}
