//! `disctale-bpm-key`: BPM and key of audio clips, one JSON line per file,
//! as DISCTALE serves them — the tempo from Beat This! (`disctale-beat`),
//! the key from `disctale-key`. Either may abstain and says why.
//!
//! Decoding is not the libraries' job: this tool reads WAV (any channel
//! count, mixed down to mono) or raw mono little-endian f32 (`.f32`, e.g.
//! `ffmpeg -i clip.mp3 -ac 1 -ar 44100 -f f32le clip.f32`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use sha2::{Digest, Sha256};

/// The official `final0` checkpoint of Beat This! (PROVENANCE.md).
const FINAL0_SHA256: &str = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331";

const USAGE: &str = "usage: disctale-bpm-key [--weights final0.ckpt] [--raw-rate HZ] FILE...

  --weights PATH   Beat This! final0 checkpoint (or BEAT_THIS_CHECKPOINT);
                   without it, only the key is read.
  --raw-rate HZ    sample rate of .f32 files (default 44100).
  FILE             .wav, or raw mono little-endian f32 (.f32).

Fetch the weights with scripts/fetch-weights.sh; they are checked against
their published checksum before use.";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut weights = std::env::var_os("BEAT_THIS_CHECKPOINT").map(PathBuf::from);
    let mut raw_rate = 44_100u32;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--weights" => weights = Some(args.next().ok_or(USAGE)?.into()),
            "--raw-rate" => {
                raw_rate = args.next().and_then(|v| v.parse().ok()).ok_or(USAGE)?;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if files.is_empty() {
        return Err(USAGE.to_string());
    }

    let tracker = match weights {
        Some(path) => Some(load_tracker(&path)?),
        None => {
            eprintln!("no weights: BPM not read (see --weights)");
            None
        }
    };

    for path in &files {
        let (samples, rate) = read_audio(path, raw_rate)?;
        let started = Instant::now();
        let (bpm, tempo_note) = match &tracker {
            None => (None, "no_weights".to_string()),
            Some(tracker) => {
                match tracker.tempo(&samples, rate, &disctale_beat::Pace::flat_out()) {
                    Ok(disctale_beat::TempoReading::Found { bpm, precise }) => {
                        (Some(bpm), format!("{precise:.2}"))
                    }
                    Ok(disctale_beat::TempoReading::Abstained(why)) => {
                        (None, why.as_str().to_string())
                    }
                    Err(error) => (None, error.to_string()),
                }
            }
        };
        let (key, open_key, key_note) =
            match disctale_key::analyze_tempo_and_key(&samples, rate).key {
                disctale_key::KeyReading::Found { key, confidence } => (
                    Some(key.traditional()),
                    Some(key.open_key()),
                    format!("{confidence:.3}"),
                ),
                disctale_key::KeyReading::Abstained(why) => (None, None, format!("{why:?}")),
            };
        let line = serde_json::json!({
            "file": path.file_name().map(|n| n.to_string_lossy()),
            "bpm": bpm,
            "tempo_note": tempo_note,
            "key": key,
            "open_key": open_key,
            "key_note": key_note,
            "ms": started.elapsed().as_millis() as u64,
        });
        println!("{line}");
    }
    Ok(())
}

fn load_tracker(path: &Path) -> Result<disctale_beat::BeatTracker, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if digest != FINAL0_SHA256 {
        return Err(format!(
            "{}: sha256 {digest} is not the official final0 checkpoint",
            path.display()
        ));
    }
    disctale_beat::BeatTracker::from_checkpoint(&bytes)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn read_audio(path: &Path, raw_rate: u32) -> Result<(Vec<f32>, u32), String> {
    let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let is_wav = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    if !is_wav {
        let bytes = std::fs::read(path).map_err(|e| fail(&e))?;
        let samples = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return Ok((samples, raw_rate));
    }
    let mut reader = hound::WavReader::open(path).map_err(|e| fail(&e))?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| fail(&e))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| fail(&e))?
        }
    };
    let channels = usize::from(spec.channels.max(1));
    let mono = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}
