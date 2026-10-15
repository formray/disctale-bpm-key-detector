//! Beat This! (CPJKU, JKU Linz; MIT) in Rust: the tempo of a clip from its
//! beats, as DISCTALE serves it. Provenance in `PROVENANCE.md`.
//!
//! One call reads one clip: decoded mono samples in, one tempo or the
//! reason there is none out. Nothing here touches the network or disk
//! except [`BeatTracker::load`], which reads the checkpoint once.

pub mod checkpoint;
pub mod mel;
pub mod model;
pub mod tempo;

pub use model::Pace;
pub use tempo::{TempoAbstention, TempoReading};

/// The production path reads at most this many frames (29.76 s): one
/// model pass, the single chunk of `split_piece` (server bench).
pub const MAX_FRAMES: usize = model::CHUNK - 2 * model::BORDER;

#[derive(Debug, thiserror::Error)]
pub enum BeatError {
    #[error("checkpoint unreadable: {0}")]
    Checkpoint(#[from] checkpoint::CheckpointError),
    #[error("checkpoint file: {0}")]
    Io(#[from] std::io::Error),
    #[error("sample rate {0} Hz is not supported (44,100 or 22,050)")]
    SampleRate(u32),
}

pub struct BeatTracker {
    model: model::BeatThis,
    mel: mel::LogMel,
}

impl BeatTracker {
    pub fn load(path: &std::path::Path) -> Result<BeatTracker, BeatError> {
        let bytes = std::fs::read(path)?;
        Self::from_checkpoint(&bytes)
    }

    pub fn from_checkpoint(bytes: &[u8]) -> Result<BeatTracker, BeatError> {
        let weights = checkpoint::read(bytes)?;
        Ok(BeatTracker {
            model: model::BeatThis::from_weights(weights)?,
            mel: mel::LogMel::new(),
        })
    }

    /// The log-mel frames the model reads, `T × 128`.
    pub fn spectrogram(
        &self,
        samples: &[f32],
        sample_rate: u32,
        pace: &Pace,
    ) -> Result<Vec<f32>, BeatError> {
        let resampled;
        let at_22k = match sample_rate {
            44_100 => {
                resampled = mel::halve(samples, pace);
                &resampled[..]
            }
            22_050 => samples,
            other => return Err(BeatError::SampleRate(other)),
        };
        Ok(self.mel.frames(at_22k, pace))
    }

    /// Frame-wise beat logits for the whole piece, chunked as upstream.
    pub fn logits(&self, spectrogram: &[f32], pace: &Pace) -> Vec<f32> {
        self.model.piece_logits(spectrogram, pace)
    }

    /// The served tempo: the first [`MAX_FRAMES`] frames, one pass.
    pub fn tempo(
        &self,
        samples: &[f32],
        sample_rate: u32,
        pace: &Pace,
    ) -> Result<TempoReading, BeatError> {
        let mut spect = self.spectrogram(samples, sample_rate, pace)?;
        spect.truncate(MAX_FRAMES * mel::N_MELS);
        let logits = self.logits(&spect, pace);
        Ok(tempo::tempo(&tempo::beats(&logits)))
    }
}
