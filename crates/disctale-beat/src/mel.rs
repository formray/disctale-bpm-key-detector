//! The model's input: `LogMelSpect` of `beat_this/preprocessing.py` —
//! 22,050 Hz mono, 1,024-point periodic Hann STFT, hop 441 (50 frames per
//! second), centred with reflection padding, magnitude scaled by
//! 1/sqrt(1024) (`normalized="frame_length"`), 128 Slaney mel bands from
//! 30 Hz to 11 kHz without area normalisation, then `log1p(1000·x)`.

use std::sync::Arc;

use crate::model::Pace;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: u32 = 22_050;
pub const N_FFT: usize = 1024;
pub const HOP: usize = 441;
pub const N_MELS: usize = 128;
const N_FREQS: usize = N_FFT / 2 + 1;
const F_MIN: f64 = 30.0;
const F_MAX: f64 = 11_000.0;

pub struct LogMel {
    window: Vec<f32>,
    /// `N_FREQS × N_MELS`, row-major, as torchaudio's `melscale_fbanks`.
    filters: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> LogMel {
        let window = (0..N_FFT)
            .map(|n| {
                let x = 2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64;
                (0.5 - 0.5 * x.cos()) as f32
            })
            .collect();
        LogMel {
            window,
            filters: slaney_filters(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// `T × 128` frames, row-major, from samples at 22,050 Hz.
    pub fn frames(&self, samples: &[f32], pace: &Pace) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let pad = N_FFT / 2;
        let padded: Vec<f32> = (0..samples.len() + 2 * pad)
            .map(|i| samples[reflect(i as isize - pad as isize, samples.len())])
            .collect();
        let count = 1 + (padded.len() - N_FFT) / HOP;
        let scale = 1.0 / (N_FFT as f32).sqrt();
        let mut out = vec![0.0f32; count * N_MELS];
        let mut buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut magnitude = vec![0.0f32; N_FREQS];
        for frame in 0..count {
            if frame.is_multiple_of(32) {
                pace.tick();
            }
            let start = frame * HOP;
            for (i, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buffer);
            for (bin, m) in magnitude.iter_mut().enumerate() {
                *m = buffer[bin].norm() * scale;
            }
            let row = &mut out[frame * N_MELS..(frame + 1) * N_MELS];
            for (bin, m) in magnitude.iter().enumerate() {
                if *m == 0.0 {
                    continue;
                }
                let filters = &self.filters[bin * N_MELS..(bin + 1) * N_MELS];
                for (acc, w) in row.iter_mut().zip(filters) {
                    *acc += m * w;
                }
            }
            for v in row.iter_mut() {
                *v = (1000.0 * *v).ln_1p();
            }
        }
        out
    }
}

/// numpy/torch "reflect": the edge sample is not repeated.
fn reflect(i: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let mut j = i.rem_euclid(period);
    if j >= len as isize {
        j = period - j;
    }
    j as usize
}

fn hz_to_mel(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

fn slaney_filters() -> Vec<f32> {
    let nyquist = SAMPLE_RATE as f64 / 2.0;
    let freqs: Vec<f64> = (0..N_FREQS)
        .map(|i| nyquist * i as f64 / (N_FREQS - 1) as f64)
        .collect();
    let (m_min, m_max) = (hz_to_mel(F_MIN), hz_to_mel(F_MAX));
    let points: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut filters = vec![0.0f32; N_FREQS * N_MELS];
    for (bin, f) in freqs.iter().enumerate() {
        for m in 0..N_MELS {
            let down = (f - points[m]) / (points[m + 1] - points[m]);
            let up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
            filters[bin * N_MELS + m] = down.min(up).max(0.0) as f32;
        }
    }
    filters
}

/// 44,100 Hz → 22,050 Hz: a 2:1 decimator behind a Kaiser-windowed sinc
/// low-pass (511 taps, cut at 10.6 kHz). Upstream uses soxr, flat to
/// 10 kHz and closed by 11 kHz; of the shapes tried on three bench clips
/// this one comes closest to it (log-mel mean difference 0.003, all of it
/// in the top two of 128 bands).
pub fn halve(samples: &[f32], pace: &Pace) -> Vec<f32> {
    const TAPS: usize = 511;
    const CUTOFF: f64 = 10_600.0 / 44_100.0;
    const BETA: f64 = 10.0;
    let centre = (TAPS / 2) as isize;
    let kernel: Vec<f32> = (0..TAPS)
        .map(|i| {
            let n = i as f64 - centre as f64;
            let sinc = if n == 0.0 {
                2.0 * CUTOFF
            } else {
                (2.0 * std::f64::consts::PI * CUTOFF * n).sin() / (std::f64::consts::PI * n)
            };
            let r = n / centre as f64;
            (sinc * bessel_i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(BETA)) as f32
        })
        .collect();
    let gain: f32 = kernel.iter().sum();
    let out_len = samples.len().div_ceil(2);
    (0..out_len)
        .map(|o| {
            if o.is_multiple_of(8192) {
                pace.tick();
            }
            let centre_at = (2 * o) as isize;
            let mut acc = 0.0f32;
            for (k, w) in kernel.iter().enumerate() {
                let i = centre_at + k as isize - centre;
                if i >= 0 && (i as usize) < samples.len() {
                    acc += samples[i as usize] * w;
                }
            }
            acc / gain
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..50 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_does_not_repeat_the_edge() {
        let got: Vec<usize> = (-3..7).map(|i| reflect(i, 4)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
    }

    #[test]
    fn frame_count_matches_a_centred_stft() {
        let mel = LogMel::new();
        let frames = mel.frames(&vec![0.1; 22_050], &Pace::flat_out());
        assert_eq!(frames.len() / N_MELS, 1 + 22_050 / HOP);
    }

    #[test]
    fn a_tone_lands_in_its_mel_band() {
        let tone: Vec<f32> = (0..22_050)
            .map(|n| (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / 22_050.0).sin())
            .collect();
        let frames = LogMel::new().frames(&tone, &Pace::flat_out());
        let row = &frames[25 * N_MELS..26 * N_MELS];
        let loudest = (0..N_MELS)
            .max_by(|a, b| row[*a].total_cmp(&row[*b]))
            .unwrap_or(0);
        let centre_hz = mel_to_hz(
            hz_to_mel(F_MIN) + (hz_to_mel(F_MAX) - hz_to_mel(F_MIN)) * (loudest + 1) as f64 / 129.0,
        );
        assert!(
            (centre_hz - 1000.0).abs() < 60.0,
            "band centre {centre_hz} Hz"
        );
    }

    #[test]
    fn halving_keeps_the_low_band_and_drops_the_high_one() {
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let tone = |hz: f32| -> Vec<f32> {
            (0..44_100)
                .map(|n| (2.0 * std::f32::consts::PI * hz * n as f32 / 44_100.0).sin())
                .collect()
        };
        let low = halve(&tone(440.0), &Pace::flat_out());
        let high = halve(&tone(15_000.0), &Pace::flat_out());
        assert_eq!(low.len(), 22_050);
        assert!((rms(&low[1000..21_000]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01);
        assert!(rms(&high[1000..21_000]) < 1e-3);
    }
}
