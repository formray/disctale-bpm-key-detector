//! From beat logits to one tempo — DISCTALE's rule (written before the
//! benches on 15/10/2026, corrected once on the development bench).
//!
//! Beats are the minimal postprocessor of `beat_this/model/postprocessor.py`:
//! a local maximum within ±3 frames whose logit is above 0. Then:
//! 1. fewer than 16 beats: abstain;
//! 2. fewer than 75 % of the beat intervals within 15 % of their median:
//!    abstain (the tempo is not steady);
//! 3. a grid: a beat closer than half a period to the last kept one is
//!    dropped, a gap counts its missing beats; a least-squares line through
//!    the beats gives the period; twice, every beat is re-placed on the
//!    fitted grid and those more than 0.2 periods off it are left out;
//! 4. outside 40–250 BPM: abstain. Never doubled or halved.

pub const FPS: f64 = 50.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempoAbstention {
    FewBeats,
    Unstable,
    OutOfRange,
}

impl TempoAbstention {
    pub fn as_str(self) -> &'static str {
        match self {
            TempoAbstention::FewBeats => "few_beats",
            TempoAbstention::Unstable => "unstable",
            TempoAbstention::OutOfRange => "out_of_range",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TempoReading {
    /// `bpm` is the served value, `precise` the line's slope before rounding.
    Found {
        bpm: u16,
        precise: f64,
    },
    Abstained(TempoAbstention),
}

/// Beat times in seconds.
pub fn beats(logits: &[f32]) -> Vec<f64> {
    (0..logits.len())
        .filter(|&i| {
            let lo = i.saturating_sub(3);
            let hi = (i + 4).min(logits.len());
            let window_max = logits[lo..hi]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            logits[i] > 0.0 && logits[i] == window_max
        })
        .map(|i| i as f64 / FPS)
        .collect()
}

pub fn tempo(beats: &[f64]) -> TempoReading {
    const MIN_BEATS: usize = 16;
    if beats.len() < MIN_BEATS {
        return TempoReading::Abstained(TempoAbstention::FewBeats);
    }
    let intervals: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    let period = median(&intervals);
    let steady = intervals
        .iter()
        .filter(|i| (*i / period - 1.0).abs() <= 0.15)
        .count();
    if (steady as f64) < 0.75 * intervals.len() as f64 {
        return TempoReading::Abstained(TempoAbstention::Unstable);
    }

    let mut index = vec![0.0f64];
    let mut kept = vec![beats[0]];
    let (mut last, mut k) = (beats[0], 0.0f64);
    for &t in &beats[1..] {
        let step = ((t - last) / period).round_ties_even();
        if step == 0.0 {
            continue;
        }
        k += step;
        last = t;
        index.push(k);
        kept.push(t);
    }
    let (mut slope, mut intercept) = line(&index, &kept);
    for _ in 0..2 {
        let mut ks = Vec::with_capacity(beats.len());
        let mut ts = Vec::with_capacity(beats.len());
        for &t in beats {
            let kk = ((t - intercept) / slope).round_ties_even();
            if (t - (intercept + slope * kk)).abs() <= 0.2 * slope {
                ks.push(kk);
                ts.push(t);
            }
        }
        if ks.len() < MIN_BEATS {
            return TempoReading::Abstained(TempoAbstention::FewBeats);
        }
        (slope, intercept) = line(&ks, &ts);
    }
    let precise = 60.0 / slope;
    if !(40.0..=250.0).contains(&precise) {
        return TempoReading::Abstained(TempoAbstention::OutOfRange);
    }
    TempoReading::Found {
        bpm: precise.round_ties_even() as u16,
        precise,
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Least squares `y = slope·x + intercept`.
fn line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(bpm: f64, count: usize) -> Vec<f64> {
        (0..count).map(|i| 0.3 + i as f64 * 60.0 / bpm).collect()
    }

    #[test]
    fn a_steady_grid_reads_its_tempo() {
        assert!(matches!(
            tempo(&grid(124.68, 60)),
            TempoReading::Found { bpm: 125, .. }
        ));
    }

    #[test]
    fn a_stray_beat_half_a_period_off_does_not_move_the_tempo() {
        let mut beats = grid(124.0, 60);
        beats.insert(30, beats[29] + 0.2);
        beats.remove(32);
        let TempoReading::Found { precise, .. } = tempo(&beats) else {
            panic!("abstained");
        };
        assert!((precise - 124.0).abs() < 0.05, "{precise}");
    }

    #[test]
    fn too_few_beats_abstain() {
        assert_eq!(
            tempo(&grid(120.0, 15)),
            TempoReading::Abstained(TempoAbstention::FewBeats)
        );
    }

    #[test]
    fn a_tempo_that_wanders_abstains() {
        let mut beats = vec![0.0];
        for i in 0..40 {
            let step = if i % 2 == 0 { 0.30 } else { 0.62 };
            beats.push(beats[beats.len() - 1] + step);
        }
        assert_eq!(
            tempo(&beats),
            TempoReading::Abstained(TempoAbstention::Unstable)
        );
    }

    #[test]
    fn peaks_are_local_maxima_above_zero() {
        let mut logits = vec![-5.0f32; 200];
        logits[10] = 2.0;
        logits[12] = 1.0;
        logits[60] = -0.5;
        logits[100] = 3.0;
        assert_eq!(beats(&logits), vec![10.0 / FPS, 100.0 / FPS]);
    }
}
