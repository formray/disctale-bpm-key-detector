//! Tempo and key of a stretch of audio, or the honest reason there is none.
//!
//! Written for DISCTALE, which serves the key from it (the served tempo comes
//! from Beat This!, crate `disctale-beat`). A pure function with no
//! dependency beyond the standard library: the FFT is the textbook radix-2
//! below. A first version was measured on records and found wanting; this
//! one rewrote both readings.
//!
//! The governing rule is the plan's: **a wrong value is worth less than no
//! value.** Each reading either gives a value or names why it abstains.
//!
//! - **Tempo** comes from how the onset envelope repeats itself: its
//!   autocorrelation, read at every period between 50 and 220 BPM and at
//!   that period's multiples (Scheirer's comb filters, 1998), times a
//!   log-normal preference for tempi near 120 (as in Klapuri, Eronen and
//!   Astola, 2006). House keeps its off-beat as loud as its beat, so nothing
//!   here asks the half-beat to be quiet. The reading abstains when the half
//!   period repeats almost as well as the period (double or half time), when
//!   a period at 3:2 repeats better (a figure in threes), or when ten-second
//!   windows read tempi that are not related.
//! - **Key** comes from the twelve-note profile of the partials of the whole
//!   clip, the bass apart and added at half weight, compared with major and
//!   minor profiles measured on electronic dance music (Faraldo, Jordà and
//!   Herrera, 2017) in all twelve transpositions. Too little held sound,
//!   fewer notes than a triad, or a best key too close to the runner-up,
//!   abstains.
//!
//! The synthetic tests fix the behaviour; the thresholds were set on 83
//! development clips, and the held-out bench reads 96 clips never used here
//! (README, «Bench»).

use std::f64::consts::PI;

/// Shortest clip either reading accepts: one tempo window.
pub const MIN_SECONDS: f64 = 10.0;

/// Both readings of one clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoKeyAnalysis {
    pub tempo: TempoReading,
    pub key: KeyReading,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TempoReading {
    /// `confidence` in 0..=1: how exactly the onsets repeat at the period,
    /// less the room left by its strongest rival at double or 3:2.
    Found {
        bpm: f64,
        confidence: f64,
    },
    Abstained(TempoAbstention),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempoAbstention {
    /// No sample rate, or a sample that is not a finite number.
    InvalidInput,
    /// Shorter than [`MIN_SECONDS`].
    TooShort,
    Silent,
    /// No periodic onsets clearly above the rest, in at least half the clip.
    NoSteadyPulse,
    /// Parts of the clip are clear but at different tempi.
    UnstableTempo,
    /// The tempo and its double (or half) fit about equally well.
    DoubleOrHalfTime,
    /// The tempo and one at 3:2 from it fit about equally well: a figure
    /// in threes across the beat, or triplets.
    ThreeAgainstTwo,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyReading {
    /// `confidence` in 0..=1: how far the best key's correlation with its
    /// profile is above the runner-up's.
    Found {
        key: MusicalKey,
        confidence: f64,
    },
    Abstained(KeyAbstention),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAbstention {
    /// No sample rate, or a sample that is not a finite number.
    InvalidInput,
    /// Shorter than [`MIN_SECONDS`].
    TooShort,
    Silent,
    /// Almost nothing in the clip is a held note: noise or percussion.
    NoTonalContent,
    /// Fewer than three distinct notes, or two keys, often relative or
    /// parallel, that fit about equally well.
    AmbiguousKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyMode {
    Major,
    Minor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MusicalKey {
    /// Pitch class of the tonic, C = 0 up to B = 11.
    pub tonic: u8,
    pub mode: KeyMode,
}

const NOTE_NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];

impl MusicalKey {
    /// Open Key notation: 1d is C major, 1m A minor, each step a fifth up.
    pub fn open_key(&self) -> String {
        let tonic = u32::from(self.tonic % 12);
        match self.mode {
            KeyMode::Major => format!("{}d", tonic * 7 % 12 + 1),
            KeyMode::Minor => format!("{}m", (tonic + 3) * 7 % 12 + 1),
        }
    }

    /// The name with sharps, "m" for minor, as GetSongBPM writes it: "A♯m".
    pub fn traditional(&self) -> String {
        let name = NOTE_NAMES[usize::from(self.tonic % 12)];
        match self.mode {
            KeyMode::Major => name.to_string(),
            KeyMode::Minor => format!("{name}m"),
        }
    }
}

/// Reads tempo and key from mono `samples` at `sample_rate` Hz.
pub fn analyze_tempo_and_key(samples: &[f32], sample_rate: u32) -> TempoKeyAnalysis {
    match prepare(samples, sample_rate) {
        Err(problem) => TempoKeyAnalysis {
            tempo: TempoReading::Abstained(match problem {
                Unreadable::Invalid => TempoAbstention::InvalidInput,
                Unreadable::TooShort => TempoAbstention::TooShort,
                Unreadable::Silent => TempoAbstention::Silent,
            }),
            key: KeyReading::Abstained(match problem {
                Unreadable::Invalid => KeyAbstention::InvalidInput,
                Unreadable::TooShort => KeyAbstention::TooShort,
                Unreadable::Silent => KeyAbstention::Silent,
            }),
        },
        Ok(signal) => {
            let rate = f64::from(sample_rate);
            TempoKeyAnalysis {
                tempo: read_tempo(&signal, rate),
                key: read_key(&signal, rate),
            }
        }
    }
}

// ------------------------------------------------------------------ input

enum Unreadable {
    Invalid,
    TooShort,
    Silent,
}

/// Below this RMS (−80 dBFS) the clip is treated as silence.
const SILENCE_RMS: f64 = 1e-4;

/// The samples as f64 scaled to unit RMS, so the log compression of the
/// spectra sees the same levels from a quiet clip and a loud one.
fn prepare(samples: &[f32], sample_rate: u32) -> Result<Vec<f64>, Unreadable> {
    if !(4_000..=384_000).contains(&sample_rate) || samples.iter().any(|s| !s.is_finite()) {
        return Err(Unreadable::Invalid);
    }
    if (samples.len() as f64) < MIN_SECONDS * f64::from(sample_rate) {
        return Err(Unreadable::TooShort);
    }
    let rms =
        (samples.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    if rms < SILENCE_RMS {
        return Err(Unreadable::Silent);
    }
    Ok(samples.iter().map(|&s| f64::from(s) / rms).collect())
}

// ------------------------------------------------------------------ spectra

/// In-place iterative radix-2 FFT; `re.len()` is a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let (step_re, step_im) = (angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let (mut w_re, mut w_im) = (1.0, 0.0);
            for k in 0..len / 2 {
                let a = start + k;
                let b = a + len / 2;
                let t_re = re[b] * w_re - im[b] * w_im;
                let t_im = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next = w_re * step_re - w_im * step_im;
                w_im = w_re * step_im + w_im * step_re;
                w_re = next;
            }
        }
        len <<= 1;
    }
}

/// Calls `each` with the magnitude spectrum (bins 0..=frame/2) of every
/// Hann-windowed frame. A sinusoid of amplitude A reads A/2 at its bin.
fn for_each_spectrum(signal: &[f64], frame: usize, hop: usize, mut each: impl FnMut(&[f64])) {
    let window: Vec<f64> = (0..frame)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / frame as f64).cos())
        .collect();
    let scale = 2.0 / frame as f64;
    let mut re = vec![0.0; frame];
    let mut im = vec![0.0; frame];
    let mut magnitude = vec![0.0; frame / 2 + 1];
    let mut start = 0;
    while start + frame <= signal.len() {
        for n in 0..frame {
            re[n] = signal[start + n] * window[n];
            im[n] = 0.0;
        }
        fft(&mut re, &mut im);
        for (k, m) in magnitude.iter_mut().enumerate() {
            *m = (re[k] * re[k] + im[k] * im[k]).sqrt() * scale;
        }
        each(&magnitude);
        start += hop;
    }
}

// ------------------------------------------------------------------ tempo

/// Onset envelope frames per second.
const ENVELOPE_RATE: f64 = 200.0;
/// The onset envelope reads this band, so every sample rate sees the same.
const ONSET_LOW_HZ: f64 = 30.0;
const ONSET_TOP_HZ: f64 = 5_000.0;
const ONSET_BANDS: usize = 40;
/// Log compression of magnitudes before the flux.
const ONSET_COMPRESSION: f64 = 100.0;
const MIN_BPM: f64 = 50.0;
const MAX_BPM: f64 = 220.0;
/// A period counts the envelope's self-similarity at this many multiples of
/// itself: a beat repeats after one, two, three and four beats.
const HARMONICS: usize = 4;
/// Candidate periods step by this fraction of an envelope frame.
const LAG_STEP: f64 = 0.05;
/// Tempi nearer than this ratio are the same peak.
const SAME_PEAK: f64 = 1.04;
/// Readings of two windows further apart than this ratio disagree.
const WINDOW_AGREEMENT: f64 = 1.02;
const WINDOW_SECONDS: f64 = 10.0;
const WINDOW_HOP_SECONDS: f64 = 5.0;
/// The preference for tempi near 120 BPM, in octaves of standard deviation.
const PRIOR_CENTRE_BPM: f64 = 120.0;
const PRIOR_WIDTH_OCTAVES: f64 = 1.0;
/// Below this self-similarity at the period itself (1 is a perfect
/// repetition, 0 none) there is no pulse to speak of.
const MIN_PERIODICITY: f64 = 0.1;
/// The envelope's self-similarity at half the period, as a share of its
/// self-similarity at the period, at or above which twice the tempo is a
/// real candidate: below about 110 BPM a record at twice the tempo is as
/// plausible, and only the accent on every other half tells them apart.
const OCTAVE_RIVAL: f64 = 0.5;
/// At or above this share the half period repeats as well as the period:
/// nothing marks every other one, so the faster reading is the beat,
/// unless the prior clearly prefers the slower one.
const NO_ACCENT: f64 = 0.9;
/// The prior of the faster reading, as a share of the slower one's, at or
/// above which it does not clearly prefer the slower one.
const PRIOR_TIE: f64 = 0.9;
/// A period 3:2 from the chosen one that repeats at least as well: only the
/// preference for tempi near 120 chose between them.
const THREE_TWO_RIVAL: f64 = 1.0;

fn read_tempo(signal: &[f64], rate: f64) -> TempoReading {
    let (envelope, fps) = onset_envelope(signal, rate);
    let window = ((WINDOW_SECONDS * fps) as usize).min(envelope.len());
    let count = if envelope.len() <= window {
        1
    } else {
        ((envelope.len() - window) as f64 / (WINDOW_HOP_SECONDS * fps)).round() as usize + 1
    };
    let windows: Vec<f64> = (0..count)
        .filter_map(|i| {
            let start = if count == 1 {
                0
            } else {
                i * (envelope.len() - window) / (count - 1)
            };
            let pulse = Pulse::of(&envelope[start..start + window], fps)?;
            (pulse.clarity >= MIN_PERIODICITY).then(|| 60.0 * fps / pulse.lag)
        })
        .collect();
    // Ten-second windows read on their own: any two that disagree beyond an
    // octave or 3:2 make the tempo unstable, and most must agree with the
    // whole clip.
    let unrelated = |a: f64, b: f64| {
        (a / b).ln().abs() > WINDOW_AGREEMENT.ln() && relation(a / b) == Relation::Other
    };
    if windows
        .iter()
        .any(|&a| windows.iter().any(|&b| unrelated(a, b)))
    {
        return TempoReading::Abstained(TempoAbstention::UnstableTempo);
    }

    let Some(pulse) = Pulse::of(&envelope, fps).filter(|p| p.clarity >= MIN_PERIODICITY) else {
        return TempoReading::Abstained(TempoAbstention::NoSteadyPulse);
    };
    let bpm = 60.0 * fps / pulse.lag;
    if windows.iter().any(|&w| unrelated(w, bpm)) {
        return TempoReading::Abstained(TempoAbstention::UnstableTempo);
    }
    let agreeing = windows
        .iter()
        .filter(|&&w| (w / bpm).ln().abs() <= WINDOW_AGREEMENT.ln())
        .count();
    if agreeing * 2 < count {
        let octave = windows
            .iter()
            .any(|&w| relation(w / bpm) == Relation::Octave);
        return TempoReading::Abstained(if octave {
            TempoAbstention::DoubleOrHalfTime
        } else {
            TempoAbstention::NoSteadyPulse
        });
    }
    if pulse.half >= OCTAVE_RIVAL {
        return TempoReading::Abstained(TempoAbstention::DoubleOrHalfTime);
    }
    if pulse.three_two >= THREE_TWO_RIVAL {
        return TempoReading::Abstained(TempoAbstention::ThreeAgainstTwo);
    }
    let rival = (pulse.half / OCTAVE_RIVAL).max(pulse.three_two / THREE_TWO_RIVAL);
    TempoReading::Found {
        bpm: 60.0 * fps / refine(&envelope, pulse.lag),
        confidence: (pulse.clarity * (1.0 - rival)).clamp(0.0, 1.0),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relation {
    Octave,
    ThreeTwo,
    Other,
}

/// How two periods (or tempi) whose ratio is `ratio` relate.
fn relation(ratio: f64) -> Relation {
    let near = |target: f64| (ratio.ln().abs() - target.ln()).abs() < SAME_PEAK.ln();
    if near(2.0) {
        Relation::Octave
    } else if near(1.5) {
        Relation::ThreeTwo
    } else {
        Relation::Other
    }
}

/// The period at which an envelope best repeats itself, and its rivals.
struct Pulse {
    /// In envelope frames.
    lag: f64,
    /// The envelope's self-similarity at the period itself.
    clarity: f64,
    /// The self-similarity at half the period as a share of that at the
    /// period, 0 when twice the tempo is out of range: near 1 nothing
    /// accents every other half, and the tempo could be twice as fast.
    half: f64,
    /// How well the periods at 3:2 and 2:3 repeat with their multiples, the
    /// better of the two, as a share of the chosen period's.
    three_two: f64,
}

impl Pulse {
    /// Every period between [`MAX_BPM`] and [`MIN_BPM`] is scored by how
    /// well the envelope repeats at it and its multiples ([`comb`]), times
    /// the prior; the best peak of that score is the period. A period twice
    /// as long repeats as well as the beat itself, so between octaves the
    /// prior decides, unless the half period repeats as well as the period.
    fn of(envelope: &[f64], fps: f64) -> Option<Pulse> {
        let lag_of = |bpm: f64| 60.0 * fps / bpm;
        let (shortest, longest) = (lag_of(MAX_BPM), lag_of(MIN_BPM));
        let reach = (HARMONICS as f64 * longest * SAME_PEAK).ceil() as usize + 2;
        if envelope.len() <= reach {
            return None;
        }
        let similarity = self_similarity(envelope, reach);
        let steps = ((longest - shortest) / LAG_STEP) as usize + 1;
        let lags: Vec<f64> = (0..steps).map(|i| shortest + i as f64 * LAG_STEP).collect();
        let score: Vec<f64> = lags
            .iter()
            .map(|&lag| comb(&similarity, lag) * prior(60.0 * fps / lag))
            .collect();
        let best = (1..steps - 1)
            .filter(|&i| score[i] > 0.0 && score[i] >= score[i - 1] && score[i] >= score[i + 1])
            .max_by(|&a, &b| score[a].total_cmp(&score[b]))?;
        let half = |lag: f64| {
            if lag / 2.0 < shortest {
                0.0
            } else {
                at(&similarity, lag / 2.0) / at(&similarity, lag)
            }
        };
        let mut lag = lags[best];
        if half(lag) >= NO_ACCENT && prior(lag_of(lag / 2.0)) >= PRIOR_TIE * prior(lag_of(lag)) {
            lag /= 2.0;
        }
        // The best repetition near a period related to the chosen one, as a
        // share of the chosen one's; 0 outside the tempo range.
        let own = comb(&similarity, lag);
        let rival = |target: f64| {
            if !(shortest..=longest).contains(&target) || own <= 0.0 {
                return 0.0;
            }
            let mut most = 0.0f64;
            let mut candidate = target / SAME_PEAK;
            while candidate <= target * SAME_PEAK {
                most = most.max(comb(&similarity, candidate));
                candidate += LAG_STEP;
            }
            most / own
        };
        Some(Pulse {
            lag,
            clarity: at(&similarity, lag),
            half: half(lag),
            three_two: rival(lag * 1.5).max(rival(lag / 1.5)),
        })
    }
}

/// The envelope's autocorrelation, mean removed, divided by its value at
/// lag zero: 1 where the envelope repeats itself exactly, 0 where it is
/// unrelated to itself.
fn self_similarity(envelope: &[f64], max_lag: usize) -> Vec<f64> {
    let n = envelope.len();
    let mean = envelope.iter().sum::<f64>() / n as f64;
    let centred: Vec<f64> = envelope.iter().map(|v| v - mean).collect();
    let r: Vec<f64> = (0..=max_lag.min(n - 1))
        .map(|lag| {
            centred[..n - lag]
                .iter()
                .zip(&centred[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / (n - lag) as f64
        })
        .collect();
    let zero = r[0];
    if zero <= 0.0 {
        return vec![0.0; r.len()];
    }
    r.iter().map(|v| v / zero).collect()
}

/// Linear interpolation at a fractional lag; 0 beyond the end.
fn at(similarity: &[f64], lag: f64) -> f64 {
    let i = lag.floor() as usize;
    if i + 1 >= similarity.len() {
        return 0.0;
    }
    let f = lag - i as f64;
    similarity[i] * (1.0 - f) + similarity[i + 1] * f
}

/// How well the envelope repeats at `lag` and its multiples: the comb of
/// Scheirer's tempo filters, read off the autocorrelation.
fn comb(similarity: &[f64], lag: f64) -> f64 {
    (1..=HARMONICS)
        .map(|k| at(similarity, k as f64 * lag).max(0.0))
        .sum::<f64>()
        / HARMONICS as f64
}

fn prior(bpm: f64) -> f64 {
    let octaves = (bpm / PRIOR_CENTRE_BPM).log2() / PRIOR_WIDTH_OCTAVES;
    (-0.5 * octaves * octaves).exp()
}

/// The period near `lag` at which the whole envelope repeats best, counting
/// every multiple that fits in half of it: a longer stretch pins the period
/// more finely than four beats do.
fn refine(envelope: &[f64], lag: f64) -> f64 {
    let multiples = ((envelope.len() / 2) as f64 / lag).floor().max(1.0) as usize;
    let reach = (multiples as f64 * lag * 1.02).ceil() as usize + 2;
    let similarity = self_similarity(envelope, reach);
    let fit = |l: f64| {
        (1..=multiples)
            .map(|k| at(&similarity, k as f64 * l))
            .sum::<f64>()
    };
    let mut best = (lag, fit(lag));
    let mut candidate = lag * 0.99;
    while candidate <= lag * 1.01 {
        let value = fit(candidate);
        if value > best.1 {
            best = (candidate, value);
        }
        candidate += 0.002;
    }
    best.0
}

/// Half-wave-rectified log spectral flux over mel bands, local mean
/// removed, lightly smoothed so a tempo slightly off the grid still gathers
/// its onsets. Bands, not bins: a kick lives in a handful of low bins and a
/// hi-hat in hundreds of high ones, and a sum over bins would let the hats
/// set the beat. Each band is the mean of its bins, as the ear groups them.
fn onset_envelope(signal: &[f64], rate: f64) -> (Vec<f64>, f64) {
    let hop = (rate / ENVELOPE_RATE).round().max(1.0) as usize;
    let fps = rate / hop as f64;
    let frame = power_of_two_within(rate * 0.06);
    let mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let (low, high) = (mel(ONSET_LOW_HZ), mel(ONSET_TOP_HZ));
    let band_of: Vec<Option<usize>> = (0..=frame / 2)
        .map(|k| {
            let hz = k as f64 * rate / frame as f64;
            (ONSET_LOW_HZ..ONSET_TOP_HZ).contains(&hz).then(|| {
                (((mel(hz) - low) / (high - low) * ONSET_BANDS as f64) as usize)
                    .min(ONSET_BANDS - 1)
            })
        })
        .collect();
    let mut width = vec![0usize; ONSET_BANDS];
    for band in band_of.iter().flatten() {
        width[*band] += 1;
    }
    let mut previous: Option<Vec<f64>> = None;
    let mut flux = Vec::new();
    let mut bands = vec![0.0; ONSET_BANDS];
    for_each_spectrum(signal, frame, hop, |magnitude| {
        bands.iter_mut().for_each(|b| *b = 0.0);
        for (m, band) in magnitude.iter().zip(&band_of) {
            if let Some(band) = band {
                bands[*band] += m;
            }
        }
        let logs: Vec<f64> = bands
            .iter()
            .zip(&width)
            .filter(|(_, w)| **w > 0)
            .map(|(b, w)| (1.0 + ONSET_COMPRESSION * b / *w as f64).ln())
            .collect();
        flux.push(match &previous {
            None => 0.0,
            Some(before) => logs
                .iter()
                .zip(before)
                .map(|(now, then)| (now - then).max(0.0))
                .sum(),
        });
        previous = Some(logs);
    });

    let reach = (0.1 * fps).round() as usize;
    let mut prefix = vec![0.0; flux.len() + 1];
    for (i, f) in flux.iter().enumerate() {
        prefix[i + 1] = prefix[i] + f;
    }
    let peaks: Vec<f64> = (0..flux.len())
        .map(|i| {
            let lo = i.saturating_sub(reach);
            let hi = (i + reach + 1).min(flux.len());
            (flux[i] - (prefix[hi] - prefix[lo]) / (hi - lo) as f64).max(0.0)
        })
        .collect();
    let kernel = [1.0, 2.0, 3.0, 2.0, 1.0];
    let envelope = (0..peaks.len())
        .map(|i| {
            let mut sum = 0.0;
            let mut weight = 0.0;
            for (j, k) in kernel.iter().enumerate() {
                if let Some(v) = (i + j).checked_sub(2).and_then(|n| peaks.get(n)) {
                    sum += k * v;
                    weight += k;
                }
            }
            sum / weight
        })
        .collect();
    (envelope, fps)
}

// ------------------------------------------------------------------ key

/// Key profiles derived from electronic dance music on Beatport, with the
/// degrees outside the scale set to zero ("bgate"): Á. Faraldo, S. Jordà
/// and P. Herrera, "A Multi-Profile Method for Key Estimation in EDM", AES
/// International Conference on Semantic Audio, 2017; values as printed in
/// Faraldo's thesis (Universitat Pompeu Fabra, 2017, fig. 6.12), tonic
/// first. Transcribed via the bpm-key crate 0.1.0 (MIT OR Apache-2.0,
/// © its authors), which prints the same table. The Krumhansl–Kessler
/// profiles, measured on classical listening, read minor records as major;
/// these hold the third of each mode apart.
const MAJOR_PROFILE: [f64; 12] = [
    1.00, 0.00, 0.42, 0.00, 0.53, 0.37, 0.00, 0.77, 0.00, 0.38, 0.21, 0.30,
];
const MINOR_PROFILE: [f64; 12] = [
    1.00, 0.00, 0.36, 0.39, 0.00, 0.38, 0.00, 0.74, 0.27, 0.00, 0.42, 0.23,
];
/// Notes read for the key: E1 (41 Hz), the bass guitar's lowest string,
/// up to C7 (2.1 kHz).
const LOWEST_NOTE: i32 = 28;
const HIGHEST_NOTE: i32 = 96;
/// Notes below E3 (165 Hz) are the bass. Dance music keeps its tonic
/// there, so the bass, scaled on its own, weighs half as much as
/// everything above it together.
const BASS_TOP_NOTE: i32 = 52;
const BASS_WEIGHT: f64 = 0.5;
/// Below this share of the note range's magnitude in held partials, the
/// clip is noise or percussion: white noise holds about 0.08 %.
const MIN_TONAL_SHARE: f64 = 0.0025;
/// The best key's correlation must exceed the runner-up's by this much. On
/// the development clips, readings below it matched the reference about
/// half the time; above it, four times in five.
const MIN_KEY_MARGIN: f64 = 0.12;
/// Hertz either side whose median magnitude is a bin's noise floor.
const FLOOR_REACH_HZ: f64 = 27.0;
/// Noise magnitudes are Rayleigh-distributed and exceed three times their
/// median about 0.2 % of the time; a held partial is far above. This
/// decides whether the clip holds notes at all.
const HELD_OVER_FLOOR: f64 = 3.0;
/// The key itself reads every partial above one and a half times its
/// floor: in a full mix the notes that decide the key are rarely alone.
const PARTIAL_OVER_FLOOR: f64 = 1.5;
/// A key needs at least a triad's worth of notes: one pitch fits a dozen
/// keys, and a root with its fifth says nothing of major or minor.
const MIN_DISTINCT_NOTES: f64 = 3.0;

fn read_key(signal: &[f64], rate: f64) -> KeyReading {
    let tones = Tones::of(signal, rate);
    if tones.held_share < MIN_TONAL_SHARE {
        return KeyReading::Abstained(KeyAbstention::NoTonalContent);
    }
    let chroma = tones.chroma();
    let mut fits: Vec<(f64, MusicalKey)> = (0..12u8)
        .flat_map(|tonic| {
            [
                (KeyMode::Major, &MAJOR_PROFILE),
                (KeyMode::Minor, &MINOR_PROFILE),
            ]
            .map(|(mode, profile)| {
                let rotated: [f64; 12] =
                    std::array::from_fn(|i| profile[(i + 12 - usize::from(tonic)) % 12]);
                (correlation(&chroma, &rotated), MusicalKey { tonic, mode })
            })
        })
        .collect();
    fits.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (best, margin) = (fits[0].1, fits[0].0 - fits[1].0);
    let held: [f64; 12] = std::array::from_fn(|i| tones.bass[i] + tones.upper[i]);
    let notes = distinct_notes(&held);
    if notes < MIN_DISTINCT_NOTES || margin < MIN_KEY_MARGIN {
        return KeyReading::Abstained(KeyAbstention::AmbiguousKey);
    }
    KeyReading::Found {
        key: best,
        confidence: margin.clamp(0.0, 1.0),
    }
}

/// The twelve-note content of a clip, from its partials: a bin counts by
/// how far it stands above a multiple of the median of its neighbours.
/// Noise, and the fixed spectrum of a drum sample struck again and again,
/// stay near that floor; a note is a narrow peak above it.
struct Tones {
    bass: [f64; 12],
    upper: [f64; 12],
    /// The share of the note range's magnitude in held partials.
    held_share: f64,
}

impl Tones {
    fn of(signal: &[f64], rate: f64) -> Tones {
        let frame = power_of_two_within(rate * 0.75);
        let reach = (FLOOR_REACH_HZ * frame as f64 / rate).round() as usize;
        let mut bins: Vec<(usize, i32, f64)> = Vec::new();
        for k in 1..=frame / 2 {
            let hz = k as f64 * rate / frame as f64;
            let pitch = 69.0 + 12.0 * (hz / 440.0).log2();
            let note = pitch.round();
            let weight = 1.0 - 2.0 * (pitch - note).abs();
            let note = note as i32;
            if (LOWEST_NOTE..HIGHEST_NOTE).contains(&note) && weight > 0.0 {
                bins.push((k, note, weight));
            }
        }
        let mut tones = Tones {
            bass: [0.0; 12],
            upper: [0.0; 12],
            held_share: 0.0,
        };
        let (mut total, mut held) = (0.0, 0.0);
        let mut neighbours = Vec::with_capacity(2 * reach + 1);
        for_each_spectrum(signal, frame, frame / 2, |magnitude| {
            for &(k, note, weight) in &bins {
                total += weight * magnitude[k];
                neighbours.clear();
                let lo = k.saturating_sub(reach);
                let hi = (k + reach + 1).min(magnitude.len());
                neighbours.extend_from_slice(&magnitude[lo..hi]);
                let middle = neighbours.len() / 2;
                let floor = *neighbours.select_nth_unstable_by(middle, f64::total_cmp).1;
                held += weight * (magnitude[k] - HELD_OVER_FLOOR * floor).max(0.0);
                let part = weight * (magnitude[k] - PARTIAL_OVER_FLOOR * floor).max(0.0);
                let class = (note % 12) as usize;
                if note < BASS_TOP_NOTE {
                    tones.bass[class] += part;
                } else {
                    tones.upper[class] += part;
                }
            }
        });
        tones.held_share = if total > 0.0 { held / total } else { 0.0 };
        tones
    }

    /// Bass and upper notes, each scaled to sum to one, added.
    fn chroma(&self) -> [f64; 12] {
        let unit = |part: &[f64; 12]| {
            let sum: f64 = part.iter().sum();
            part.map(|v| if sum > 0.0 { v / sum } else { 0.0 })
        };
        let (bass, upper) = (unit(&self.bass), unit(&self.upper));
        std::array::from_fn(|i| BASS_WEIGHT * bass[i] + upper[i])
    }
}

/// The largest power of two not above `length` samples.
fn power_of_two_within(length: f64) -> usize {
    1 << (length.max(2.0).log2().floor() as u32)
}

/// How many pitch classes the clip really holds, above its flat floor:
/// the inverse participation ratio, 1 for one note, 12 for all alike.
fn distinct_notes(chroma: &[f64; 12]) -> f64 {
    let floor = chroma.iter().copied().fold(f64::INFINITY, f64::min);
    let above: Vec<f64> = chroma.iter().map(|c| c - floor).collect();
    let total: f64 = above.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    1.0 / above.iter().map(|a| (a / total).powi(2)).sum::<f64>()
}

fn correlation(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / 12.0;
    let mean_b = b.iter().sum::<f64>() / 12.0;
    let (mut cross, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        cross += (x - mean_a) * (y - mean_b);
        var_a += (x - mean_a).powi(2);
        var_b += (y - mean_b).powi(2);
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        0.0
    } else {
        cross / (var_a * var_b).sqrt()
    }
}
