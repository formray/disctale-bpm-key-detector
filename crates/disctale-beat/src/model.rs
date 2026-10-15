//! The Beat This! network (`beat_this/model/beat_tracker.py` and
//! `roformer.py`, checkpoint `final0`), forward pass only, in plain f32.
//!
//! Written for a small server: attention is computed one (sequence, head)
//! at a time, so the largest temporary is one 1,500 × 1,500 score matrix
//! (9 MB) instead of every head of every frequency band at once.

use crate::checkpoint::{CheckpointError, Weights};
use crate::mel::N_MELS;

const HEAD_DIM: usize = 32;
const STEM_DIM: usize = 32;
const DIM: usize = 512;
const LAYERS: usize = 6;
const BN_EPS: f32 = 1e-5;
/// `Spect2Frames`: chunks of 1,500 frames overlapping by 6 on each side.
pub const CHUNK: usize = 1500;
pub const BORDER: usize = 6;

pub struct BeatThis {
    stem_bn1d: Norm,
    stem_conv: Conv,
    stem_bn2d: Norm,
    blocks: Vec<FrontendBlock>,
    frontend_linear: Linear,
    layers: Vec<(Attention, FeedForward)>,
    final_norm: Vec<f32>,
    head: Linear,
    rotary: Rotary,
}

impl BeatThis {
    pub fn from_weights(mut w: Weights) -> Result<BeatThis, CheckpointError> {
        let rotary =
            Rotary::new(w.take("frontend.blocks.0.partial.attnF.rotary_embed.freqs", &[16])?);
        let stem_bn1d = Norm::batch(&mut w, "frontend.stem.bn1d", N_MELS)?;
        let stem_conv = Conv::new(&mut w, "frontend.stem.conv2d", 1, STEM_DIM, 4)?;
        let stem_bn2d = Norm::batch(&mut w, "frontend.stem.bn2d", STEM_DIM)?;
        let mut blocks = Vec::new();
        let mut dim = STEM_DIM;
        for i in 0..3 {
            let p = format!("frontend.blocks.{i}");
            blocks.push(FrontendBlock {
                attn_f: Attention::new(&mut w, &format!("{p}.partial.attnF"), dim)?,
                ff_f: FeedForward::new(&mut w, &format!("{p}.partial.ffF"), dim)?,
                attn_t: Attention::new(&mut w, &format!("{p}.partial.attnT"), dim)?,
                ff_t: FeedForward::new(&mut w, &format!("{p}.partial.ffT"), dim)?,
                conv: Conv::new(&mut w, &format!("{p}.conv2d"), dim, dim * 2, 2)?,
                norm: Norm::batch(&mut w, &format!("{p}.norm"), dim * 2)?,
            });
            dim *= 2;
        }
        let frontend_linear = Linear::new(&mut w, "frontend.linear", 1024, DIM, true)?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let p = format!("transformer_blocks.layers.{i}");
            layers.push((
                Attention::new(&mut w, &format!("{p}.0"), DIM)?,
                FeedForward::new(&mut w, &format!("{p}.1"), DIM)?,
            ));
        }
        let final_norm = w.take("transformer_blocks.norm.gamma", &[DIM])?;
        let head = Linear::new(&mut w, "task_heads.beat_downbeat_lin", DIM, 2, true)?;
        Ok(BeatThis {
            stem_bn1d,
            stem_conv,
            stem_bn2d,
            blocks,
            frontend_linear,
            layers,
            final_norm,
            head,
            rotary,
        })
    }

    /// Beat logits for a whole piece of `T × 128` frames, split and
    /// re-joined exactly as `split_predict_aggregate` (keep_first). Frames
    /// no chunk covers stay at -1000, as upstream.
    pub fn piece_logits(&self, spect: &[f32], pace: &Pace) -> Vec<f32> {
        let len = spect.len() / N_MELS;
        let mut out = vec![-1000.0f32; len];
        if len == 0 {
            return out;
        }
        let step = CHUNK - 2 * BORDER;
        let mut starts: Vec<isize> = (0..)
            .map(|i| -(BORDER as isize) + (i * step) as isize)
            .take_while(|&s| s < len as isize - BORDER as isize)
            .collect();
        if len > step
            && let Some(last) = starts.last_mut()
        {
            *last = len as isize - (CHUNK - BORDER) as isize;
        }
        // keep_first: later chunks are written first, earlier ones win.
        for &start in starts.iter().rev() {
            let from = start.max(0) as usize;
            let to = ((start + CHUNK as isize) as usize).min(len);
            let left = (-start).max(0) as usize;
            let right = (BORDER as isize)
                .min(start + CHUNK as isize - len as isize)
                .max(0) as usize;
            let frames = left + (to - from) + right;
            let mut chunk = vec![0.0f32; frames * N_MELS];
            chunk[left * N_MELS..(left + to - from) * N_MELS]
                .copy_from_slice(&spect[from * N_MELS..to * N_MELS]);
            let logits = self.forward(&chunk, frames, pace);
            let kept = &logits[BORDER..frames - BORDER];
            let first = start + BORDER as isize;
            for (i, v) in kept.iter().enumerate() {
                let at = first + i as isize;
                if at >= 0 && (at as usize) < len && at < start + (CHUNK - BORDER) as isize {
                    out[at as usize] = *v;
                }
            }
        }
        out
    }

    /// One chunk: `t × 128` frames in, `t` beat logits out (beat + downbeat,
    /// the `SumHead`).
    pub fn forward(&self, spect: &[f32], t: usize, pace: &Pace) -> Vec<f32> {
        // Stem: b t f -> b f t, BatchNorm1d over the 128 bands.
        let mut x = vec![0.0f32; N_MELS * t];
        for frame in 0..t {
            for band in 0..N_MELS {
                x[band * t + frame] = spect[frame * N_MELS + band];
            }
        }
        self.stem_bn1d.apply_channels(&mut x, t);
        let mut x = self.stem_conv.apply(&x, N_MELS, t, pace);
        let mut freqs = N_MELS / 4;
        self.stem_bn2d.apply_channels(&mut x, freqs * t);
        gelu(&mut x);

        let mut channels = STEM_DIM;
        for block in &self.blocks {
            let mut tokens = to_tokens(&x, channels, freqs, t);
            // Across frequencies: t sequences of `freqs` tokens.
            let a = block
                .attn_f
                .apply(&tokens, channels, t, freqs, freqs, 1, &self.rotary, pace);
            add(&mut tokens, &a);
            let f = block.ff_f.apply(&tokens, channels, pace);
            add(&mut tokens, &f);
            // Across time: `freqs` sequences of t tokens.
            let a = block
                .attn_t
                .apply(&tokens, channels, freqs, t, 1, freqs, &self.rotary, pace);
            add(&mut tokens, &a);
            let f = block.ff_t.apply(&tokens, channels, pace);
            add(&mut tokens, &f);
            let back = from_tokens(&tokens, channels, freqs, t);
            x = block.conv.apply(&back, freqs, t, pace);
            freqs /= 2;
            channels *= 2;
            block.norm.apply_channels(&mut x, freqs * t);
            gelu(&mut x);
        }

        // b c f t -> b t (c f), then the projection to the transformer.
        let mut flat = vec![0.0f32; t * channels * freqs];
        for c in 0..channels {
            for f in 0..freqs {
                for frame in 0..t {
                    flat[frame * channels * freqs + c * freqs + f] = x[(c * freqs + f) * t + frame];
                }
            }
        }
        let mut h = self.frontend_linear.apply(&flat, t);
        for (attn, ff) in &self.layers {
            let a = attn.apply(&h, DIM, 1, t, 0, 1, &self.rotary, pace);
            add(&mut h, &a);
            let f = ff.apply(&h, DIM, pace);
            add(&mut h, &f);
        }
        rms_norm(&mut h, DIM, &self.final_norm);
        let both = self.head.apply(&h, t);
        (0..t).map(|i| both[2 * i] + both[2 * i + 1]).collect()
    }
}

/// Leaves the processor to others at regular intervals: after about `work`
/// of computing, the thread sleeps for `rest`. A server that runs the model
/// beside latency-bound requests on one vCPU caps its share this way; the
/// operating system's priorities do not hold under a CPU quota (server bench).
pub struct Pace {
    work: std::time::Duration,
    rest: std::time::Duration,
    since: std::cell::Cell<std::time::Instant>,
}

impl Pace {
    /// Never pauses.
    pub fn flat_out() -> Pace {
        Pace::new(std::time::Duration::MAX, std::time::Duration::ZERO)
    }

    pub fn new(work: std::time::Duration, rest: std::time::Duration) -> Pace {
        Pace {
            work,
            rest,
            since: std::cell::Cell::new(std::time::Instant::now()),
        }
    }

    pub(crate) fn tick(&self) {
        if self.since.get().elapsed() >= self.work {
            std::thread::sleep(self.rest);
            self.since.set(std::time::Instant::now());
        }
    }
}

struct FrontendBlock {
    attn_f: Attention,
    ff_f: FeedForward,
    attn_t: Attention,
    ff_t: FeedForward,
    conv: Conv,
    norm: Norm,
}

/// (C, F, T) → tokens (T, F) × C.
fn to_tokens(x: &[f32], c: usize, f: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ch in 0..c {
        for band in 0..f {
            let src = &x[(ch * f + band) * t..(ch * f + band + 1) * t];
            for (frame, v) in src.iter().enumerate() {
                out[(frame * f + band) * c + ch] = *v;
            }
        }
    }
    out
}

fn from_tokens(tokens: &[f32], c: usize, f: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens.len()];
    for frame in 0..t {
        for band in 0..f {
            let src = &tokens[(frame * f + band) * c..(frame * f + band + 1) * c];
            for (ch, v) in src.iter().enumerate() {
                out[(ch * f + band) * t + frame] = *v;
            }
        }
    }
    out
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// `F.normalize(x) * sqrt(d) * gamma` over each row of `d`.
fn rms_norm(x: &mut [f32], d: usize, gamma: &[f32]) {
    let scale = (d as f32).sqrt();
    for row in x.chunks_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for (v, g) in row.iter_mut().zip(gamma) {
            *v = *v / norm * scale * g;
        }
    }
}

fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// erf through the Chebyshev fit of erfc in Numerical Recipes (`erfcc`,
/// relative error below 1.2e-7), in f64 for the margin.
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

#[allow(clippy::too_many_arguments)]
fn sgemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    a_rows: (isize, isize),
    b: &[f32],
    b_rows: (isize, isize),
    c: &mut [f32],
) {
    debug_assert!(c.len() >= m * n);
    // SAFETY: the strides describe matrices that lie inside `a`, `b` and
    // `c`, which every caller sizes from the same `m`, `k`, `n`.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            a_rows.0,
            a_rows.1,
            b.as_ptr(),
            b_rows.0,
            b_rows.1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    inputs: usize,
    outputs: usize,
}

impl Linear {
    fn new(
        w: &mut Weights,
        name: &str,
        inputs: usize,
        outputs: usize,
        bias: bool,
    ) -> Result<Linear, CheckpointError> {
        Ok(Linear {
            weight: w.take(&format!("{name}.weight"), &[outputs, inputs])?,
            bias: if bias {
                Some(w.take(&format!("{name}.bias"), &[outputs])?)
            } else {
                None
            },
            inputs,
            outputs,
        })
    }

    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * self.outputs];
        sgemm(
            rows,
            self.inputs,
            self.outputs,
            x,
            (self.inputs as isize, 1),
            &self.weight,
            (1, self.inputs as isize),
            &mut out,
        );
        if let Some(bias) = &self.bias {
            for row in out.chunks_mut(self.outputs) {
                for (v, b) in row.iter_mut().zip(bias) {
                    *v += b;
                }
            }
        }
        out
    }
}

/// Inference batch norm, folded to one scale and shift per channel.
struct Norm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl Norm {
    fn batch(w: &mut Weights, name: &str, channels: usize) -> Result<Norm, CheckpointError> {
        let gamma = w.take(&format!("{name}.weight"), &[channels])?;
        let beta = w.take(&format!("{name}.bias"), &[channels])?;
        let mean = w.take(&format!("{name}.running_mean"), &[channels])?;
        let var = w.take(&format!("{name}.running_var"), &[channels])?;
        let scale: Vec<f32> = gamma
            .iter()
            .zip(&var)
            .map(|(g, v)| g / (v + BN_EPS).sqrt())
            .collect();
        let shift = beta
            .iter()
            .zip(&mean)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        Ok(Norm { scale, shift })
    }

    /// `x` is channel-major with `inner` values per channel.
    fn apply_channels(&self, x: &mut [f32], inner: usize) {
        for (ch, block) in x.chunks_mut(inner).enumerate() {
            let (s, b) = (self.scale[ch], self.shift[ch]);
            for v in block {
                *v = *v * s + b;
            }
        }
    }
}

/// Conv2d over (C, F, T) with kernel (kf, 3), stride (kf, 1), padding
/// (0, 1) and no bias — the only shape the model uses.
struct Conv {
    /// `out × (in · kf · 3)`, PyTorch order.
    weight: Vec<f32>,
    inputs: usize,
    outputs: usize,
    kf: usize,
}

impl Conv {
    fn new(
        w: &mut Weights,
        name: &str,
        inputs: usize,
        outputs: usize,
        kf: usize,
    ) -> Result<Conv, CheckpointError> {
        Ok(Conv {
            weight: w.take(&format!("{name}.weight"), &[outputs, inputs, kf, 3])?,
            inputs,
            outputs,
            kf,
        })
    }

    /// `x`: (inputs, f, t) → (outputs, f / kf, t).
    fn apply(&self, x: &[f32], f: usize, t: usize, pace: &Pace) -> Vec<f32> {
        let fo = f / self.kf;
        let k = self.inputs * self.kf * 3;
        let mut out = vec![0.0f32; self.outputs * fo * t];
        let mut cols = vec![0.0f32; t * k];
        let mut res = vec![0.0f32; t * self.outputs];
        for band in 0..fo {
            for c in 0..self.inputs {
                for kf in 0..self.kf {
                    let src = &x
                        [(c * f + band * self.kf + kf) * t..(c * f + band * self.kf + kf + 1) * t];
                    for kt in 0..3 {
                        let col = (c * self.kf + kf) * 3 + kt;
                        for frame in 0..t {
                            let at = frame as isize + kt as isize - 1;
                            cols[frame * k + col] = if at >= 0 && (at as usize) < t {
                                src[at as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
            sgemm(
                t,
                k,
                self.outputs,
                &cols,
                (k as isize, 1),
                &self.weight,
                (1, k as isize),
                &mut res,
            );
            pace.tick();
            for o in 0..self.outputs {
                let dst = &mut out[(o * fo + band) * t..(o * fo + band + 1) * t];
                for (frame, v) in dst.iter_mut().enumerate() {
                    *v = res[frame * self.outputs + o];
                }
            }
        }
        out
    }
}

/// Rotary position embedding of `rotary_embedding_torch` (theta 10,000,
/// interleaved pairs), with the frequencies stored in the checkpoint.
struct Rotary {
    freqs: Vec<f32>,
}

impl Rotary {
    fn new(freqs: Vec<f32>) -> Rotary {
        Rotary { freqs }
    }

    fn table(&self, len: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.freqs.len();
        let mut cos = vec![0.0f32; len * half];
        let mut sin = vec![0.0f32; len * half];
        for p in 0..len {
            for (i, f) in self.freqs.iter().enumerate() {
                let angle = p as f32 * f;
                cos[p * half + i] = angle.cos();
                sin[p * half + i] = angle.sin();
            }
        }
        (cos, sin)
    }
}

fn rotate(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    for i in 0..cos.len() {
        let (a, b) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = a * cos[i] - b * sin[i];
        v[2 * i + 1] = b * cos[i] + a * sin[i];
    }
}

struct Attention {
    gamma: Vec<f32>,
    qkv: Linear,
    gates: Linear,
    out: Linear,
    heads: usize,
    dim: usize,
}

impl Attention {
    fn new(w: &mut Weights, name: &str, dim: usize) -> Result<Attention, CheckpointError> {
        let heads = dim / HEAD_DIM;
        Ok(Attention {
            gamma: w.take(&format!("{name}.norm.gamma"), &[dim])?,
            qkv: Linear::new(w, &format!("{name}.to_qkv"), dim, 3 * dim, false)?,
            gates: Linear::new(w, &format!("{name}.to_gates"), dim, heads, true)?,
            out: Linear::new(w, &format!("{name}.to_out.0"), dim, dim, false)?,
            heads,
            dim,
        })
    }

    /// Attention over `groups` sequences of `len` tokens; token `s` of
    /// group `g` is row `g·group_stride + s·step` of `x` (rows of `dim`).
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        x: &[f32],
        dim: usize,
        groups: usize,
        len: usize,
        group_stride: usize,
        step: usize,
        rotary: &Rotary,
        pace: &Pace,
    ) -> Vec<f32> {
        debug_assert_eq!(dim, self.dim);
        let rows = x.len() / dim;
        let mut normed = x.to_vec();
        rms_norm(&mut normed, dim, &self.gamma);
        let qkv = self.qkv.apply(&normed, rows);
        let gates = self.gates.apply(&normed, rows);
        pace.tick();
        let (cos, sin) = rotary.table(len);
        let half = HEAD_DIM / 2;
        let scale = (HEAD_DIM as f32).powf(-0.5);

        let mut mixed = vec![0.0f32; rows * dim];
        let mut q = vec![0.0f32; len * HEAD_DIM];
        let mut k = vec![0.0f32; len * HEAD_DIM];
        let mut v = vec![0.0f32; len * HEAD_DIM];
        let mut scores = vec![0.0f32; len * len];
        let mut o = vec![0.0f32; len * HEAD_DIM];
        for g in 0..groups {
            for h in 0..self.heads {
                for s in 0..len {
                    let row = g * group_stride + s * step;
                    let base = row * 3 * dim + h * HEAD_DIM;
                    let qs = &mut q[s * HEAD_DIM..(s + 1) * HEAD_DIM];
                    qs.copy_from_slice(&qkv[base..base + HEAD_DIM]);
                    rotate(
                        qs,
                        &cos[s * half..(s + 1) * half],
                        &sin[s * half..(s + 1) * half],
                    );
                    for val in qs.iter_mut() {
                        *val *= scale;
                    }
                    let ks = &mut k[s * HEAD_DIM..(s + 1) * HEAD_DIM];
                    ks.copy_from_slice(&qkv[base + dim..base + dim + HEAD_DIM]);
                    rotate(
                        ks,
                        &cos[s * half..(s + 1) * half],
                        &sin[s * half..(s + 1) * half],
                    );
                    v[s * HEAD_DIM..(s + 1) * HEAD_DIM]
                        .copy_from_slice(&qkv[base + 2 * dim..base + 2 * dim + HEAD_DIM]);
                }
                sgemm(
                    len,
                    HEAD_DIM,
                    len,
                    &q,
                    (HEAD_DIM as isize, 1),
                    &k,
                    (1, HEAD_DIM as isize),
                    &mut scores,
                );
                for row in scores.chunks_mut(len) {
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for val in row.iter_mut() {
                        *val = (*val - max).exp();
                        sum += *val;
                    }
                    let inv = 1.0 / sum;
                    for val in row.iter_mut() {
                        *val *= inv;
                    }
                }
                sgemm(
                    len,
                    len,
                    HEAD_DIM,
                    &scores,
                    (len as isize, 1),
                    &v,
                    (HEAD_DIM as isize, 1),
                    &mut o,
                );
                for s in 0..len {
                    let row = g * group_stride + s * step;
                    let gate = 1.0 / (1.0 + (-gates[row * self.heads + h]).exp());
                    let dst = &mut mixed[row * dim + h * HEAD_DIM..row * dim + (h + 1) * HEAD_DIM];
                    for (d, val) in dst.iter_mut().zip(&o[s * HEAD_DIM..(s + 1) * HEAD_DIM]) {
                        *d = val * gate;
                    }
                }
                pace.tick();
            }
        }
        self.out.apply(&mixed, rows)
    }
}

struct FeedForward {
    gamma: Vec<f32>,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(w: &mut Weights, name: &str, dim: usize) -> Result<FeedForward, CheckpointError> {
        Ok(FeedForward {
            gamma: w.take(&format!("{name}.net.0.gamma"), &[dim])?,
            up: Linear::new(w, &format!("{name}.net.1"), dim, 4 * dim, true)?,
            down: Linear::new(w, &format!("{name}.net.4"), 4 * dim, dim, true)?,
        })
    }

    fn apply(&self, x: &[f32], dim: usize, pace: &Pace) -> Vec<f32> {
        let rows = x.len() / dim;
        let mut normed = x.to_vec();
        rms_norm(&mut normed, dim, &self.gamma);
        let mut inner = self.up.apply(&normed, rows);
        gelu(&mut inner);
        pace.tick();
        let out = self.down.apply(&inner, rows);
        pace.tick();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
        ] {
            assert!((erf(x) - want).abs() < 1e-6, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn rotation_turns_pairs_by_their_angle() {
        let mut v = vec![1.0, 0.0, 0.0, 1.0];
        let half = std::f32::consts::FRAC_PI_2;
        rotate(&mut v, &[half.cos(), 1.0], &[half.sin(), 0.0]);
        assert!((v[0]).abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6);
        assert_eq!(&v[2..], &[0.0, 1.0]);
    }
}
