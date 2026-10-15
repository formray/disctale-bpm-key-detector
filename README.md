# disctale-bpm-key-detector

[![CI](https://github.com/formray/disctale-bpm-key-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/formray/disctale-bpm-key-detector/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#licences-and-sources)
[![Rust 2024](https://img.shields.io/badge/rust-2024%20edition-b7410e.svg)](Cargo.toml)
[![BPM: Beat This!](https://img.shields.io/badge/BPM-Beat%20This!-6e7781.svg)](https://github.com/CPJKU/beat_this)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="disctale-bpm-key-detector: BPM and key in pure Rust, by Formray" src="docs/assets/logo-light.svg" width="480">
</picture>

> The BPM and the musical key of a short clip of music, in pure Rust.
> When the evidence does not decide, it says so instead of guessing.

Part of [DISCTALE](https://disctale.app), by [Formray](https://formray.io).

## Features

- **BPM** from [Beat This!](https://github.com/CPJKU/beat_this) (ISMIR 2024)
  ported to Rust: no PyTorch, no Python, the official weights checked by
  checksum.
- **Key** from the twelve notes of the clip against profiles measured on
  dance music, with Open Key notation; no weights, no dependencies.
- **It abstains instead of guessing**, and names why (`unstable`,
  `AmbiguousKey`, …). It never doubles or halves a tempo to make it fit.
- **Pure functions** over mono samples: no decoder, no network, no files
  except the model weights.
- **A command-line tool**, `disctale-bpm-key`: one JSON line per file.
- **Measured on 983 real clips**, with the numbers and the limits written
  down below.

## What is here

| Crate | What it does | Licence |
|---|---|---|
| [`disctale-beat`](crates/disctale-beat) | [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, JKU Linz) ported to Rust without PyTorch, and our rule from its beats to one BPM | MIT |
| [`disctale-key`](crates/disctale-key) | Our key reading: the twelve-note profile of the clip against major and minor profiles measured on dance music | MIT OR Apache-2.0 |
| [`disctale-bpm-key-detector`](crates/disctale-bpm-key-detector) | The command-line tool `disctale-bpm-key`: one JSON line per file | MIT OR Apache-2.0 |

DISCTALE, an iOS app that recognises vinyl records, runs the same code on
its server for the tracks no catalogue knows: the BPM from `disctale-beat`,
the key from `disctale-key`, on the 30-second preview of each track. This
repository is that code, given to anyone who wants to use it or make it
better; on the clips below it reads exactly what DISCTALE reads.

The rule behind every choice: **a wrong value is worth less than no value.**
Each reading either answers or names why it abstains.

## Usage

### Command line

```sh
git clone https://github.com/formray/disctale-bpm-key-detector.git
cd disctale-bpm-key-detector
scripts/fetch-weights.sh final0.ckpt      # official Beat This! weights, 81 MB, checksum verified
cargo build --release
ffmpeg -i track.mp3 -ac 1 -ar 44100 -f f32le track.f32
target/release/disctale-bpm-key --weights final0.ckpt track.f32
```

```json
{"file":"track.f32","bpm":124,"tempo_note":"124.02","key":"Am","open_key":"1m","key_note":"0.143","ms":3100}
```

`bpm` and `key` are `null` when the reading abstains; `tempo_note` and
`key_note` then hold the reason (for example `unstable`, `AmbiguousKey`).
WAV files are read directly. Without `--weights` only the key is read.

### As a library

```rust
// Key: no weights, no dependencies.
let reading = disctale_key::analyze_tempo_and_key(&samples, 44_100);
if let disctale_key::KeyReading::Found { key, .. } = reading.key {
    println!("{} ({})", key.traditional(), key.open_key());
}

// BPM: Beat This! with the official weights.
let tracker = disctale_beat::BeatTracker::load("final0.ckpt".as_ref())?;
let tempo = tracker.tempo(&samples, 44_100, &disctale_beat::Pace::flat_out())?;
```

`Pace::new(work, rest)` makes the model yield the processor at regular
intervals, for a server that runs it beside requests on one CPU.

## How it works

**BPM.** Beat This! (Foscarin, Schlüter and Widmer, ISMIR 2024) finds the
beats: a log-mel spectrogram at 50 frames per second through a transformer,
ported from the official repository at a fixed commit (see
[PROVENANCE](crates/disctale-beat/PROVENANCE.md)). Our rule turns beats into
one tempo: fewer than 16 beats, abstain; fewer than 75 % of the intervals
within 15 % of their median, abstain (the tempo is not steady); otherwise a
least-squares line through the beats placed on a grid, beats more than 0.2
periods off it left out, twice; outside 40–250 BPM, abstain. It never
doubles or halves a tempo. The served value is the rounded integer. The
port reads the first 29.76 s of a clip in one pass and resamples 44.1 kHz to
22.05 kHz with its own filter, where upstream uses soxr.

**Key.** The partials of the whole clip are folded into twelve notes, with
the bass (below E3) kept apart and added at half weight, and compared with
the major and minor profiles of Faraldo, Jordà and Herrera (2017), measured
on electronic dance music, in all twelve transpositions. Too little held
sound, fewer than three notes, or a best key too close to the runner-up:
abstain. The FFT is a textbook radix-2, written here.

`disctale-key` also reads a tempo (the autocorrelation of the onset envelope
with comb filters, after Scheirer, 1998, and a preference for tempi near
120, after Klapuri, Eronen and Astola, 2006). DISCTALE does not serve it:
on the same clips Beat This! is right more often and halves less.

## Measured

Two sets of 30-second clips cut from one DJ's own library, mostly house,
techno and other electronic music: 83 clips used to tune the readings
(*development*) and 96 clips never looked at until a single final run
(*held out*). The reference for each track is GetSongBPM where it knew the
track, otherwise the DJ software's analysis of the full track. The audio,
the track names and the per-clip reference stay private; the method to
repeat the measurement on your own clips is in [bench](bench/README.md).

| Values served | Development (83) | Held out (96) |
|---|---|---|
| BPM right within ±1, of those given | 75 of 77 (97.4 %) | 85 of 87 (97.7 %) |
| BPM at double or half | 1 half | 1 half |
| BPM abstentions | 6 (7.2 %) | 9 (9.4 %) |
| Key exact, of those given | 20 of 24 (83.3 %) | 19 of 34 (55.9 %) |
| Key abstentions | 59 | 62 |

The key was tuned on the development set: the held-out number is the honest
one. **These numbers are the floor**: a change must not go below BPM 94 %
right with at most 3.5 % double or half and at most 25 % abstentions, and
key 55 % exact, on either set.

**A larger set: 804 tracks bought on Beatport.** One 30-second clip per
track, cut at 30 % of its length; two references per track, the key and BPM
the store writes into the file and the DJ software's analysis of the full
track. Nothing was tuned on these tracks.

| Values served | All 804 | 354 where store and DJ software agree on the key |
|---|---|---|
| BPM right within ±1 of the DJ software, of those given | 782 of 790 (99.0 %) | 346 of 349 (99.1 %) |
| BPM at double or half | none | none |
| BPM abstentions | 14 (1.7 %) | 5 (1.4 %) |
| Key exact against the store, of those given | 173 of 279 (62.0 %) | — |
| Key exact against the DJ software, of those given | 182 of 279 (65.2 %) | 137 of 160 (85.6 %) |
| Key abstentions | 525 (65 %) | 194 (55 %) |

Against the store's BPM, 727 of 790 are right: 22 more are at double the
store's value, all tracks the store lists between 65 and 73 BPM, where the
DJ software reads the same double tempo we do.

For comparison on the held-out clips: the tempo of `disctale-key` 84 of 89
(94.4 %, 3 halves); [`bpm-key`](https://crates.io/crates/bpm-key) 0.1.0
BPM 83 of 96, key 34 of 96; `stratum-dsp` 1.0.0 BPM 73 of 96, key 16 of 96
(both always answer).

Cost, on one core of an Apple M2 Max: about 3 s per clip for the BPM, 0.3–0.5 s
for the key. In a Linux container with 1 CPU and 512 MiB, the BPM peaks at
186 MB of memory.

## Known limits

- **The key reference is weak.** On the 804 Beatport tracks the store and
  the DJ software agree on the key 354 times: on those the key is exact
  85.6 % of the time it answers; on all 804, 62–65 % depending on the reference.
  Part of the gap is the reference, not the reading; a reference written by
  ear on a sample of clips would be the way to tell them apart.
- **The key abstains often** (about two clips in three): below the margin it
  keeps, its answers were right less than half the time.
- **Fast tempi can read at half**: the one half in each set is drum and bass
  (167 read as 84, 156 as 78).
- **Spurious beats at half a period** sometimes push the BPM rule to abstain
  or miss by a few BPM; a sturdier rule is possible but must be measured on
  clips it was not built on.
- Only tempo and key. No mood, no energy, no beat grid in the output.

## Licences and sources

- `disctale-key`, the command-line tool, the bench and the docs: MIT OR
  Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)),
  © 2026 Formray.
- `disctale-beat`: MIT, with the notice of the Institute of Computational
  Perception, JKU Linz ([LICENSE](crates/disctale-beat/LICENSE)). Its own
  files `src/checkpoint.rs` and `src/tempo.rs` are also offered under
  Apache-2.0.
- **The Beat This! weights are not in this repository and are never
  redistributed.** `scripts/fetch-weights.sh` downloads `final0` from the
  official URL and checks its SHA-256; the tool checks it again.
- The key profiles are Á. Faraldo, S. Jordà and P. Herrera, «A
  Multi-Profile Method for Key Estimation in EDM», AES Semantic Audio, 2017
  (values from Faraldo's thesis, UPF 2017), transcribed via the
  [`bpm-key`](https://crates.io/crates/bpm-key) crate 0.1.0 (MIT OR
  Apache-2.0).
- F. Foscarin, J. Schlüter, G. Widmer, «Beat this! Accurate beat tracking
  without DBN postprocessing», ISMIR 2024.
- No code from GPL or AGPL projects, and nothing taken from closed DJ
  software.
- The logo is drawn for this repository; its lettering is set in
  JetBrains Mono and Inter (SIL Open Font License) and turned into outlines.

## Contributing

Improvements are welcome, the key above all. See
[CONTRIBUTING](CONTRIBUTING.md): a failing synthetic test first, then the
bench before and after on your own clips.

---

<sub>Made by [Formray](https://formray.io). Technology, thoughtfully.</sub>
