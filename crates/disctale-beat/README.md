# disctale-beat

[Beat This!](https://github.com/CPJKU/beat_this) (Foscarin, Schlüter,
Widmer, ISMIR 2024; CPJKU, JKU Linz) in pure Rust, without PyTorch, and a
rule from its beats to one BPM that abstains when the tempo is not steady.

```rust
let tracker = disctale_beat::BeatTracker::load("final0.ckpt".as_ref())?;
match tracker.tempo(&samples, 44_100, &disctale_beat::Pace::flat_out())? {
    disctale_beat::TempoReading::Found { bpm, .. } => println!("{bpm} BPM"),
    disctale_beat::TempoReading::Abstained(why) => println!("no tempo: {}", why.as_str()),
}
```

The weights are not included and never redistributed: fetch the official
`final0` checkpoint with `scripts/fetch-weights.sh` (checksum in
`PROVENANCE.md`). Method and measurements:
[the repository README](https://github.com/formray/disctale-bpm-key-detector).

Licence: MIT, with the notice of CPJKU (`LICENSE`).
