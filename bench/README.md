# Bench: measure it on your own music

The numbers in the main README come from this procedure on two private sets
of clips. Run it on yours: the clips, their names and your reference never
need to leave your machine, and only the counts are worth sharing.

## 1. Choose the clips, then write the reference

- Cut 30-second clips (DISCTALE reads 30-second previews), one per track,
  named `NN - anything.mp3` (`NN` = 01, 02, …).
- Write `reference.tsv` before running anything: `id`, `bpm`, `key`,
  `genre`, tab-separated, as in [reference.example.tsv](reference.example.tsv).
  Keys as `Am`, `F#`, `Bb`… The reference is per track and holds for the clip.
- Choose the clips without listening to the tool's answers, for example at
  random with a fixed seed from your library, and keep a set aside that you
  measure **once**, after you stop changing the code.

A reference is only as good as its source. For the published numbers: an
online BPM database where it knew the track, otherwise the DJ software's
analysis of the full track. That key source agreed with the stores' key tags
3 times out of 12: a key written by ear is worth much more.

## 2. Convert and run

```sh
bench/convert.sh path/to/clips out/clips             # needs ffmpeg
scripts/fetch-weights.sh final0.ckpt
cargo build --release
target/release/disctale-bpm-key --weights final0.ckpt out/clips/*.f32 > out/results.jsonl
```

## 3. Score

```sh
bench/score.py reference.tsv out/results.jsonl       # -v lists every miss
```

```text
clips 96
BPM  given 87, right 85 (97.7 %, 95 % CI 92.0-99.4), double 0, half 1, other ratio 0, wrong 1, abstained 9 (9.4 %)
KEY  given 34, exact 19 (55.9 %, 95 % CI 39.5-71.1), relative 1, fifth 1, parallel 2, wrong 11, abstained 62
floor: BPM met, key met
```

Rules: a BPM is right within ±1 of the reference; double or half within 2 %;
3:2, 2:3, 4:3 and 3:4 count apart, like anything else that is off. A key is
exact, relative, a fifth away, parallel, or wrong. Rates are over the values
given; abstentions are counted apart. The floor is the one in the README.
