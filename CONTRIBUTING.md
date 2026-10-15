# Contributing

Improvements are welcome, above all to the key: on dance music it is right
a little more than half the time it answers (README, «Measured»). A change
that reads more records correctly is worth more than any refactor.

## The one rule

**A wrong value is worth less than no value.** Each reading either answers
or names why it abstains. A change that answers more often must not make the
answers less right: show both numbers.

## Before a pull request

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

With the weights (`scripts/fetch-weights.sh`), also:

```sh
BEAT_THIS_CHECKPOINT=$PWD/final0.ckpt cargo test --release -p disctale-beat
```

## Changing a reading

1. Add a synthetic test that fails for the reason you expect (the tests in
   `crates/disctale-key/tests` show how: clicks, chords and grooves whose
   answer is known by construction).
2. Change the code until it passes, with every other test still green.
3. Run the bench (`bench/README.md`) on your own clips before and after,
   and put both lines of `score.py` in the pull request, with what kind of
   music the clips are. Keep a part of your clips aside and measure it only
   once, at the end: numbers on the clips you tuned on flatter the change.

Never commit audio, track names or per-clip references: only the counts.

## Licences

By contributing you agree that your contribution is licensed like the files
it touches: `disctale-key` and the command-line tool under MIT OR
Apache-2.0; `disctale-beat` under MIT (its own files `src/checkpoint.rs` and
`src/tempo.rs` also under Apache-2.0). Code from GPL or AGPL projects (for
example Essentia or aubio) cannot be accepted, and neither can anything
taken from closed DJ software.
