# Provenance of `disctale-beat`

`disctale-beat` runs **Beat This!** (Foscarin, Schlüter, Widmer, ISMIR 2024), the
beat tracker of the Institute of Computational Perception, JKU Linz, in
Rust, as the DISCTALE server uses it. Only the official sources are used:

| What | Where | Pinned |
|---|---|---|
| Code followed by the port | https://github.com/CPJKU/beat_this | commit `b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c` (28/5/2026), version 1.1.0 |
| Weights | checkpoint `final0`, from the repository's own URL (`beat_this/inference.py`, `CHECKPOINT_URL`) | 81,058,141 bytes, sha256 `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331` |
| Licence | code and published weights MIT (README, «License»; `LICENSE`) | copied in `LICENSE` |

Ported by hand from the files of that commit, nothing else:

- `beat_this/preprocessing.py` (`LogMelSpect`) → `src/mel.rs`. The resampling
  to 22,050 Hz is ours (a windowed-sinc 2:1 decimator); upstream uses soxr.
- `beat_this/model/beat_tracker.py`, `beat_this/model/roformer.py` and the
  rotary embedding they import (`rotary_embedding_torch`, MIT) → `src/model.rs`.
- `beat_this/inference.py` (`split_piece`, `aggregate_prediction`) and
  `beat_this/model/postprocessor.py` (minimal peak picking) → `src/model.rs`,
  `src/tempo.rs`.

Ours, not upstream: reading the checkpoint (`src/checkpoint.rs`), the rule
from beats to one BPM and its abstentions (`src/tempo.rs`).
No code or data from any other beat or key tracker is in this crate.

The weights are not in the repository and are never redistributed: fetch
them from the official URL and check them against the checksum above
(`scripts/fetch-weights.sh`; the command-line tool checks it again).
