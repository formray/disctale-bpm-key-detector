//! The checkpoint reader on a file PyTorch wrote (`tests/fixtures/tiny.ckpt`,
//! `torch.save` of a Lightning-shaped dict with protocol 2: a matrix, a
//! transposed view of the same storage, an int64 counter, a bias), and —
//! only when `BEAT_THIS_CHECKPOINT` names the official `final0` file — the
//! whole tracker on it.

use disctale_beat::checkpoint;

#[test]
fn a_pytorch_checkpoint_reads_with_shapes_strides_and_shared_storage() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny.ckpt"
    ))
    .expect("fixture");
    let mut weights = checkpoint::read(&bytes).expect("readable");
    assert_eq!(
        weights.take("layer.weight", &[2, 3]).expect("weight"),
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    );
    assert_eq!(
        weights
            .take("layer.weight_t", &[3, 2])
            .expect("transposed view"),
        vec![0.0, 0.75, 0.25, 1.0, 0.5, 1.25]
    );
    assert_eq!(
        weights.take("layer.bias", &[2]).expect("bias"),
        vec![-1.5, 2.25]
    );
    // The int64 counter is not a weight.
    assert!(weights.take("norm.num_batches_tracked", &[]).is_err());
    assert!(matches!(
        weights.take("layer.bias", &[2]),
        Err(checkpoint::CheckpointError::Missing(_))
    ));
}

#[test]
fn a_wrong_shape_is_refused() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny.ckpt"
    ))
    .expect("fixture");
    let mut weights = checkpoint::read(&bytes).expect("readable");
    assert!(matches!(
        weights.take("layer.weight", &[3, 2]),
        Err(checkpoint::CheckpointError::Shape { .. })
    ));
}

#[test]
fn something_that_is_not_a_checkpoint_is_refused() {
    assert!(checkpoint::read(b"not a zip at all").is_err());
}

/// Clicks at a known tempo through the real model. Opt-in: the 81 MB file
/// is not in the repository (PROVENANCE.md).
#[test]
fn the_official_checkpoint_reads_a_click_track() {
    let Ok(path) = std::env::var("BEAT_THIS_CHECKPOINT") else {
        eprintln!("BEAT_THIS_CHECKPOINT not set: skipped");
        return;
    };
    let tracker = disctale_beat::BeatTracker::load(std::path::Path::new(&path)).expect("final0");
    let rate = 44_100usize;
    let period = rate as f64 * 60.0 / 124.0;
    let mut samples = vec![0.0f32; rate * 30];
    let mut beat = 0usize;
    while (beat as f64 * period) as usize + 2_000 < samples.len() {
        let start = (beat as f64 * period) as usize;
        let accent = if beat.is_multiple_of(4) { 1.0 } else { 0.6 };
        for (i, s) in samples[start..start + 2_000].iter_mut().enumerate() {
            let t = i as f32 / rate as f32;
            *s += accent * (-t * 60.0).exp() * (2.0 * std::f32::consts::PI * 80.0 * t).sin();
        }
        beat += 1;
    }
    let reading = tracker
        .tempo(&samples, 44_100, &disctale_beat::Pace::flat_out())
        .expect("44.1 kHz");
    assert!(
        matches!(reading, disctale_beat::TempoReading::Found { bpm: 124, .. }),
        "{reading:?}"
    );
}
