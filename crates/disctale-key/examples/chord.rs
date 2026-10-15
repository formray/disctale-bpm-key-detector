//! Reads the key of twelve seconds of a synthesised cadence in A minor
//! (i–iv–V–i). One chord alone would rightly abstain: A minor and C major
//! share its notes. `cargo run --example chord -p disctale-key`

use disctale_key::{KeyReading, analyze_tempo_and_key};

/// A held note with a few harmonics, struck at `start` seconds.
fn add_note(out: &mut [f32], rate: u32, start: f32, length: f32, midi: i32) {
    let freq = 440.0 * 2f32.powf((midi - 69) as f32 / 12.0);
    let first = (start * rate as f32) as usize;
    let count = (length * rate as f32) as usize;
    for (n, sample) in out[first..first + count].iter_mut().enumerate() {
        let t = n as f32 / rate as f32;
        let partials: f32 = (1..=6)
            .map(|h| (2.0 * std::f32::consts::PI * freq * h as f32 * t).sin() / h as f32)
            .sum();
        *sample += 0.2 * (t / 0.01).min(1.0) * (-t / 1.5).exp() * partials;
    }
}

fn main() {
    let rate = 22_050u32;
    // A minor, D minor, E major (harmonic minor's dominant), A minor.
    let cadence = [[45, 48, 52], [50, 53, 57], [52, 56, 59], [45, 48, 52]];
    let mut samples = vec![0.0f32; rate as usize * 12];
    for bar in 0..6 {
        let chord = cadence[bar % 4];
        let start = bar as f32 * 2.0;
        add_note(&mut samples, rate, start, 2.0, chord[0] - 12);
        for note in chord {
            add_note(&mut samples, rate, start, 2.0, note);
        }
    }
    match analyze_tempo_and_key(&samples, rate).key {
        KeyReading::Found { key, confidence } => println!(
            "{} ({}), confidence {confidence:.2}",
            key.traditional(),
            key.open_key()
        ),
        KeyReading::Abstained(why) => println!("no key: {why:?}"),
    }
}
