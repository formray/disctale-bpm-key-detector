//! Tempo and key read from audio by `disctale-key`.
//!
//! Every signal here is synthesised, so the right answer is known by
//! construction: clicks at a known tempo, chords in a known key, noise, and
//! the patterns that push a tempo reader to double or half. The rule under
//! test is the plan's: a wrong value is worth less than no value, so where
//! the evidence does not decide, the reading must abstain and say why.

use disctale_key::{
    KeyAbstention, KeyMode, KeyReading, MusicalKey, TempoAbstention, TempoReading,
    analyze_tempo_and_key,
};

const SR: u32 = 22_050;

// ---------------------------------------------------------------- signals

/// Deterministic white noise in [-1, 1], so every run hears the same thing.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn silence(seconds: f32, sr: u32) -> Vec<f32> {
    vec![0.0; (seconds * sr as f32) as usize]
}

fn add(into: &mut [f32], from: &[f32]) {
    for (a, b) in into.iter_mut().zip(from) {
        *a += b;
    }
}

fn add_at(into: &mut [f32], at_seconds: f64, sr: u32, sound: &[f32]) {
    let start = (at_seconds * sr as f64).round() as usize;
    if start < into.len() {
        let end = (start + sound.len()).min(into.len());
        add(&mut into[start..end], sound);
    }
}

fn white_noise(seconds: f32, sr: u32, amplitude: f32, seed: u32) -> Vec<f32> {
    let mut noise = Noise(seed);
    (0..(seconds * sr as f32) as usize)
        .map(|_| noise.next() * amplitude)
        .collect()
}

/// A metronome click: 15 ms of bright noise and a 2 kHz ping, fast decay.
fn click(sr: u32, amplitude: f32) -> Vec<f32> {
    let mut noise = Noise(7);
    (0..(0.015 * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let ping = (2.0 * std::f32::consts::PI * 2000.0 * t).sin();
            amplitude * (-t / 0.003).exp() * (0.5 * noise.next() + 0.5 * ping)
        })
        .collect()
}

/// A kick drum: a sine falling from 150 to 50 Hz over 150 ms.
fn kick(sr: u32) -> Vec<f32> {
    let mut phase = 0.0f32;
    (0..(0.15 * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let freq = 50.0 + 100.0 * (-t / 0.03).exp();
            phase += 2.0 * std::f32::consts::PI * freq / sr as f32;
            (-t / 0.05).exp() * phase.sin()
        })
        .collect()
}

/// A snare: noise with a 200 Hz body, 80 ms.
fn snare(sr: u32) -> Vec<f32> {
    let mut noise = Noise(99);
    (0..(0.08 * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let body = (2.0 * std::f32::consts::PI * 200.0 * t).sin();
            0.8 * (-t / 0.02).exp() * (0.6 * noise.next() + 0.4 * body)
        })
        .collect()
}

/// A closed hi-hat: 20 ms of noise brightened by a first difference.
fn hat(sr: u32, amplitude: f32) -> Vec<f32> {
    let mut noise = Noise(1234);
    let mut previous = 0.0f32;
    (0..(0.02 * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let value = noise.next();
            let bright = value - previous;
            previous = value;
            amplitude * 0.5 * (-t / 0.005).exp() * bright
        })
        .collect()
}

/// Clicks on every beat; `accent(beat)` scales each one.
fn clicks(bpm: f64, seconds: f32, sr: u32, accent: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut out = silence(seconds, sr);
    let beat = 60.0 / bpm;
    let mut index = 0;
    while index as f64 * beat < seconds as f64 {
        add_at(&mut out, index as f64 * beat, sr, &click(sr, accent(index)));
        index += 1;
    }
    out
}

/// Clicks whose onsets follow `beat_times` seconds.
fn clicks_at(times: &[f64], seconds: f32, sr: u32) -> Vec<f32> {
    let mut out = silence(seconds, sr);
    for &time in times {
        add_at(&mut out, time, sr, &click(sr, 1.0));
    }
    out
}

fn midi_hz(note: i32) -> f32 {
    440.0 * 2f32.powf((note - 69) as f32 / 12.0)
}

/// A held note with six harmonics falling as 1/n, struck at the start.
fn tone(note: i32, seconds: f32, sr: u32) -> Vec<f32> {
    let freq = midi_hz(note);
    (0..(seconds * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let envelope = (t / 0.01).min(1.0) * (-t / 1.5).exp();
            let partials: f32 = (1..=6)
                .filter(|h| freq * (*h as f32) < sr as f32 / 2.0)
                .map(|h| (2.0 * std::f32::consts::PI * freq * h as f32 * t).sin() / h as f32)
                .sum();
            0.2 * envelope * partials
        })
        .collect()
}

fn sine(note: i32, seconds: f32, sr: u32) -> Vec<f32> {
    let freq = midi_hz(note);
    (0..(seconds * sr as f32) as usize)
        .map(|n| 0.3 * (2.0 * std::f32::consts::PI * freq * n as f32 / sr as f32).sin())
        .collect()
}

/// Chords of `notes`, one every `chord_seconds`, the progression repeated
/// until `seconds`; each chord also sounds its root an octave below.
fn chords(progression: &[Vec<i32>], chord_seconds: f32, seconds: f32, sr: u32) -> Vec<f32> {
    let mut out = silence(seconds, sr);
    let mut index = 0;
    while index as f32 * chord_seconds < seconds {
        let chord = &progression[index % progression.len()];
        let mut sound = tone(chord[0] - 12, chord_seconds, sr);
        for &note in chord {
            add(&mut sound, &tone(note, chord_seconds, sr));
        }
        add_at(&mut out, (index as f32 * chord_seconds) as f64, sr, &sound);
        index += 1;
    }
    out
}

/// I–IV–V–I in a major key, or i–iv–V–i (harmonic minor) in a minor key,
/// voiced around the octave below middle C.
fn cadence(key: MusicalKey) -> Vec<Vec<i32>> {
    let root = 48 + key.tonic as i32 - if key.tonic >= 6 { 12 } else { 0 };
    let third = match key.mode {
        KeyMode::Major => 4,
        KeyMode::Minor => 3,
    };
    let tonic = vec![root, root + third, root + 7];
    let subdominant = vec![root + 5, root + 5 + third, root + 12];
    let dominant = vec![root + 7, root + 11, root + 14];
    vec![tonic.clone(), subdominant, dominant, tonic]
}

/// Kick on every beat and, if `hat_level` > 0, hats on the eighths.
fn kicks_and_hats(bpm: f64, seconds: f32, sr: u32, hat_level: f32) -> Vec<f32> {
    let mut out = silence(seconds, sr);
    let beat = 60.0 / bpm;
    let mut index = 0;
    while index as f64 * beat < seconds as f64 {
        let at = index as f64 * beat;
        add_at(&mut out, at, sr, &kick(sr));
        if hat_level > 0.0 {
            add_at(&mut out, at, sr, &hat(sr, hat_level));
            add_at(&mut out, at + beat / 2.0, sr, &hat(sr, hat_level));
        }
        index += 1;
    }
    out
}

/// Kick on beats 1 and 3, snare on 2 and 4.
fn backbeat(bpm: f64, seconds: f32, sr: u32) -> Vec<f32> {
    let mut out = silence(seconds, sr);
    let beat = 60.0 / bpm;
    let mut index = 0;
    while index as f64 * beat < seconds as f64 {
        let at = index as f64 * beat;
        if index % 2 == 0 {
            add_at(&mut out, at, sr, &kick(sr));
        } else {
            add_at(&mut out, at, sr, &snare(sr));
        }
        index += 1;
    }
    out
}

/// An open hi-hat: 90 ms of bright noise, the off-beat "tss" of house.
fn open_hat(sr: u32, amplitude: f32) -> Vec<f32> {
    let mut noise = Noise(4321);
    let mut previous = 0.0f32;
    (0..(0.09 * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let value = noise.next();
            let bright = value - previous;
            previous = value;
            amplitude * 0.5 * (-t / 0.03).exp() * bright
        })
        .collect()
}

/// A short bass note, 110 ms with a soft attack: the off-beat bass of house.
fn bass_stab(note: i32, sr: u32, amplitude: f32) -> Vec<f32> {
    let freq = midi_hz(note);
    (0..(0.11 * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let envelope = (t / 0.004).min(1.0) * (-t / 0.05).exp();
            amplitude * envelope * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// A house groove as the first bench measured it on records: kick on every beat, and on
/// every off-beat an open hat with a bass note, `offbeat` times as loud as
/// the kick; quiet closed hats on the sixteenths and a little noise.
fn house_groove(bpm: f64, seconds: f32, sr: u32, offbeat: f32) -> Vec<f32> {
    let mut out = white_noise(seconds, sr, 0.02, 11);
    let beat = 60.0 / bpm;
    let mut index = 0;
    while index as f64 * beat < seconds as f64 {
        let at = index as f64 * beat;
        add_at(&mut out, at, sr, &kick(sr));
        add_at(&mut out, at + beat / 2.0, sr, &open_hat(sr, 2.0 * offbeat));
        add_at(&mut out, at + beat / 2.0, sr, &bass_stab(33, sr, offbeat));
        for sixteenth in 0..4 {
            add_at(
                &mut out,
                at + sixteenth as f64 * beat / 4.0,
                sr,
                &hat(sr, 0.15),
            );
        }
        index += 1;
    }
    out
}

/// A shuffle with a figure in threes: kick on every beat, snare on two and
/// four, a quiet shaker on every triplet, an open hat on the last triplet
/// of each beat, and a stab every dotted quarter, `riff` times as loud as
/// the kick: three against two, the way a riff or a delay crosses the beat.
fn shuffle_groove(bpm: f64, seconds: f32, sr: u32, riff: f32) -> Vec<f32> {
    let mut out = white_noise(seconds, sr, 0.02, 12);
    let beat = 60.0 / bpm;
    let mut index = 0;
    while index as f64 * beat < seconds as f64 {
        let at = index as f64 * beat;
        add_at(&mut out, at, sr, &kick(sr));
        if index % 2 == 1 {
            add_at(&mut out, at, sr, &snare(sr));
        }
        add_at(&mut out, at + beat * 2.0 / 3.0, sr, &open_hat(sr, 0.8));
        for triplet in 0..3 {
            add_at(
                &mut out,
                at + triplet as f64 * beat / 3.0,
                sr,
                &hat(sr, 0.15),
            );
        }
        index += 1;
    }
    let mut stab_at = 0.0;
    while stab_at < seconds as f64 {
        add_at(&mut out, stab_at, sr, &bass_stab(45, sr, riff));
        add_at(&mut out, stab_at, sr, &open_hat(sr, riff));
        stab_at += 1.5 * beat;
    }
    out
}

/// A synth bass: a sawtooth's sixteen harmonics falling as 1/n, held. Its
/// fifth harmonic is the major third of its own note, two octaves up.
fn saw_bass(note: i32, seconds: f32, sr: u32, amplitude: f32) -> Vec<f32> {
    let freq = midi_hz(note);
    (0..(seconds * sr as f32) as usize)
        .map(|n| {
            let t = n as f32 / sr as f32;
            let envelope = (t / 0.01).min(1.0) * (-t / 3.0).exp();
            let partials: f32 = (1..=16)
                .filter(|h| freq * (*h as f32) < sr as f32 / 2.0)
                .map(|h| (2.0 * std::f32::consts::PI * freq * h as f32 * t).sin() / h as f32)
                .sum();
            amplitude * envelope * partials
        })
        .collect()
}

/// i–VI–III–VII, the minor loop of dance music, over a synth bass that
/// strikes the tonic with every chord, under a loud kick and hats at 124
/// BPM. The chords use only the notes of the relative major, the bass's
/// harmonics add the parallel major's third, and the drums cover most of
/// the spectrum: the bass on the tonic is what makes it minor.
fn minor_loop_over_tonic(tonic: u8, seconds: f32, sr: u32) -> Vec<f32> {
    let root = 48 + tonic as i32 - if tonic >= 6 { 12 } else { 0 };
    let progression = [
        [root, root + 3, root + 7],
        [root - 4, root, root + 3],
        [root + 3, root + 7, root + 10],
        [root - 2, root + 2, root + 5],
    ];
    let chord_seconds = 2.0;
    let mut out = silence(seconds, sr);
    let mut index = 0;
    while index as f32 * chord_seconds < seconds {
        let chord = &progression[index % progression.len()];
        let mut sound = saw_bass(root - 12, chord_seconds, sr, 0.2);
        for &note in chord {
            add(&mut sound, &tone(note, chord_seconds, sr));
        }
        add_at(&mut out, (index as f32 * chord_seconds) as f64, sr, &sound);
        index += 1;
    }
    let groove: Vec<f32> = kicks_and_hats(124.0, seconds, sr, 1.0)
        .iter()
        .map(|v| 2.0 * v)
        .collect();
    add(&mut out, &groove);
    out
}

// ---------------------------------------------------------------- readings

fn bpm_of(reading: TempoReading) -> f64 {
    match reading {
        TempoReading::Found { bpm, confidence } => {
            assert!(
                (0.0..=1.0).contains(&confidence),
                "confidence {confidence} outside 0..1"
            );
            bpm
        }
        TempoReading::Abstained(reason) => panic!("expected a tempo, abstained: {reason:?}"),
    }
}

fn key_of(reading: KeyReading) -> MusicalKey {
    match reading {
        KeyReading::Found { key, confidence } => {
            assert!(
                (0.0..=1.0).contains(&confidence),
                "confidence {confidence} outside 0..1"
            );
            key
        }
        KeyReading::Abstained(reason) => panic!("expected a key, abstained: {reason:?}"),
    }
}

fn assert_bpm(reading: TempoReading, expected: f64) {
    let bpm = bpm_of(reading);
    assert!(
        (bpm - expected).abs() <= 1.0,
        "read {bpm:.2} BPM, expected {expected} ± 1"
    );
}

fn key(tonic: u8, mode: KeyMode) -> MusicalKey {
    MusicalKey { tonic, mode }
}

// ---------------------------------------------------------------- tempo

/// The plan's precision bar is ±1 BPM; tempi off the integer grid make sure
/// the reading is not a grid artefact.
#[test]
fn clicks_read_their_tempo_within_one_bpm() {
    for bpm in [72.0, 97.3, 120.0, 123.4, 128.0, 140.0, 174.0] {
        let audio = clicks(bpm, 24.0, SR, |_| 1.0);
        let reading = analyze_tempo_and_key(&audio, SR);
        assert_bpm(reading.tempo, bpm);
    }
}

#[test]
fn the_sample_rate_does_not_change_the_reading() {
    for sr in [11_025, 22_050, 44_100, 48_000] {
        let audio = clicks(126.0, 20.0, sr, |_| 1.0);
        assert_bpm(analyze_tempo_and_key(&audio, sr).tempo, 126.0);
    }
}

#[test]
fn clicks_under_loud_noise_still_read_their_tempo() {
    let mut audio = clicks(132.0, 24.0, SR, |_| 1.0);
    add(&mut audio, &white_noise(24.0, SR, 0.3, 42));
    assert_bpm(analyze_tempo_and_key(&audio, SR).tempo, 132.0);
}

#[test]
fn noise_alone_has_neither_tempo_nor_key() {
    for seed in [1, 2, 3] {
        let audio = white_noise(24.0, SR, 0.5, seed);
        let reading = analyze_tempo_and_key(&audio, SR);
        assert_eq!(
            reading.tempo,
            TempoReading::Abstained(TempoAbstention::NoSteadyPulse)
        );
        assert_eq!(
            reading.key,
            KeyReading::Abstained(KeyAbstention::NoTonalContent)
        );
    }
}

#[test]
fn silence_is_silent() {
    let reading = analyze_tempo_and_key(&silence(20.0, SR), SR);
    assert_eq!(
        reading.tempo,
        TempoReading::Abstained(TempoAbstention::Silent)
    );
    assert_eq!(reading.key, KeyReading::Abstained(KeyAbstention::Silent));
}

#[test]
fn a_short_clip_is_too_short() {
    let audio = clicks(120.0, 4.0, SR, |_| 1.0);
    let reading = analyze_tempo_and_key(&audio, SR);
    assert_eq!(
        reading.tempo,
        TempoReading::Abstained(TempoAbstention::TooShort)
    );
    assert_eq!(reading.key, KeyReading::Abstained(KeyAbstention::TooShort));
}

#[test]
fn input_that_is_not_audio_is_refused() {
    let audio = clicks(120.0, 20.0, SR, |_| 1.0);
    let reading = analyze_tempo_and_key(&audio, 0);
    assert_eq!(
        reading.tempo,
        TempoReading::Abstained(TempoAbstention::InvalidInput)
    );
    assert_eq!(
        reading.key,
        KeyReading::Abstained(KeyAbstention::InvalidInput)
    );

    let mut broken = audio.clone();
    broken[1000] = f32::NAN;
    let reading = analyze_tempo_and_key(&broken, SR);
    assert_eq!(
        reading.tempo,
        TempoReading::Abstained(TempoAbstention::InvalidInput)
    );
    assert_eq!(
        reading.key,
        KeyReading::Abstained(KeyAbstention::InvalidInput)
    );
}

/// 110 BPM for half the clip, then 125: there is no one tempo to give.
#[test]
fn a_tempo_that_changes_is_unstable() {
    let mut times = Vec::new();
    let mut at = 0.0;
    while at < 30.0 {
        times.push(at);
        at += if at < 15.0 {
            60.0 / 110.0
        } else {
            60.0 / 125.0
        };
    }
    let audio = clicks_at(&times, 30.0, SR);
    assert_eq!(
        analyze_tempo_and_key(&audio, SR).tempo,
        TempoReading::Abstained(TempoAbstention::UnstableTempo)
    );
}

/// From 100 to 130 BPM in one smooth accelerando: no value is right.
#[test]
fn an_accelerando_has_no_tempo() {
    let mut times = Vec::new();
    let mut at = 0.0;
    while at < 30.0 {
        times.push(at);
        at += 60.0 / (100.0 + at);
    }
    let audio = clicks_at(&times, 30.0, SR);
    let reading = analyze_tempo_and_key(&audio, SR).tempo;
    assert!(
        matches!(reading, TempoReading::Abstained(_)),
        "an accelerando gave {reading:?}"
    );
}

/// Hats on the eighths are the classic way to read double tempo.
#[test]
fn eighth_note_hats_do_not_double_the_tempo() {
    for bpm in [90.0, 122.0] {
        let audio = kicks_and_hats(bpm, 24.0, SR, 0.3);
        assert_bpm(analyze_tempo_and_key(&audio, SR).tempo, bpm);
    }
}

/// Kick on 1 and 3 is the classic way to read half tempo.
#[test]
fn a_backbeat_does_not_halve_the_tempo() {
    for bpm in [100.0, 128.0] {
        let audio = backbeat(bpm, 24.0, SR);
        assert_bpm(analyze_tempo_and_key(&audio, SR).tempo, bpm);
    }
}

/// Strong clicks at 75 BPM with weaker clicks halfway between them, the
/// weak ones growing from nothing to as loud as the strong ones. At one end
/// the beat is 75, at the other 150; in between both readings fit. The
/// reading may move from 75 to 150 once, never back, and only through an
/// abstention: jumping straight from one to the other would give a doubled
/// or halved value to a signal one step away from the other answer.
#[test]
fn double_and_half_change_over_only_through_an_abstention() {
    let mut readings = Vec::new();
    for step in 0..=20 {
        let weak = step as f32 / 20.0;
        let audio = clicks(
            150.0,
            20.0,
            11_025,
            |beat| {
                if beat % 2 == 0 { 1.0 } else { weak }
            },
        );
        let reading = analyze_tempo_and_key(&audio, 11_025).tempo;
        let class = match reading {
            TempoReading::Found { bpm, .. } if (bpm - 75.0).abs() <= 1.0 => 0,
            TempoReading::Abstained(TempoAbstention::DoubleOrHalfTime) => 1,
            TempoReading::Found { bpm, .. } if (bpm - 150.0).abs() <= 1.0 => 2,
            other => panic!("weak clicks at {weak}: {other:?}"),
        };
        readings.push(class);
    }
    assert_eq!(readings.first(), Some(&0), "{readings:?}");
    assert_eq!(readings.last(), Some(&2), "{readings:?}");
    assert!(readings.windows(2).all(|w| w[0] <= w[1]), "{readings:?}");
    assert!(readings.contains(&1), "{readings:?}");
}

/// The first bench's main finding: on records the off-beat of house and techno is not
/// quiet, it carries the hat and often the bass, at 60 % of the kick's
/// weight in the median and up to 90 %. The beat is still the kick.
#[test]
fn a_loud_off_beat_keeps_the_house_tempo() {
    for offbeat in [0.6, 0.75, 0.9] {
        for bpm in [122.0, 125.0, 128.0] {
            let audio = house_groove(bpm, 30.0, SR, offbeat);
            let reading = analyze_tempo_and_key(&audio, SR).tempo;
            let bpm_read = bpm_of(reading);
            assert!(
                (bpm_read - bpm).abs() <= 1.0,
                "off-beat at {offbeat}: read {bpm_read:.2} BPM, expected {bpm} ± 1"
            );
        }
    }
}

/// The first bench saw records read at 2/3 of their tempo: a figure every beat and a
/// half repeats, too. A shuffle with such a figure keeps its tempo.
#[test]
fn a_shuffle_with_a_figure_in_threes_is_not_read_at_two_thirds() {
    for riff in [0.6, 0.9] {
        for bpm in [96.0, 118.0, 124.0] {
            let audio = shuffle_groove(bpm, 30.0, SR, riff);
            let bpm_read = bpm_of(analyze_tempo_and_key(&audio, SR).tempo);
            assert!(
                (bpm_read - bpm).abs() <= 1.0,
                "figure at {riff}: read {bpm_read:.2} BPM, expected {bpm} ± 1"
            );
        }
    }
}

// ---------------------------------------------------------------- key

/// The first bench saw minor records read as major, and most records left without a
/// key under their drums. A minor loop under a groove reads minor.
#[test]
fn a_minor_loop_under_a_groove_reads_minor() {
    for tonic in [0, 2, 4, 6, 9] {
        let audio = minor_loop_over_tonic(tonic, 16.0, 11_025);
        let found = key_of(analyze_tempo_and_key(&audio, 11_025).key);
        assert_eq!(
            found,
            key(tonic, KeyMode::Minor),
            "minor loop on {}",
            key(tonic, KeyMode::Minor).traditional()
        );
    }
}

#[test]
fn chords_name_every_major_and_minor_key() {
    for tonic in 0..12 {
        for mode in [KeyMode::Major, KeyMode::Minor] {
            let expected = key(tonic, mode);
            let audio = chords(&cadence(expected), 2.0, 16.0, 11_025);
            let found = key_of(analyze_tempo_and_key(&audio, 11_025).key);
            assert_eq!(found, expected, "cadence in {}", expected.traditional());
        }
    }
}

/// A chord every two seconds is 30 BPM: below any tempo a DJ would set, and
/// nothing between the chords marks a beat.
#[test]
fn sustained_chords_have_no_tempo() {
    let audio = chords(&cadence(key(0, KeyMode::Major)), 2.0, 16.0, 11_025);
    let reading = analyze_tempo_and_key(&audio, 11_025).tempo;
    assert!(
        matches!(reading, TempoReading::Abstained(_)),
        "sustained chords gave {reading:?}"
    );
}

/// One pitch fits a major and a minor key on it, and more besides.
#[test]
fn a_lone_pitch_has_no_key() {
    let audio = sine(57, 16.0, 11_025);
    assert_eq!(
        analyze_tempo_and_key(&audio, 11_025).key,
        KeyReading::Abstained(KeyAbstention::AmbiguousKey)
    );
}

/// The seven white notes held at equal strength are C major and A minor
/// alike: without a tonic there is no key to give.
#[test]
fn seven_notes_without_a_tonic_have_no_key() {
    let mut audio = silence(16.0, 11_025);
    for note in [60, 62, 64, 65, 67, 69, 71] {
        add(&mut audio, &sine(note, 16.0, 11_025));
    }
    assert_eq!(
        analyze_tempo_and_key(&audio, 11_025).key,
        KeyReading::Abstained(KeyAbstention::AmbiguousKey)
    );
}

#[test]
fn clicks_have_no_key() {
    let audio = clicks(120.0, 20.0, SR, |_| 1.0);
    assert_eq!(
        analyze_tempo_and_key(&audio, SR).key,
        KeyReading::Abstained(KeyAbstention::NoTonalContent)
    );
}

/// A whole groove: kick, snare, hats and chords in E minor at 124 BPM, under
/// noise. Both readings at once, from one call.
#[test]
fn a_groove_over_chords_reads_tempo_and_key() {
    let bpm = 124.0;
    let seconds = 24.0;
    let bar = 4.0 * 60.0 / bpm as f32;
    let mut audio = chords(&cadence(key(4, KeyMode::Minor)), bar, seconds, SR);
    add(&mut audio, &backbeat(bpm, seconds, SR));
    add(&mut audio, &kicks_and_hats(bpm, seconds, SR, 0.2));
    add(&mut audio, &white_noise(seconds, SR, 0.05, 5));
    let reading = analyze_tempo_and_key(&audio, SR);
    assert_bpm(reading.tempo, bpm);
    assert_eq!(key_of(reading.key), key(4, KeyMode::Minor));
}

// ---------------------------------------------------------------- names

/// The Open Key wheel as DJ software prints it, and the plain names.
#[test]
fn keys_are_named_in_open_key_and_traditional_notation() {
    let wheel = [
        (0, KeyMode::Major, "1d", "C"),
        (7, KeyMode::Major, "2d", "G"),
        (2, KeyMode::Major, "3d", "D"),
        (9, KeyMode::Major, "4d", "A"),
        (4, KeyMode::Major, "5d", "E"),
        (11, KeyMode::Major, "6d", "B"),
        (6, KeyMode::Major, "7d", "F♯"),
        (1, KeyMode::Major, "8d", "C♯"),
        (8, KeyMode::Major, "9d", "G♯"),
        (3, KeyMode::Major, "10d", "D♯"),
        (10, KeyMode::Major, "11d", "A♯"),
        (5, KeyMode::Major, "12d", "F"),
        (9, KeyMode::Minor, "1m", "Am"),
        (4, KeyMode::Minor, "2m", "Em"),
        (11, KeyMode::Minor, "3m", "Bm"),
        (6, KeyMode::Minor, "4m", "F♯m"),
        (1, KeyMode::Minor, "5m", "C♯m"),
        (8, KeyMode::Minor, "6m", "G♯m"),
        (3, KeyMode::Minor, "7m", "D♯m"),
        (10, KeyMode::Minor, "8m", "A♯m"),
        (5, KeyMode::Minor, "9m", "Fm"),
        (0, KeyMode::Minor, "10m", "Cm"),
        (7, KeyMode::Minor, "11m", "Gm"),
        (2, KeyMode::Minor, "12m", "Dm"),
    ];
    for (tonic, mode, open, traditional) in wheel {
        let k = key(tonic, mode);
        assert_eq!(k.open_key(), open, "{traditional}");
        assert_eq!(k.traditional(), traditional);
    }
}

/// A root and its fifth, with no third, say nothing of major or minor:
/// the power chord of rock and of many dance basslines.
#[test]
fn a_root_and_its_fifth_have_no_mode() {
    let mut audio = sine(48, 16.0, 11_025);
    add(&mut audio, &sine(55, 16.0, 11_025));
    add(&mut audio, &sine(60, 16.0, 11_025));
    assert_eq!(
        analyze_tempo_and_key(&audio, 11_025).key,
        KeyReading::Abstained(KeyAbstention::AmbiguousKey)
    );
}
