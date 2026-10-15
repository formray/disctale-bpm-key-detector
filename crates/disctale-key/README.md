# disctale-key

The musical key of a clip of mono audio, or the reason there is none. One
pure function, no dependencies:

```rust
let reading = disctale_key::analyze_tempo_and_key(&samples, 44_100);
if let disctale_key::KeyReading::Found { key, .. } = reading.key {
    println!("{} ({})", key.traditional(), key.open_key()); // "Am (1m)"
}
```

It also reads a tempo, which DISCTALE does not serve: its tempo comes from
`disctale-beat`. Method, measurements, limits and sources:
[the repository README](https://github.com/formray/disctale-bpm-key-detector).

Licence: MIT OR Apache-2.0.
