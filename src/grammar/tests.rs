use super::*;
use serde_json::json;

/// Bytes 0..=255, a few multi-byte pieces, one control and two EOG tokens.
const PIECES: &[&[u8]] = &[
    b"{\"", b"\": ", b", \"", b"true", b"null", b"\"}", b"name", b"  ",
];
const CONTROL: usize = 256 + PIECES.len();
const EOG: [usize; 2] = [CONTROL + 1, CONTROL + 2];
const VOCAB: usize = CONTROL + 3;

fn bytes(id: usize) -> Option<&'static [u8]> {
    const SINGLE: [u8; 256] = {
        let mut all = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            all[i] = i as u8;
            i += 1;
        }
        all
    };
    match id {
        0..256 => Some(&SINGLE[id..id + 1]),
        _ if id < CONTROL => Some(PIECES[id - 256]),
        _ => None,
    }
}

fn decoder() -> JsonDecoder {
    JsonDecoder::new(VOCAB, bytes, |id| EOG.contains(&(id as usize))).unwrap()
}

/// SplitMix64, for reproducible random logits.
struct Noise(u64);

impl Noise {
    fn logits(&mut self) -> Vec<f32> {
        (0..VOCAB)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 * 8.0
            })
            .collect()
    }
}

/// Decodes with random logits until an EOG token; returns the text.
fn decode(schema: &JsonSchema, seed: u64, sampled: bool) -> String {
    let mut constraint = decoder().constraint(schema).unwrap();
    let mut sampler = crate::sampling::Sampler::new(crate::sampling::Sampling {
        temperature: if sampled { 1.0 } else { 0.0 },
        seed: Some(seed),
        ..crate::sampling::Sampling::GREEDY
    })
    .unwrap();
    let mut noise = Noise(seed);
    let mut text = Vec::new();
    for _ in 0..2000 {
        let mut logits = noise.logits();
        // Favour closing strings and containers so values stay short.
        for byte in [b'"', b'}', b']'] {
            logits[byte as usize] += 3.0;
        }
        let token = constraint.sample(&logits, |l| sampler.sample(l)).unwrap() as usize;
        if EOG.contains(&token) {
            assert!(constraint.is_complete());
            return String::from_utf8(text).unwrap();
        }
        assert_ne!(token, CONTROL, "control tokens are never allowed");
        text.extend_from_slice(bytes(token).unwrap());
    }
    panic!("no end of generation: {}", String::from_utf8_lossy(&text));
}

fn schemas() -> Vec<Value> {
    vec![
        json!({"type": "object"}),
        json!({
            "type": "object",
            "properties": {"name": {"type": "string", "maxLength": 12}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": false
        }),
        json!({
            "type": "array",
            "items": {"type": "number", "minimum": -1000, "maximum": 1000},
            "maxItems": 5
        }),
        json!({"enum": ["red", "green", 3, null]}),
        json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string", "maxLength": 6}, "maxItems": 3},
                "note": {"type": ["string", "null"], "maxLength": 8}
            },
            "required": ["ok", "tags", "note"],
            "additionalProperties": false
        }),
    ]
}

#[test]
fn greedy_and_sampled_output_always_validates() {
    for schema in schemas() {
        let validator = jsonschema::validator_for(&schema).unwrap();
        let constrained = JsonSchema::new(schema.clone()).unwrap();
        for seed in 0..12 {
            for sampled in [false, true] {
                let text = decode(&constrained, seed, sampled);
                let value: Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    // Valid JSON, but an unbounded `number` can take an
                    // exponent beyond f64 (documented; bound it to avoid).
                    Err(e) if e.to_string().contains("number out of range") => continue,
                    Err(e) => panic!("{text:?} does not parse: {e}"),
                };
                assert!(validator.is_valid(&value), "{text} violates {schema}");
            }
        }
    }
}

#[test]
fn eog_is_allowed_only_once_the_value_is_complete() {
    let schema = JsonSchema::new(json!({"type": "boolean"})).unwrap();
    let mut constraint = decoder().constraint(&schema).unwrap();
    // The model prefers an EOG token, then "{", then "true".
    let mut logits = vec![0.0f32; VOCAB];
    logits[EOG[1]] = 9.0;
    logits[b'{' as usize] = 8.0;
    logits[256 + 3] = 7.0;
    let greedy = |l: &[f32]| crate::sampling::Sampler::new(Default::default())?.sample(l);
    assert_eq!(constraint.sample(&logits, greedy).unwrap(), 256 + 3);
    assert!(constraint.is_complete());
    // Both EOG tokens are allowed now; the model's favourite wins.
    assert_eq!(constraint.sample(&logits, greedy).unwrap(), EOG[1] as i32);
}

#[test]
fn a_draw_outside_the_mask_falls_back_to_the_best_allowed_token() {
    let schema = JsonSchema::new(json!({"type": "null"})).unwrap();
    let mut constraint = decoder().constraint(&schema).unwrap();
    let mut logits = vec![0.0f32; VOCAB];
    logits[b'n' as usize] = 1.0;
    logits[256 + 4] = 2.0;
    let token = constraint.sample(&logits, |_| Ok(CONTROL as i32)).unwrap();
    assert_eq!(token, 256 + 4);
}

#[test]
fn rejects_unsupported_and_malformed_schemas() {
    for (schema, needle) in [
        (json!(true), "must be an object"),
        (json!([1]), "must be an object"),
        (
            json!({"type": "object", "x-guidance": {"lenient": true}}),
            "x-guidance",
        ),
        (json!({"type": "strange"}), "unsupported JSON schema"),
        (
            json!({"type": "object", "dependentSchemas": {"a": {}}}),
            "unsupported JSON schema",
        ),
        (json!({"type": "array", "uniqueItems": true}), "uniqueItems"),
        (json!({"not": {"type": "string"}}), "Unimplemented keys"),
        (
            json!({"type": "string", "format": "weird"}),
            "Unknown format",
        ),
    ] {
        let error = JsonSchema::new(schema.clone()).unwrap_err().to_string();
        assert!(error.contains(needle), "{schema}: {error}");
    }
    assert!(
        JsonSchema::parse("{not json")
            .unwrap_err()
            .to_string()
            .contains("not valid JSON")
    );
    // Annotations are fine.
    JsonSchema::new(json!({"title": "T", "description": "d", "type": "integer", "default": 1}))
        .unwrap();
    assert_eq!(
        JsonSchema::any_object().schema(),
        &json!({"type": "object"})
    );
}

#[test]
fn needs_an_end_of_generation_token() {
    let error = JsonDecoder::new(VOCAB, bytes, |_| false).err().unwrap();
    assert!(error.to_string().contains("end-of-generation"));
}

#[test]
fn whitespace_runs_are_bounded() {
    let schema =
        JsonSchema::new(json!({"type": "array", "items": {"type": "boolean"}, "maxItems": 2}))
            .unwrap();
    let mut constraint = decoder().constraint(&schema).unwrap();
    let greedy = |l: &[f32]| crate::sampling::Sampler::new(Default::default())?.sample(l);
    let mut noise = Noise(1);
    let mut text = Vec::new();
    for _ in 0..1000 {
        let mut logits = noise.logits();
        // A model that would rather print blank space than anything else.
        for token in [b' ' as usize, b'\n' as usize, 256 + 7] {
            logits[token] += 20.0;
        }
        let token = constraint.sample(&logits, greedy).unwrap() as usize;
        if EOG.contains(&token) {
            let text = String::from_utf8(text).unwrap();
            let longest = text.split(|c: char| !c.is_whitespace()).map(str::len).max();
            assert!(longest <= Some(MAX_WHITESPACE), "{text:?}");
            serde_json::from_str::<Value>(&text).unwrap();
            return;
        }
        text.extend_from_slice(bytes(token).unwrap());
    }
    panic!(
        "whitespace never ended: {:?}",
        String::from_utf8_lossy(&text)
    );
}
