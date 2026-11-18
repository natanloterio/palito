use super::{Segmenter, Tokenizer, invalid, phi3_stop_piece};
use crate::{Result, gguf::Value};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap},
};

pub(super) struct SentencePiece {
    add_space_prefix: bool,
    trim_phi_special_right: bool,
    vocab: HashMap<String, (i32, f32)>,
    bytes: [i32; 256],
}

#[derive(PartialEq)]
struct Score(f32);

impl Eq for Score {}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("finite vocabulary scores")
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SentencePiece {
    pub(super) fn from_metadata(
        metadata: &HashMap<String, Value>,
        tokens: &[String],
        types: &[u64],
        bos: Option<i32>,
        eos: Option<i32>,
        unused: Vec<bool>,
    ) -> Result<Tokenizer> {
        let phi = matches!(metadata.get("general.architecture"), Some(Value::Text(arch)) if arch == "phi3");
        let add_space_prefix = phi
            && !matches!(
                metadata.get("tokenizer.ggml.add_space_prefix"),
                Some(Value::Bool(false))
            );
        let trim_phi_special_right = phi
            && matches!(metadata.get("general.name"), Some(Value::Text(name))
                if { let name = name.to_ascii_lowercase(); name.contains("phi-3") || name.contains("phi3") });
        let scores = match metadata.get("tokenizer.ggml.scores") {
            Some(Value::Floats(scores))
                if scores.len() == tokens.len() && scores.iter().all(|v| v.is_finite()) =>
            {
                scores
            }
            _ => {
                return Err(invalid(
                    "SentencePiece scores must be finite and match vocabulary",
                ));
            }
        };
        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut bytes = [-1; 256];
        let mut special = Vec::new();
        let mut decoded = Vec::with_capacity(tokens.len());
        for (index, ((token, &kind), &score)) in tokens.iter().zip(types).zip(scores).enumerate() {
            let id = i32::try_from(index).map_err(invalid)?;
            if token.is_empty() || vocab.insert(token.clone(), (id, score)).is_some() {
                return Err(invalid("empty or duplicate vocabulary token"));
            }
            let piece = match kind {
                1 => Some(token.replace('\u{2581}', " ").into_bytes()),
                2 if phi => {
                    special.push((token.clone(), id));
                    None
                }
                2 | 5 => None,
                3 | 4 => {
                    special.push((token.clone(), id));
                    (kind == 4 && token != "<end_of_turn>" && !(phi && phi3_stop_piece(token)))
                        .then(|| token.as_bytes().to_vec())
                }
                6 => {
                    let byte = token
                        .strip_prefix("<0x")
                        .and_then(|s| s.strip_suffix('>'))
                        .filter(|s| s.len() == 2)
                        .and_then(|s| u8::from_str_radix(s, 16).ok())
                        .filter(|&b| bytes[b as usize] == -1)
                        .ok_or_else(|| invalid("invalid or duplicate SentencePiece byte token"))?;
                    bytes[byte as usize] = id;
                    Some(vec![byte])
                }
                _ => {
                    return Err(invalid(format!(
                        "unsupported SentencePiece token type {kind}"
                    )));
                }
            };
            decoded.push(piece);
        }
        if bytes.contains(&-1) {
            return Err(invalid("SentencePiece requires all 256 byte tokens"));
        }
        Ok(Tokenizer {
            segmenter: Segmenter::SentencePiece(Box::new(Self {
                vocab,
                bytes,
                add_space_prefix,
                trim_phi_special_right,
            })),
            decoded,
            unused,
            special,
            bos,
            eos,
        })
    }

    pub(super) fn after_special<'a>(&self, piece: &str, rest: &'a str) -> &'a str {
        if self.trim_phi_special_right && !matches!(piece, "<unk>" | "<s>" | "<|endoftext|>") {
            rest.trim_start_matches(|c: char| {
                matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
            })
        } else {
            rest
        }
    }

    pub(super) fn encode(&self, text: &str, output: &mut Vec<i32>) {
        let escaped = text.replace(' ', "\u{2581}");
        let text = if self.add_space_prefix {
            format!("\u{2581}{escaped}")
        } else {
            escaped
        };
        let mut nodes: Vec<_> = text
            .char_indices()
            .enumerate()
            .map(|(index, (start, ch))| Node {
                start,
                end: start + ch.len_utf8(),
                previous: index.checked_sub(1),
                next: Some(index + 1).filter(|_| start + ch.len_utf8() < text.len()),
            })
            .collect();
        let mut candidates = BinaryHeap::new();
        let enqueue = |left: usize, right: usize, nodes: &[Node], heap: &mut BinaryHeap<_>| {
            let end = nodes[right].end;
            if let Some(&(_, score)) = self.vocab.get(&text[nodes[left].start..end]) {
                heap.push((Score(score), Reverse(left), right, end));
            }
        };
        for left in 0..nodes.len().saturating_sub(1) {
            enqueue(left, left + 1, &nodes, &mut candidates);
        }
        while let Some((_, Reverse(left), right, end)) = candidates.pop() {
            if nodes[left].next != Some(right) || nodes[right].end != end {
                continue;
            }
            nodes[left].end = end;
            nodes[left].next = nodes[right].next.take();
            if let Some(next) = nodes[left].next {
                nodes[next].previous = Some(left);
                enqueue(left, next, &nodes, &mut candidates);
            }
            if let Some(previous) = nodes[left].previous {
                enqueue(previous, left, &nodes, &mut candidates);
            }
        }
        let mut cursor = (!nodes.is_empty()).then_some(0);
        while let Some(index) = cursor {
            let node = &nodes[index];
            let piece = &text[node.start..node.end];
            if let Some(&(id, _)) = self.vocab.get(piece) {
                output.push(id);
            } else {
                output.extend(piece.bytes().map(|b| self.bytes[b as usize]));
            }
            cursor = node.next;
        }
    }
}

struct Node {
    start: usize,
    end: usize,
    previous: Option<usize>,
    next: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> HashMap<String, Value> {
        let mut tokens: Vec<_> = (0..256).map(|byte| format!("<0x{byte:02X}>")).collect();
        let mut types = vec![6; 256];
        let mut scores = vec![0.0; 256];
        for (text, kind, score) in [
            ("<bos>", 3, 0.0),
            ("<end_of_turn>", 3, 0.0),
            ("a", 1, 0.0),
            ("b", 1, 0.0),
            ("c", 1, 0.0),
            ("ab", 1, 1.0),
            ("bc", 1, 2.0),
            ("▁a", 1, 3.0),
        ] {
            tokens.push(text.into());
            types.push(kind);
            scores.push(score);
        }
        HashMap::from([
            ("general.architecture".into(), Value::Text("gemma3".into())),
            ("tokenizer.ggml.model".into(), Value::Text("llama".into())),
            ("tokenizer.ggml.pre".into(), Value::Text("default".into())),
            ("tokenizer.ggml.tokens".into(), Value::Strings(tokens)),
            ("tokenizer.ggml.token_type".into(), Value::Integers(types)),
            ("tokenizer.ggml.scores".into(), Value::Floats(scores)),
            ("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)),
            ("tokenizer.ggml.bos_token_id".into(), Value::Integer(256)),
        ])
    }

    #[test]
    fn scored_pairs_space_control_and_utf8_fallback_round_trip() {
        let tokenizer = Tokenizer::from_metadata(&fixture()).unwrap();
        let ids = tokenizer.tokenize("abc<end_of_turn> a🦀").unwrap();
        assert_eq!(ids, [256, 258, 262, 257, 263, 240, 159, 166, 128]);
        assert_eq!(tokenizer.detokenize(&ids).unwrap(), "abc a🦀");
        assert_eq!(tokenizer.tokenize("").unwrap(), [256]);
    }

    #[test]
    fn rejects_nonfinite_scores() {
        let mut metadata = fixture();
        let Some(Value::Floats(scores)) = metadata.get_mut("tokenizer.ggml.scores") else {
            panic!("missing test scores");
        };
        scores[261] = f32::NAN;
        assert!(Tokenizer::from_metadata(&metadata).is_err());
    }
}
