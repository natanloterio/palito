//! Offline tokenization using a GGUF's embedded vocabulary.
mod sentencepiece;

use crate::{
    Error, Result,
    gguf::{Gguf, Value},
};
use fancy_regex::Regex;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

use sentencepiece::SentencePiece;

pub(crate) fn phi3_stop_piece(text: &str) -> bool {
    matches!(text, "</s>" | "<|end|>" | "<|endoftext|>")
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("GGUF tokenizer: {message}"))
}

fn byte_alphabet() -> [char; 256] {
    let mut extra = 256;
    std::array::from_fn(|byte| {
        let code = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
            byte as u32
        } else {
            let code = extra;
            extra += 1;
            code
        };
        char::from_u32(code).expect("byte alphabet is within Unicode")
    })
}

/// Offline tokenizer: no external tokenizer files, service, or native inference code.
pub struct Tokenizer {
    segmenter: Segmenter,
    decoded: Vec<Option<Vec<u8>>>,
    unused: Vec<bool>,
    special: Vec<(String, i32)>,
    bos: Option<i32>,
    eos: Option<i32>,
}

enum Segmenter {
    ByteBpe(Box<ByteBpe>),
    SentencePiece(Box<SentencePiece>),
}

struct ByteBpe {
    whole_pieces: Option<HashMap<String, i32>>,
    bytes: [i32; 256],
    merges: HashMap<(i32, i32), (usize, i32)>,
    normalize_nfc: bool,
    digits: Option<Regex>,
    pieces: Regex,
}

impl Tokenizer {
    /// Reads only the GGUF metadata; does not expand or evaluate model weights.
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_metadata(&Gguf::read_metadata(path)?)
    }
    pub(crate) fn from_metadata(metadata: &HashMap<String, Value>) -> Result<Self> {
        let text = |key| match metadata.get(key) {
            Some(Value::Text(value)) => Ok(value.as_str()),
            _ => Err(invalid(format!("missing string {key}"))),
        };
        let flag = |key| match metadata.get(key) {
            Some(Value::Bool(value)) => Ok(*value),
            None => Ok(false),
            _ => Err(invalid(format!("invalid boolean {key}"))),
        };
        let model = text("tokenizer.ggml.model")?;
        // Older SentencePiece conversions predate the pre-tokenizer key.
        let pre = match (model, metadata.contains_key("tokenizer.ggml.pre")) {
            ("llama", false) => "default",
            _ => text("tokenizer.ggml.pre")?,
        };
        let spm = model == "llama" && pre == "default";
        let llama_spm = spm
            && matches!(metadata.get("general.architecture"), Some(Value::Text(arch)) if arch == "llama");
        let phi = spm
            && matches!(metadata.get("general.architecture"), Some(Value::Text(arch)) if arch == "phi3");
        let gemma = spm
            && matches!(metadata.get("general.architecture"), Some(Value::Text(arch)) if matches!(arch.as_str(), "gemma2" | "gemma3"));
        let llama3 = model == "gpt2" && matches!(pre, "llama-bpe" | "llama3" | "llama-v3");
        let split_digits = matches!(pre, "smollm" | "starcoder");
        if !gemma
            && !phi
            && !llama_spm
            && !llama3
            && (model != "gpt2"
                || (!split_digits
                    && !matches!(
                        pre,
                        "qwen2" | "qwen35" | "gpt-2" | "bloom" | "mpt" | "olmo" | "dbrx"
                    )))
        {
            return Err(invalid(
                "supported tokenizers: gpt2/gpt-2, gpt2/bloom, gpt2/mpt, gpt2/olmo, gpt2/dbrx, gpt2/smollm, gpt2/starcoder, gpt2/qwen2, gpt2/qwen35, gpt2/llama-bpe, and Llama/Gemma 3/Phi-3 SentencePiece",
            ));
        }
        for (key, supported) in [
            ("tokenizer.ggml.remove_extra_whitespaces", false),
            ("tokenizer.ggml.add_space_prefix", phi || llama_spm),
            ("tokenizer.ggml.ignore_merges", llama3),
        ] {
            if flag(key)? && !supported {
                return Err(invalid(format!("unsupported option {key}")));
            }
        }
        let tokens = match metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Strings(values)) if !values.is_empty() => values,
            _ => return Err(invalid("missing vocabulary")),
        };
        let types = match metadata.get("tokenizer.ggml.token_type") {
            Some(Value::Integers(values)) if values.len() == tokens.len() => values,
            _ => return Err(invalid("token types must match vocabulary length")),
        };
        let added = |flag_key, id_key| -> Result<Option<i32>> {
            if !flag(flag_key)? {
                return Ok(None);
            }
            match metadata.get(id_key) {
                Some(Value::Integer(id)) if *id < tokens.len() as u64 => Ok(Some(*id as i32)),
                _ => Err(invalid(format!("missing or invalid {id_key}"))),
            }
        };
        let bos = match metadata.get("tokenizer.ggml.add_bos_token") {
            None if llama3 || phi || llama_spm => match metadata.get("tokenizer.ggml.bos_token_id") {
                Some(Value::Integer(id)) if *id < tokens.len() as u64 => Some(*id as i32),
                _ => return Err(invalid("missing or invalid tokenizer.ggml.bos_token_id")),
            },
            _ => added(
                "tokenizer.ggml.add_bos_token",
                "tokenizer.ggml.bos_token_id",
            )?,
        };
        let eos = added(
            "tokenizer.ggml.add_eos_token",
            "tokenizer.ggml.eos_token_id",
        )?;
        let unused = types.iter().map(|kind| *kind == 5).collect();
        if gemma || phi || llama_spm {
            return SentencePiece::from_metadata(metadata, tokens, types, bos, eos, unused);
        }
        let mut vocab = HashMap::new();
        for (index, token) in tokens.iter().enumerate() {
            let id = i32::try_from(index).map_err(invalid)?;
            if token.is_empty() || vocab.insert(token.as_str(), id).is_some() {
                return Err(invalid("empty or duplicate vocabulary token"));
            }
        }
        let alphabet = byte_alphabet();
        let inverse: HashMap<_, _> = alphabet
            .iter()
            .enumerate()
            .map(|(byte, &character)| (character, byte as u8))
            .collect();
        let mut bytes = [-1; 256];
        for (index, character) in alphabet.iter().enumerate() {
            let Some(&id) = vocab.get(character.to_string().as_str()) else {
                continue;
            };
            if types[id as usize] != 1 {
                return Err(invalid("byte tokens must be normal tokens"));
            }
            bytes[index] = id;
        }
        let mut special = Vec::new();
        let mut decoded = Vec::with_capacity(tokens.len());
        for (index, (token, kind)) in tokens.iter().zip(types).enumerate() {
            match kind {
                1 => decoded.push(Some(
                    token
                        .chars()
                        .map(|character| {
                            inverse
                                .get(&character)
                                .copied()
                                .ok_or_else(|| invalid("non-byte character in normal token"))
                        })
                        .collect::<Result<_>>()?,
                )),
                3 | 4 => {
                    special.push((token.clone(), index as i32));
                    decoded.push((*kind == 4).then(|| token.as_bytes().to_vec()));
                }
                5 => decoded.push(None),
                _ => return Err(invalid(format!("unsupported token type {kind}"))),
            }
        }
        let merge_strings = match metadata.get("tokenizer.ggml.merges") {
            Some(Value::Strings(values)) => values,
            _ => return Err(invalid("missing BPE merges")),
        };
        let mut merges = HashMap::new();
        for (rank, merge) in merge_strings.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .filter(|(left, right)| {
                    !left.is_empty() && !right.is_empty() && !right.contains(' ')
                })
                .ok_or_else(|| invalid("malformed BPE merge"))?;
            let id = |piece: &str| {
                vocab
                    .get(piece)
                    .copied()
                    .filter(|id| types[*id as usize] == 1)
                    .ok_or_else(|| invalid(format!("merge references non-normal token {piece:?}")))
            };
            let pair = (id(left)?, id(right)?);
            let merged = id(&format!("{left}{right}"))?;
            if merges.insert(pair, (rank, merged)).is_some() {
                return Err(invalid("duplicate BPE merge"));
            }
        }
        let segmenter = Segmenter::ByteBpe(Box::new(ByteBpe {
            whole_pieces: llama3.then(|| {
                tokens.iter().zip(types).enumerate()
                    .filter(|(_, (_, kind))| **kind == 1)
                    .map(|(id, (token, _))| (token.clone(), id as i32))
                    .collect()
            }),
            bytes,
            merges,
            normalize_nfc: pre == "qwen2",
            digits: split_digits.then(|| Regex::new(r"\p{N}").map_err(invalid)).transpose()?,
            pieces: Regex::new(if pre == "bloom" {
                r" ?[^(\s|.,!?…。，、।۔،)]+"
            } else if pre == "qwen35" {
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            } else if pre == "qwen2" {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            } else if llama3 || pre == "dbrx" {
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            } else {
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
            })
            .map_err(invalid)?,
        }));
        Ok(Self {
            segmenter,
            decoded,
            unused,
            special,
            bos,
            eos,
        })
    }
}

impl ByteBpe {
    fn bpe(&self, text: &str, output: &mut Vec<i32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if let Some(pieces) = &self.whole_pieces {
            let alphabet = byte_alphabet();
            let encoded: String = text.bytes().map(|b| alphabet[b as usize]).collect();
            if let Some(&id) = pieces.get(&encoded) {
                output.push(id);
                return Ok(());
            }
        }
        if let Some(byte) = text.bytes().find(|&byte| self.bytes[byte as usize] < 0) {
            return Err(invalid(format!(
                "vocabulary has no token for input byte {byte}"
            )));
        }
        struct Node {
            id: i32,
            previous: Option<usize>,
            next: Option<usize>,
            active: bool,
        }
        let mut nodes: Vec<_> = text
            .bytes()
            .enumerate()
            .map(|(index, byte)| Node {
                id: self.bytes[byte as usize],
                previous: index.checked_sub(1),
                next: (index + 1 < text.len()).then_some(index + 1),
                active: true,
            })
            .collect();
        let mut candidates = BinaryHeap::new();
        let enqueue = |left: usize, right: usize, nodes: &[Node], heap: &mut BinaryHeap<_>| {
            if let Some(&(rank, _)) = self.merges.get(&(nodes[left].id, nodes[right].id)) {
                heap.push(Reverse((rank, left, right)));
            }
        };
        for left in 0..nodes.len() - 1 {
            enqueue(left, left + 1, &nodes, &mut candidates);
        }
        while let Some(Reverse((rank, left, right))) = candidates.pop() {
            if !nodes[left].active || !nodes[right].active || nodes[left].next != Some(right) {
                continue;
            }
            let Some(&(current_rank, merged)) = self.merges.get(&(nodes[left].id, nodes[right].id))
            else {
                continue;
            };
            if rank != current_rank {
                continue;
            }
            nodes[left].id = merged;
            nodes[left].next = nodes[right].next;
            nodes[right].active = false;
            if let Some(next) = nodes[right].next {
                nodes[next].previous = Some(left);
                enqueue(left, next, &nodes, &mut candidates);
            }
            if let Some(previous) = nodes[left].previous {
                enqueue(previous, left, &nodes, &mut candidates);
            }
        }
        output.extend(nodes.iter().filter(|node| node.active).map(|node| node.id));
        Ok(())
    }

    fn ordinary(&self, text: &str, output: &mut Vec<i32>) -> Result<()> {
        let normalized = self.normalize_nfc.then(|| text.nfc().collect::<String>());
        let text = normalized.as_deref().unwrap_or(text);
        let split_words = |text: &str, output: &mut Vec<i32>| -> Result<()> {
            let mut cursor = 0;
            for matched in self.pieces.find_iter(text) {
                let matched = matched.map_err(invalid)?;
                self.bpe(&text[cursor..matched.start()], output)?;
                self.bpe(matched.as_str(), output)?;
                cursor = matched.end();
            }
            self.bpe(&text[cursor..], output)?;
            Ok(())
        };
        // SmolLM isolates each Unicode digit before applying GPT-2 word splitting.
        let mut cursor = 0;
        let Some(digits) = &self.digits else {
            return split_words(text, output);
        };
        for digit in digits.find_iter(text) {
            let digit = digit.map_err(invalid)?;
            split_words(&text[cursor..digit.start()], output)?;
            self.bpe(digit.as_str(), output)?;
            cursor = digit.end();
        }
        split_words(&text[cursor..], output)
    }
}

impl Tokenizer {
    fn ordinary(&self, text: &str, output: &mut Vec<i32>) -> Result<()> {
        match &self.segmenter {
            Segmenter::ByteBpe(bpe) => bpe.ordinary(text, output),
            Segmenter::SentencePiece(spm) => {
                if !text.is_empty() {
                    spm.encode(text, output);
                }
                Ok(())
            }
        }
    }

    /// Recognizes embedded control/user-defined tokens and applies configured BOS/EOS.
    /// Qwen2 uses NFC normalization. No chat template is applied.
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let mut output: Vec<_> = self.bos.into_iter().collect();
        let mut remaining = text;
        while !remaining.is_empty() {
            let next = self
                .special
                .iter()
                .filter_map(|(token, id)| {
                    remaining
                        .find(token)
                        .map(|start| (start, Reverse(token.len()), *id))
                })
                .min();
            match next {
                Some((start, Reverse(length), id)) => {
                    self.ordinary(&remaining[..start], &mut output)?;
                    output.push(id);
                    let piece = &remaining[start..start + length];
                    remaining = &remaining[start + length..];
                    if let Segmenter::SentencePiece(spm) = &self.segmenter {
                        remaining = spm.after_special(piece, remaining);
                    }
                }
                None => {
                    self.ordinary(remaining, &mut output)?;
                    break;
                }
            }
        }
        output.extend(self.eos);
        Ok(output)
    }

    /// Decodes a complete sequence, omitting controls. Incomplete UTF-8 bytes use
    /// the standard replacement character; decode together, not token by token.
    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in tokens {
            let piece = usize::try_from(id)
                .ok()
                .and_then(|id| self.decoded.get(id))
                .ok_or_else(|| invalid(format!("token ID {id} is outside the vocabulary")))?;
            if self.unused[id as usize] {
                return Err(invalid(format!("token ID {id} is marked unused")));
            }
            if let Some(piece) = piece {
                bytes.extend_from_slice(piece);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn vocab_tokens(&self) -> usize {
        self.decoded.len()
    }

    /// The bytes `detokenize` emits for one token; None for controls, unused
    /// tokens and IDs outside the vocabulary.
    pub fn token_bytes(&self, id: usize) -> Option<&[u8]> {
        if self.unused.get(id).copied().unwrap_or(true) {
            return None;
        }
        self.decoded.get(id)?.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_id(tokenizer: &Tokenizer, byte: u8) -> i32 {
        let Segmenter::ByteBpe(bpe) = &tokenizer.segmenter else {
            panic!("expected byte BPE");
        };
        bpe.bytes[byte as usize]
    }

    fn metadata(merges: &[(&str, &str)]) -> HashMap<String, Value> {
        let mut tokens: Vec<_> = byte_alphabet().iter().map(char::to_string).collect();
        let mut types = vec![1; tokens.len()];
        for (text, kind) in [("<s>", 3), ("<s>long", 3), ("<literal>", 4)] {
            tokens.push(text.into());
            types.push(kind);
        }
        for (left, right) in merges {
            let token = format!("{left}{right}");
            if !tokens.contains(&token) {
                tokens.push(token);
                types.push(1);
            }
        }
        HashMap::from([
            ("tokenizer.ggml.model".into(), Value::Text("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::Text("smollm".into())),
            ("tokenizer.ggml.tokens".into(), Value::Strings(tokens)),
            ("tokenizer.ggml.token_type".into(), Value::Integers(types)),
            (
                "tokenizer.ggml.merges".into(),
                Value::Strings(merges.iter().map(|(a, b)| format!("{a} {b}")).collect()),
            ),
        ])
    }

    #[test]
    fn bpe_uses_rank_then_leftmost_position_and_discards_stale_pairs() {
        let metadata = metadata(&[("b", "c"), ("a", "b"), ("a", "a"), ("aa", "aa")]);
        let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
        let tokens = tokenizer.tokenize("abc").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokenizer.detokenize(&tokens[..1]).unwrap(), "a");
        assert_eq!(tokenizer.detokenize(&tokens[1..]).unwrap(), "bc");
        assert_eq!(tokenizer.tokenize("aaaa").unwrap().len(), 1);
        assert_eq!(tokenizer.tokenize("aaaaaa").unwrap().len(), 2);
        assert_eq!(
            tokenizer
                .detokenize(&tokenizer.tokenize("aaaaaa").unwrap())
                .unwrap(),
            "aaaaaa"
        );
    }

    #[test]
    fn digit_and_word_boundaries_prevent_cross_piece_merges() {
        let metadata = metadata(&[
            ("1", "2"),
            ("a", "1"),
            ("'", "s"),
            ("a", "'s"),
            ("\u{120}", "a"),
        ]);
        let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(
            tokenizer.tokenize("a12").unwrap(),
            [
                byte_id(&tokenizer, b'a'),
                byte_id(&tokenizer, b'1'),
                byte_id(&tokenizer, b'2')
            ]
        );
        assert_eq!(tokenizer.tokenize("a's").unwrap().len(), 2);
        assert_eq!(tokenizer.tokenize(" a").unwrap().len(), 1);
        for text in [
            "",
            "  a \t b\r\n\n ",
            "I'm WE'RE can't",
            "123 \u{661}\u{662} \u{ff11}\u{ff12}",
            "\u{e9} cafe\u{301} \u{4e2d} \u{1f642} \0",
        ] {
            assert_eq!(
                tokenizer
                    .detokenize(&tokenizer.tokenize(text).unwrap())
                    .unwrap(),
                text
            );
        }
    }

    #[test]
    fn qwen_contractions_digits_and_unused_padding_have_distinct_semantics() {
        let mut metadata = metadata(&[("'", "S"), ("1", "2")]);
        metadata.insert("tokenizer.ggml.pre".into(), Value::Text("qwen2".into()));
        if let Some(Value::Strings(tokens)) = metadata.get_mut("tokenizer.ggml.tokens") {
            tokens.push("<unused>".into());
        }
        if let Some(Value::Integers(types)) = metadata.get_mut("tokenizer.ggml.token_type") {
            types.push(5);
        }
        let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.tokenize("'S").unwrap().len(), 1);
        assert_eq!(tokenizer.tokenize("12").unwrap().len(), 2);
        assert_eq!(
            tokenizer
                .detokenize(&tokenizer.tokenize("e\u{301} a\u{308}").unwrap())
                .unwrap(),
            "\u{e9} \u{e4}"
        );
        assert!(tokenizer.tokenize("<unused>").unwrap().len() > 1);
        assert!(
            tokenizer
                .detokenize(&[(tokenizer.vocab_tokens() - 1) as i32])
                .is_err()
        );
    }

    #[test]
    fn specials_use_longest_match_and_configured_bos_eos() {
        let mut metadata = metadata(&[]);
        for (key, value) in [
            ("tokenizer.ggml.add_bos_token", Value::Bool(true)),
            ("tokenizer.ggml.bos_token_id", Value::Integer(256)),
            ("tokenizer.ggml.add_eos_token", Value::Bool(true)),
            ("tokenizer.ggml.eos_token_id", Value::Integer(257)),
        ] {
            metadata.insert(key.into(), value);
        }
        let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
        let tokens = tokenizer.tokenize("<s>longa<literal><s>").unwrap();
        assert_eq!(tokens, [256, 257, byte_id(&tokenizer, b'a'), 258, 256, 257]);
        assert_eq!(tokenizer.detokenize(&tokens).unwrap(), "a<literal>");
        assert_eq!(tokenizer.tokenize("").unwrap(), [256, 257]);
    }

    #[test]
    fn decoding_reassembles_utf8_and_rejects_out_of_range_ids() {
        let tokenizer = Tokenizer::from_metadata(&metadata(&[])).unwrap();
        let tokens = [byte_id(&tokenizer, 0xc3), byte_id(&tokenizer, 0xa9)];
        assert_eq!(tokenizer.detokenize(&tokens).unwrap(), "\u{e9}");
        assert_eq!(tokenizer.detokenize(&tokens[..1]).unwrap(), "\u{fffd}");
        assert!(tokenizer.detokenize(&[-1]).is_err());
        assert!(
            tokenizer
                .detokenize(&[tokenizer.vocab_tokens() as i32])
                .is_err()
        );
        let mut metadata = metadata(&[]);
        if let Some(Value::Strings(tokens)) = metadata.get_mut("tokenizer.ggml.tokens") {
            tokens[4] = "unused".into();
        }
        let pruned = Tokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(pruned.tokenize("a").unwrap(), [byte_id(&pruned, b'a')]);
        assert!(
            pruned
                .tokenize("\u{4}")
                .unwrap_err()
                .to_string()
                .contains("input byte 4")
        );
    }

    #[test]
    fn malformed_or_unsupported_tokenizers_fail_explicitly() {
        for (key, value) in [
            ("tokenizer.ggml.pre", Value::Text("unsupported".into())),
            (
                "tokenizer.ggml.tokens",
                Value::Strings(vec!["duplicate".into(); 259]),
            ),
            ("tokenizer.ggml.token_type", Value::Integers(vec![1; 2])),
            ("tokenizer.ggml.token_type", Value::Integers(vec![3; 259])),
            (
                "tokenizer.ggml.merges",
                Value::Strings(vec!["a missing".into()]),
            ),
            (
                "tokenizer.ggml.merges",
                Value::Strings(vec!["a b extra".into()]),
            ),
            ("tokenizer.ggml.add_space_prefix", Value::Bool(true)),
            ("tokenizer.ggml.add_bos_token", Value::Integer(1)),
            ("tokenizer.ggml.add_bos_token", Value::Bool(true)),
        ] {
            let mut metadata = metadata(&[]);
            metadata.insert(key.into(), value);
            assert!(Tokenizer::from_metadata(&metadata).is_err(), "{key}");
        }
    }
}
