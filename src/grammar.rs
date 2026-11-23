//! Grammar-constrained decoding: output that always matches a JSON Schema
//! (`response_format` in the OpenAI API).
//!
//! The grammar engine is llguidance (the one llama.cpp and vLLM use). Before
//! every token it computes the set of vocabulary entries that keep the output
//! a valid prefix of the grammar; the others get `MASKED_LOGIT` before the
//! sampler runs, so greedy and sampled decoding both stay inside the grammar.
//! When the JSON value is complete only end-of-generation tokens are allowed.
//!
//! Whitespace between JSON tokens is allowed (models like to pretty-print,
//! and forbidding a newline they want pushes them into runaway digits or
//! strings) but each run is at most `MAX_WHITESPACE` characters, so a model
//! cannot loop on blank lines forever.

use crate::{Error, Result};
use llguidance::{
    JsonCompileOptions, Matcher, ParserFactory,
    api::TopLevelGrammar,
    toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
};
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};

/// Score given to tokens the grammar forbids: finite, so the sampler's
/// nonfinite-logit check still guards real logits, and `exp` of it is 0.
const MASKED_LOGIT: f32 = f32::MIN;

/// Longest whitespace run between two JSON tokens: a newline plus indentation.
const MAX_WHITESPACE: usize = 32;

/// Keyword llguidance reads its own compile options from; the request must not
/// override ours.
const OPTIONS_KEYWORD: &str = "x-guidance";

/// A JSON Schema checked to compile into a grammar.
///
/// Supported: `type` (object, array, string, number, integer, boolean, null,
/// or a list of them), `properties`, `required`, `additionalProperties`,
/// `items`, `enum`, `const`, and what llguidance adds on top (`anyOf`,
/// `oneOf`, `allOf`, local `$ref`, `pattern`, `minLength`/`maxLength`,
/// `minimum`/`maximum`, `minItems`/`maxItems`, common `format`s).
/// Annotations (`title`, `description`, `default`, `examples`, `$schema`, ...)
/// are ignored. Any other JSON Schema keyword is an error, never silently
/// dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonSchema(Value);

impl JsonSchema {
    /// Any JSON object (`response_format: {"type": "json_object"}`).
    pub fn any_object() -> Self {
        Self(json!({"type": "object"}))
    }

    pub fn new(schema: Value) -> Result<Self> {
        let Some(object) = schema.as_object() else {
            return Err(invalid("JSON schema must be an object"));
        };
        if object.contains_key(OPTIONS_KEYWORD) {
            return Err(invalid(format!(
                "unsupported JSON schema keyword {OPTIONS_KEYWORD:?}"
            )));
        }
        let schema = Self(schema);
        schema.matcher(validator())?;
        Ok(schema)
    }

    /// Parses JSON Schema text.
    pub fn parse(text: &str) -> Result<Self> {
        let value = serde_json::from_str(text)
            .map_err(|error| invalid(format!("JSON schema is not valid JSON: {error}")))?;
        Self::new(value)
    }

    pub fn schema(&self) -> &Value {
        &self.0
    }

    fn grammar(&self) -> TopLevelGrammar {
        let mut schema = self.0.clone();
        if let Some(object) = schema.as_object_mut() {
            object.insert(OPTIONS_KEYWORD.into(), json_compile_options());
        }
        TopLevelGrammar::from_json_schema(schema)
    }

    fn matcher(&self, factory: &ParserFactory) -> Result<Matcher> {
        let parser = factory
            .create_parser(self.grammar())
            .map_err(|error| invalid(format!("unsupported JSON schema: {error}")))?;
        let matcher = Matcher::new(Ok(parser));
        match matcher.get_error() {
            Some(error) => Err(invalid(format!("unsupported JSON schema: {error}"))),
            None => Ok(matcher),
        }
    }
}

/// llguidance options for JSON: bounded whitespace between tokens.
pub(crate) fn json_compile_options() -> Value {
    let options = JsonCompileOptions {
        whitespace_pattern: Some(format!("[\\x20\\x0A\\x0D\\x09]{{1,{MAX_WHITESPACE}}}")),
        ..JsonCompileOptions::default()
    };
    serde_json::to_value(options).unwrap_or_else(|_| json!({}))
}

/// Schema errors do not depend on the vocabulary: a byte-level one checks
/// requests without a model (the server answers 400 before queueing).
fn validator() -> &'static ParserFactory {
    static FACTORY: OnceLock<ParserFactory> = OnceLock::new();
    FACTORY.get_or_init(|| {
        let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
            .expect("byte-level grammar factory");
        factory.quiet();
        factory
    })
}

/// A model's vocabulary prepared for grammar masks; built once per engine and
/// reused by every request.
pub struct JsonDecoder {
    factory: ParserFactory,
    /// End-of-generation tokens; `eog[0]` is the grammar's EOS.
    eog: Arc<[u32]>,
}

impl JsonDecoder {
    /// `vocab_tokens` is the logits width; `bytes` gives the text of a token
    /// (None for controls and unused tokens); `is_eog` marks end-of-generation
    /// tokens.
    pub fn new<'a>(
        vocab_tokens: usize,
        bytes: impl Fn(usize) -> Option<&'a [u8]>,
        is_eog: impl Fn(i32) -> bool,
    ) -> Result<Self> {
        let vocab_size = u32::try_from(vocab_tokens)
            .map_err(|_| invalid("vocabulary too large for grammar masks"))?;
        let eog: Arc<[u32]> = (0..vocab_size).filter(|&id| is_eog(id as i32)).collect();
        let Some(&eos) = eog.first() else {
            return Err(invalid(
                "JSON output needs a model with an end-of-generation token",
            ));
        };
        let words: Vec<Vec<u8>> = (0..vocab_tokens)
            .map(|id| word(bytes(id), id, eog.contains(&(id as u32))))
            .collect();
        let trie = TokTrie::from(&TokRxInfo::new(vocab_size, eos), &words);
        let env: TokEnv = Arc::new(TrieEnv(trie));
        let mut factory = ParserFactory::new_simple(&env)
            .map_err(|error| Error::Backend(format!("grammar vocabulary: {error}")))?;
        factory.quiet();
        Ok(Self { factory, eog })
    }

    pub fn constraint(&self, schema: &JsonSchema) -> Result<Constraint> {
        Ok(Constraint {
            matcher: schema.matcher(&self.factory)?,
            eog: self.eog.clone(),
            mask: SimpleVob::new(),
            masked: Vec::new(),
        })
    }

    /// A constraint from a Lark grammar.
    pub(crate) fn lark_constraint(&self, grammar: &str) -> Result<Constraint> {
        let parser = self
            .factory
            .create_parser(TopLevelGrammar::from_lark(grammar.to_owned()))
            .map_err(|error| invalid(format!("tool call grammar: {error}")))?;
        let matcher = Matcher::new(Ok(parser));
        if let Some(error) = matcher.get_error() {
            return Err(invalid(format!("tool call grammar: {error}")));
        }
        Ok(Constraint {
            matcher,
            eog: self.eog.clone(),
            mask: SimpleVob::new(),
            masked: Vec::new(),
        })
    }
}

/// Trie bytes for one token. End-of-generation tokens get the special-token
/// form (never matched by grammar text); controls and unused tokens get no
/// bytes, so no grammar ever allows them.
fn word(bytes: Option<&[u8]>, id: usize, eog: bool) -> Vec<u8> {
    if eog {
        let mut special = vec![TokTrie::SPECIAL_TOKEN_MARKER];
        special.extend_from_slice(format!("[{id}]").as_bytes());
        return special;
    }
    bytes.map(<[u8]>::to_vec).unwrap_or_default()
}

struct TrieEnv(TokTrie);

impl TokenizerEnv for TrieEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.0
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.0.greedy_tokenize(bytes)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

/// One request's grammar state.
pub struct Constraint {
    matcher: Matcher,
    eog: Arc<[u32]>,
    /// Tokens allowed at the current step.
    mask: SimpleVob,
    masked: Vec<f32>,
}

impl Constraint {
    /// An independent copy in the current state.
    pub(crate) fn fresh(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
            eog: self.eog.clone(),
            mask: SimpleVob::new(),
            masked: Vec::new(),
        }
    }

    /// Draws the next token with `pick` over `logits` restricted to the grammar,
    /// then advances the grammar past it.
    pub fn sample(
        &mut self,
        logits: &[f32],
        pick: impl FnOnce(&[f32]) -> Result<i32>,
    ) -> Result<i32> {
        self.mask_logits(logits)?;
        let mut token = pick(&self.masked)?;
        if !self.allowed(token) {
            // Rounding in a sampler's cumulative draw can land past the last
            // allowed token; the best allowed one replaces it.
            token = self.best_allowed()?;
        }
        self.accept(token)?;
        Ok(token)
    }

    fn mask_logits(&mut self, logits: &[f32]) -> Result<()> {
        self.mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|error| Error::Backend(format!("grammar mask: {error}")))?;
        if self.mask.is_allowed(self.eog[0]) {
            for &token in self.eog.iter() {
                self.mask.allow_token(token);
            }
        }
        self.masked.clear();
        self.masked.resize(logits.len(), MASKED_LOGIT);
        let mut any = false;
        let masked = &mut self.masked;
        self.mask.iter_set_entries(|index| {
            if let Some(&logit) = logits.get(index) {
                masked[index] = logit;
                any = true;
            }
        });
        if !any {
            return Err(Error::Backend("grammar allows no token".into()));
        }
        Ok(())
    }

    fn allowed(&self, token: i32) -> bool {
        u32::try_from(token)
            .is_ok_and(|token| (token as usize) < self.masked.len() && self.mask.is_allowed(token))
    }

    fn best_allowed(&self) -> Result<i32> {
        let mut best = None::<(usize, f32)>;
        self.mask.iter_set_entries(|index| {
            let logit = self.masked.get(index).copied().unwrap_or(MASKED_LOGIT);
            if index < self.masked.len() && best.is_none_or(|(_, max)| logit > max) {
                best = Some((index, logit));
            }
        });
        best.map(|(index, _)| index as i32)
            .ok_or_else(|| Error::Backend("grammar allows no token".into()))
    }

    fn accept(&mut self, token: i32) -> Result<()> {
        let token = token as u32;
        if self.eog.contains(&token) {
            return Ok(());
        }
        self.matcher
            .consume_token(token)
            .map_err(|error| Error::Backend(format!("grammar rejected token {token}: {error}")))
    }

    /// Whether the output so far is a complete value of the grammar.
    pub fn is_complete(&mut self) -> bool {
        self.matcher.is_stopped() || self.matcher.is_accepting().unwrap_or(false)
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

#[cfg(test)]
mod tests;
