//! Token sampling from host logits, in llama.cpp's order: logit bias,
//! repetition penalties, top-k, top-p (nucleus), min-p, temperature, draw.
//!
//! `temperature == 0` (the default) or `top_k == 1` is greedy. Greedy with no
//! logit bias and no active penalty lets backends keep their token-only fast
//! path (the CUDA decode graph's device argmax). Anything else needs the full
//! logits on the host for every generated token (`Sampling::needs_host_logits`).

mod penalties;

use crate::{Error, Result};
use penalties::History;
use std::time::{SystemTime, UNIX_EPOCH};

/// Candidates whose probability is below this fraction of the most likely
/// token's are skipped by the first nucleus pass (an exact full pass follows
/// whenever they could matter).
const NUCLEUS_PREFILTER: f32 = 1e-6;

/// llama.cpp's default repetition window.
pub const DEFAULT_REPEAT_LAST_N: u32 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    /// 0: greedy. Logits are divided by this before the final draw.
    pub temperature: f32,
    /// Keep the smallest set of most likely tokens whose probability reaches
    /// this mass (1: off).
    pub top_p: f32,
    /// Keep only the k most likely tokens (0: off).
    pub top_k: u32,
    /// Drop tokens less likely than this fraction of the most likely one, in
    /// [0, 1] (0: off).
    pub min_p: f32,
    /// > 0. Logits of tokens in the last `repeat_last_n` are divided by it
    /// when positive and multiplied when not (1: off).
    pub repeat_penalty: f32,
    /// Window of recent tokens (prompt included) the penalties look at (0: off).
    pub repeat_last_n: u32,
    /// Subtracted once from the logit of every token in the window (0: off).
    pub presence_penalty: f32,
    /// Subtracted from a token's logit per occurrence in the window (0: off).
    pub frequency_penalty: f32,
    /// (token id, bias) added to the logits first; -inf bans the token.
    pub logit_bias: Vec<(u32, f32)>,
    /// None: seeded from the clock, so every request differs.
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self::GREEDY
    }
}

impl Sampling {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        repeat_last_n: DEFAULT_REPEAT_LAST_N,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: Vec::new(),
        seed: None,
    };

    /// Picks the most likely token (after bias and penalties).
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }

    /// Repetition penalties change the logits (llama.cpp's `is_disabled`).
    pub fn penalizes(&self) -> bool {
        self.repeat_last_n > 0
            && (self.repeat_penalty != 1.0
                || self.presence_penalty != 0.0
                || self.frequency_penalty != 0.0)
    }

    /// False only for plain greedy decoding, which backends may run on the
    /// device without copying logits to the host.
    pub fn needs_host_logits(&self) -> bool {
        !self.is_greedy() || self.penalizes() || !self.logit_bias.is_empty()
    }

    pub fn validate(&self) -> Result<()> {
        let invalid = |message: &str| Err(Error::Invalid(message.into()));
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return invalid("temperature must be a finite number >= 0");
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return invalid("top_p must be in (0, 1]");
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return invalid("min_p must be in [0, 1]");
        }
        if !self.repeat_penalty.is_finite()
            || self.repeat_penalty <= 0.0
            || !(1.0 / self.repeat_penalty).is_finite()
        {
            return invalid("repeat_penalty must be finite and greater than 0");
        }
        if !self.presence_penalty.is_finite() || !self.frequency_penalty.is_finite() {
            return invalid("presence_penalty and frequency_penalty must be finite");
        }
        if self
            .logit_bias
            .iter()
            .any(|&(_, bias)| bias.is_nan() || bias == f32::INFINITY)
        {
            return invalid("logit_bias values must be finite (or -inf to ban a token)");
        }
        Ok(())
    }

    /// Every logit_bias token is below `vocab_tokens`.
    pub fn check_vocabulary(&self, vocab_tokens: u32) -> Result<()> {
        match self.logit_bias.iter().find(|&&(token, _)| token >= vocab_tokens) {
            Some(&(token, _)) => Err(Error::Invalid(format!(
                "logit_bias token {token} is outside the {vocab_tokens}-token vocabulary"
            ))),
            None => Ok(()),
        }
    }
}

pub struct Sampler {
    settings: Sampling,
    rng: SplitMix64,
    history: History,
    /// Reused per-token buffers.
    scores: Vec<f32>,
    weights: Vec<f32>,
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(settings: Sampling) -> Result<Self> {
        settings.validate()?;
        let seed = settings.seed.unwrap_or_else(clock_seed);
        let window = if settings.penalizes() { settings.repeat_last_n } else { 0 };
        Ok(Self {
            settings,
            rng: SplitMix64(seed),
            history: History::new(window),
            scores: Vec::new(),
            weights: Vec::new(),
            candidates: Vec::new(),
        })
    }

    pub fn is_greedy(&self) -> bool {
        self.settings.is_greedy()
    }

    pub fn needs_host_logits(&self) -> bool {
        self.settings.needs_host_logits()
    }

    pub fn check_vocabulary(&self, vocab_tokens: u32) -> Result<()> {
        self.settings.check_vocabulary(vocab_tokens)
    }

    /// Starts a request: the penalty window is the end of `prompt`.
    pub fn begin(&mut self, prompt: &[i32]) {
        self.history.clear();
        let window = self.settings.repeat_last_n as usize;
        for &token in &prompt[prompt.len().saturating_sub(window)..] {
            self.history.accept(token);
        }
    }

    /// Records a generated token for the penalties.
    pub fn accept(&mut self, token: i32) {
        self.history.accept(token);
    }

    /// Draws the next token from `logits` (one score per vocabulary entry).
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32> {
        check_finite(logits)?;
        if !self.settings.penalizes() && self.settings.logit_bias.is_empty() {
            return self.pick(logits);
        }
        let mut scores = std::mem::take(&mut self.scores);
        scores.clear();
        scores.extend_from_slice(logits);
        penalties::bias(&mut scores, &self.settings.logit_bias);
        if self.settings.penalizes() {
            let s = &self.settings;
            penalties::penalize(
                &mut scores,
                &self.history,
                s.repeat_penalty,
                s.frequency_penalty,
                s.presence_penalty,
            );
        }
        let token = self.pick(&scores);
        self.scores = scores;
        token
    }

    /// top-k, top-p, min-p and temperature over adjusted `scores`.
    fn pick(&mut self, scores: &[f32]) -> Result<i32> {
        let (best, max) = argmax(scores)?;
        if self.is_greedy() {
            return Ok(best as i32);
        }
        let inverse = 1.0 / self.settings.temperature;
        let top_k = self.settings.top_k as usize;
        let top_p = self.settings.top_p;
        let min_p = self.settings.min_p;
        self.weights.clear();
        if top_k == 0 && top_p >= 1.0 && min_p <= 0.0 {
            self.weights
                .extend(scores.iter().map(|&x| ((x - max) * inverse).exp()));
            let total: f64 = self.weights.iter().map(|&w| w as f64).sum();
            return Ok(self.draw_all(total));
        }
        // Truncation works on untempered probabilities; temperature comes last.
        self.weights.extend(scores.iter().map(|&x| (x - max).exp()));
        self.truncate(top_k, top_p, min_p);
        for candidate in &mut self.candidates {
            candidate.1 = ((scores[candidate.0 as usize] - max) * inverse).exp();
        }
        let kept: f64 = self.candidates.iter().map(|c| c.1 as f64).sum();
        let mut target = self.rng.unit() * kept;
        for &(token, weight) in &self.candidates {
            target -= weight as f64;
            if target < 0.0 {
                return Ok(token as i32);
            }
        }
        Ok(self.candidates.last().map_or(best, |c| c.0 as usize) as i32)
    }

    /// Leaves the top-k, then top-p, then min-p survivors in `candidates`
    /// (weights relative to the most likely token, which always survives).
    fn truncate(&mut self, top_k: usize, top_p: f32, min_p: f32) {
        self.candidates.clear();
        let all = self.weights.iter().enumerate().map(|(i, &w)| (i as u32, w));
        if top_k > 0 && top_k < self.weights.len() {
            self.candidates.extend(all);
            self.candidates
                .select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(top_k);
        } else if top_p < 1.0 {
            self.candidates
                .extend(all.filter(|&(_, w)| w >= NUCLEUS_PREFILTER));
        } else {
            // min-p alone: exact, no sort needed.
            self.candidates.extend(all.filter(|&(_, w)| w >= min_p));
            return;
        }
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if top_p < 1.0 {
            // top-p is taken over the top-k set when both are on (llama.cpp order).
            let mass: f64 = if top_k > 0 && top_k < self.weights.len() {
                self.candidates.iter().map(|c| c.1 as f64).sum()
            } else {
                self.weights.iter().map(|&w| w as f64).sum()
            };
            if !self.keep_nucleus(mass * top_p as f64) {
                // The prefiltered set holds less than the nucleus: redo it exactly.
                self.candidates.clear();
                self.candidates
                    .extend(self.weights.iter().enumerate().map(|(i, &w)| (i as u32, w)));
                self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                self.keep_nucleus(mass * top_p as f64);
            }
        }
        if min_p > 0.0 {
            self.candidates.retain(|c| c.1 >= min_p);
        }
    }

    /// Truncates the sorted candidates to the smallest prefix reaching `mass`;
    /// false when all of them together stay below it.
    fn keep_nucleus(&mut self, mass: f64) -> bool {
        let mut sum = 0.0;
        for (index, candidate) in self.candidates.iter().enumerate() {
            sum += candidate.1 as f64;
            if sum >= mass {
                self.candidates.truncate(index + 1);
                return true;
            }
        }
        false
    }

    fn draw_all(&mut self, total: f64) -> i32 {
        let mut target = self.rng.unit() * total;
        for (index, &weight) in self.weights.iter().enumerate() {
            target -= weight as f64;
            if target < 0.0 {
                return index as i32;
            }
        }
        (self.weights.len() - 1) as i32
    }
}

fn check_finite(logits: &[f32]) -> Result<()> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::Backend("nonfinite logit".into()));
    }
    Ok(())
}

/// Most likely token; -inf scores (banned by logit_bias) never win.
fn argmax(scores: &[f32]) -> Result<(usize, f32)> {
    if scores.is_empty() {
        return Err(Error::Backend("empty logits".into()));
    }
    let mut best = None::<(usize, f32)>;
    for (index, &value) in scores.iter().enumerate() {
        if value > f32::NEG_INFINITY && best.is_none_or(|(_, max)| value > max) {
            best = Some((index, value));
        }
    }
    best.ok_or_else(|| Error::Invalid("logit_bias bans every token".into()))
}

fn clock_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    nanos ^ COUNTER.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
}

/// Small, fast, seedable generator (Steele et al., SplitMix64).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests;
