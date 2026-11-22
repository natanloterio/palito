//! Token sampling from host logits: temperature, top-k and top-p (nucleus).
//!
//! `temperature == 0` (the default) or `top_k == 1` is greedy and lets backends
//! keep their token-only fast path (the CUDA decode graph). Any other setting
//! needs the full logits on the host for every generated token.

use crate::{Error, Result};
use std::time::{SystemTime, UNIX_EPOCH};

/// Candidates whose probability is below this fraction of the most likely
/// token's are skipped by the first nucleus pass (an exact full pass follows
/// whenever they could matter).
const NUCLEUS_PREFILTER: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// 0: greedy. Logits are divided by this before the softmax.
    pub temperature: f32,
    /// Keep the smallest set of most likely tokens whose probability reaches
    /// this mass (1: off).
    pub top_p: f32,
    /// Keep only the k most likely tokens (0: off).
    pub top_k: u32,
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
        seed: None,
    };

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::Invalid(
                "temperature must be a finite number >= 0".into(),
            ));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(Error::Invalid("top_p must be in (0, 1]".into()));
        }
        Ok(())
    }
}

pub struct Sampler {
    settings: Sampling,
    rng: SplitMix64,
    /// Reused per-token buffers.
    weights: Vec<f32>,
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(settings: Sampling) -> Result<Self> {
        settings.validate()?;
        let seed = settings.seed.unwrap_or_else(clock_seed);
        Ok(Self {
            settings,
            rng: SplitMix64(seed),
            weights: Vec::new(),
            candidates: Vec::new(),
        })
    }

    pub fn is_greedy(&self) -> bool {
        self.settings.is_greedy()
    }

    /// Draws the next token from `logits` (one score per vocabulary entry).
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32> {
        let (best, max) = argmax(logits)?;
        if self.is_greedy() {
            return Ok(best as i32);
        }
        let inverse = 1.0 / self.settings.temperature;
        self.weights.clear();
        self.weights
            .extend(logits.iter().map(|&x| ((x - max) * inverse).exp()));
        let total: f64 = self.weights.iter().map(|&w| w as f64).sum();
        let top_k = self.settings.top_k as usize;
        let top_p = self.settings.top_p;
        if top_k == 0 && top_p >= 1.0 {
            return Ok(self.draw_all(total));
        }
        self.candidates.clear();
        if top_k > 0 && top_k < self.weights.len() {
            self.candidates
                .extend(self.weights.iter().enumerate().map(|(i, &w)| (i as u32, w)));
            self.candidates
                .select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(top_k);
        } else {
            self.candidates.extend(
                self.weights
                    .iter()
                    .enumerate()
                    .filter(|&(_, &w)| w >= NUCLEUS_PREFILTER)
                    .map(|(i, &w)| (i as u32, w)),
            );
        }
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        // top-p is taken over the top-k set when both are on (llama.cpp order).
        let mass: f64 = if top_k > 0 {
            self.candidates.iter().map(|c| c.1 as f64).sum()
        } else {
            total
        };
        if top_p < 1.0 && !self.keep_nucleus(mass * top_p as f64) {
            // The prefiltered set holds less than the nucleus: redo it exactly.
            self.candidates.clear();
            self.candidates
                .extend(self.weights.iter().enumerate().map(|(i, &w)| (i as u32, w)));
            self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            self.keep_nucleus(mass * top_p as f64);
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

fn argmax(logits: &[f32]) -> Result<(usize, f32)> {
    let mut best = None::<(usize, f32)>;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::Backend("nonfinite logit".into()));
        }
        if best.is_none_or(|(_, max)| value > max) {
            best = Some((index, value));
        }
    }
    best.ok_or_else(|| Error::Backend("empty logits".into()))
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
mod tests {
    use super::*;

    fn sampler(temperature: f32, top_p: f32, top_k: u32, seed: u64) -> Sampler {
        Sampler::new(Sampling {
            temperature,
            top_p,
            top_k,
            seed: Some(seed),
        })
        .unwrap()
    }

    fn counts(sampler: &mut Sampler, logits: &[f32], draws: usize) -> Vec<usize> {
        let mut counts = vec![0; logits.len()];
        for _ in 0..draws {
            counts[sampler.sample(logits).unwrap() as usize] += 1;
        }
        counts
    }

    #[test]
    fn greedy_settings_pick_the_argmax() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        for mut s in [
            sampler(0.0, 1.0, 0, 1),
            sampler(1.0, 1.0, 1, 1),
            sampler(1.0, 1e-6, 0, 1),
        ] {
            for _ in 0..50 {
                assert_eq!(s.sample(&logits).unwrap(), 1);
            }
        }
    }

    #[test]
    fn same_seed_same_tokens_and_temperature_follows_softmax() {
        let logits: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.3).collect();
        let a: Vec<_> = (0..40).map(|_| 0).scan(sampler(0.8, 1.0, 0, 7), |s, _| s.sample(&logits).ok()).collect();
        let b: Vec<_> = (0..40).map(|_| 0).scan(sampler(0.8, 1.0, 0, 7), |s, _| s.sample(&logits).ok()).collect();
        assert_eq!(a, b);
        // ln 3 apart at temperature 1: the larger is chosen ~3x as often.
        let two = [0.0, 3f32.ln()];
        let c = counts(&mut sampler(1.0, 1.0, 0, 3), &two, 40_000);
        let ratio = c[1] as f64 / c[0] as f64;
        assert!((ratio - 3.0).abs() < 0.15, "ratio {ratio}");
    }

    #[test]
    fn top_k_and_top_p_drop_the_tail() {
        let logits = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let c = counts(&mut sampler(1.0, 1.0, 2, 5), &logits, 5_000);
        assert!(c[0] > 0 && c[1] > 0 && c[2..].iter().all(|&n| n == 0));
        // Probabilities ~0.636, 0.234, 0.086...: top_p 0.8 keeps two tokens.
        let c = counts(&mut sampler(1.0, 0.8, 0, 9), &logits, 5_000);
        assert!(c[0] > 0 && c[1] > 0 && c[2..].iter().all(|&n| n == 0), "{c:?}");
    }

    #[test]
    fn nucleus_falls_back_to_an_exact_pass_for_flat_distributions() {
        // One dominant token and a huge flat tail below the prefilter.
        // exp(-14) = 8.3e-7 per tail token: 500k of them hold ~29% of the mass.
        let mut logits = vec![-14.0f32; 500_000];
        logits[0] = 0.0;
        let mut s = sampler(1.0, 0.99, 0, 11);
        let c = counts(&mut s, &logits, 300);
        let tail: usize = c[1..].iter().sum();
        assert!((40..140).contains(&tail), "tail drawn {tail} of 300 times");
    }

    #[test]
    fn rejects_bad_settings_and_nonfinite_logits() {
        for (t, p) in [(-0.1, 1.0), (f32::NAN, 1.0), (1.0, 0.0), (1.0, 1.5)] {
            assert!(Sampler::new(Sampling { temperature: t, top_p: p, top_k: 0, seed: None }).is_err());
        }
        assert!(sampler(1.0, 1.0, 0, 1).sample(&[0.0, f32::INFINITY]).is_err());
    }
}
