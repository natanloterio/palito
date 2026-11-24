//! Logit bias and repetition penalties, applied to a copy of the logits before
//! any truncation (llama.cpp's `logit_bias` and `penalties` samplers).

use std::collections::{HashMap, VecDeque};

/// The last `window` tokens seen (prompt, then generated) with their counts.
pub(super) struct History {
    window: usize,
    recent: VecDeque<i32>,
    counts: HashMap<i32, u32>,
}

impl History {
    pub(super) fn new(window: u32) -> Self {
        Self {
            window: window as usize,
            recent: VecDeque::new(),
            counts: HashMap::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.recent.clear();
        self.counts.clear();
    }

    pub(super) fn accept(&mut self, token: i32) {
        if self.window == 0 {
            return;
        }
        self.recent.push_back(token);
        *self.counts.entry(token).or_insert(0) += 1;
        if self.recent.len() > self.window {
            let Some(old) = self.recent.pop_front() else {
                return;
            };
            if let Some(count) = self.counts.get_mut(&old) {
                *count -= 1;
                if *count == 0 {
                    self.counts.remove(&old);
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn count(&self, token: i32) -> u32 {
        self.counts.get(&token).copied().unwrap_or(0)
    }
}

/// Adds each bias to its token's logit (-inf bans the token).
pub(super) fn bias(scores: &mut [f32], biases: &[(u32, f32)]) {
    for &(token, bias) in biases {
        if let Some(score) = scores.get_mut(token as usize) {
            *score += bias;
        }
    }
}

/// llama.cpp's penalty for every token of the window: a repeat penalty that
/// pushes the logit towards "less likely" whatever its sign, then OpenAI's
/// `count * frequency + presence`.
pub(super) fn penalize(
    scores: &mut [f32],
    history: &History,
    repeat: f32,
    frequency: f32,
    presence: f32,
) {
    for (&token, &count) in &history.counts {
        let Some(score) = usize::try_from(token).ok().and_then(|t| scores.get_mut(t)) else {
            continue;
        };
        if *score <= 0.0 {
            *score *= repeat;
        } else {
            *score /= repeat;
        }
        *score -= count as f32 * frequency + presence;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_keeps_the_last_window_tokens() {
        let mut history = History::new(3);
        for token in [5, 5, 7, 5, 9] {
            history.accept(token);
        }
        // Window: 7, 5, 9.
        assert_eq!((history.count(5), history.count(7), history.count(9)), (1, 1, 1));
        history.accept(9);
        assert_eq!((history.count(7), history.count(9)), (0, 2));
        let mut off = History::new(0);
        off.accept(1);
        assert_eq!(off.count(1), 0);
        history.clear();
        assert_eq!(history.count(9), 0);
    }

    #[test]
    fn penalty_math_matches_llama_cpp() {
        let mut history = History::new(64);
        for token in [0, 1, 1, 3] {
            history.accept(token);
        }
        let mut scores = [2.0, -1.0, 4.0, 0.0, 1.0];
        penalize(&mut scores, &history, 2.0, 0.5, 0.25);
        // 2/2 - (1*0.5 + 0.25); -1*2 - (2*0.5 + 0.25); untouched; 0*2 - 0.75; untouched.
        assert_eq!(scores, [0.25, -3.25, 4.0, -0.75, 1.0]);
        let mut scores = [1.0, 1.0, 1.0];
        bias(&mut scores, &[(0, 2.5), (2, f32::NEG_INFINITY), (9, 1.0)]);
        assert_eq!(scores, [3.5, 1.0, f32::NEG_INFINITY]);
    }
}
