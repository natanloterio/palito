//! One cached prompt whose K/V stays in the device buffers between requests.
//!
//! Nothing is copied: the next request starts evaluating at a restored position
//! and the K/V of the positions before it is the previous prompt's. For the
//! result to be exactly the uncached one, a request may only resume at the start
//! of one of the previous prompt's prefill chunks. The chunks before that start
//! are then the same tokens, at the same positions, in the same batches, as an
//! uncached prefill of the new prompt would evaluate, and the chunks from it on
//! are batched exactly as they would be without the cache. Kernels never see
//! a batch shape the uncached run would not, so no batch invariance is assumed.

#[derive(Debug, Default)]
pub(super) struct PrefixCache {
    /// Tokens whose K/V the device buffers hold at positions `[0, len)`, all
    /// written by prompt prefill.
    tokens: Vec<i32>,
    /// Ascending prompt-chunk start positions, each below `tokens.len()`.
    starts: Vec<usize>,
    /// Between `restore` and `store`: evaluations are prompt chunks.
    recording: bool,
}

impl PrefixCache {
    /// Positions of `prompt` whose K/V can be reused: the latest chunk start at
    /// or before the shared prefix, leaving at least the last token to evaluate
    /// (its logits are not kept). Drops everything after it and starts recording
    /// the new prompt's chunks.
    pub(super) fn restore(&mut self, prompt: &[i32]) -> usize {
        let shared = self
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(cached, token)| cached == token)
            .count()
            .min(prompt.len().saturating_sub(1));
        let restored = self
            .starts
            .iter()
            .rev()
            .copied()
            .find(|&start| start <= shared)
            .unwrap_or(0);
        self.truncate(restored);
        self.recording = true;
        restored
    }

    /// Before evaluating at `position`: its K/V is about to be overwritten.
    pub(super) fn overwrite(&mut self, position: usize) {
        if position < self.tokens.len() {
            self.truncate(position);
        }
    }

    /// After `tokens` were evaluated at `position`.
    pub(super) fn evaluated(&mut self, tokens: &[i32], position: usize) {
        if self.recording && position == self.tokens.len() {
            self.starts.push(position);
            self.tokens.extend_from_slice(tokens);
        }
    }

    /// The prompt is evaluated: stop recording (generated tokens are evaluated
    /// one at a time, unlike an uncached prefill, so they are never reused).
    pub(super) fn store(&mut self, prompt: &[i32]) {
        if !self.recording || self.tokens != prompt {
            self.truncate(0);
        }
        self.recording = false;
    }

    #[cfg(test)]
    pub(super) fn cached_tokens(&self) -> usize {
        self.tokens.len()
    }

    fn truncate(&mut self, length: usize) {
        self.tokens.truncate(length);
        self.starts.retain(|&start| start < length);
        self.recording = false;
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;

    /// Mimics `Executor::prefill`: restore, evaluate the rest in `chunk`-token
    /// pieces, store. Returns (restored, chunk starts evaluated).
    fn prefill(cache: &mut PrefixCache, prompt: &[i32], chunk: usize) -> (usize, Vec<usize>) {
        let restored = cache.restore(prompt);
        let mut starts = Vec::new();
        for (index, piece) in prompt[restored..].chunks(chunk).enumerate() {
            let position = restored + index * chunk;
            cache.overwrite(position);
            starts.push(position);
            cache.evaluated(piece, position);
        }
        cache.store(prompt);
        (restored, starts)
    }

    fn decode(cache: &mut PrefixCache, tokens: &[i32], mut position: usize) {
        for &token in tokens {
            cache.overwrite(position);
            cache.evaluated(&[token], position);
            position += 1;
        }
    }

    fn ids(range: std::ops::Range<i32>) -> Vec<i32> {
        range.collect()
    }

    #[test]
    fn empty_cache_misses() {
        let mut cache = PrefixCache::default();
        assert_eq!(prefill(&mut cache, &ids(0..10), 4), (0, vec![0, 4, 8]));
        assert_eq!(cache.cached_tokens(), 10);
    }

    #[test]
    fn extended_prompt_resumes_at_the_last_chunk_start() {
        let mut cache = PrefixCache::default();
        prefill(&mut cache, &ids(0..10), 4);
        decode(&mut cache, &[50, 51], 10);
        assert_eq!(cache.cached_tokens(), 10, "generated tokens are not cached");
        // The uncached run would evaluate [0,4) [4,8) [8,12) [12,15).
        assert_eq!(prefill(&mut cache, &ids(0..15), 4), (8, vec![8, 12]));
        assert_eq!(cache.cached_tokens(), 15);
        assert_eq!(prefill(&mut cache, &ids(0..30), 4).0, 12);
    }

    #[test]
    fn divergence_resumes_at_the_chunk_start_before_it() {
        let mut cache = PrefixCache::default();
        prefill(&mut cache, &ids(0..20), 4);
        let mut prompt = ids(0..20);
        prompt[9] = 99;
        assert_eq!(prefill(&mut cache, &prompt, 4), (8, vec![8, 12, 16]));
        prompt[2] = 99;
        assert_eq!(prefill(&mut cache, &prompt, 4), (0, vec![0, 4, 8, 12, 16]));
    }

    #[test]
    fn the_last_prompt_token_is_always_evaluated() {
        let mut cache = PrefixCache::default();
        prefill(&mut cache, &ids(0..12), 4);
        // Same prompt: 8 is the latest start below 11 (its last-token index).
        assert_eq!(prefill(&mut cache, &ids(0..12), 4), (8, vec![8]));
        assert_eq!(prefill(&mut cache, &ids(0..9), 4), (8, vec![8]));
        assert_eq!(prefill(&mut cache, &ids(0..8), 4), (4, vec![4]));
        assert_eq!(prefill(&mut cache, &[0], 4), (0, vec![0]));
    }

    #[test]
    fn evaluation_outside_a_prefill_invalidates_what_it_overwrites() {
        let mut cache = PrefixCache::default();
        prefill(&mut cache, &ids(0..12), 4);
        cache.overwrite(5);
        cache.evaluated(&[7], 5);
        assert_eq!(cache.cached_tokens(), 5);
        assert_eq!(prefill(&mut cache, &ids(0..12), 4).0, 4);
    }

    #[test]
    fn failed_prefill_keeps_only_completed_chunks() {
        let mut cache = PrefixCache::default();
        prefill(&mut cache, &ids(0..12), 4);
        let prompt = ids(100..112);
        assert_eq!(cache.restore(&prompt), 0);
        cache.overwrite(0);
        cache.evaluated(&prompt[..4], 0);
        cache.overwrite(4); // fails here; no evaluated(), no store()
        assert_eq!(cache.cached_tokens(), 4);
        assert_eq!(prefill(&mut cache, &prompt, 4), (0, vec![0, 4, 8]));
        assert_eq!(prefill(&mut cache, &prompt, 4).0, 8);
    }

    #[test]
    fn store_of_a_different_prompt_drops_the_cache() {
        let mut cache = PrefixCache::default();
        prefill(&mut cache, &ids(0..12), 4);
        cache.restore(&ids(0..12));
        cache.store(&ids(0..3));
        assert_eq!(cache.cached_tokens(), 0);
    }
}
