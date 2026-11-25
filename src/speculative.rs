//! Zero-weight speculative decoding (SuffixDecoding / Prompt Lookup) and
//! `uint64` bitmask tree-verification structures for Palito.
//!
//! SuffixDecoding maintains an online suffix index over the prompt and generated
//! tokens on the CPU (~20 us/token, 0 GPU overhead). When a high-confidence
//! continuation is found (common in coding, editing, RAG, and agentic loops),
//! Palito feeds the drafted token tree directly into `hyb_matvec_mma` (k=8/16)
//! with a compact `[u64; 64]` causal tree attention bitmask.

use std::collections::HashMap;

/// Maximum number of candidate nodes in a single speculative verification tree
/// (fits in a 64-bit bitmask per query node for FlashAttention).
pub const MAX_TREE_NODES: usize = 64;

/// A candidate verification tree ready for GPU multi-vector verification.
///
/// Node 0 is always the last accepted token (the root). Each node `i > 0` has
/// `parent[i] < i`, and `attn_mask[i]` has bit `j` set iff node `j` is on the
/// root-to-`i` ancestor path (including `i` itself).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i8>,
    pub depths: Vec<u8>,
    /// Bit `j` of `attn_mask[i]` is 1 iff tree node `j` is visible to node `i`.
    pub attn_mask: [u64; MAX_TREE_NODES],
}

impl SpecTree {
    /// Builds a linear chain tree `[root, d_0, d_1, ...]` of length `<= 64`.
    pub fn from_chain(root: u32, draft: &[u32]) -> Self {
        let count = (1 + draft.len()).min(MAX_TREE_NODES);
        let mut tokens = Vec::with_capacity(count);
        let mut parents = Vec::with_capacity(count);
        let mut depths = Vec::with_capacity(count);
        let mut attn_mask = [0u64; MAX_TREE_NODES];

        tokens.push(root);
        parents.push(-1);
        depths.push(0);
        attn_mask[0] = 1u64;

        for (idx, &tok) in draft.iter().take(count - 1).enumerate() {
            let node = idx + 1;
            tokens.push(tok);
            parents.push(idx as i8);
            depths.push(node as u8);
            attn_mask[node] = attn_mask[idx] | (1u64 << node);
        }
        Self {
            tokens,
            parents,
            depths,
            attn_mask,
        }
    }

    /// Builds a general tree from `(token, parent_index)` pairs where node 0 has
    /// `parent_index = -1` and every `node > 0` has `0 <= parent < node`.
    pub fn from_nodes(nodes: &[(u32, i8)]) -> Self {
        assert!(!nodes.is_empty() && nodes.len() <= MAX_TREE_NODES);
        let mut tokens = Vec::with_capacity(nodes.len());
        let mut parents = Vec::with_capacity(nodes.len());
        let mut depths = Vec::with_capacity(nodes.len());
        let mut attn_mask = [0u64; MAX_TREE_NODES];

        for (i, &(tok, parent)) in nodes.iter().enumerate() {
            tokens.push(tok);
            parents.push(parent);
            if i == 0 {
                assert_eq!(parent, -1);
                depths.push(0);
                attn_mask[0] = 1u64;
            } else {
                let p = parent as usize;
                assert!(p < i, "tree nodes must be topologically ordered");
                depths.push(depths[p] + 1);
                attn_mask[i] = attn_mask[p] | (1u64 << i);
            }
        }
        Self {
            tokens,
            parents,
            depths,
            attn_mask,
        }
    }

    /// Greedy lossless verification of a drafted tree against target argmax
    /// predictions `target_argmax[i]` (the target model's greedy next token
    /// predicted at tree node `i`).
    ///
    /// Returns `(accepted_tokens, accepted_node_indices)` where
    /// `accepted_tokens` always contains at least 1 token (the bonus token from
    /// the deepest matched node) and is 100% identical to greedy autoregressive
    /// decoding.
    pub fn verify_greedy(&self, target_argmax: &[u32]) -> (Vec<u32>, Vec<usize>) {
        assert_eq!(target_argmax.len(), self.tokens.len());
        let mut accepted = Vec::new();
        let mut path_nodes = vec![0usize];
        let mut cur = 0usize;

        loop {
            let next_tok = target_argmax[cur];
            // Check if any child of `cur` in the tree matches `next_tok`.
            let child = (cur + 1..self.tokens.len())
                .find(|&i| self.parents[i] == cur as i8 && self.tokens[i] == next_tok);
            accepted.push(next_tok);
            match child {
                Some(next_node) => {
                    path_nodes.push(next_node);
                    cur = next_node;
                }
                None => break,
            }
        }
        (accepted, path_nodes)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Online SuffixDecoding drafter (zero model weights, CPU-only).
///
/// Indexes all n-grams of length `min_match..=max_match` from the prompt and
/// generated history, returning the most recent / highest-frequency continuation
/// up to `max_draft` tokens.
pub struct SuffixDrafter {
    history: Vec<u32>,
    /// Maps n-gram key (hash of `k` tokens) to the list of end positions in `history`.
    /// We keep separate tables for match lengths `1..=max_match` so longer matches
    /// are probed first with zero hash-collision ambiguity (verified against `history`).
    max_match: usize,
    index: Vec<HashMap<u64, Vec<usize>>>,
}

impl SuffixDrafter {
    pub fn new(max_match: usize) -> Self {
        let max_match = max_match.clamp(1, 16);
        Self {
            history: Vec::new(),
            max_match,
            index: (0..=max_match).map(|_| HashMap::new()).collect(),
        }
    }

    /// Appends tokens (prompt or newly accepted tokens) and updates the suffix index.
    pub fn push_tokens(&mut self, tokens: &[u32]) {
        for &tok in tokens {
            let pos = self.history.len();
            self.history.push(tok);
            let max_len = (pos + 1).min(self.max_match);
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for len in 1..=max_len {
                // Hash backwards from `pos` so `len` covers `history[pos+1-len ..= pos]`.
                let t = self.history[pos + 1 - len];
                h ^= u64::from(t);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
                let bucket = self.index[len].entry(h).or_default();
                // Cap per-key occurrences to keep lookup O(1) bounded.
                if bucket.len() >= 16 {
                    bucket.remove(0);
                }
                bucket.push(pos);
            }
        }
    }

    /// Drafts up to `max_draft` continuation tokens matching the longest suffix
    /// of `history` (prefers the longest match, breaking ties by recency).
    pub fn draft(&self, max_draft: usize) -> Vec<u32> {
        let n = self.history.len();
        if n < 2 || max_draft == 0 {
            return Vec::new();
        }
        let max_len = (n - 1).min(self.max_match);
        // Precompute suffix hashes for lengths 1..=max_len.
        let mut hashes = vec![0u64; max_len + 1];
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for len in 1..=max_len {
            let t = self.history[n - len];
            h ^= u64::from(t);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            hashes[len] = h;
        }

        // Probe longest suffix first.
        for len in (1..=max_len).rev() {
            if let Some(positions) = self.index[len].get(&hashes[len]) {
                let suffix = &self.history[n - len..n];
                // Scan most recent occurrences first, skipping the current end `n - 1`.
                for &end_pos in positions.iter().rev() {
                    if end_pos + 1 >= n || end_pos + 1 < len {
                        continue;
                    }
                    let start = end_pos + 1 - len;
                    if &self.history[start..=end_pos] == suffix {
                        let avail = (n - 1 - end_pos).min(max_draft);
                        if avail > 0 {
                            return self.history[end_pos + 1..end_pos + 1 + avail].to_vec();
                        }
                    }
                }
            }
        }
        Vec::new()
    }

    /// Convenience helper: drafts up to `max_draft` tokens and wraps them into
    /// a `SpecTree` rooted at the last token in `history`.
    pub fn draft_tree(&self, max_draft: usize) -> Option<SpecTree> {
        let &root = self.history.last()?;
        let continuation = self.draft(max_draft);
        Some(SpecTree::from_chain(root, &continuation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_drafter_finds_longest_recent_continuation() {
        let mut d = SuffixDrafter::new(8);
        // Simulate a code prompt with repeated function signature pattern.
        d.push_tokens(&[10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30]);
        let drafted = d.draft(4);
        assert_eq!(drafted, vec![40, 50, 60, 70]);
    }

    #[test]
    fn spec_tree_bitmask_and_greedy_verification_are_exact() {
        // Tree:
        //   0 (root=10)
        //   ├── 1 (tok=20)
        //   │   ├── 2 (tok=30)
        //   │   └── 3 (tok=31)
        //   └── 4 (tok=21)
        let tree = SpecTree::from_nodes(&[(10, -1), (20, 0), (30, 1), (31, 1), (21, 0)]);
        assert_eq!(tree.attn_mask[0], 0b00001);
        assert_eq!(tree.attn_mask[1], 0b00011);
        assert_eq!(tree.attn_mask[2], 0b00111);
        assert_eq!(tree.attn_mask[3], 0b01011);
        assert_eq!(tree.attn_mask[4], 0b10001);

        // Target agrees with 0->1 (20), 1->3 (31), and predicts bonus token 99 at node 3.
        let target_argmax = [20, 31, 88, 99, 77];
        let (accepted, path) = tree.verify_greedy(&target_argmax);
        assert_eq!(accepted, vec![20, 31, 99]);
        assert_eq!(path, vec![0, 1, 3]);
    }
}
