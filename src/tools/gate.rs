//! Decides, token by token, when a reply is free text and when it is a call.
//!
//! The decision uses token IDs only: text that merely looks like `<tool_call>`
//! never starts a call, and `<tool_call>` inside the reasoning is reasoning.

use super::ToolMode;
use crate::{
    Error, Result,
    grammar::{Constraint, LogitConstraint, MASKED_LOGIT},
};

/// Whitespace tokens allowed in a row where only a call or the end may follow.
/// Without a limit, a model can emit newlines until the budget runs out.
const MAX_GAP_TOKENS: u8 = 2;

/// The vocabulary entries the gate needs.
#[derive(Clone, Debug)]
pub struct ToolTokens {
    pub call_open: i32,
    pub call_close: i32,
    pub think_open: Option<i32>,
    pub think_close: Option<i32>,
    pub eog: Vec<i32>,
    /// Per token: its text is only whitespace.
    pub whitespace: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatePhase {
    Free,
    Reasoning,
    Call,
    AfterCall,
}

/// What the last sampled token is in the reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenRole {
    Text,
    CallOpen,
    CallBody,
    CallClose,
}

pub struct ToolGate {
    tokens: ToolTokens,
    /// Unused grammar state, copied for every call. None only when forbidden.
    grammar: Option<Constraint>,
    active: Option<Constraint>,
    mode: ToolMode,
    parallel: bool,
    phase: GatePhase,
    role: TokenRole,
    started: bool,
    /// Only whitespace so far, so `<think>` can still open the reasoning.
    leading: bool,
    /// Whitespace tokens since the last other token.
    gap: u8,
    masked: Vec<f32>,
}

impl ToolGate {
    /// `call` is the grammar of one call (None only with `ToolMode::Forbidden`).
    pub fn new(
        tokens: ToolTokens,
        call: Option<Constraint>,
        mode: ToolMode,
        parallel: bool,
        prompt_opens_reasoning: bool,
    ) -> Self {
        Self {
            tokens,
            grammar: call,
            active: None,
            mode,
            parallel,
            phase: if prompt_opens_reasoning { GatePhase::Reasoning } else { GatePhase::Free },
            role: TokenRole::Text,
            started: false,
            leading: !prompt_opens_reasoning,
            gap: 0,
            masked: Vec::new(),
        }
    }

    pub fn phase(&self) -> GatePhase {
        self.phase
    }

    /// Whether a call has started in this reply.
    pub fn call_started(&self) -> bool {
        self.started
    }

    /// The role of the last sampled token.
    pub fn last_role(&self) -> TokenRole {
        self.role
    }

    fn is_space(&self, token: i32) -> bool {
        usize::try_from(token)
            .ok()
            .and_then(|index| self.tokens.whitespace.get(index))
            .copied()
            .unwrap_or(false)
    }

    fn gap_allows(&self, token: i32) -> bool {
        self.gap < MAX_GAP_TOKENS && self.is_space(token)
    }

    /// Whether `token` may come next, outside a call.
    fn allows(&self, token: i32) -> bool {
        let tokens = &self.tokens;
        if self.mode == ToolMode::Forbidden && token == tokens.call_open {
            return false;
        }
        let must_call = matches!(self.mode, ToolMode::Required | ToolMode::Named(_)) && !self.started;
        match self.phase {
            GatePhase::Reasoning => !(must_call && tokens.eog.contains(&token)),
            GatePhase::Free if must_call => {
                token == tokens.call_open
                    || self.gap_allows(token)
                    || (self.leading && Some(token) == tokens.think_open)
            }
            GatePhase::Free => true,
            GatePhase::AfterCall => {
                self.gap_allows(token)
                    || tokens.eog.contains(&token)
                    || (self.parallel && token == tokens.call_open)
            }
            GatePhase::Call => false,
        }
    }

    fn sample_outside_call(
        &mut self,
        logits: &[f32],
        pick: &mut dyn FnMut(&[f32]) -> Result<i32>,
    ) -> Result<i32> {
        let free = self.mode == ToolMode::Auto
            && matches!(self.phase, GatePhase::Free | GatePhase::Reasoning);
        if free {
            return pick(logits);
        }
        let mut masked = std::mem::take(&mut self.masked);
        masked.clear();
        masked.extend(logits.iter().enumerate().map(|(index, &logit)| {
            if self.allows(index as i32) { logit } else { MASKED_LOGIT }
        }));
        let picked = pick(&masked);
        let token = picked.and_then(|token| {
            if self.allows(token) && (token as usize) < masked.len() {
                Ok(token)
            } else {
                best_allowed(&masked)
            }
        });
        self.masked = masked;
        token
    }

    fn advance(&mut self, token: i32) -> Result<()> {
        self.role = TokenRole::Text;
        self.gap = if self.is_space(token) { self.gap.saturating_add(1) } else { 0 };
        match self.phase {
            GatePhase::Reasoning => {
                if Some(token) == self.tokens.think_close {
                    self.phase = GatePhase::Free;
                }
            }
            GatePhase::Free | GatePhase::AfterCall => {
                if token == self.tokens.call_open && self.mode != ToolMode::Forbidden {
                    self.open_call()?;
                } else if self.phase == GatePhase::Free
                    && self.leading
                    && Some(token) == self.tokens.think_open
                {
                    self.phase = GatePhase::Reasoning;
                }
                self.leading = self.leading && self.is_space(token);
            }
            GatePhase::Call => {}
        }
        Ok(())
    }

    fn open_call(&mut self) -> Result<()> {
        let grammar = self
            .grammar
            .as_ref()
            .ok_or_else(|| Error::Backend("tool call without a grammar".into()))?;
        self.active = Some(grammar.fresh());
        self.phase = GatePhase::Call;
        self.role = TokenRole::CallOpen;
        self.started = true;
        Ok(())
    }

    fn sample_call(
        &mut self,
        logits: &[f32],
        pick: &mut dyn FnMut(&[f32]) -> Result<i32>,
    ) -> Result<i32> {
        let close = self.tokens.call_close;
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| Error::Backend("tool call state missing".into()))?;
        let (token, closed) = active.sample_call(logits, close, pick)?;
        if closed {
            self.active = None;
            self.gap = 0;
            self.phase = GatePhase::AfterCall;
            self.role = TokenRole::CallClose;
        } else {
            self.role = TokenRole::CallBody;
        }
        Ok(token)
    }
}

impl LogitConstraint for ToolGate {
    fn sample(
        &mut self,
        logits: &[f32],
        pick: &mut dyn FnMut(&[f32]) -> Result<i32>,
    ) -> Result<i32> {
        if self.phase == GatePhase::Call {
            return self.sample_call(logits, pick);
        }
        let token = self.sample_outside_call(logits, pick)?;
        self.advance(token)?;
        Ok(token)
    }
}

fn best_allowed(masked: &[f32]) -> Result<i32> {
    masked
        .iter()
        .enumerate()
        .filter(|&(_, &logit)| logit != MASKED_LOGIT)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as i32)
        .ok_or_else(|| Error::Backend("tool gate allows no token".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammar::JsonDecoder,
        tools::{ToolDef, call_grammar::call_grammar},
    };
    use serde_json::json;

    const CALL_OPEN: i32 = 256;
    const CALL_CLOSE: i32 = 257;
    const THINK_OPEN: i32 = 258;
    const THINK_CLOSE: i32 = 259;
    const WORD: i32 = 260;
    const EOG: i32 = 261;
    const VOCAB: usize = 262;
    const NEWLINE: i32 = b'\n' as i32;
    const CALL: &str = "{\"name\": \"a\", \"arguments\": {}}";

    fn bytes(id: usize) -> Option<&'static [u8]> {
        static SINGLE: [u8; 256] = {
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
            256 => Some(b"<tool_call>"),
            257 => Some(b"</tool_call>"),
            258 => Some(b"<think>"),
            259 => Some(b"</think>"),
            260 => Some(b"hello"),
            _ => None,
        }
    }

    fn tokens() -> ToolTokens {
        let whitespace = (0..VOCAB)
            .map(|id| bytes(id).is_some_and(|b| b.iter().all(u8::is_ascii_whitespace)))
            .collect();
        ToolTokens {
            call_open: CALL_OPEN,
            call_close: CALL_CLOSE,
            think_open: Some(THINK_OPEN),
            think_close: Some(THINK_CLOSE),
            eog: vec![EOG],
            whitespace,
        }
    }

    fn build(mode: ToolMode, parallel: bool, opened: bool) -> ToolGate {
        let decoder = JsonDecoder::new(VOCAB, bytes, |id| id == EOG).unwrap();
        let defs = [ToolDef { name: "a".into(), parameters: json!({"type": "object"}) }];
        let call = (mode != ToolMode::Forbidden)
            .then(|| decoder.lark_constraint(&call_grammar(&defs, &mode)).unwrap());
        ToolGate::new(tokens(), call, mode, parallel, opened)
    }

    fn gate(mode: ToolMode, parallel: bool) -> ToolGate {
        build(mode, parallel, false)
    }

    fn allowed(gate: &ToolGate) -> Vec<i32> {
        (0..VOCAB as i32).filter(|&token| gate.allows(token)).collect()
    }

    fn feed(gate: &mut ToolGate, token: i32) -> i32 {
        gate.sample(&[0.0; VOCAB], &mut |_| Ok(token)).unwrap()
    }

    fn feed_text(gate: &mut ToolGate, text: &str) {
        for byte in text.bytes() {
            assert_eq!(feed(gate, i32::from(byte)), i32::from(byte));
        }
    }

    fn gate_after_one_call(mode: ToolMode, parallel: bool) -> ToolGate {
        let mut gate = gate(mode, parallel);
        feed(&mut gate, CALL_OPEN);
        feed_text(&mut gate, CALL);
        feed(&mut gate, CALL_CLOSE);
        gate
    }

    #[test]
    fn auto_is_free_until_the_call_token_and_reasoning_ignores_it() {
        let mut gate = gate(ToolMode::Auto, true);
        assert!(allowed(&gate).contains(&WORD));
        feed(&mut gate, THINK_OPEN);
        assert_eq!(gate.phase(), GatePhase::Reasoning);
        feed(&mut gate, CALL_OPEN);
        assert_eq!(gate.phase(), GatePhase::Reasoning);
        assert_eq!(gate.last_role(), TokenRole::Text);
        feed(&mut gate, THINK_CLOSE);
        feed(&mut gate, CALL_OPEN);
        assert_eq!(gate.phase(), GatePhase::Call);
        assert_eq!(gate.last_role(), TokenRole::CallOpen);
        assert!(gate.call_started());
    }

    #[test]
    fn think_token_after_text_is_plain_text() {
        let mut gate = gate(ToolMode::Auto, true);
        feed(&mut gate, WORD);
        feed(&mut gate, THINK_OPEN);
        assert_eq!(gate.phase(), GatePhase::Free);
    }

    #[test]
    fn after_a_call_only_space_another_call_or_the_end() {
        let gate = gate_after_one_call(ToolMode::Auto, true);
        assert_eq!(gate.phase(), GatePhase::AfterCall);
        assert_eq!(gate.last_role(), TokenRole::CallClose);
        let next = allowed(&gate);
        assert!(next.contains(&CALL_OPEN) && next.contains(&EOG) && next.contains(&NEWLINE));
        assert!(!next.contains(&WORD));
    }

    #[test]
    fn without_parallel_a_second_call_is_forbidden() {
        let gate = gate_after_one_call(ToolMode::Auto, false);
        assert!(!allowed(&gate).contains(&CALL_OPEN));
    }

    #[test]
    fn a_second_parallel_call_is_constrained_from_a_fresh_grammar() {
        let mut gate = gate_after_one_call(ToolMode::Auto, true);
        feed(&mut gate, NEWLINE);
        feed(&mut gate, CALL_OPEN);
        feed_text(&mut gate, CALL);
        feed(&mut gate, CALL_CLOSE);
        assert_eq!(gate.phase(), GatePhase::AfterCall);
    }

    #[test]
    fn required_forbids_the_end_and_text_before_the_call() {
        let mut gate = gate(ToolMode::Required, true);
        let first = allowed(&gate);
        assert!(first.contains(&THINK_OPEN) && first.contains(&CALL_OPEN));
        assert!(!first.contains(&EOG) && !first.contains(&WORD));
        feed(&mut gate, THINK_OPEN);
        let inside = allowed(&gate);
        assert!(inside.contains(&WORD) && !inside.contains(&EOG));
        feed(&mut gate, THINK_CLOSE);
        let after = allowed(&gate);
        assert!(after.contains(&CALL_OPEN) && !after.contains(&EOG) && !after.contains(&WORD));
        assert!(!after.contains(&THINK_OPEN));
    }

    #[test]
    fn required_cannot_loop_on_whitespace_before_the_call() {
        let mut gate = gate(ToolMode::Required, true);
        feed(&mut gate, THINK_OPEN);
        feed(&mut gate, WORD);
        feed(&mut gate, THINK_CLOSE);
        feed(&mut gate, NEWLINE);
        assert!(allowed(&gate).contains(&NEWLINE));
        feed(&mut gate, NEWLINE);
        assert_eq!(allowed(&gate), vec![CALL_OPEN]);
    }

    #[test]
    fn whitespace_after_a_call_is_limited() {
        let mut gate = gate_after_one_call(ToolMode::Auto, true);
        feed(&mut gate, NEWLINE);
        feed(&mut gate, NEWLINE);
        assert_eq!(allowed(&gate), vec![CALL_OPEN, EOG]);
    }

    #[test]
    fn a_forbidden_pick_is_replaced_by_an_allowed_token() {
        let mut gate = gate(ToolMode::Required, true);
        let mut logits = [0.0; VOCAB];
        logits[CALL_OPEN as usize] = 5.0;
        let token = gate.sample(&logits, &mut |_| Ok(WORD)).unwrap();
        assert_eq!(token, CALL_OPEN);
    }

    #[test]
    fn pick_only_sees_allowed_logits() {
        let mut gate = gate(ToolMode::Required, true);
        let mut logits = [1.0; VOCAB];
        logits[WORD as usize] = 100.0;
        let mut seen = 0.0;
        gate.sample(&logits, &mut |masked| {
            seen = masked[WORD as usize];
            Ok(CALL_OPEN)
        })
        .unwrap();
        assert!(seen < -1e30);
    }

    #[test]
    fn forbidden_only_masks_the_call_token() {
        let mut gate = gate(ToolMode::Forbidden, true);
        let free = allowed(&gate);
        assert!(!free.contains(&CALL_OPEN) && free.contains(&WORD) && free.contains(&EOG));
        feed(&mut gate, THINK_OPEN);
        assert!(!allowed(&gate).contains(&CALL_OPEN));
    }

    #[test]
    fn a_prompt_that_opens_reasoning_starts_in_reasoning() {
        assert_eq!(build(ToolMode::Auto, true, true).phase(), GatePhase::Reasoning);
    }

    #[test]
    fn close_token_ends_a_call_only_when_the_json_is_complete() {
        let mut gate = gate(ToolMode::Auto, true);
        feed(&mut gate, CALL_OPEN);
        feed_text(&mut gate, "{\"name\": \"a\", \"arguments\": {\"x\": \"");
        assert_eq!(feed(&mut gate, CALL_CLOSE), CALL_CLOSE);
        assert_eq!(gate.phase(), GatePhase::Call);
        assert_eq!(gate.last_role(), TokenRole::CallBody);
        feed_text(&mut gate, "\"}}");
        feed(&mut gate, NEWLINE);
        assert_eq!(gate.phase(), GatePhase::Call);
        assert_eq!(feed(&mut gate, CALL_CLOSE), CALL_CLOSE);
        assert_eq!(gate.phase(), GatePhase::AfterCall);
    }

    #[test]
    fn the_end_of_generation_is_masked_while_a_call_is_complete() {
        let mut gate = gate(ToolMode::Auto, true);
        feed(&mut gate, CALL_OPEN);
        feed_text(&mut gate, CALL);
        let mut logits = [0.0; VOCAB];
        logits[CALL_CLOSE as usize] = 1.0;
        assert_eq!(gate.sample(&logits, &mut |_| Ok(EOG)).unwrap(), CALL_CLOSE);
    }
}
