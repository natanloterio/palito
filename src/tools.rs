//! OpenAI-style tool calling for templates that use Qwen's `<tool_call>` JSON.

pub(crate) mod call_grammar;
pub mod gate;
pub mod spec;

pub use gate::{GatePhase, TokenRole, ToolGate, ToolTokens};
pub use spec::{ToolDef, ToolMode, ToolSpec};
