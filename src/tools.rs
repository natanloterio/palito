//! OpenAI-style tool calling for templates that use Qwen's `<tool_call>` JSON.

pub(crate) mod call_grammar;
pub mod spec;

pub use spec::{ToolDef, ToolMode, ToolSpec};
