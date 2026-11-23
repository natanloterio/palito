//! The Lark grammar of one tool call: the JSON inside `<tool_call>`.
//!
//! The name and the `arguments` key are literals, so their order is fixed and
//! only declared names are possible. Arguments are a nested `%json` schema.

use super::{ToolDef, ToolMode};
use crate::grammar::json_compile_options;
use serde_json::Value;

const OPTIONS_KEYWORD: &str = "x-guidance";

/// Grammar for `{"name": <tool>, "arguments": <schema>}` with up to two
/// spaces or newlines around it (Qwen writes one newline on each side).
pub(crate) fn call_grammar(defs: &[ToolDef], mode: &ToolMode) -> String {
    let alternatives: Vec<String> = defs
        .iter()
        .filter(|def| !matches!(mode, ToolMode::Named(name) if *name != def.name))
        .map(alternative)
        .collect();
    format!(
        "start: SPACE? call SPACE?\ncall: {}\nSPACE: /[ \\n]{{1,2}}/\n",
        alternatives.join("\n    | ")
    )
}

fn alternative(def: &ToolDef) -> String {
    let name = Value::String(def.name.clone()).to_string();
    let open = format!("{{\"name\": {name}, \"arguments\": ");
    let mut schema = def.parameters.clone();
    if let Some(object) = schema.as_object_mut() {
        object.insert(OPTIONS_KEYWORD.into(), json_compile_options());
    }
    format!("{} %json {schema} \"}}\"", Value::String(open))
}
