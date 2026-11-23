//! Tool definitions from a request, and the tolerant schema policy.
//!
//! Unlike `response_format`, a tool schema never fails a request because of a
//! keyword the grammar cannot enforce: the keyword is dropped with a warning.

use crate::grammar::JsonSchema;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashSet};

/// JSON Schema keywords the grammar does not enforce.
const UNSUPPORTED_KEYWORDS: [&str; 14] = [
    "uniqueItems",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "minContains",
    "maxContains",
    "dependentRequired",
    "dependentSchemas",
    "propertyNames",
    "unevaluatedProperties",
    "unevaluatedItems",
    "multipleOf",
];
/// Keywords whose value maps names to schemas.
const SCHEMA_MAPS: [&str; 4] = ["properties", "$defs", "definitions", "patternProperties"];
/// Keywords whose value is a schema or a list of schemas.
const SCHEMA_VALUES: [&str; 6] = ["items", "additionalProperties", "prefixItems", "anyOf", "oneOf", "allOf"];
/// Keywords that already say what a schema accepts, so `type` is not implied.
const SHAPE_KEYWORDS: [&str; 7] = ["type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolMode {
    /// The model decides whether to call a tool.
    Auto,
    /// The reply must contain at least one call.
    Required,
    /// The reply must call this tool.
    Named(String),
    /// The model must not call any tool.
    Forbidden,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: String,
    /// Schema of the arguments, cleaned by the tolerant policy.
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    /// Definitions as the client sent them, for the chat template.
    pub raw: Vec<Value>,
    pub defs: Vec<ToolDef>,
    pub mode: ToolMode,
    pub parallel: bool,
    /// Schema parts that are not enforced, for the server log.
    pub warnings: Vec<String>,
}

impl ToolSpec {
    /// Validates `tools`, `tool_choice` and `parallel_tool_calls`.
    pub fn parse(
        tools: &[Value],
        choice: Option<&Value>,
        parallel: Option<bool>,
    ) -> Result<Self, String> {
        let mut defs = Vec::with_capacity(tools.len());
        let mut warnings = Vec::new();
        let mut names = HashSet::new();
        for tool in tools {
            let (name, parameters) = definition(tool)?;
            if !names.insert(name.to_owned()) {
                return Err(format!("duplicate tool name {name:?}"));
            }
            let (parameters, notes) = tolerant_schema(name, &parameters);
            warnings.extend(notes);
            defs.push(ToolDef { name: name.to_owned(), parameters });
        }
        let mode = tool_mode(choice, &names)?;
        Ok(Self {
            raw: tools.to_vec(),
            defs,
            mode,
            parallel: parallel.unwrap_or(true),
            warnings,
        })
    }

    /// For a history with tool turns and no tools in the request.
    pub fn forbidden() -> Self {
        Self {
            raw: Vec::new(),
            defs: Vec::new(),
            mode: ToolMode::Forbidden,
            parallel: false,
            warnings: Vec::new(),
        }
    }

    /// The definitions the chat template should show the model.
    pub fn template_tools(&self) -> Option<&[Value]> {
        (self.mode != ToolMode::Forbidden && !self.raw.is_empty()).then_some(self.raw.as_slice())
    }
}

/// (name, parameters with an object shape) of one `tools` entry.
fn definition(tool: &Value) -> Result<(&str, Value), String> {
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return Err("tools entries must have type \"function\"".into());
    }
    let function = tool.get("function").ok_or("tools entry without \"function\"")?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or("tool function needs a non-empty string \"name\"")?;
    let parameters = match function.get("parameters") {
        None | Some(Value::Null) => json!({"type": "object"}),
        Some(Value::Object(schema)) => {
            let mut schema = schema.clone();
            if !SHAPE_KEYWORDS.iter().any(|keyword| schema.contains_key(*keyword)) {
                schema.insert("type".into(), json!("object"));
            }
            Value::Object(schema)
        }
        Some(_) => return Err(format!("parameters of tool {name:?} must be an object")),
    };
    Ok((name, parameters))
}

fn tool_mode(choice: Option<&Value>, names: &HashSet<String>) -> Result<ToolMode, String> {
    let mode = match choice {
        None | Some(Value::Null) => ToolMode::Auto,
        Some(Value::String(text)) => match text.as_str() {
            "auto" => ToolMode::Auto,
            "none" => ToolMode::Forbidden,
            "required" => ToolMode::Required,
            other => return Err(format!("unsupported tool_choice {other:?}")),
        },
        Some(value) => {
            let name = value
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .filter(|_| value.get("type").and_then(Value::as_str) == Some("function"))
                .ok_or("tool_choice object needs type \"function\" and function.name")?;
            if !names.contains(name) {
                return Err(format!("tool_choice names undeclared tool {name:?}"));
            }
            ToolMode::Named(name.to_owned())
        }
    };
    if names.is_empty() && mode == ToolMode::Required {
        return Err("tool_choice \"required\" needs at least one tool".into());
    }
    Ok(mode)
}

/// `schema` without keywords the grammar cannot enforce; any object if it
/// still does not compile. The second value holds warnings for the log.
pub(crate) fn tolerant_schema(tool: &str, schema: &Value) -> (Value, Vec<String>) {
    let mut clean = schema.clone();
    let mut removed = BTreeSet::new();
    strip_keywords(&mut clean, &mut removed);
    let mut warnings = Vec::new();
    if !removed.is_empty() {
        let keywords: Vec<_> = removed.into_iter().collect();
        warnings.push(format!(
            "tool {tool:?}: schema keywords not enforced: {}",
            keywords.join(", ")
        ));
    }
    if let Err(error) = JsonSchema::new(clean.clone()) {
        warnings.push(format!(
            "tool {tool:?}: schema not enforced, accepting any object: {error}"
        ));
        clean = json!({"type": "object"});
    }
    (clean, warnings)
}

/// Visits schema positions only, so a property named `not` is kept.
fn strip_keywords(schema: &mut Value, removed: &mut BTreeSet<String>) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    object.retain(|key, _| {
        let unsupported = UNSUPPORTED_KEYWORDS.contains(&key.as_str());
        if unsupported {
            removed.insert(key.clone());
        }
        !unsupported
    });
    for keyword in SCHEMA_MAPS {
        if let Some(Value::Object(entries)) = object.get_mut(keyword) {
            entries.values_mut().for_each(|entry| strip_keywords(entry, removed));
        }
    }
    for keyword in SCHEMA_VALUES {
        match object.get_mut(keyword) {
            Some(Value::Array(list)) => list.iter_mut().for_each(|entry| strip_keywords(entry, removed)),
            Some(entry) => strip_keywords(entry, removed),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, parameters: Value) -> Value {
        json!({"type": "function", "function": {"name": name, "parameters": parameters}})
    }

    #[test]
    fn parses_every_tool_choice() {
        let tools = [tool("a", json!({"type": "object"})), tool("b", json!({"type": "object"}))];
        assert_eq!(ToolSpec::parse(&tools, None, None).unwrap().mode, ToolMode::Auto);
        assert_eq!(
            ToolSpec::parse(&tools, Some(&json!("auto")), None).unwrap().mode,
            ToolMode::Auto
        );
        assert_eq!(
            ToolSpec::parse(&tools, Some(&json!("required")), None).unwrap().mode,
            ToolMode::Required
        );
        let none = ToolSpec::parse(&tools, Some(&json!("none")), None).unwrap();
        assert_eq!(none.mode, ToolMode::Forbidden);
        assert!(none.template_tools().is_none());
        let named = json!({"type": "function", "function": {"name": "b"}});
        assert_eq!(
            ToolSpec::parse(&tools, Some(&named), None).unwrap().mode,
            ToolMode::Named("b".into())
        );
        assert!(ToolSpec::parse(&tools, None, None).unwrap().parallel);
        assert!(!ToolSpec::parse(&tools, None, Some(false)).unwrap().parallel);
        assert_eq!(ToolSpec::parse(&tools, None, None).unwrap().template_tools().unwrap().len(), 2);
    }

    #[test]
    fn rejects_invalid_definitions() {
        let ok = tool("a", json!({"type": "object"}));
        for (tools, choice) in [
            (vec![json!({"type": "web", "function": {"name": "a"}})], None),
            (vec![json!({"type": "function"})], None),
            (vec![tool("", json!({"type": "object"}))], None),
            (vec![ok.clone(), ok.clone()], None),
            (vec![tool("a", json!("x"))], None),
            (vec![ok.clone()], Some(json!({"type": "function", "function": {"name": "z"}}))),
            (vec![ok.clone()], Some(json!("sometimes"))),
            (vec![], Some(json!("required"))),
        ] {
            assert!(ToolSpec::parse(&tools, choice.as_ref(), None).is_err(), "{tools:?} {choice:?}");
        }
    }

    #[test]
    fn missing_parameters_mean_any_object_and_names_keep_any_character() {
        let tools = [json!({"type": "function", "function": {"name": "a\"b\\ç"}})];
        let spec = ToolSpec::parse(&tools, None, None).unwrap();
        assert_eq!(spec.defs[0].name, "a\"b\\ç");
        assert_eq!(spec.defs[0].parameters, json!({"type": "object"}));
    }

    #[test]
    fn forbidden_spec_has_no_tools_for_the_template() {
        let spec = ToolSpec::forbidden();
        assert_eq!(spec.mode, ToolMode::Forbidden);
        assert!(spec.template_tools().is_none() && spec.defs.is_empty());
    }

    #[test]
    fn removes_unsupported_keywords_but_not_properties_with_the_same_name() {
        let schema = json!({"type": "object", "not": {"required": ["x"]}, "properties": {
            "not": {"type": "string"},
            "list": {"type": "array", "uniqueItems": true, "items": {"type": "integer"}}
        }});
        let (clean, warnings) = tolerant_schema("t", &schema);
        assert!(clean.get("not").is_none());
        assert_eq!(clean["properties"]["not"], json!({"type": "string"}));
        assert!(clean["properties"]["list"].get("uniqueItems").is_none());
        assert!(warnings.iter().any(|w| w.contains("t") && w.contains("uniqueItems")));
    }

    #[test]
    fn a_schema_that_still_does_not_compile_becomes_any_object() {
        let (clean, warnings) = tolerant_schema("t", &json!({"$ref": "#/missing"}));
        assert_eq!(clean, json!({"type": "object"}));
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn a_supported_schema_is_kept_without_warnings() {
        let schema = json!({"type": "object", "properties": {"a": {"enum": ["x", "y"]}}, "required": ["a"]});
        let (clean, warnings) = tolerant_schema("t", &schema);
        assert_eq!(clean, schema);
        assert!(warnings.is_empty());
    }
}
