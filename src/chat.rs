//! Chat prompts rendered from the GGUF `tokenizer.chat_template` (Hugging Face Jinja).
//!
//! Rendering follows the Hugging Face conventions: `trim_blocks`, `lstrip_blocks`,
//! Python string methods, `raise_exception` and `strftime_now`. The rendered text
//! is then tokenized normally, so embedded control tokens become their IDs.

mod tojson;

use crate::{
    Error, Result,
    gguf::{Gguf, Value},
};
use minijinja::{
    Environment, ErrorKind, Value as TemplateValue, context, syntax::SyntaxConfig, value::Serde,
};
use serde::Serialize;
use std::{collections::HashMap, path::Path, time::SystemTime};

const TEMPLATE_NAME: &str = "chat";
const TEMPLATE_KEY: &str = "tokenizer.chat_template";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// OpenAI-shaped calls of an assistant turn. Absent fields stay undefined
    /// in the template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// A template that writes tool calls as Qwen's JSON inside `<tool_call>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolFormat {
    /// The template inserts string `arguments` as they are; otherwise it
    /// applies `tojson` to them.
    pub string_arguments: bool,
}

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
    bos_token: String,
    eos_token: String,
    enable_thinking: bool,
}

impl ChatTemplate {
    /// Compiles `source`; `bos_token`/`eos_token` are the texts the template may emit.
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
        let mut env = Environment::new();
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(template_error)?;
        env.set_syntax(syntax);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson::tojson);
        env.add_function("raise_exception", |message: String| {
            Err::<String, _>(minijinja::Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_function("strftime_now", |format: String| {
            strftime(&format, unix_days_now())
        });
        env.add_template_owned(TEMPLATE_NAME, source.to_owned())
            .map_err(template_error)?;
        Ok(Self {
            env,
            source: source.to_owned(),
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
            enable_thinking: true,
        })
    }

    /// Sets `enable_thinking` for reasoning templates (Qwen3); others ignore it.
    pub fn with_thinking(self, enable_thinking: bool) -> Self {
        Self {
            enable_thinking,
            ..self
        }
    }

    /// Reads the template and BOS/EOS texts from GGUF metadata.
    ///
    /// When the tokenizer already prepends BOS, the template's `bos_token` renders
    /// empty so the prompt does not start with two BOS tokens. `override_source`
    /// replaces the embedded template (for models without one).
    pub fn from_model(
        path: &Path,
        override_source: Option<&str>,
        tokenizer_adds_bos: bool,
    ) -> Result<Self> {
        let metadata = Gguf::read_metadata(path)?;
        let source = match (override_source, metadata.get(TEMPLATE_KEY)) {
            (Some(source), _) => source.to_owned(),
            (None, Some(Value::Text(source))) => source.clone(),
            (None, _) => {
                return Err(Error::Invalid(format!(
                    "model has no {TEMPLATE_KEY}; supply --chat-template-file"
                )));
            }
        };
        let bos = special_token(&metadata, "tokenizer.ggml.bos_token_id")?;
        let eos = special_token(&metadata, "tokenizer.ggml.eos_token_id")?;
        let bos = if tokenizer_adds_bos { "" } else { bos };
        Self::new(&source, bos, eos)
    }

    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        self.render_thinking(messages, add_generation_prompt, self.enable_thinking)
    }

    /// `render` with an explicit `enable_thinking` for this call.
    pub fn render_thinking(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String> {
        let has_tool_turns =
            messages.iter().any(|m| m.tool_calls.is_some() || m.tool_call_id.is_some());
        let template_messages = if has_tool_turns {
            TemplateValue::from(Serde(messages))
        } else {
            TemplateValue::from_iter(messages.iter().map(|m| context! {
                role => &m.role,
                content => &m.content,
            }))
        };
        self.env
            .get_template(TEMPLATE_NAME)
            .and_then(|template| {
                template.render(context! {
                    messages => template_messages,
                    add_generation_prompt => add_generation_prompt,
                    bos_token => &self.bos_token,
                    eos_token => &self.eos_token,
                    enable_thinking => enable_thinking,
                })
            })
            .map_err(template_error)
    }

    /// Renders a conversation with tool fields and, when given, tool
    /// definitions, ending with the generation prompt.
    pub fn render_tools(
        &self,
        messages: &[Message],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<String> {
        let tools = tools.map_or(TemplateValue::UNDEFINED, |tools| {
            TemplateValue::from(Serde(tools))
        });
        self.env
            .get_template(TEMPLATE_NAME)
            .and_then(|template| {
                template.render(context! {
                    messages => TemplateValue::from(Serde(messages)),
                    tools => tools,
                    add_generation_prompt => true,
                    bos_token => &self.bos_token,
                    eos_token => &self.eos_token,
                    enable_thinking => enable_thinking,
                })
            })
            .map_err(template_error)
    }

    /// The tool call format of this template, if Palito supports it.
    pub fn tool_format(&self) -> Option<ToolFormat> {
        let source = &self.source;
        let qwen_json = source.contains("<tool_call>")
            && source.contains("<function-name>")
            && !source.contains("<function=");
        qwen_json.then(|| ToolFormat {
            string_arguments: source.contains("arguments is string"),
        })
    }
}

/// Drops the oldest non-system turns until `fits` accepts the conversation.
///
/// System messages are kept, and the remaining turns always start with a user
/// message because strict templates reject a leading assistant turn.
pub fn fit_history(
    messages: &[Message],
    mut fits: impl FnMut(&[Message]) -> Result<bool>,
) -> Result<Vec<Message>> {
    let (system, turns): (Vec<_>, Vec<_>) =
        messages.iter().cloned().partition(|m| m.role == "system");
    for start in 0..turns.len() {
        if turns[start].role != "user" {
            continue;
        }
        let candidate = [system.as_slice(), &turns[start..]].concat();
        if fits(&candidate)? {
            return Ok(candidate);
        }
    }
    Err(Error::Invalid(
        "latest message does not fit in --context with --tokens reserved for the reply".into(),
    ))
}

fn special_token<'a>(metadata: &'a HashMap<String, Value>, key: &str) -> Result<&'a str> {
    let tokens = match metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Strings(tokens)) => tokens,
        _ => return Err(Error::Invalid("missing tokenizer.ggml.tokens".into())),
    };
    match metadata.get(key) {
        None => Ok(""),
        Some(Value::Integer(id)) => usize::try_from(*id)
            .ok()
            .and_then(|id| tokens.get(id))
            .map(String::as_str)
            .ok_or_else(|| Error::Invalid(format!("{key} is outside the vocabulary"))),
        Some(_) => Err(Error::Invalid(format!("invalid {key}"))),
    }
}

fn template_error(error: minijinja::Error) -> Error {
    Error::Invalid(format!("chat template: {error:#}"))
}

fn unix_days_now() -> i64 {
    let seconds = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    i64::try_from(seconds / 86_400).unwrap_or(0)
}

/// Date-only `strftime` (UTC), enough for chat templates that print today's date.
fn strftime(format: &str, unix_days: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    ];
    let (year, month, day) = civil_from_days(unix_days);
    let name = MONTHS[(month - 1) as usize];
    let mut output = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            output.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => output.push_str(&format!("{day:02}")),
            Some('m') => output.push_str(&format!("{month:02}")),
            Some('Y') => output.push_str(&year.to_string()),
            Some('b') => output.push_str(&name[..3]),
            Some('B') => output.push_str(name),
            Some('%') => output.push('%'),
            Some(other) => {
                output.push('%');
                output.push(other);
            }
            None => output.push('%'),
        }
    }
    output
}

/// Howard Hinnant's days-to-civil conversion for the proleptic Gregorian calendar.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn renders_chatml_with_generation_prompt() {
        let template = ChatTemplate::new(CHATML, "", "<|im_end|>").unwrap();
        let text = template
            .render(&[Message::system("Be brief."), Message::user("Hi")], true)
            .unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn exposes_bos_and_eos_tokens_and_block_trimming() {
        let source = "{{ bos_token }}\n  {% for m in messages %}\n[{{ m.content }}]{{ eos_token }}\n  {% endfor %}";
        let template = ChatTemplate::new(source, "<s>", "</s>").unwrap();
        let text = template.render(&[Message::user("a")], false).unwrap();
        assert_eq!(text, "<s>\n[a]</s>\n");
    }

    #[test]
    fn supports_python_string_methods_used_by_qwen_templates() {
        let source = "{% for m in messages %}{{ m.content.strip().split('|')[0] }}{% if m.content.startswith(' x') %}!{% endif %}{% endfor %}";
        let template = ChatTemplate::new(source, "", "").unwrap();
        assert_eq!(template.render(&[Message::user(" x|y ")], false).unwrap(), "x!");
    }

    #[test]
    fn passes_enable_thinking_to_the_template() {
        let source = "{% if enable_thinking is defined and enable_thinking is false %}off{% else %}on{% endif %}";
        let template = ChatTemplate::new(source, "", "").unwrap();
        assert_eq!(template.render(&[], true).unwrap(), "on");
        let template = template.with_thinking(false);
        assert_eq!(template.render(&[], true).unwrap(), "off");
    }

    #[test]
    fn raise_exception_becomes_an_error() {
        let source = "{{ raise_exception('roles must alternate') }}";
        let template = ChatTemplate::new(source, "", "").unwrap();
        let error = template.render(&[], false).unwrap_err().to_string();
        assert!(error.contains("roles must alternate"), "{error}");
    }

    #[test]
    fn invalid_template_fails_at_construction() {
        assert!(ChatTemplate::new("{% for %}", "", "").is_err());
    }

    #[test]
    fn strftime_formats_dates() {
        // 2024-07-26 is 19930 days after the Unix epoch.
        assert_eq!(strftime("%d %b %Y", 19_930), "26 Jul 2024");
        assert_eq!(strftime("%Y-%m-%d %B %% %H", 19_930), "2024-07-26 July % %H");
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
    }

    #[test]
    fn fit_history_keeps_everything_when_it_fits() {
        let messages = [Message::system("s"), Message::user("u")];
        assert_eq!(fit_history(&messages, |_| Ok(true)).unwrap(), messages);
    }

    #[test]
    fn fit_history_drops_oldest_turns_and_keeps_system() {
        let messages = [
            Message::system("s"),
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
            Message::user("u3"),
        ];
        let fitted = fit_history(&messages, |m| Ok(m.len() <= 4)).unwrap();
        assert_eq!(
            fitted,
            [
                Message::system("s"),
                Message::user("u2"),
                Message::assistant("a2"),
                Message::user("u3"),
            ]
        );
    }

    #[test]
    fn fit_history_fails_when_latest_message_cannot_fit() {
        let messages = [Message::user("long")];
        assert!(fit_history(&messages, |_| Ok(false)).is_err());
    }

    #[test]
    fn fit_history_propagates_errors() {
        let messages = [Message::user("u")];
        let result = fit_history(&messages, |_| Err(Error::Invalid("boom".into())));
        assert!(result.unwrap_err().to_string().contains("boom"));
    }

    /// The parts of the Qwen3-30B-A3B template that handle tools.
    const QWEN3_TOOLS_TEMPLATE: &str = r#"{%- if tools %}
{{- '<|im_start|>system\n# Tools\n<tools>' }}
{%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}
{{- '\n</tools>\n<tool_call>\n{"name": <function-name>}\n</tool_call><|im_end|>\n' }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == "user" or (message.role == "system" and not loop.first) %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == "assistant" %}
{{- '<|im_start|>assistant\n' + message.content }}
{%- if message.tool_calls %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function %}{%- set tool_call = tool_call.function %}{%- endif %}
{{- '<tool_call>\n{"name": "' }}{{- tool_call.name }}{{- '", "arguments": ' }}
{%- if tool_call.arguments is string %}{{- tool_call.arguments }}{%- else %}{{- tool_call.arguments | tojson }}{%- endif %}
{{- '}\n</tool_call>' }}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- elif message.role == "tool" %}
{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"#;

    #[test]
    fn renders_tools_calls_and_results_with_the_qwen3_template() {
        let template = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "<|im_end|>").unwrap();
        let tools = [json!({"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}})];
        let mut call = Message::assistant("");
        call.tool_calls = Some(json!([{"id": "call_1", "type": "function",
            "function": {"name": "weather", "arguments": "{\"city\":\"Rio\"}"}}]));
        let messages = [Message::user("Tempo?"), call, Message::new("tool", "28C")];
        let prompt = template.render_tools(&messages, Some(&tools), true).unwrap();
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("{\"name\": \"weather\", \"arguments\": {\"city\":\"Rio\"}}"));
        assert!(prompt.contains("<tool_response>\n28C\n</tool_response>"));
    }

    #[test]
    fn render_without_tool_fields_is_unchanged() {
        let template = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "<|im_end|>").unwrap();
        let messages = [Message::system("S"), Message::user("Oi")];
        assert_eq!(
            template.render_tools(&messages, None, true).unwrap(),
            template.render_thinking(&messages, true, true).unwrap()
        );
    }

    #[test]
    fn absent_reasoning_content_is_undefined_in_the_template() {
        let source = "{% for m in messages %}{% if m.reasoning_content is defined %}set{% else %}unset{% endif %}{% endfor %}";
        let template = ChatTemplate::new(source, "", "").unwrap();
        let mut with = Message::assistant("a");
        with.reasoning_content = Some("r".into());
        let prompt = template.render_tools(&[Message::assistant("a"), with], None, true).unwrap();
        assert_eq!(prompt, "unsetset");
    }

    #[test]
    fn plain_render_keeps_tool_history() {
        let template = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "<|im_end|>").unwrap();
        let mut call = Message::assistant("");
        call.tool_calls = Some(serde_json::json!([{"id": "1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Rio\"}"}}]));
        let messages = [Message::user("Tempo?"), call, Message::new("tool", "22 C")];
        let prompt = template.render_thinking(&messages, true, false).unwrap();
        assert!(prompt.contains("get_weather"), "{prompt}");
        assert_eq!(prompt, template.render_tools(&messages, None, false).unwrap());
    }

    #[test]
    fn tool_json_keeps_the_key_order_the_client_sent() {
        let template = ChatTemplate::new("{{ tools[0] | tojson }}", "", "").unwrap();
        let tool = serde_json::json!({"type": "function", "function": {"name": "a", "description": "b"}});
        let prompt = template.render_tools(&[], Some(&[tool]), true).unwrap();
        assert_eq!(prompt, r#"{"type": "function", "function": {"name": "a", "description": "b"}}"#);
    }

    #[test]
    fn tool_json_is_not_html_escaped() {
        let template = ChatTemplate::new("{{ tools[0] | tojson }}", "", "").unwrap();
        let tool = serde_json::json!({"a": "<b>&'ç😊\"\n", "n": [1, 2.5, null, true]});
        let prompt = template.render_tools(&[], Some(&[tool]), true).unwrap();
        assert_eq!(prompt, r#"{"a": "<b>&'ç😊\"\n", "n": [1, 2.5, null, true]}"#);
    }

    #[test]
    fn tojson_with_options_still_works() {
        let template = ChatTemplate::new("{{ tools[0] | tojson(indent=2) }}", "", "").unwrap();
        let tool = serde_json::json!({"a": 1});
        let prompt = template.render_tools(&[], Some(&[tool]), true).unwrap();
        assert_eq!(prompt, "{\n  \"a\": 1\n}");
    }

    #[test]
    fn detects_only_the_qwen_json_tool_format() {
        let qwen3 = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "").unwrap();
        assert_eq!(qwen3.tool_format(), Some(ToolFormat { string_arguments: true }));
        let qwen25 = ChatTemplate::new("<tool_call> <function-name> {{ a | tojson }}", "", "").unwrap();
        assert_eq!(qwen25.tool_format(), Some(ToolFormat { string_arguments: false }));
        let qwen35 = ChatTemplate::new("<tool_call><function=x> <function-name>", "", "").unwrap();
        assert_eq!(qwen35.tool_format(), None);
        assert_eq!(ChatTemplate::new("{{ messages }}", "", "").unwrap().tool_format(), None);
    }
}
