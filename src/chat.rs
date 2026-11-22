//! Chat prompts rendered from the GGUF `tokenizer.chat_template` (Hugging Face Jinja).
//!
//! Rendering follows the Hugging Face conventions: `trim_blocks`, `lstrip_blocks`,
//! Python string methods, `raise_exception` and `strftime_now`. The rendered text
//! is then tokenized normally, so embedded control tokens become their IDs.

use crate::{
    Error, Result,
    gguf::{Gguf, Value},
};
use minijinja::{Environment, ErrorKind, Value as TemplateValue, context, syntax::SyntaxConfig};
use serde::Serialize;
use std::{collections::HashMap, path::Path, time::SystemTime};

const TEMPLATE_NAME: &str = "chat";
const TEMPLATE_KEY: &str = "tokenizer.chat_template";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
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

pub struct ChatTemplate {
    env: Environment<'static>,
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
        self.env
            .get_template(TEMPLATE_NAME)
            .and_then(|template| {
                template.render(context! {
                    messages => TemplateValue::from_iter(messages.iter().map(|m| context! {
                        role => &m.role,
                        content => &m.content,
                    })),
                    add_generation_prompt => add_generation_prompt,
                    bos_token => &self.bos_token,
                    eos_token => &self.eos_token,
                    enable_thinking => enable_thinking,
                })
            })
            .map_err(template_error)
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
}
