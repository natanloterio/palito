//! Tool calling on a real model. Needs a Qwen3 GGUF:
//! `PALITO_TOOL_MODEL=/path/to/Qwen3-0.6B-Q4_K_M.gguf`, or
//! models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf.
#![cfg(feature = "cuda")]

use palito::{
    chat::Message,
    engine::{Engine, EngineOptions, FinishReason, GenerationOptions},
    tools::{ToolEvent, ToolMode, ToolSpec},
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard, PoisonError},
};

/// One engine on the GPU at a time.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(PoisonError::into_inner)
}

fn model() -> PathBuf {
    std::env::var_os("PALITO_TOOL_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from,
    )
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "weather",
        "description": "Current weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }
    }})
}

fn engine_with_weather(mode: ToolMode) -> (Engine, ToolSpec) {
    let choice = match &mode {
        ToolMode::Auto => json!("auto"),
        ToolMode::Required => json!("required"),
        ToolMode::Forbidden => json!("none"),
        ToolMode::Named(name) => json!({"type": "function", "function": {"name": name}}),
    };
    let spec = ToolSpec::parse(&[weather_tool()], Some(&choice), None).unwrap();
    let options = EngineOptions { context_tokens: Some(2048), ..EngineOptions::default() };
    (Engine::open(&model(), &options).unwrap(), spec)
}

fn ask(question: &str) -> Vec<Message> {
    vec![Message::user(question)]
}

fn history_with_a_call() -> Vec<Message> {
    let mut call = Message::assistant("");
    call.tool_calls = Some(json!([{"id": "call_1", "type": "function",
        "function": {"name": "weather", "arguments": "{\"city\": \"Rio\"}"}}]));
    vec![
        Message::user("Tempo no Rio?"),
        call,
        Message::new("tool", "{\"temperature\": 28}"),
    ]
}

fn capped(max_tokens: u32) -> GenerationOptions {
    GenerationOptions { max_tokens: Some(max_tokens), ..GenerationOptions::default() }
}

fn case_message(value: &Value, string_arguments: bool) -> Message {
    let mut message = Message::new(
        value["role"].as_str().unwrap(),
        value["content"].as_str().unwrap_or_default(),
    );
    message.tool_call_id = value["tool_call_id"].as_str().map(str::to_owned);
    message.tool_calls = value.get("tool_calls").cloned().map(|mut calls| {
        for call in calls.as_array_mut().unwrap() {
            let arguments = &mut call["function"]["arguments"];
            if let (false, Some(text)) = (string_arguments, arguments.as_str()) {
                *arguments = serde_json::from_str(text).unwrap();
            }
        }
        calls
    });
    message
}

/// For scripts/compare_tool_prompts.py: reads cases from PALITO_PROMPT_CASES
/// and writes the prompt token IDs of each to PALITO_PROMPT_IDS.
#[test]
#[ignore = "run by scripts/compare_tool_prompts.py"]
fn dump_prompt_ids() {
    let cases: Vec<Value> =
        serde_json::from_slice(&std::fs::read(std::env::var("PALITO_PROMPT_CASES").unwrap()).unwrap())
            .unwrap();
    let _gpu = gpu_lock();
    let options = EngineOptions { context_tokens: Some(2048), ..EngineOptions::default() };
    let engine = Engine::open(&model(), &options).unwrap();
    let string_arguments = engine.tool_format().unwrap().string_arguments;
    let ids: Vec<Vec<i32>> = cases
        .iter()
        .map(|case| {
            let messages: Vec<Message> = case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|message| case_message(message, string_arguments))
                .collect();
            let spec = ToolSpec::parse(case["tools"].as_array().unwrap(), None, None).unwrap();
            let prompt = engine.render_chat_tools(&messages, &spec, true).unwrap();
            engine.tokenize(&prompt).unwrap()
        })
        .collect();
    std::fs::write(std::env::var("PALITO_PROMPT_IDS").unwrap(), json!(ids).to_string()).unwrap();
}

#[test]
#[ignore = "requires a GPU and a Qwen3 GGUF"]
fn required_produces_a_valid_call() {
    let _gpu = gpu_lock();
    let (mut engine, spec) = engine_with_weather(ToolMode::Required);
    assert!(engine.supports_tools());
    let done = engine.chat_tools(&ask("Tempo no Rio?"), &spec, &capped(256), |_| true).unwrap();
    assert!(done.calls[0].complete, "{:?}", done.calls);
    assert_eq!(done.calls[0].name, "weather");
    let args: Value = serde_json::from_str(&done.calls[0].arguments).unwrap();
    assert!(args["city"].is_string());
    assert_eq!(done.completion.finish_reason, FinishReason::Stop);
}

#[test]
#[ignore = "requires a GPU and a Qwen3 GGUF"]
fn named_without_reasoning_streams_the_same_call() {
    let _gpu = gpu_lock();
    let (mut engine, spec) = engine_with_weather(ToolMode::Named("weather".into()));
    let options = GenerationOptions { enable_thinking: false, ..capped(128) };
    let mut arguments = String::new();
    let mut names = Vec::new();
    let done = engine
        .chat_tools(&ask("Tempo em Lisboa?"), &spec, &options, |event| {
            match event {
                ToolEvent::CallStart { name, .. } => names.push(name),
                ToolEvent::CallArguments { index: 0, text } => arguments.push_str(&text),
                _ => {}
            }
            true
        })
        .unwrap();
    assert_eq!(names[0], "weather");
    assert_eq!(arguments, done.calls[0].arguments);
    assert!(!done.completion.text.contains("tool_call"));
}

#[test]
#[ignore = "requires a GPU and a Qwen3 GGUF"]
fn forbidden_never_emits_the_call_token() {
    let _gpu = gpu_lock();
    let (mut engine, spec) = engine_with_weather(ToolMode::Forbidden);
    let done = engine.chat_tools(&history_with_a_call(), &spec, &capped(128), |_| true).unwrap();
    assert!(done.calls.is_empty() && !done.completion.text.contains("<tool_call>"));
}

#[test]
#[ignore = "requires a GPU and a Qwen3 GGUF"]
fn stop_does_not_cut_a_call() {
    let _gpu = gpu_lock();
    let (mut engine, spec) = engine_with_weather(ToolMode::Required);
    let options = GenerationOptions { stop: vec!["\n".into()], ..capped(256) };
    let done = engine.chat_tools(&ask("Tempo no Rio?"), &spec, &options, |_| true).unwrap();
    assert!(done.calls[0].complete);
}

#[test]
#[ignore = "requires a GPU and a Qwen3 GGUF"]
fn max_tokens_inside_a_call_keeps_a_partial_call() {
    let _gpu = gpu_lock();
    let (mut engine, spec) = engine_with_weather(ToolMode::Required);
    let options = GenerationOptions { enable_thinking: false, ..capped(12) };
    let done = engine.chat_tools(&ask("Tempo no Rio?"), &spec, &options, |_| true).unwrap();
    assert_eq!(done.completion.finish_reason, FinishReason::Length);
    assert!(done.calls.iter().all(|call| !call.complete));
}
