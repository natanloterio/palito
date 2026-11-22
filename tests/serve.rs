//! `palito serve` end to end on the CPU backend (needs
//! models/SmolLM2-135M-Instruct-Q4_K_M.gguf from scripts/download_model.py).

use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    process::{Child, Command, Stdio},
    time::Duration,
};

struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(port: u16, extra: &[&str]) -> Server {
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/models/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    let mut child = Command::new(env!("CARGO_BIN_EXE_palito"))
        .args(["serve", "--backend", "cpu", "--context", "512", "--model", model])
        .args(["--port", &port.to_string()])
        .args(extra)
        .env_remove("PALITO_API_KEY")
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    loop {
        let line = lines.next().expect("server exited").unwrap();
        if line.starts_with("Serving") {
            break;
        }
    }
    // Keep draining stderr so the server never blocks on a full pipe.
    std::thread::spawn(move || lines.for_each(drop));
    Server { child, port }
}

/// Minimal HTTP/1.1 request; returns (status, body).
fn request(server: &Server, method: &str, path: &str, body: Option<Value>, key: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    let auth = key.map(|k| format!("Authorization: Bearer {k}\r\n")).unwrap_or_default();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let status = response[9..12].parse().unwrap();
    let body = response.split_once("\r\n\r\n").map_or("", |(_, b)| b).to_owned();
    (status, body)
}

fn chunked_body(raw: &str) -> String {
    // Decode HTTP chunked transfer encoding.
    let mut out = String::new();
    let mut rest = raw;
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        out.push_str(&tail[..size]);
        rest = &tail[size + 2..];
    }
    out
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn serves_openai_chat_streaming_and_completions() {
    let server = start(18093, &[]);
    let (status, body) = request(&server, "GET", "/v1/models", None, None);
    assert_eq!(status, 200);
    let models: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(models["data"][0]["id"], "SmolLM2-135M-Instruct-Q4_K_M");

    let chat = json!({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16});
    let (status, body) = request(&server, "POST", "/v1/chat/completions", Some(chat.clone()), None);
    assert_eq!(status, 200, "{body}");
    let reply: Value = serde_json::from_str(&body).unwrap();
    let text = reply["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
    assert!(!text.is_empty());
    assert!(reply["usage"]["completion_tokens"].as_u64().unwrap() <= 16);

    let mut streaming = chat.clone();
    streaming["stream"] = json!(true);
    streaming["stream_options"] = json!({"include_usage": true});
    let (status, raw) = request(&server, "POST", "/v1/chat/completions", Some(streaming), None);
    assert_eq!(status, 200);
    let events: Vec<String> = chunked_body(&raw)
        .lines()
        .filter_map(|line| line.strip_prefix("data: ").map(str::to_owned))
        .collect();
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let streamed: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(streamed, text, "streamed and non-streamed replies differ");
    assert!(chunks.iter().any(|c| c["choices"][0]["finish_reason"].is_string()));
    assert!(chunks.last().unwrap()["usage"]["total_tokens"].as_u64().unwrap() > 0);

    let completion = json!({"prompt": "The capital of France is", "max_tokens": 6});
    let (status, body) = request(&server, "POST", "/v1/completions", Some(completion), None);
    assert_eq!(status, 200);
    let reply: Value = serde_json::from_str(&body).unwrap();
    assert!(reply["choices"][0]["text"].as_str().unwrap().contains("Paris"));
    assert_eq!(reply["choices"][0]["finish_reason"], "length");

    let (status, body) = request(&server, "POST", "/v1/chat/completions", Some(json!({"messages": []})), None);
    assert_eq!(status, 400);
    assert!(body.contains("invalid_request_error"));
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn api_key_guards_v1_but_not_health() {
    let server = start(18094, &["--api-key", "s3cret"]);
    assert_eq!(request(&server, "GET", "/v1/models", None, None).0, 401);
    assert_eq!(request(&server, "GET", "/v1/models", None, Some("nope")).0, 401);
    assert_eq!(request(&server, "GET", "/v1/models", None, Some("s3cret")).0, 200);
    assert_eq!(request(&server, "GET", "/health", None, None).0, 200);
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn stop_sequences_and_seeded_sampling() {
    let server = start(18095, &["--temperature", "0.9", "--top-k", "40", "--min-p", "0.05",
        "--repeat-penalty", "1.1", "--repeat-last-n", "32"]);
    let text = |body: Value| {
        let (status, body) = request(&server, "POST", "/v1/completions", Some(body), None);
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).unwrap();
        let choice = &reply["choices"][0];
        (choice["text"].as_str().unwrap().to_owned(), choice["finish_reason"].clone())
    };
    let greedy = json!({"prompt": "1, 2, 3, 4,", "max_tokens": 20, "temperature": 0,
        "repeat_penalty": 1, "stop": [" 7"]});
    let (cut, finish) = text(greedy);
    assert!(!cut.contains(" 7") && cut.contains('6'), "{cut:?}");
    assert_eq!(finish, "stop");
    // Server defaults (temperature 0.9) with a request seed: reproducible.
    let seeded = json!({"prompt": "Once upon a time", "max_tokens": 20, "seed": 42});
    assert_eq!(text(seeded.clone()).0, text(seeded).0);
    // Request penalties: the prompt's repeated " 1" is avoided under greedy.
    let ones = |extra: Value| {
        let mut body = json!({"prompt": "1, 1, 1, 1, 1, 1,", "max_tokens": 4, "temperature": 0});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        text(body).0
    };
    assert!(ones(json!({"repeat_penalty": 1.0})).starts_with(" 1"));
    let penalized = ones(json!({"frequency_penalty": 2.0, "presence_penalty": 0.5}));
    assert!(!penalized.starts_with(" 1"), "{penalized:?}");
    for bad in [json!({"top_p": 2}), json!({"min_p": -1}), json!({"repeat_penalty": 0}),
        json!({"logit_bias": {"99999999": 1}}), json!({"logit_bias": {"x": 1}})] {
        let mut body = json!({"prompt": "x"});
        body.as_object_mut().unwrap().extend(bad.as_object().unwrap().clone());
        let (status, reply) = request(&server, "POST", "/v1/completions", Some(body), None);
        assert_eq!(status, 400, "{bad}: {reply}");
    }
    let streamed = json!({"prompt": "x", "stream": true, "logit_bias": {"99999999": 1}});
    assert_eq!(request(&server, "POST", "/v1/completions", Some(streamed), None).0, 400);
}

/// Bounded schemas: a small model always finishes them within the token cap.
fn json_schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 20},
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }),
        json!({
            "type": "array",
            "items": {"type": "string", "maxLength": 12},
            "minItems": 1,
            "maxItems": 3
        }),
        json!({"enum": ["red", "green", "blue"]}),
        json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 10},
                "note": {"type": ["string", "null"], "maxLength": 16}
            },
            "required": ["ok", "score", "note"],
            "additionalProperties": false
        }),
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "maxLength": 16}},
                    "required": ["name"],
                    "additionalProperties": false
                },
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2}
            },
            "required": ["city", "tags"],
            "additionalProperties": false
        }),
    ]
}

fn json_chat(schema: &Value, extra: Value) -> Value {
    let mut body = json!({
        "messages": [{"role": "user", "content": "Describe a person as JSON."}],
        "max_tokens": 200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    });
    for (key, value) in extra.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn response_format_constrains_replies_to_the_schema() {
    let server = start(18096, &[]);
    for schema in json_schemas() {
        let validator = jsonschema::validator_for(&schema).unwrap();
        for extra in [json!({}), json!({"temperature": 1.0, "seed": 3})] {
            let (status, body) =
                request(&server, "POST", "/v1/chat/completions", Some(json_chat(&schema, extra)), None);
            assert_eq!(status, 200, "{body}");
            let reply: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(reply["choices"][0]["finish_reason"], "stop", "{body}");
            let content = reply["choices"][0]["message"]["content"].as_str().unwrap();
            let value: Value = serde_json::from_str(content)
                .unwrap_or_else(|e| panic!("{content:?} is not JSON: {e}"));
            assert!(validator.is_valid(&value), "{content} violates {schema}");
        }
    }

    // json_object, streamed: the chunks concatenate to one JSON object.
    let body = json!({
        "messages": [{"role": "user", "content": "Reply with a JSON object with one key, \"ok\"."}],
        "max_tokens": 400,
        "stream": true,
        "response_format": {"type": "json_object"}
    });
    let (status, raw) = request(&server, "POST", "/v1/chat/completions", Some(body), None);
    assert_eq!(status, 200);
    let streamed: String = chunked_body(&raw)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| {
            let chunk: Value = serde_json::from_str(data).unwrap();
            chunk["choices"][0]["delta"]["content"].as_str().map(str::to_owned)
        })
        .collect();
    let value: Value = serde_json::from_str(&streamed)
        .unwrap_or_else(|e| panic!("{streamed:?} is not JSON: {e}"));
    assert!(value.is_object(), "{streamed}");

    // Unsupported keywords and formats are refused before generation, streamed or not.
    let unsupported = json!({"type": "array", "uniqueItems": true});
    for stream in [false, true] {
        let body = json_chat(&unsupported, json!({"stream": stream}));
        let (status, body) = request(&server, "POST", "/v1/chat/completions", Some(body), None);
        assert_eq!(status, 400, "{body}");
        assert!(body.contains("uniqueItems"), "{body}");
    }
    let bad = json!({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "xml"}});
    let (status, _) = request(&server, "POST", "/v1/chat/completions", Some(bad), None);
    assert_eq!(status, 400);
}
