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
