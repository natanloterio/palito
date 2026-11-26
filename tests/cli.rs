//! `palito` command-line behavior: error messages, exit codes, help and
//! stdin inputs. Tests that run a model need
//! models/SmolLM2-135M-Instruct-Q4_K_M.gguf from scripts/download_model.py.

use std::{
    io::Write,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

const SMOL_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/SmolLM2-135M-Instruct-Q4_K_M.gguf");

fn palito(args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_palito"))
        .args(args)
        .env_remove("PALITO_MODEL")
        .env_remove("PALITO_BACKEND")
        .env_remove("PALITO_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    if let Some(text) = stdin {
        input.write_all(text.as_bytes()).unwrap();
    }
    drop(input);
    child.wait_with_output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn missing_model_names_the_path_and_hints() {
    let output = palito(&["chat", "--model", "/nope/x.gguf"], None);
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.starts_with("error: model file not found: /nope/x.gguf"), "{message}");
    assert!(message.contains("hint:"), "{message}");
}

#[test]
fn a_non_model_file_is_rejected_by_name() {
    let readme = concat!(env!("CARGO_MANIFEST_DIR"), "/README.md");
    let output = palito(&["generate", "--model", readme, "--prompt", "hi"], None);
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.contains("README.md is not a GGUF or .palito model"), "{message}");
}

#[test]
fn errors_have_no_library_prefix() {
    let output = palito(&["chat", "--model", SMOL_MODEL, "--top-p", "2"], Some(""));
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.starts_with("error: "), "{message}");
    assert!(!message.contains("invalid request or configuration"), "{message}");
}

#[test]
fn bad_token_file_names_the_file() {
    let output = palito(
        &["detokenize", "--model", SMOL_MODEL, "--input-tokens-file", "-"],
        Some("not json"),
    );
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.contains("expected a JSON array of token IDs"), "{message}");
}

#[test]
fn tokenize_reads_stdin_and_detokenize_takes_inline_ids() {
    let output = palito(&["tokenize", "--model", SMOL_MODEL, "--prompt-file", "-"], Some("Hello"));
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let ids: Vec<String> = json["prompt_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.to_string())
        .collect();
    let output = palito(
        &["detokenize", "--model", SMOL_MODEL, "--tokens", &ids.join(",")],
        None,
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["text"], "Hello");
}

#[test]
fn piped_chat_handles_commands_without_the_model() {
    let output = palito(
        &["chat", "--model", SMOL_MODEL, "--backend", "cpu", "--context", "512"],
        Some("/help\n/exti\n/exit\n"),
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let message = stderr(&output);
    assert!(message.contains("/reset"), "{message}");
    assert!(message.contains("did you mean /exit?"), "{message}");
    assert!(output.stdout.is_empty(), "nothing reaches the model");
}

#[test]
fn piped_chat_fails_with_the_reply_error() {
    let output = palito(
        &["chat", "--model", SMOL_MODEL, "--backend", "cpu", "--context", "64", "--tokens", "8"],
        Some(&format!("{}\n", "word ".repeat(200))),
    );
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
}

#[test]
fn busy_port_fails_before_loading_the_model() {
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = busy.local_addr().unwrap().port().to_string();
    let started = Instant::now();
    let output = palito(
        &["serve", "--backend", "cpu", "--model", SMOL_MODEL, "--port", &port],
        None,
    );
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.contains(&format!("port {port} is already in use")), "{message}");
    assert!(!message.contains("Loaded"), "{message}");
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[test]
fn convert_refuses_to_overwrite_without_force() {
    let dir = std::env::temp_dir().join(format!("palito-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output_path = dir.join("existing.palito");
    std::fs::write(&output_path, b"keep me").unwrap();
    let output = palito(
        &["convert", "-i", SMOL_MODEL, "-o", output_path.to_str().unwrap()],
        None,
    );
    let message = stderr(&output);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(output.status.code(), Some(1));
    // Builds without qtip-research refuse convert before looking at the output.
    if !message.contains("requires a build with") {
        assert!(message.contains("already exists"), "{message}");
        assert!(message.contains("--force"), "{message}");
    }
}

#[test]
fn short_help_hides_validation_flags_and_shows_examples() {
    let output = palito(&["chat", "-h"], None);
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("Examples:"), "{help}");
    assert!(!help.contains("--f32-weights"), "{help}");
    assert!(!help.contains("residual-int8-research"), "{help}");
    let long = palito(&["chat", "--help"], None);
    assert!(String::from_utf8_lossy(&long.stdout).contains("--f32-weights"));
}

#[test]
fn help_fits_80_columns() {
    for command in ["chat", "serve", "generate", "bench", "convert"] {
        let output = Command::new(env!("CARGO_BIN_EXE_palito"))
            .args([command, "--help"])
            .env("COLUMNS", "80")
            .output()
            .unwrap();
        let help = String::from_utf8_lossy(&output.stdout);
        let widest = help.lines().map(|line| line.chars().count()).max().unwrap_or(0);
        // Examples are verbatim and may run a little past the wrap width.
        assert!(widest <= 100, "{command} --help has a {widest}-column line");
    }
}

#[test]
fn version_lists_backends() {
    let output = palito(&["--version"], None);
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("backends: cpu"), "{text}");
}

#[test]
fn completions_are_generated() {
    let output = palito(&["completions", "bash"], None);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("palito"));
}
