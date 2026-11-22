#![cfg(feature = "ktransformers")]

use palito::{
    GenerationMode,
    backend::{CompletionRequest, GenerationInput, RequestBackend},
    ktransformers::{KTransformersBackend, KTransformersConfig},
};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    process::Command,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

struct Received {
    path: String,
    body: Value,
}

fn server_info() -> Value {
    json!({
        "served_model_name": "moe-test", "model_path": "/model/gpu",
        "version": "fixture-version", "kt_method": "LLAMAFILE",
        "kt_weight_path": "/model/gguf", "kt_cpuinfer": 4,
        "kt_num_gpu_experts": 0, "kt_max_deferred_experts_per_token": 0,
        "context_length": 128, "disable_radix_cache": true, "skip_tokenizer_init": false,
        "api_key": "fixture-secret-must-not-be-serialized"
    })
}

fn model_info() -> Value {
    json!({
        "model_path": "/model/gpu", "tokenizer_path": "/model/gpu",
        "model_type": "qwen3_moe", "is_generation": true, "weight_version": "v1"
    })
}

fn completion() -> Value {
    json!({
        "text": "test output", "output_ids": [7, 8],
        "meta_info": {
            "prompt_tokens": 3, "completion_tokens": 2, "cached_tokens": 0,
            "weight_version": "v1", "finish_reason": {"type": "length"}
        }
    })
}

fn receive(stream: &mut TcpStream) -> Received {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let path = line.split_whitespace().nth(1).unwrap().to_owned();
    let mut size = 0;
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        let (key, value) = line.split_once(':').unwrap();
        if key.eq_ignore_ascii_case("content-length") {
            size = value.trim().parse::<usize>().unwrap();
        }
    }
    let mut bytes = vec![0; size];
    reader.read_exact(&mut bytes).unwrap();
    Received {
        path,
        body: if size == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        },
    }
}

fn serve(responses: Vec<(u16, String)>) -> (String, JoinHandle<Vec<Received>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let task = thread::spawn(move || {
        let mut received = Vec::new();
        for (status, body) in responses {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "client did not send expected request"
                        );
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("accept failed: {error}"),
                }
            };
            received.push(receive(&mut stream));
            write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        }
        received
    });
    (endpoint, task)
}

fn responses(server: Value, model: Value, generations: Vec<Value>) -> Vec<(u16, String)> {
    [server, model]
        .into_iter()
        .chain(generations)
        .map(|body| (200, body.to_string()))
        .collect()
}

fn config(endpoint: String) -> KTransformersConfig {
    KTransformersConfig {
        endpoint,
        expected_model: "moe-test".into(),
        expected_method: "llamafile".into(),
        timeout: Duration::from_secs(2),
        allow_expert_deferral: false,
    }
}

fn request() -> CompletionRequest {
    CompletionRequest {
        input: GenerationInput::TokenIds(vec![1, 2, 3]),
        max_tokens: 2,
        mode: GenerationMode::FixedLength,
    }
}

#[test]
fn verifies_identity_and_uses_stateless_fixed_length_requests() {
    let (endpoint, task) = serve(responses(
        server_info(),
        model_info(),
        vec![completion(), completion()],
    ));
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    assert!(!backend.info().capabilities.token_step);
    assert!(!backend.info().capabilities.chunked_prefill);
    assert!(
        !serde_json::to_string(backend.info())
            .unwrap()
            .contains("fixture-secret")
    );
    for _ in 0..2 {
        let result = backend.generate(&request()).unwrap();
        assert_eq!(result.generated_tokens, vec![7, 8]);
        assert!(result.elapsed_ns > 0);
    }
    let received = task.join().unwrap();
    assert_eq!(received[0].path, "/server_info");
    assert_eq!(received[1].path, "/model_info");
    for input in &received[2..] {
        assert_eq!(input.path, "/generate");
        assert_eq!(input.body["input_ids"], json!([1, 2, 3]));
        assert_eq!(input.body["sampling_params"]["ignore_eos"], true);
        assert_eq!(
            input.body["sampling_params"]["temperature"].as_f64(),
            Some(0.0)
        );
        assert_eq!(input.body["stream"], false);
        assert!(input.body.get("session_params").is_none());
    }
}

#[test]
fn rejects_non_kt_wrong_identity_and_unapproved_deferral() {
    for (field, value) in [
        ("kt_method", Value::Null),
        ("kt_method", json!("AMXINT4")),
        ("kt_weight_path", json!("")),
        ("served_model_name", json!("other-model")),
        ("kt_max_deferred_experts_per_token", json!(1)),
        ("kt_max_deferred_experts_per_token", Value::Null),
    ] {
        let mut server = server_info();
        server[field] = value;
        let (endpoint, task) = serve(responses(server, model_info(), vec![]));
        assert!(KTransformersBackend::connect(config(endpoint)).is_err());
        task.join().unwrap();
    }
}

#[test]
fn rejects_inconsistent_model_and_missing_generation_support() {
    for (field, value) in [
        ("model_path", json!("/model/other")),
        ("weight_version", json!("")),
        ("is_generation", json!(false)),
        ("skip_tokenizer_init", json!(true)),
        ("context_length", json!(0)),
        ("kt_cpuinfer", json!(0)),
    ] {
        let mut server = server_info();
        let mut model = model_info();
        if model.get(field).is_some() {
            model[field] = value;
        } else {
            server[field] = value;
        }
        let (endpoint, task) = serve(responses(server, model, vec![]));
        assert!(
            KTransformersBackend::connect(config(endpoint)).is_err(),
            "accepted invalid {field}"
        );
        assert_eq!(task.join().unwrap().len(), 2);
    }
}

#[test]
fn admits_exact_reported_context_but_rejects_one_token_over_without_io() {
    let mut server = server_info();
    server["context_length"] = json!(5);
    let (endpoint, task) = serve(responses(server, model_info(), vec![completion()]));
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    let error = backend
        .generate(&CompletionRequest {
            max_tokens: 3,
            ..request()
        })
        .unwrap_err();
    assert!(matches!(error, palito::Error::Invalid(_)));
    assert_eq!(
        backend.generate(&request()).unwrap().generated_tokens,
        vec![7, 8]
    );
    let received = task.join().unwrap();
    assert_eq!(received.len(), 3);
    assert_eq!(received[2].body["sampling_params"]["max_new_tokens"], 2);
}

#[test]
fn missing_context_does_not_invent_a_capacity_or_disable_accounting_checks() {
    let mut server = server_info();
    server.as_object_mut().unwrap().remove("context_length");
    let mut result = completion();
    result["meta_info"]["prompt_tokens"] = json!(256);
    let mut inconsistent = result.clone();
    inconsistent["meta_info"]["completion_tokens"] = json!(1);
    let (endpoint, task) = serve(responses(server, model_info(), vec![inconsistent, result]));
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    assert_eq!(backend.info().context_tokens, None);
    let request = CompletionRequest {
        input: GenerationInput::TokenIds(vec![1; 256]),
        ..request()
    };
    assert!(matches!(
        backend.generate(&request),
        Err(palito::Error::Backend(_))
    ));
    let result = backend.generate(&request).unwrap();
    assert_eq!(result.prompt_tokens, 256);
    assert_eq!(result.generated_tokens, vec![7, 8]);
    assert_eq!(task.join().unwrap().len(), 4);
}

#[test]
fn expert_deferral_requires_explicit_opt_in_and_is_reported() {
    let mut server = server_info();
    server["kt_max_deferred_experts_per_token"] = json!(1);
    let (endpoint, task) = serve(responses(server, model_info(), vec![]));
    let mut config = config(endpoint);
    config.allow_expert_deferral = true;
    let backend = KTransformersBackend::connect(config).unwrap();
    assert_eq!(backend.info().deferred_experts_per_token, 1);
    task.join().unwrap();
}

#[test]
fn rejects_partial_aborted_changed_weight_and_invalid_accounting_results() {
    for (pointer, value) in [
        ("/output_ids", json!([7])),
        (
            "/meta_info/finish_reason",
            json!({"type": "abort", "message": "worker failed"}),
        ),
        ("/meta_info/weight_version", json!("v2")),
        ("/meta_info/prompt_tokens", json!(2)),
        ("/meta_info/cached_tokens", json!(4)),
        ("/output_ids", json!([-1, 8])),
    ] {
        let mut result = completion();
        *result.pointer_mut(pointer).unwrap() = value;
        let (endpoint, task) = serve(responses(server_info(), model_info(), vec![result]));
        let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
        assert!(backend.generate(&request()).is_err());
        task.join().unwrap();
    }
}

#[test]
fn supports_text_and_eog_without_forcing_fixed_length() {
    let mut result = completion();
    result["output_ids"] = json!([7]);
    result["meta_info"]["completion_tokens"] = json!(1);
    result["meta_info"]["finish_reason"] = json!({"type": "stop"});
    let (endpoint, task) = serve(responses(server_info(), model_info(), vec![result]));
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    let result = backend
        .generate(&CompletionRequest {
            input: GenerationInput::Text("a test".into()),
            max_tokens: 8,
            mode: GenerationMode::UntilEog,
        })
        .unwrap();
    assert_eq!(result.generated_tokens, vec![7]);
    let inputs = task.join().unwrap();
    assert_eq!(inputs[2].body["text"], "a test");
    assert_eq!(inputs[2].body["sampling_params"]["ignore_eos"], false);
}

#[test]
fn rejects_silent_length_clamping_and_post_tokenization_context_overflow() {
    let mut short = completion();
    short["output_ids"] = json!([7]);
    short["meta_info"]["completion_tokens"] = json!(1);
    for mode in [GenerationMode::FixedLength, GenerationMode::UntilEog] {
        let (endpoint, task) = serve(responses(server_info(), model_info(), vec![short.clone()]));
        let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
        let mut request = request();
        request.mode = mode;
        assert!(backend.generate(&request).is_err());
        task.join().unwrap();
    }
    let mut overflow = completion();
    overflow["meta_info"]["prompt_tokens"] = json!(127);
    let (endpoint, task) = serve(responses(server_info(), model_info(), vec![overflow]));
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    assert!(
        backend
            .generate(&CompletionRequest {
                input: GenerationInput::Text("server-tokenized prompt".into()),
                ..request()
            })
            .is_err()
    );
    task.join().unwrap();
}

#[test]
fn rejects_invalid_requests_before_generation_io() {
    let (endpoint, task) = serve(responses(server_info(), model_info(), vec![]));
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    for request in [
        CompletionRequest {
            input: GenerationInput::Text(String::new()),
            ..request()
        },
        CompletionRequest {
            input: GenerationInput::TokenIds(vec![-1]),
            ..request()
        },
        CompletionRequest {
            max_tokens: 0,
            ..request()
        },
        CompletionRequest {
            max_tokens: 126,
            ..request()
        },
    ] {
        assert!(backend.generate(&request).is_err());
    }
    task.join().unwrap();
}

#[test]
fn rejects_remote_origins_redirects_http_errors_and_malformed_json() {
    for endpoint in [
        "http://example.com",
        "http://192.168.1.1",
        "https://127.0.0.1",
        "http://127.0.0.1/path",
        "http://user:pass@127.0.0.1",
    ] {
        assert!(KTransformersBackend::connect(config(endpoint.into())).is_err());
    }
    for (status, body) in [
        (302, "{}"),
        (500, "{\"error\":\"worker failure\"}"),
        (200, "not json"),
        (200, "{\"error\":{\"message\":\"configuration failed\"}}"),
    ] {
        let (endpoint, task) = serve(vec![(status, body.into())]);
        assert!(KTransformersBackend::connect(config(endpoint)).is_err());
        task.join().unwrap();
    }
}

#[test]
fn http_deadline_is_bounded() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut options = config(format!("http://{}", listener.local_addr().unwrap()));
    options.timeout = Duration::from_millis(50);
    listener.set_nonblocking(true).unwrap();
    let task = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "client did not connect");
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("accept failed: {error}"),
            }
        };
        receive(&mut stream);
        thread::sleep(Duration::from_millis(150));
    });
    let start = Instant::now();
    let error = match KTransformersBackend::connect(options) {
        Ok(_) => panic!("a server that sends no response must time out"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("timed out"));
    assert!(start.elapsed() < Duration::from_secs(1));
    task.join().unwrap();
}

#[test]
fn failed_generation_is_not_retried_and_next_request_is_independent() {
    let mut replies = responses(server_info(), model_info(), vec![]);
    replies.push((
        500,
        json!({"error": {"message": "worker failure"}}).to_string(),
    ));
    replies.push((200, completion().to_string()));
    let (endpoint, task) = serve(replies);
    let mut backend = KTransformersBackend::connect(config(endpoint)).unwrap();
    let error = backend.generate(&request()).unwrap_err().to_string();
    assert!(error.contains("500") && error.contains("worker failure"));
    assert_eq!(
        backend.generate(&request()).unwrap().generated_tokens,
        vec![7, 8]
    );
    let received = task.join().unwrap();
    assert_eq!(received.len(), 4);
    assert_eq!(received[2].body, received[3].body);
}

#[test]
fn benchmark_rejects_cached_or_non_repeatable_work_without_flushing_server() {
    for mode in ["configured-cache", "observed-cache", "changed-output"] {
        let mut server = server_info();
        let mut generations = vec![];
        if mode == "configured-cache" {
            server["disable_radix_cache"] = json!(false);
        } else if mode == "observed-cache" {
            let mut cached = completion();
            cached["meta_info"]["cached_tokens"] = json!(1);
            generations.push(cached);
        } else {
            generations.push(completion());
            let mut changed = completion();
            changed["output_ids"] = json!([7, 9]);
            generations.push(changed);
        }
        let (endpoint, task) = serve(responses(server, model_info(), generations));
        let output_path = std::env::temp_dir().join(format!(
            "kt-adapter-rejection-{}-{mode}.json",
            std::process::id()
        ));
        let prompt = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/prompt.txt");
        let output = Command::new(env!("CARGO_BIN_EXE_ktransformers"))
            .args([
                "--endpoint",
                &endpoint,
                "--model",
                "moe-test",
                "--kt-method",
                "LLAMAFILE",
                "bench",
                "--tokens",
                "2",
                "--warmups",
                "1",
                "--repetitions",
                "1",
                "--prompt-file",
            ])
            .arg(prompt)
            .arg("--output")
            .arg(&output_path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!output_path.exists());
        assert!(
            task.join()
                .unwrap()
                .iter()
                .all(|input| input.path != "/flush_cache")
        );
    }
}

#[test]
fn cli_benchmark_requires_exactly_one_input_source_before_connecting() {
    for inputs in [
        vec![],
        vec![
            "--prompt-file",
            "unused.txt",
            "--input-tokens-file",
            "unused.json",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_ktransformers"))
            .args([
                "--endpoint",
                "http://example.com",
                "--model",
                "moe-test",
                "--kt-method",
                "LLAMAFILE",
                "bench",
                "--output",
                "unused-report.json",
            ])
            .args(inputs)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--prompt-file"), "{stderr}");
        assert!(stderr.contains("--input-tokens-file"), "{stderr}");
        assert!(!stderr.contains("KTransformers:"), "{stderr}");
    }
}

#[test]
fn unified_cli_uses_request_adapter_without_inventing_token_step_timings() {
    let (endpoint, task) = serve(responses(server_info(), model_info(), vec![completion()]));
    let output = Command::new(env!("CARGO_BIN_EXE_palito"))
        .args([
            "remote",
            "--endpoint",
            &endpoint,
            "--model",
            "moe-test",
            "--kt-method",
            "LLAMAFILE",
            "--prompt",
            "test input",
            "--tokens",
            "2",
            "--fixed-length",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["completion"]["generated_tokens"], json!([7, 8]));
    assert!(report["completion"].get("prefill_ns").is_none());
    assert!(
        !report["model"]["capabilities"]["token_step"]
            .as_bool()
            .unwrap()
    );
    assert!(report.get("memory").is_none());
    let calls = task.join().unwrap();
    assert_eq!(calls.last().unwrap().path, "/generate");
    assert_eq!(
        calls.last().unwrap().body["sampling_params"]["ignore_eos"],
        true
    );
}

#[test]
fn cli_benchmark_preserves_existing_report_without_generation() {
    let (endpoint, task) = serve(responses(server_info(), model_info(), vec![]));
    let output_path =
        std::env::temp_dir().join(format!("kt-existing-report-{}.json", std::process::id()));
    let original = b"existing benchmark evidence\n";
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .unwrap();
    file.write_all(original).unwrap();
    drop(file);
    let output = Command::new(env!("CARGO_BIN_EXE_ktransformers"))
        .args([
            "--endpoint",
            &endpoint,
            "--model",
            "moe-test",
            "--kt-method",
            "LLAMAFILE",
            "bench",
            "--prompt-file",
            "unused.txt",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .unwrap();
    let preserved = std::fs::read(&output_path).unwrap();
    std::fs::remove_file(output_path).unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("report exists"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(preserved, original);
    assert_eq!(task.join().unwrap().len(), 2);
}

#[test]
fn cli_benchmark_persists_request_metrics_without_native_phase_labels() {
    let (endpoint, task) = serve(responses(
        server_info(),
        model_info(),
        vec![completion(), completion(), completion()],
    ));
    let output_path =
        std::env::temp_dir().join(format!("kt-adapter-test-{}.json", std::process::id()));
    let input_path =
        std::env::temp_dir().join(format!("kt-adapter-input-{}.json", std::process::id()));
    std::fs::write(&input_path, "[1,2,3]").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ktransformers"))
        .args([
            "--endpoint",
            &endpoint,
            "--model",
            "moe-test",
            "--kt-method",
            "LLAMAFILE",
            "bench",
            "--tokens",
            "2",
            "--warmups",
            "1",
            "--repetitions",
            "2",
            "--input-tokens-file",
        ])
        .arg(&input_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
    std::fs::remove_file(output_path).unwrap();
    std::fs::remove_file(input_path).unwrap();
    assert_eq!(
        report["input"],
        json!({"kind": "token_ids", "value": [1, 2, 3]})
    );
    assert_eq!(report["samples"].as_array().unwrap().len(), 2);
    assert!(report["samples"][0].get("elapsed_ns").is_some());
    assert!(report["samples"][0].get("prefill_ns").is_none());
    assert!(report["samples"][0].get("decode_ns").is_none());
    let received = task.join().unwrap();
    assert_eq!(received.len(), 5);
    for request in &received[2..] {
        assert_eq!(request.body["input_ids"], json!([1, 2, 3]));
        assert!(request.body.get("text").is_none());
    }
}

#[test]
#[ignore = "requires a real local KTransformers/SGLang server and LI_KT_MODEL/LI_KT_METHOD"]
fn real_ktransformers_service_generates_repeatable_fixed_length_tokens() {
    let mut options = config(std::env::var("LI_KT_ENDPOINT").expect("set LI_KT_ENDPOINT"));
    options.expected_model = std::env::var("LI_KT_MODEL").expect("set LI_KT_MODEL");
    options.expected_method = std::env::var("LI_KT_METHOD").expect("set LI_KT_METHOD");
    options.timeout = Duration::from_secs(300);
    let mut backend = KTransformersBackend::connect(options).unwrap();
    let request = CompletionRequest {
        input: GenerationInput::Text("Explain memory bandwidth briefly.".into()),
        max_tokens: 8,
        mode: GenerationMode::FixedLength,
    };
    let first = backend.generate(&request).unwrap();
    let second = backend.generate(&request).unwrap();
    assert_eq!(first.generated_tokens, second.generated_tokens);
}
