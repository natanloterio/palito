//! Engine and C API tests. Model-backed tests need
//! models/SmolLM2-135M-Instruct-Q4_K_M.gguf (scripts/download_model.py).

use palito::{
    chat::Message,
    engine::{Device, Engine, EngineOptions, FinishReason, GenerationOptions},
    ffi::*,
    sampling::Sampling,
};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::PathBuf,
    ptr,
};

fn smollm() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
}

fn cpu() -> EngineOptions {
    EngineOptions {
        device: Device::Cpu,
        context_tokens: Some(512),
        ..EngineOptions::default()
    }
}

/// Serializes the tests that load a model on the GPU: the test harness runs
/// tests in parallel threads, and two Qwen3-30B-A3B engines do not fit in GPU
/// memory at once. Poison-tolerant so one failing test does not fail the rest.
#[cfg(feature = "cuda")]
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn capped(max_tokens: u32) -> GenerationOptions {
    GenerationOptions {
        max_tokens: Some(max_tokens),
        ..GenerationOptions::default()
    }
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn chat_streams_exactly_the_returned_text() {
    let mut engine = Engine::open(&smollm(), &cpu()).unwrap();
    assert_eq!((engine.context_tokens(), engine.profile()), (512, "cpu"));
    let messages = [Message::user("Write a sentence about the sea, with an emoji.")];
    let mut streamed = String::new();
    let reply = engine
        .chat(&messages, &capped(40), |piece| {
            streamed.push_str(piece);
            true
        })
        .unwrap();
    assert_eq!(streamed, reply.text);
    assert_eq!(reply.text, reply.text.trim());
    assert!(reply.generated_tokens > 0 && reply.generated_tokens <= 40);
    if reply.finish_reason == FinishReason::Length {
        assert_eq!(reply.generated_tokens, 40);
    }
    // Same request without a callback: same text (greedy, state reset per request).
    let again = engine.chat(&messages, &capped(40), |_| true).unwrap();
    assert_eq!(again.text, reply.text);
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn cancel_completion_and_full_context() {
    let mut engine = Engine::open(&smollm(), &cpu()).unwrap();
    let story = [Message::user("Write a long story about the sea.")];
    let mut pieces = 0;
    let cancelled = engine
        .chat(&story, &GenerationOptions::default(), |_| {
            pieces += 1;
            pieces < 3
        })
        .unwrap();
    assert_eq!(cancelled.finish_reason, FinishReason::Cancelled);
    assert_eq!(pieces, 3);

    let completion = engine
        .complete("The capital of France is", &capped(8), |_| true)
        .unwrap();
    assert_eq!(completion.finish_reason, FinishReason::Length);
    assert!(completion.text.contains("Paris"), "{}", completion.text);

    let mut small = Engine::open(
        &smollm(),
        &EngineOptions {
            context_tokens: Some(48),
            ..cpu()
        },
    )
    .unwrap();
    let full = small
        .complete("Once upon a time", &GenerationOptions::default(), |_| true)
        .unwrap();
    assert!(matches!(
        full.finish_reason,
        FinishReason::ContextFull | FinishReason::Stop
    ));
    let long = "word ".repeat(100);
    assert!(small.complete(&long, &capped(4), |_| true).is_err());
}

unsafe extern "C" fn collect(text: *const c_char, length: usize, user: *mut c_void) -> c_int {
    let out = unsafe { &mut *(user as *mut Vec<u8>) };
    out.extend_from_slice(unsafe { std::slice::from_raw_parts(text.cast::<u8>(), length) });
    0
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn c_api_chat_round_trip() {
    let path = CString::new(smollm().to_str().unwrap()).unwrap();
    let mut options = std::mem::MaybeUninit::uninit();
    let engine = unsafe {
        palito_options_init(options.as_mut_ptr());
        let mut options = options.assume_init();
        options.device = PALITO_DEVICE_CPU;
        options.context_tokens = 512;
        palito_open(path.as_ptr(), &options, ptr::null_mut())
    };
    assert!(!engine.is_null());
    unsafe {
        assert_eq!(palito_context_tokens(engine), 512);
        assert_eq!(CStr::from_ptr(palito_profile(engine)).to_str(), Ok("cpu"));
    }
    let (role, content) = (c"user", c"Say hello.");
    let messages = [PalitoMessage {
        role: role.as_ptr(),
        content: content.as_ptr(),
    }];
    let mut generation = std::mem::MaybeUninit::<PalitoGeneration>::uninit();
    unsafe { palito_generation_init(generation.as_mut_ptr()) };
    let mut generation = unsafe { generation.assume_init() };
    generation.max_tokens = 24;
    generation.enable_thinking = 0;
    let mut streamed: Vec<u8> = Vec::new();
    let mut result = std::mem::MaybeUninit::<PalitoResult>::uninit();
    let status = unsafe {
        palito_chat(
            engine,
            messages.as_ptr(),
            1,
            &generation,
            Some(collect),
            (&mut streamed as *mut Vec<u8>).cast(),
            result.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, 0);
    let mut result = unsafe { result.assume_init() };
    let text = unsafe { CStr::from_ptr(result.text) }.to_str().unwrap().to_owned();
    assert_eq!(String::from_utf8(streamed).unwrap(), text);
    assert!(result.generated_tokens > 0 && result.generated_tokens <= 24);
    unsafe {
        palito_result_free(&mut result);
        assert!(result.text.is_null());
        palito_close(engine);
    }
}

#[test]
fn c_api_reports_errors_without_a_model() {
    let mut error: *mut c_char = ptr::null_mut();
    let missing = c"/nonexistent/model.gguf";
    let engine = unsafe { palito_open(missing.as_ptr(), ptr::null(), &mut error) };
    assert!(engine.is_null());
    assert!(!error.is_null());
    unsafe { palito_string_free(error) };

    let mut error: *mut c_char = ptr::null_mut();
    assert!(unsafe { palito_open(ptr::null(), ptr::null(), &mut error) }.is_null());
    let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap().to_owned();
    unsafe { palito_string_free(error) };
    assert!(message.contains("model_path is NULL"), "{message}");

    let mut error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        palito_complete(
            ptr::null_mut(),
            c"x".as_ptr(),
            ptr::null(),
            None,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut error,
        )
    };
    assert_eq!(status, -1);
    unsafe { palito_string_free(error) };
    unsafe {
        palito_close(ptr::null_mut());
        palito_string_free(ptr::null_mut());
        assert_eq!(palito_context_tokens(ptr::null()), 0);
    }
    let version = unsafe { CStr::from_ptr(palito_version()) }.to_str().unwrap();
    assert_eq!(version, env!("CARGO_PKG_VERSION"));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA 13 and models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn cuda_engine_uses_the_trained_context_and_matches_cpu_text() {
    let _gpu = gpu_lock();
    let mut gpu = Engine::open(
        &smollm(),
        &EngineOptions {
            device: Device::Cuda,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    assert_eq!(gpu.profile(), "int8-fast");
    assert_eq!(gpu.context_tokens(), 8192);
    let completion = gpu
        .complete("The capital of France is", &capped(8), |_| true)
        .unwrap();
    assert!(completion.text.contains("Paris"), "{}", completion.text);
    assert!(completion.ttft_ns > 0);
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn stop_sequences_end_the_text_before_the_stop() {
    let mut engine = Engine::open(&smollm(), &cpu()).unwrap();
    let full = engine
        .complete("1, 2, 3, 4,", &capped(24), |_| true)
        .unwrap()
        .text;
    let cut = full.find(" 7").expect("the model counts past 7");
    let options = GenerationOptions {
        stop: vec![" 7".into(), "never appears".into()],
        ..capped(24)
    };
    let mut streamed = String::new();
    let stopped = engine
        .complete("1, 2, 3, 4,", &options, |piece| {
            streamed.push_str(piece);
            true
        })
        .unwrap();
    assert_eq!(stopped.text, full[..cut]);
    assert_eq!(streamed, stopped.text);
    assert_eq!(stopped.finish_reason, FinishReason::Stop);
    assert!(stopped.generated_tokens < 24);
    assert!(stopped.ttft_ns > 0);
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn sampling_is_seeded_and_near_zero_temperature_is_greedy() {
    let mut engine = Engine::open(&smollm(), &cpu()).unwrap();
    let messages = [Message::user("Tell me a short story about a robot.")];
    let greedy = engine.chat(&messages, &capped(40), |_| true).unwrap().text;
    let sampled = |engine: &mut Engine, temperature: f32, seed: u64| {
        let options = GenerationOptions {
            sampling: Sampling {
                temperature,
                top_p: 0.95,
                top_k: 40,
                seed: Some(seed),
                ..Sampling::GREEDY
            },
            ..capped(40)
        };
        let mut streamed = String::new();
        let reply = engine
            .chat(&messages, &options, |piece| {
                streamed.push_str(piece);
                true
            })
            .unwrap();
        assert_eq!(streamed, reply.text);
        reply.text
    };
    assert_eq!(sampled(&mut engine, 1e-4, 1), greedy);
    let a = sampled(&mut engine, 1.2, 7);
    assert_eq!(sampled(&mut engine, 1.2, 7), a);
    let others: Vec<String> = (8..12).map(|seed| sampled(&mut engine, 1.2, seed)).collect();
    assert!(others.iter().any(|text| *text != a), "seeds should change the text");
    let bad = GenerationOptions {
        sampling: Sampling {
            temperature: -1.0,
            ..Sampling::GREEDY
        },
        ..capped(4)
    };
    assert!(engine.chat(&messages, &bad, |_| true).is_err());
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn penalties_min_p_and_logit_bias_are_seeded_and_change_the_text() {
    let mut engine = Engine::open(&smollm(), &cpu()).unwrap();
    let prompt = "1, 1, 1, 1, 1, 1,";
    let run = |engine: &mut Engine, sampling: Sampling| {
        let options = GenerationOptions { sampling, ..capped(12) };
        engine.complete(prompt, &options, |_| true).unwrap().text
    };
    let greedy = run(&mut engine, Sampling::GREEDY);
    assert!(greedy.starts_with(" 1"), "{greedy:?}");
    // Neutral new fields leave greedy alone, whatever the window.
    let neutral = Sampling { repeat_last_n: 0, min_p: 0.3, ..Sampling::GREEDY };
    assert_eq!(run(&mut engine, neutral), greedy);
    // The prompt is inside the window: a strong frequency penalty avoids " 1".
    let penalized = Sampling { frequency_penalty: 2.0, ..Sampling::GREEDY };
    let text = run(&mut engine, penalized.clone());
    assert!(!text.starts_with(" 1"), "{text:?}");
    assert_eq!(run(&mut engine, penalized), text);
    let one = engine.tokenize(" 1").unwrap();
    let banned = Sampling {
        logit_bias: one.iter().map(|&t| (t as u32, f32::NEG_INFINITY)).collect(),
        ..Sampling::GREEDY
    };
    assert!(!run(&mut engine, banned).starts_with(" 1"));
    let seeded = |seed| Sampling {
        temperature: 1.0,
        min_p: 0.05,
        repeat_penalty: 1.3,
        presence_penalty: 0.5,
        seed: Some(seed),
        ..Sampling::GREEDY
    };
    assert_eq!(run(&mut engine, seeded(5)), run(&mut engine, seeded(5)));
    let out_of_vocabulary = Sampling { logit_bias: vec![(u32::MAX, 1.0)], ..Sampling::GREEDY };
    let options = GenerationOptions { sampling: out_of_vocabulary, ..capped(4) };
    assert!(engine.complete(prompt, &options, |_| true).is_err());
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn c_api_sampling_and_stop_fields() {
    let path = CString::new(smollm().to_str().unwrap()).unwrap();
    let mut options = std::mem::MaybeUninit::<PalitoOptions>::uninit();
    unsafe { palito_options_init(options.as_mut_ptr()) };
    let mut options = unsafe { options.assume_init() };
    options.device = PALITO_DEVICE_CPU;
    options.context_tokens = 256;
    let engine = unsafe { palito_open(path.as_ptr(), &options, ptr::null_mut()) };
    assert!(!engine.is_null());
    let run = |temperature: f32, seed: i64, stops: &[&CStr], bias: &[PalitoLogitBias]| {
        let pointers: Vec<*const c_char> = stops.iter().map(|s| s.as_ptr()).collect();
        let mut generation = std::mem::MaybeUninit::<PalitoGeneration>::uninit();
        unsafe { palito_generation_init(generation.as_mut_ptr()) };
        let mut generation = unsafe { generation.assume_init() };
        generation.max_tokens = 24;
        generation.temperature = temperature;
        generation.seed = seed;
        generation.stop = pointers.as_ptr();
        generation.stop_count = pointers.len();
        generation.logit_bias = bias.as_ptr();
        generation.logit_bias_count = bias.len();
        assert_eq!((generation.repeat_penalty, generation.repeat_last_n), (1.0, 64));
        let mut result = std::mem::MaybeUninit::<PalitoResult>::uninit();
        let prompt = c"1, 2, 3, 4,";
        let status = unsafe {
            palito_complete(engine, prompt.as_ptr(), &generation, None, ptr::null_mut(),
                result.as_mut_ptr(), ptr::null_mut())
        };
        assert_eq!(status, 0);
        let mut result = unsafe { result.assume_init() };
        let text = unsafe { CStr::from_ptr(result.text) }.to_str().unwrap().to_owned();
        let finish = result.finish_reason;
        unsafe { palito_result_free(&mut result) };
        (text, finish)
    };
    let (text, finish) = run(0.0, -1, &[c" 6"], &[]);
    assert!(!text.contains(" 6") && text.contains('5'), "{text:?}");
    assert_eq!(finish, PALITO_FINISH_STOP);
    assert_eq!(run(0.9, 3, &[], &[]), run(0.9, 3, &[], &[]));
    // Banning the tokens of " 5" moves the greedy reply off it.
    let five = Engine::open(&smollm(), &cpu()).unwrap().tokenize(" 5").unwrap();
    let banned: Vec<PalitoLogitBias> = five
        .iter()
        .map(|&token| PalitoLogitBias { token, bias: f32::NEG_INFINITY })
        .collect();
    let (text, _) = run(0.0, -1, &[], &banned);
    assert!(!text.starts_with(" 5"), "{text:?}");
    unsafe { palito_close(engine) };
}

/// The language bindings (bindings/*) mirror these layouts; Java uses the
/// offsets directly. Changing a struct means updating every binding.
#[test]
fn c_abi_layout_matches_the_bindings() {
    use std::mem::{offset_of, size_of};
    assert_eq!(size_of::<PalitoOptions>(), 32);
    assert_eq!(offset_of!(PalitoOptions, gpu_memory_limit_bytes), 16);
    assert_eq!(offset_of!(PalitoOptions, kernel_cache_dir), 24);
    assert_eq!(size_of::<PalitoMessage>(), 16);
    assert_eq!(size_of::<PalitoGeneration>(), 88);
    assert_eq!(offset_of!(PalitoGeneration, top_k), 16);
    assert_eq!(offset_of!(PalitoGeneration, seed), 24);
    assert_eq!(offset_of!(PalitoGeneration, stop), 32);
    assert_eq!(offset_of!(PalitoGeneration, stop_count), 40);
    assert_eq!(offset_of!(PalitoGeneration, min_p), 48);
    assert_eq!(offset_of!(PalitoGeneration, repeat_penalty), 52);
    assert_eq!(offset_of!(PalitoGeneration, repeat_last_n), 56);
    assert_eq!(offset_of!(PalitoGeneration, presence_penalty), 60);
    assert_eq!(offset_of!(PalitoGeneration, frequency_penalty), 64);
    assert_eq!(offset_of!(PalitoGeneration, logit_bias), 72);
    assert_eq!(offset_of!(PalitoGeneration, logit_bias_count), 80);
    assert_eq!(size_of::<PalitoLogitBias>(), 8);
    assert_eq!(size_of::<PalitoResult>(), 40);
    assert_eq!(offset_of!(PalitoResult, generated_tokens), 16);
    assert_eq!(offset_of!(PalitoResult, ttft_ms), 24);
    assert_eq!(offset_of!(PalitoResult, tokens_per_second), 32);
}

#[cfg(feature = "cuda")]
fn qwen3(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// (text, prompt tokens, cached tokens) for each prompt of a scripted chat:
/// three turns, then the second turn asked differently, then a new system
/// prompt. Replies are appended as the model gave them.
#[cfg(feature = "cuda")]
fn scripted_chat(engine: &mut Engine) -> Vec<(String, usize, usize)> {
    let options = GenerationOptions {
        max_tokens: Some(24),
        enable_thinking: false,
        ..GenerationOptions::default()
    };
    let mut results = Vec::new();
    let mut ask = |engine: &mut Engine, messages: &[Message]| {
        let completion = engine.chat(messages, &options, |_| true).unwrap();
        results.push((
            completion.text.clone(),
            completion.prompt_tokens,
            completion.cached_tokens,
        ));
        Message::new("assistant", completion.text)
    };
    let system = Message::system("You are a terse assistant. Answer in one short sentence.");
    let mut messages = vec![system, Message::user("Name a primary color.")];
    for question in ["Name another one.", "Which of the two is warmer?"] {
        let reply = ask(engine, &messages);
        messages.push(reply);
        messages.push(Message::user(question));
    }
    ask(engine, &messages);
    messages.truncate(3);
    messages.push(Message::user("Name a fruit of that color instead."));
    ask(engine, &messages);
    ask(
        engine,
        &[
            Message::system("Reply in French."),
            Message::user("Name a color."),
        ],
    );
    results
}

#[cfg(feature = "cuda")]
fn assert_prefix_cache_is_exact(model: &str, profiles: &[(palito::cuda::CudaMatmul, bool)]) {
    for &(matmul, graph) in profiles {
        let open = |prefix_cache| {
            Engine::open(
                &qwen3(model),
                &EngineOptions {
                    device: Device::Cuda,
                    context_tokens: Some(1024),
                    batch_tokens: Some(64),
                    prefill_chunk: Some(16),
                    cuda_matmul: Some(matmul),
                    decode_graph: Some(graph),
                    prefix_cache,
                    ..EngineOptions::default()
                },
            )
            .unwrap()
        };
        let uncached = scripted_chat(&mut open(false));
        let cached = scripted_chat(&mut open(true));
        let label = format!("{matmul:?} graph={graph}");
        for ((text, prompt, hit), (expected, expected_prompt, miss)) in cached.iter().zip(&uncached)
        {
            assert_eq!(text, expected, "{label}: cached reply differs");
            assert_eq!(
                prompt, expected_prompt,
                "{label}: prompt_tokens must be the full prompt"
            );
            assert_eq!(*miss, 0, "{label}: disabled cache reused tokens");
            assert!(hit % 16 == 0 && hit < prompt, "{label}: {hit} of {prompt}");
        }
        let hits: Vec<usize> = cached.iter().map(|r| r.2).collect();
        assert_eq!(hits[0], 0, "{label}: first prompt");
        assert!(hits[1] > 0 && hits[2] > hits[1], "{label}: turns {hits:?}");
        assert!(
            hits[3] > 0 && hits[3] < hits[2],
            "{label}: diverging prompt {hits:?}"
        );
        assert_eq!(hits[4], 0, "{label}: unrelated prompt {hits:?}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA 13 and models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf"]
fn cuda_prefix_cache_hit_partial_and_miss_keep_greedy_output() {
    let _gpu = gpu_lock();
    use palito::cuda::CudaMatmul;
    assert_prefix_cache_is_exact(
        "compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf",
        &[
            (CudaMatmul::Int8Fast, true),
            (CudaMatmul::Int8Fast, false),
            (CudaMatmul::F32Tiled, true),
            (CudaMatmul::F32, false),
            (CudaMatmul::F32, true),
        ],
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA 13, ~20 GB of GPU memory and Qwen3-30B-A3B-Q4_K_M"]
fn cuda_prefix_cache_keeps_qwen3_30b_greedy_output() {
    let _gpu = gpu_lock();
    assert_prefix_cache_is_exact(
        "benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
        &[(palito::cuda::CudaMatmul::Int8Fast, true)],
    );
}

/// Time to first token of the second and third turns of a chat with ~10k tokens
/// of history, with and without the prefix cache (Palito's chat defaults).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "benchmark: requires CUDA 13, ~20 GB of GPU memory and Qwen3-30B-A3B-Q4_K_M"]
fn cuda_prefix_cache_speeds_up_long_conversations() {
    let _gpu = gpu_lock();
    let document: String = (0..480)
        .map(|i| {
            format!(
                "Record {i}: sensor {} read {} at site {}.\n",
                i * 7 % 113,
                i * 31 % 997,
                i % 17
            )
        })
        .collect();
    let options = GenerationOptions {
        max_tokens: Some(16),
        enable_thinking: false,
        ..GenerationOptions::default()
    };
    let run = |prefix_cache| {
        let mut engine = Engine::open(
            &qwen3("benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf"),
            &EngineOptions {
                device: Device::Cuda,
                context_tokens: Some(16384),
                prefix_cache,
                ..EngineOptions::default()
            },
        )
        .unwrap();
        let mut messages = vec![Message::user(format!(
            "{document}\nWhich site appears most?"
        ))];
        let mut turns = Vec::new();
        for question in [
            "Which sensor read the highest value?",
            "Summarize in five words.",
            "Thanks.",
        ] {
            let completion = engine.chat(&messages, &options, |_| true).unwrap();
            turns.push((
                completion.prompt_tokens,
                completion.cached_tokens,
                completion.ttft_ns,
                completion.text.clone(),
            ));
            messages.push(Message::new("assistant", completion.text));
            messages.push(Message::user(question));
        }
        turns
    };
    let cold = run(false);
    let warm = run(true);
    for (index, (cold, warm)) in cold.iter().zip(&warm).enumerate() {
        eprintln!(
            "turn {}: prompt {} tokens, cached {}, ttft {:.1} ms -> {:.1} ms ({:.2}x)",
            index + 1,
            warm.0,
            warm.1,
            cold.2 as f64 / 1e6,
            warm.2 as f64 / 1e6,
            cold.2 as f64 / warm.2 as f64
        );
        assert_eq!(cold.3, warm.3, "turn {} text differs", index + 1);
    }
    for turn in &warm[1..] {
        assert!(turn.0 > 9000, "history too short: {}", turn.0);
    }
    let speedup = cold[1].2 as f64 / warm[1].2 as f64;
    assert!(speedup >= 3.0, "second-turn speedup {speedup:.2}x");
}

#[cfg(feature = "cuda")]
fn json_schemas() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::json;
    vec![
        ("Describe a fictional person.", json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": false
        })),
        ("List three fruits.", json!({
            "type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3
        })),
        ("What colour is the sky on a clear day?", json!({"enum": ["red", "green", "blue"]})),
        ("Rate the movie Inception.", json!({
            "type": "object",
            "properties": {
                "liked": {"type": "boolean"},
                "score": {"type": "number"},
                "comment": {"type": ["string", "null"]}
            },
            "required": ["liked", "score", "comment"],
            "additionalProperties": false
        })),
        ("Describe Paris.", json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "population": {"type": "integer"}},
                    "required": ["name", "population"],
                    "additionalProperties": false
                },
                "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "capital": {"type": "boolean"}
            },
            "required": ["city", "landmarks", "capital"],
            "additionalProperties": false
        })),
    ]
}

/// Runs every schema greedy and sampled; returns the constrained decode rates.
#[cfg(feature = "cuda")]
fn check_json_replies(engine: &mut Engine, max_tokens: u32) -> Vec<f64> {
    let mut rates = Vec::new();
    for (prompt, schema) in json_schemas() {
        let validator = jsonschema::validator_for(&schema).unwrap();
        for temperature in [0.0, 0.8] {
            let options = GenerationOptions {
                max_tokens: Some(max_tokens),
                sampling: Sampling { temperature, seed: Some(5), ..Sampling::GREEDY },
                json_schema: Some(palito::grammar::JsonSchema::new(schema.clone()).unwrap()),
                ..GenerationOptions::default()
            };
            let reply = engine.chat(&[Message::user(prompt)], &options, |_| true).unwrap();
            assert_eq!(reply.finish_reason, FinishReason::Stop, "{}", reply.text);
            let value: serde_json::Value = serde_json::from_str(&reply.text)
                .unwrap_or_else(|e| panic!("{:?} is not JSON: {e}", reply.text));
            assert!(validator.is_valid(&value), "{} violates {schema}", reply.text);
            eprintln!(
                "t={temperature} {} tokens, {:.1} tok/s: {}",
                reply.generated_tokens, reply.tokens_per_second, reply.text
            );
            rates.push(reply.tokens_per_second);
        }
    }
    rates
}

#[test]
#[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn json_schema_replies_validate_and_the_c_api_takes_a_schema() {
    let mut engine = Engine::open(&smollm(), &cpu()).unwrap();
    // A small model drifts on unbounded strings; bounded reruns live in tests/serve.rs.
    let options = GenerationOptions {
        max_tokens: Some(64),
        json_schema: Some(palito::grammar::JsonSchema::parse(r#"{"enum": ["yes", "no"]}"#).unwrap()),
        ..GenerationOptions::default()
    };
    let reply = engine.chat(&[Message::user("Is water wet?")], &options, |_| true).unwrap();
    assert!(reply.text == r#""yes""# || reply.text == r#""no""#, "{}", reply.text);
    // Penalties and bias act on the grammar-masked logits; the reply stays valid.
    let penalized = GenerationOptions {
        sampling: Sampling {
            temperature: 0.8,
            seed: Some(2),
            min_p: 0.05,
            repeat_penalty: 1.3,
            frequency_penalty: 1.0,
            ..Sampling::GREEDY
        },
        ..options.clone()
    };
    let reply = engine.chat(&[Message::user("Is water wet?")], &penalized, |_| true).unwrap();
    assert!(reply.text == r#""yes""# || reply.text == r#""no""#, "{}", reply.text);
    drop(engine);

    let path = CString::new(smollm().to_str().unwrap()).unwrap();
    let mut open = std::mem::MaybeUninit::<PalitoOptions>::uninit();
    let engine = unsafe {
        palito_options_init(open.as_mut_ptr());
        let mut open = open.assume_init();
        open.device = PALITO_DEVICE_CPU;
        open.context_tokens = 512;
        palito_open(path.as_ptr(), &open, ptr::null_mut())
    };
    assert!(!engine.is_null());
    let mut generation = std::mem::MaybeUninit::<PalitoGeneration>::uninit();
    let mut generation = unsafe {
        palito_generation_init(generation.as_mut_ptr());
        generation.assume_init()
    };
    generation.max_tokens = 64;
    let role = CString::new("user").unwrap();
    let content = CString::new("Is water wet?").unwrap();
    let messages = [PalitoMessage { role: role.as_ptr(), content: content.as_ptr() }];
    let schema = CString::new(r#"{"type": "boolean"}"#).unwrap();
    let mut result = std::mem::MaybeUninit::<PalitoResult>::zeroed();
    let status = unsafe {
        palito_chat_json(engine, messages.as_ptr(), 1, &generation, schema.as_ptr(), None,
            ptr::null_mut(), result.as_mut_ptr(), ptr::null_mut())
    };
    assert_eq!(status, 0);
    let mut result = unsafe { result.assume_init() };
    let text = unsafe { CStr::from_ptr(result.text) }.to_str().unwrap().to_owned();
    assert!(text == "true" || text == "false", "{text}");
    unsafe { palito_result_free(&mut result) };

    let unsupported = CString::new(r#"{"type": "array", "uniqueItems": true}"#).unwrap();
    let mut error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        palito_complete_json(engine, content.as_ptr(), &generation, unsupported.as_ptr(), None,
            ptr::null_mut(), ptr::null_mut(), &mut error)
    };
    assert_eq!(status, -1);
    let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap().to_owned();
    assert!(message.contains("uniqueItems"), "{message}");
    unsafe {
        palito_string_free(error);
        palito_close(engine);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA 13 and models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf"]
fn qwen3_30b_json_schema_replies_validate() {
    // Two 30B models do not fit in 32 GB: serialize with the other GPU tests.
    let _gpu = gpu_lock();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf");
    let mut engine = Engine::open(
        &path,
        &EngineOptions { device: Device::Cuda, context_tokens: Some(4096), ..EngineOptions::default() },
    )
    .unwrap();
    let rates = check_json_replies(&mut engine, 400);
    let free = GenerationOptions { max_tokens: Some(200), enable_thinking: false, ..GenerationOptions::default() };
    let greedy = engine.chat(&[Message::user("Describe Paris.")], &free, |_| true).unwrap();
    let sampled = GenerationOptions {
        sampling: Sampling { temperature: 0.8, seed: Some(5), ..Sampling::GREEDY },
        ..free.clone()
    };
    let sampled = engine.chat(&[Message::user("Describe Paris.")], &sampled, |_| true).unwrap();
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    eprintln!(
        "constrained mean {mean:.1} tok/s (min {:.1}); unconstrained greedy {:.1}, sampled {:.1}",
        rates.iter().copied().fold(f64::INFINITY, f64::min),
        greedy.tokens_per_second,
        sampled.tokens_per_second
    );
}
