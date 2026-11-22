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
fn c_api_sampling_and_stop_fields() {
    let path = CString::new(smollm().to_str().unwrap()).unwrap();
    let mut options = std::mem::MaybeUninit::<PalitoOptions>::uninit();
    unsafe { palito_options_init(options.as_mut_ptr()) };
    let mut options = unsafe { options.assume_init() };
    options.device = PALITO_DEVICE_CPU;
    options.context_tokens = 256;
    let engine = unsafe { palito_open(path.as_ptr(), &options, ptr::null_mut()) };
    assert!(!engine.is_null());
    let run = |temperature: f32, seed: i64, stops: &[&CStr]| {
        let pointers: Vec<*const c_char> = stops.iter().map(|s| s.as_ptr()).collect();
        let mut generation = std::mem::MaybeUninit::<PalitoGeneration>::uninit();
        unsafe { palito_generation_init(generation.as_mut_ptr()) };
        let mut generation = unsafe { generation.assume_init() };
        generation.max_tokens = 24;
        generation.temperature = temperature;
        generation.seed = seed;
        generation.stop = pointers.as_ptr();
        generation.stop_count = pointers.len();
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
    let (text, finish) = run(0.0, -1, &[c" 6"]);
    assert!(!text.contains(" 6") && text.contains('5'), "{text:?}");
    assert_eq!(finish, PALITO_FINISH_STOP);
    assert_eq!(run(0.9, 3, &[]), run(0.9, 3, &[]));
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
    assert_eq!(size_of::<PalitoGeneration>(), 48);
    assert_eq!(offset_of!(PalitoGeneration, top_k), 16);
    assert_eq!(offset_of!(PalitoGeneration, seed), 24);
    assert_eq!(offset_of!(PalitoGeneration, stop), 32);
    assert_eq!(offset_of!(PalitoGeneration, stop_count), 40);
    assert_eq!(size_of::<PalitoResult>(), 40);
    assert_eq!(offset_of!(PalitoResult, generated_tokens), 16);
    assert_eq!(offset_of!(PalitoResult, ttft_ms), 24);
    assert_eq!(offset_of!(PalitoResult, tokens_per_second), 32);
}
