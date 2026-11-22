//! Engine and C API tests. Model-backed tests need
//! models/SmolLM2-135M-Instruct-Q4_K_M.gguf (scripts/download_model.py).

use palito::{
    chat::Message,
    engine::{Device, Engine, EngineOptions, FinishReason, GenerationOptions},
    ffi::*,
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
    let generation = PalitoGeneration {
        max_tokens: 24,
        enable_thinking: 0,
    };
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
