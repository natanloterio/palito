//! C API over `engine::Engine`; declared in `include/palito.h`.
//!
//! Conventions: functions return 0 on success and -1 on error. On error, when
//! `error` is not NULL, `*error` receives a message to release with
//! `palito_string_free`. An engine handle must not be used from two threads at
//! once. Panics never cross the boundary: they become errors.
//!
//! ABI: struct layouts never change (`c_abi_layout_matches_the_bindings` in
//! tests/engine.rs); new options come as new functions, so programs built
//! against an older header keep working (`palito_chat_json` and
//! `palito_complete_json` add the JSON Schema argument).

use crate::{
    Error,
    chat::Message,
    engine::{Completion, Device, Engine, EngineOptions, FinishReason, GenerationOptions},
    grammar::JsonSchema,
    sampling::Sampling,
};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    ptr,
};

pub const PALITO_DEVICE_DEFAULT: c_int = 0;
pub const PALITO_DEVICE_CPU: c_int = 1;
pub const PALITO_DEVICE_CUDA: c_int = 2;
pub const PALITO_DEVICE_VULKAN: c_int = 3;
pub const PALITO_DEVICE_METAL: c_int = 4;

pub const PALITO_FINISH_STOP: c_int = 0;
pub const PALITO_FINISH_LENGTH: c_int = 1;
pub const PALITO_FINISH_CONTEXT_FULL: c_int = 2;
pub const PALITO_FINISH_CANCELLED: c_int = 3;

#[repr(C)]
pub struct PalitoOptions {
    /// PALITO_DEVICE_*.
    pub device: c_int,
    /// 0: the model's trained context (CUDA: fitted to free device memory).
    pub context_tokens: u32,
    /// 0: 4 CPU threads.
    pub threads: u32,
    pub gpu_device: u32,
    /// 0: free device memory minus 2 GiB.
    pub gpu_memory_limit_bytes: u64,
    /// NULL: $PALITO_CUDA_KERNEL_CACHE or ~/.cache/palito/kernels.
    pub kernel_cache_dir: *const c_char,
}

#[repr(C)]
pub struct PalitoMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
pub struct PalitoGeneration {
    /// 0: until end of turn or a full context.
    pub max_tokens: u32,
    /// Nonzero: reasoning on for templates that support it (Qwen3).
    pub enable_thinking: c_int,
    /// 0: greedy.
    pub temperature: f32,
    /// Nucleus mass in (0, 1]; 1: off.
    pub top_p: f32,
    /// 0: off.
    pub top_k: u32,
    /// Negative: a different seed every call.
    pub seed: i64,
    /// `stop_count` NUL-terminated stop sequences (may be NULL when 0).
    pub stop: *const *const c_char,
    pub stop_count: usize,
}

#[repr(C)]
pub struct PalitoResult {
    /// UTF-8, NUL-terminated; released by palito_result_free.
    pub text: *mut c_char,
    /// PALITO_FINISH_*.
    pub finish_reason: c_int,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub ttft_ms: f64,
    pub tokens_per_second: f64,
}

/// Receives `length` bytes of UTF-8 text (not NUL-terminated, never splitting a
/// character); returns 0 to continue, nonzero to stop generation.
pub type PalitoTextCallback =
    Option<unsafe extern "C" fn(text: *const c_char, length: usize, user_data: *mut c_void) -> c_int>;

pub struct PalitoEngine(Engine);

static VERSION: &CStr = match CStr::from_bytes_with_nul(
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes(),
) {
    Ok(version) => version,
    Err(_) => panic!("version contains NUL"),
};

#[unsafe(no_mangle)]
pub extern "C" fn palito_version() -> *const c_char {
    VERSION.as_ptr()
}

/// Fills `options` with defaults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_options_init(options: *mut PalitoOptions) {
    if let Some(options) = unsafe { options.as_mut() } {
        *options = PalitoOptions {
            device: PALITO_DEVICE_DEFAULT,
            context_tokens: 0,
            threads: 0,
            gpu_device: 0,
            gpu_memory_limit_bytes: 0,
            kernel_cache_dir: ptr::null(),
        };
    }
}

/// Fills `generation` with defaults (no token cap, thinking on, greedy, no stops).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_generation_init(generation: *mut PalitoGeneration) {
    if let Some(generation) = unsafe { generation.as_mut() } {
        *generation = PalitoGeneration {
            max_tokens: 0,
            enable_thinking: 1,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: -1,
            stop: ptr::null(),
            stop_count: 0,
        };
    }
}

/// Opens a GGUF model. `options` may be NULL for defaults. Returns NULL on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_open(
    model_path: *const c_char,
    options: *const PalitoOptions,
    error: *mut *mut c_char,
) -> *mut PalitoEngine {
    let opened = guard(|| {
        let path = PathBuf::from(unsafe { text(model_path, "model_path") }?);
        let options = match unsafe { options.as_ref() } {
            Some(options) => unsafe { engine_options(options) }?,
            None => EngineOptions::default(),
        };
        Engine::open(&path, &options)
    });
    match opened {
        Ok(engine) => Box::into_raw(Box::new(PalitoEngine(engine))),
        Err(failure) => {
            unsafe { report(error, &failure) };
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_close(engine: *mut PalitoEngine) {
    if !engine.is_null() {
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// Context tokens actually opened; 0 for NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_context_tokens(engine: *const PalitoEngine) -> u32 {
    unsafe { engine.as_ref() }.map_or(0, |engine| engine.0.context_tokens())
}

fn profile_cstr(profile: Option<&str>) -> &'static CStr {
    match profile {
        Some("cpu") => c"cpu",
        Some("int8-fast") => c"int8-fast",
        Some("int8-prefill") => c"int8-prefill",
        Some("f32-tiled") => c"f32-tiled",
        Some("f32") => c"f32",
        Some("vulkan") => c"vulkan",
        Some("metal") => c"metal",
        Some("cuda") => c"cuda",
        _ => c"",
    }
}

/// Numerical profile in use ("cpu", "metal", "vulkan", "int8-fast", "int8-prefill",
/// "f32-tiled", "f32" or "cuda"); static string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_profile(engine: *const PalitoEngine) -> *const c_char {
    profile_cstr(unsafe { engine.as_ref() }.map(|engine| engine.0.profile())).as_ptr()
}

/// Generates the assistant reply to `messages` (rendered with the model's chat
/// template). `generation` may be NULL for defaults; `callback` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_chat(
    engine: *mut PalitoEngine,
    messages: *const PalitoMessage,
    message_count: usize,
    generation: *const PalitoGeneration,
    callback: PalitoTextCallback,
    user_data: *mut c_void,
    result: *mut PalitoResult,
    error: *mut *mut c_char,
) -> c_int {
    unsafe {
        palito_chat_json(
            engine,
            messages,
            message_count,
            generation,
            ptr::null(),
            callback,
            user_data,
            result,
            error,
        )
    }
}

/// palito_chat whose reply is JSON matching `json_schema` (JSON Schema text;
/// `{"type":"object"}` for any object; NULL: unconstrained). Reasoning is off.
/// An unsupported schema fails before generation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_chat_json(
    engine: *mut PalitoEngine,
    messages: *const PalitoMessage,
    message_count: usize,
    generation: *const PalitoGeneration,
    json_schema: *const c_char,
    callback: PalitoTextCallback,
    user_data: *mut c_void,
    result: *mut PalitoResult,
    error: *mut *mut c_char,
) -> c_int {
    let outcome = guard(|| {
        let engine = unsafe { engine_mut(engine) }?;
        if messages.is_null() && message_count > 0 {
            return Err(Error::Invalid("messages is NULL".into()));
        }
        let messages = if message_count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(messages, message_count) }
        };
        let messages = messages
            .iter()
            .map(|m| {
                Ok(Message::new(
                    &unsafe { text(m.role, "role") }?,
                    unsafe { text(m.content, "content") }?,
                ))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let options = unsafe { generation_options(generation, json_schema) }?;
        engine.0.chat(&messages, &options, stream_to(callback, user_data))
    });
    unsafe { finish(outcome, result, error) }
}

/// Continues `prompt` without a chat template. Arguments as in palito_chat.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_complete(
    engine: *mut PalitoEngine,
    prompt: *const c_char,
    generation: *const PalitoGeneration,
    callback: PalitoTextCallback,
    user_data: *mut c_void,
    result: *mut PalitoResult,
    error: *mut *mut c_char,
) -> c_int {
    unsafe {
        palito_complete_json(
            engine,
            prompt,
            generation,
            ptr::null(),
            callback,
            user_data,
            result,
            error,
        )
    }
}

/// palito_complete whose continuation is JSON matching `json_schema` (as in
/// palito_chat_json).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_complete_json(
    engine: *mut PalitoEngine,
    prompt: *const c_char,
    generation: *const PalitoGeneration,
    json_schema: *const c_char,
    callback: PalitoTextCallback,
    user_data: *mut c_void,
    result: *mut PalitoResult,
    error: *mut *mut c_char,
) -> c_int {
    let outcome = guard(|| {
        let engine = unsafe { engine_mut(engine) }?;
        let prompt = unsafe { text(prompt, "prompt") }?;
        let options = unsafe { generation_options(generation, json_schema) }?;
        engine.0.complete(&prompt, &options, stream_to(callback, user_data))
    });
    unsafe { finish(outcome, result, error) }
}

/// Releases the text of a result filled by palito_chat or palito_complete.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_result_free(result: *mut PalitoResult) {
    if let Some(result) = unsafe { result.as_mut() } {
        unsafe { palito_string_free(result.text) };
        result.text = ptr::null_mut();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn palito_string_free(string: *mut c_char) {
    if !string.is_null() {
        drop(unsafe { CString::from_raw(string) });
    }
}

fn guard<T>(work: impl FnOnce() -> crate::Result<T>) -> crate::Result<T> {
    catch_unwind(AssertUnwindSafe(work))
        .unwrap_or_else(|_| Err(Error::Backend("internal panic".into())))
}

unsafe fn text(pointer: *const c_char, name: &str) -> crate::Result<String> {
    if pointer.is_null() {
        return Err(Error::Invalid(format!("{name} is NULL")));
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| Error::Invalid(format!("{name} is not UTF-8")))
}

unsafe fn engine_mut<'a>(engine: *mut PalitoEngine) -> crate::Result<&'a mut PalitoEngine> {
    unsafe { engine.as_mut() }.ok_or_else(|| Error::Invalid("engine is NULL".into()))
}

unsafe fn engine_options(options: &PalitoOptions) -> crate::Result<EngineOptions> {
    let defaults = EngineOptions::default();
    let device = match options.device {
        PALITO_DEVICE_DEFAULT => defaults.device,
        PALITO_DEVICE_CPU => Device::Cpu,
        PALITO_DEVICE_CUDA => Device::Cuda,
        PALITO_DEVICE_VULKAN => Device::Vulkan,
        PALITO_DEVICE_METAL => Device::Metal,
        other => return Err(Error::Invalid(format!("unknown device {other}"))),
    };
    let kernel_cache_dir = if options.kernel_cache_dir.is_null() {
        None
    } else {
        Some(PathBuf::from(unsafe {
            text(options.kernel_cache_dir, "kernel_cache_dir")
        }?))
    };
    Ok(EngineOptions {
        device,
        context_tokens: (options.context_tokens > 0).then_some(options.context_tokens),
        threads: if options.threads == 0 {
            defaults.threads
        } else {
            options.threads
        },
        gpu_device: options.gpu_device as usize,
        gpu_memory_limit_bytes: (options.gpu_memory_limit_bytes > 0)
            .then_some(options.gpu_memory_limit_bytes),
        kernel_cache_dir,
        ..defaults
    })
}

unsafe fn generation_options(
    generation: *const PalitoGeneration,
    json_schema: *const c_char,
) -> crate::Result<GenerationOptions> {
    let json_schema = if json_schema.is_null() {
        None
    } else {
        Some(JsonSchema::parse(&unsafe { text(json_schema, "json_schema") }?)?)
    };
    let Some(generation) = (unsafe { generation.as_ref() }) else {
        return Ok(GenerationOptions {
            json_schema,
            ..GenerationOptions::default()
        });
    };
    if generation.stop.is_null() && generation.stop_count > 0 {
        return Err(Error::Invalid("stop is NULL".into()));
    }
    let stop = (0..generation.stop_count)
        .map(|i| unsafe { text(*generation.stop.add(i), "stop sequence") })
        .collect::<crate::Result<Vec<_>>>()?;
    Ok(GenerationOptions {
        max_tokens: (generation.max_tokens > 0).then_some(generation.max_tokens),
        enable_thinking: generation.enable_thinking != 0,
        sampling: Sampling {
            temperature: generation.temperature,
            top_p: generation.top_p,
            top_k: generation.top_k,
            seed: u64::try_from(generation.seed).ok(),
        },
        stop,
        json_schema,
    })
}

fn stream_to(callback: PalitoTextCallback, user_data: *mut c_void) -> impl FnMut(&str) -> bool {
    move |piece: &str| match callback {
        None => true,
        Some(callback) => {
            unsafe { callback(piece.as_ptr().cast(), piece.len(), user_data) == 0 }
        }
    }
}

unsafe fn finish(
    outcome: crate::Result<Completion>,
    result: *mut PalitoResult,
    error: *mut *mut c_char,
) -> c_int {
    let completion = match outcome {
        Ok(completion) => completion,
        Err(failure) => {
            unsafe { report(error, &failure) };
            return -1;
        }
    };
    if let Some(result) = unsafe { result.as_mut() } {
        *result = PalitoResult {
            text: c_string(&completion.text).into_raw(),
            finish_reason: match completion.finish_reason {
                FinishReason::Stop => PALITO_FINISH_STOP,
                FinishReason::Length => PALITO_FINISH_LENGTH,
                FinishReason::ContextFull => PALITO_FINISH_CONTEXT_FULL,
                FinishReason::Cancelled => PALITO_FINISH_CANCELLED,
            },
            prompt_tokens: completion.prompt_tokens as u32,
            generated_tokens: completion.generated_tokens as u32,
            ttft_ms: completion.ttft_ns as f64 / 1e6,
            tokens_per_second: completion.tokens_per_second,
        };
    }
    0
}

unsafe fn report(error: *mut *mut c_char, failure: &Error) {
    if let Some(slot) = unsafe { error.as_mut() } {
        *slot = c_string(&failure.to_string()).into_raw();
    }
}

/// Model text may contain NUL bytes; C strings cannot.
fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', "")).expect("NUL bytes removed")
}

#[cfg(test)]
mod tests {
    use super::profile_cstr;

    #[test]
    fn profile_cstr_maps_all_backend_profiles() {
        for name in [
            "cpu",
            "metal",
            "vulkan",
            "int8-fast",
            "int8-prefill",
            "f32-tiled",
            "f32",
            "cuda",
        ] {
            assert_eq!(profile_cstr(Some(name)).to_str().unwrap(), name);
        }
        assert_eq!(profile_cstr(None).to_str().unwrap(), "");
    }
}
