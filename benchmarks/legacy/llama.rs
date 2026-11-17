use crate::{
    Config, Error, Result,
    backend::{Backend, Capabilities, ModelInfo},
};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    marker::PhantomData,
    path::Path,
    ptr::{self, NonNull},
    rc::Rc,
};

#[repr(C)]
#[derive(Default)]
struct NativeInfo {
    weight_bytes: u64,
    parameters: u64,
    tensor_type_mask: u64,
    context_tokens: u32,
    batch_tokens: u32,
    vocab_tokens: u32,
}

unsafe extern "C" {
    fn li_error() -> *const c_char;
    fn li_open(
        path: *const c_char,
        context: u32,
        batch: u32,
        threads: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn li_close(handle: *mut c_void);
    fn li_get_info(handle: *mut c_void, out: *mut NativeInfo) -> i32;
    fn li_tokenize(
        handle: *mut c_void,
        text: *const c_char,
        length: i32,
        out: *mut i32,
        capacity: i32,
        count: *mut i32,
    ) -> i32;
    fn li_detokenize(
        handle: *mut c_void,
        tokens: *const i32,
        count: i32,
        out: *mut c_char,
        capacity: i32,
        length: *mut i32,
    ) -> i32;
    fn li_reset(handle: *mut c_void) -> i32;
    fn li_eval(handle: *mut c_void, tokens: *const i32, count: i32, position: i32) -> i32;
    fn li_sample_greedy(handle: *mut c_void, token: *mut i32) -> i32;
    fn li_is_eog(handle: *mut c_void, token: i32) -> i32;
}

fn checked(status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: the bridge returns a thread-local, NUL-terminated error string.
    let message = unsafe { CStr::from_ptr(li_error()) }
        .to_string_lossy()
        .into_owned();
    Err(Error::Backend(message))
}

fn length(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Invalid("buffer exceeds i32::MAX".into()))
}

struct Handle(NonNull<c_void>);

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: this is the unique owner of a successfully opened handle.
        unsafe { li_close(self.0.as_ptr()) };
    }
}

pub struct LlamaBackend {
    handle: Handle,
    info: ModelInfo,
    _single_owner: PhantomData<Rc<()>>,
}

impl LlamaBackend {
    pub fn open(path: &Path, config: &Config) -> Result<Self> {
        config.validate()?;
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::Invalid("model path contains NUL".into()))?;
        let mut raw = ptr::null_mut();
        // SAFETY: path is terminated, the output pointer is writable, and all
        // scalar values were checked before crossing the ABI.
        checked(unsafe {
            li_open(
                path.as_ptr(),
                config.context_tokens,
                config.batch_tokens,
                config.threads as i32,
                &mut raw,
            )
        })?;
        let handle =
            Handle(NonNull::new(raw).ok_or_else(|| Error::Backend("null model handle".into()))?);
        let mut info = NativeInfo::default();
        // SAFETY: handle is live and NativeInfo matches the bridge's C layout.
        checked(unsafe { li_get_info(handle.0.as_ptr(), &mut info) })?;
        let tensor_types = [
            (0, "F32"),
            (1, "F16"),
            (6, "Q5_0"),
            (8, "Q8_0"),
            (12, "Q4_K"),
            (14, "Q6_K"),
        ]
        .into_iter()
        .filter(|(bit, _)| info.tensor_type_mask & (1_u64 << bit) != 0)
        .map(|(_, name)| name.into())
        .collect();
        Ok(Self {
            handle,
            info: ModelInfo {
                backend: "llama.cpp-cpu".into(),
                revision: env!("LI_LLAMA_REVISION").into(),
                architecture: "llama".into(),
                tensor_types,
                weight_bytes: info.weight_bytes,
                parameters: info.parameters,
                context_tokens: info.context_tokens,
                batch_tokens: info.batch_tokens,
                vocab_tokens: info.vocab_tokens,
                capabilities: Capabilities {
                    token_step: true,
                    chunked_prefill: true,
                },
            },
            _single_owner: PhantomData,
        })
    }
}

impl Backend for LlamaBackend {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let mut count = 0;
        let size = length(text.len())?;
        // SAFETY: inputs are live for the calls; the first call sizes the
        // output and the second writes into a vector of that exact capacity.
        unsafe {
            checked(li_tokenize(
                self.handle.0.as_ptr(),
                text.as_ptr().cast(),
                size,
                ptr::null_mut(),
                0,
                &mut count,
            ))?;
            let mut tokens = vec![
                0;
                usize::try_from(count)
                    .map_err(|_| Error::Backend("negative token count".into()))?
            ];
            checked(li_tokenize(
                self.handle.0.as_ptr(),
                text.as_ptr().cast(),
                size,
                tokens.as_mut_ptr(),
                count,
                &mut count,
            ))?;
            tokens.truncate(count as usize);
            Ok(tokens)
        }
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let mut count = 0;
        let size = length(tokens.len())?;
        // SAFETY: both calls receive live token slices; only the second writes
        // text, into a buffer sized by the first call.
        unsafe {
            checked(li_detokenize(
                self.handle.0.as_ptr(),
                tokens.as_ptr(),
                size,
                ptr::null_mut(),
                0,
                &mut count,
            ))?;
            let mut text = vec![
                0_u8;
                usize::try_from(count)
                    .map_err(|_| Error::Backend("negative text length".into()))?
            ];
            checked(li_detokenize(
                self.handle.0.as_ptr(),
                tokens.as_ptr(),
                size,
                text.as_mut_ptr().cast(),
                count,
                &mut count,
            ))?;
            text.truncate(count as usize);
            String::from_utf8(text)
                .map_err(|error| Error::Backend(format!("generated text is not UTF-8: {error}")))
        }
    }

    fn reset(&mut self) -> Result<()> {
        // SAFETY: handle is live and mutably borrowed.
        checked(unsafe { li_reset(self.handle.0.as_ptr()) })
    }

    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        let count = length(tokens.len())?;
        let position = i32::try_from(position)
            .map_err(|_| Error::Invalid("position exceeds i32::MAX".into()))?;
        // SAFETY: tokens are valid for count entries, and the handle is unique.
        checked(unsafe { li_eval(self.handle.0.as_ptr(), tokens.as_ptr(), count, position) })
    }

    fn sample_greedy(&mut self) -> Result<i32> {
        let mut token = 0;
        // SAFETY: output is writable and the handle is live and unique.
        checked(unsafe { li_sample_greedy(self.handle.0.as_ptr(), &mut token) })?;
        Ok(token)
    }

    fn is_eog(&self, token: i32) -> bool {
        // SAFETY: the model vocabulary belongs to the live handle.
        unsafe { li_is_eog(self.handle.0.as_ptr(), token) != 0 }
    }
}
