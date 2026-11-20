use super::*;
use cudarc::driver::{DevicePtr, DevicePtrMut};

type Init = unsafe extern "C" fn(*mut i32, *mut usize, *mut usize) -> i32;
type Pack = unsafe extern "C" fn(sys::CUstream, u64, u64, u64, i32, i32) -> i32;
type Project =
    unsafe extern "C" fn(sys::CUstream, u64, u64, u64, u64, u64, u64, i32, i32, i32, i32) -> i32;
const DELTA_ELEMENTS: usize = 512 * 8192;
const JOINED: bool = cfg!(feature = "cuda-llama-ffn-mmq-joined-research");

pub(super) fn supports_chunk(n: i32) -> bool {
    n == 512 || (JOINED && matches!(n, 46 | 256))
}

pub(super) fn supports_attention(weight: &DeviceTensor, rows: i32) -> bool {
    cfg!(feature = "cuda-llama-attn-mmq-joined-research")
        && (weight.kind, weight.width, weight.rows) == (8, 2048, rows)
        && matches!(rows, 512 | 2048)
}

pub(super) struct Bridge {
    _library: libloading::Library,
    pack: Pack,
    project: Project,
    sm: i32,
    packed: usize,
    fixup: usize,
}

fn cuda_status(stage: &str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(failure(format!("FFN MMQ {stage}: CUDA status {status}")))
    }
}

impl Bridge {
    pub(super) fn open(context: &CudaContext) -> Result<Self> {
        context.bind_to_thread().map_err(failure)?;
        let library =
            unsafe { libloading::Library::new("libli_ffn_mmq_research.so") }.map_err(failure)?;
        let (init, pack, project) = unsafe {
            (
                *library.get::<Init>(b"li_ffn_mmq_init\0").map_err(failure)?,
                *library
                    .get::<Pack>(if JOINED {
                        b"li_ffn_mmq_pack_joined\0"
                    } else {
                        b"li_ffn_mmq_pack_corrected\0"
                    })
                    .map_err(failure)?,
                *library
                    .get::<Project>(if JOINED {
                        b"li_ffn_mmq_project_joined\0"
                    } else {
                        b"li_ffn_mmq_project_corrected\0"
                    })
                    .map_err(failure)?,
            )
        };
        let (mut sm, mut packed, mut fixup) = (0, 0, 0);
        cuda_status("init", unsafe { init(&mut sm, &mut packed, &mut fixup) })?;
        if sm != 170 || packed != 4_737_024 || fixup != 11_141_120 {
            return Err(invalid("unexpected RTX 5090 FFN MMQ workspace geometry"));
        }
        Ok(Self {
            _library: library,
            pack,
            project,
            sm,
            packed,
            fixup,
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        let delta = if JOINED {
            2 * DELTA_ELEMENTS
        } else {
            DELTA_ELEMENTS
        };
        (2 * self.packed + self.fixup + delta * size_of::<f32>()) as u64
    }

    pub(super) fn allocate(self, s: &Arc<CudaStream>) -> Result<ReplayKernel> {
        Ok(ReplayKernel {
            packed: s
                .alloc_zeros(if JOINED { 2 * self.packed } else { self.packed })
                .map_err(failure)?,
            residual: (!JOINED)
                .then(|| s.alloc_zeros(self.packed))
                .transpose()
                .map_err(failure)?,
            fixup: s.alloc_zeros(self.fixup).map_err(failure)?,
            delta: s
                .alloc_zeros(if JOINED {
                    2 * DELTA_ELEMENTS
                } else {
                    DELTA_ELEMENTS
                })
                .map_err(failure)?,
            stream: s.clone(),
            bridge: self,
        })
    }
}

pub(super) struct ReplayKernel {
    packed: CudaSlice<u8>,
    residual: Option<CudaSlice<u8>>,
    fixup: CudaSlice<u8>,
    delta: CudaSlice<f32>,
    stream: Arc<CudaStream>,
    bridge: Bridge,
}

impl Drop for ReplayKernel {
    fn drop(&mut self) {
        // Keep the dynamic CUDA module loaded until all its work has finished.
        if let Err(error) = self.stream.synchronize() {
            eprintln!("FFN MMQ synchronization on drop: {error}");
        }
    }
}

impl ReplayKernel {
    pub(super) fn pack(
        &mut self,
        s: &CudaStream,
        x: &CudaSlice<f32>,
        width: i32,
        n: i32,
    ) -> Result<()> {
        if !supports_chunk(n) || !matches!(width, 2048 | 8192) || x.len() < (n * width) as usize {
            return Err(invalid("invalid FFN MMQ activation shape"));
        }
        let (input, _read) = x.device_ptr(s);
        let (packed, _write) = self.packed.device_ptr_mut(s);
        let residual = self.residual.as_mut().map(|p| p.device_ptr_mut(s));
        cuda_status("pack", unsafe {
            (self.bridge.pack)(
                s.cu_stream(),
                input,
                packed,
                residual.as_ref().map_or(0, |p| p.0),
                width,
                n,
            )
        })
    }

    pub(super) fn project(
        &mut self,
        s: &CudaStream,
        weight: &DeviceTensor,
        y: &mut CudaSlice<f32>,
        n: i32,
    ) -> Result<()> {
        let valid_shape = matches!(
            (weight.kind, weight.width, weight.rows),
            (8, 2048, 8192) | (8, 8192, 2048)
        ) || (n == 512 && supports_attention(weight, weight.rows));
        if !valid_shape
            || weight.bytes.len() < (weight.width * weight.rows / 32 * 34) as usize
            || !supports_chunk(n)
            || y.len() < (n * weight.rows) as usize
        {
            return Err(invalid("invalid FFN MMQ projection shape"));
        }
        let (w, _read_w) = weight.bytes.device_ptr(s);
        let (packed, _read_x) = self.packed.device_ptr(s);
        let residual = self.residual.as_ref().map(|p| p.device_ptr(s));
        let (output, _write_y) = y.device_ptr_mut(s);
        let (fixup, _write_tmp) = self.fixup.device_ptr_mut(s);
        let (delta, _write_delta) = self.delta.device_ptr_mut(s);
        cuda_status("projection", unsafe {
            (self.bridge.project)(
                s.cu_stream(),
                w,
                packed,
                residual.as_ref().map_or(0, |p| p.0),
                output,
                fixup,
                delta,
                weight.width,
                weight.rows,
                n,
                self.bridge.sm,
            )
        })
    }
}
