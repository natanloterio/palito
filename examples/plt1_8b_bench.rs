//! Full 32-layer Llama 3.1 8B `PLT1 HYB Q=5` (2.0 bpw, folded RHT, 4 fused
//! launches/layer: QKV, O, GateUp, Down+SwiGLU) real-VRAM cold-L2 benchmark
//! for `k = 1`, `k = 8`, and `k = 16` verification steps on RTX 5090 (SM_120).
//!
//! Allocates all 32 distinct layers (~1.95 GB of real HYB Q=5 2.0 bpw tiles) in
//! GPU VRAM so every layer streams cold from GDDR7, matching real 8B execution.

#[cfg(not(all(feature = "cuda", feature = "qtip-research")))]
fn main() {
    eprintln!("plt1-8b-bench requires --features cuda,qtip-research");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", feature = "qtip-research"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::{
        driver::{CudaContext, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg, sys::CUevent_flags},
        nvrtc::{CompileOptions, compile_ptx_with_opts},
    };
    use palito::qtip::hyb::{HybCodebook, hyb_tile_words};

    const LAYERS: usize = 32;
    const H: usize = 4096;
    const KV: usize = 1024;
    const FFN: usize = 14336;
    const PAIR_BITS: u32 = 4; // 2.0 bpw (64B/tile)

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let wpt = hyb_tile_words(PAIR_BITS);

    // Compile HYB Q=5 (L=12, B=4 = 2.0 bpw) variants for folded RHT:
    // - skip_in=1, skip_out=0 (QKV, GateUp: residual input already rotated!)
    // - skip_in=0, skip_out=1 (O, Down+SwiGLU: residual output stays rotated!)
    let compile = |skip_in: usize, skip_out: usize| -> Result<_, Box<dyn std::error::Error>> {
        let src = format!(
            "#define HYB_L 12\n#define HYB_B {PAIR_BITS}\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
             #define HYB_SKIP_IN_RHT {skip_in}\n#define HYB_SKIP_OUT_RHT {skip_out}\n\
             #define HYB_VEC_K 8\n{}",
            include_str!("../src/qtip/hyb_matvec.cu")
        );
        let ptx = compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )?;
        let module = ctx.load_module(ptx)?;
        Ok((
            module.load_function("hyb_small_k_multi")?,
            module.load_function("hyb_small_k_multi_swiglu")?,
        ))
    };

    let (fn_in_folded, _) = compile(1, 0)?;
    let (fn_out_folded, fn_down_swiglu) = compile(0, 1)?;

    let cb = HybCodebook::gaussian_for_q(5);
    let d_lut: CudaSlice<u32> = stream.clone_htod(&cb.lut_f16x2)?;

    // Allocate 32 distinct layers of Llama 3.1 8B in VRAM (real ~1.75 GB footprint).
    struct LayerWeights {
        q: CudaSlice<u32>,
        k: CudaSlice<u32>,
        v: CudaSlice<u32>,
        o: CudaSlice<u32>,
        gate: CudaSlice<u32>,
        up: CudaSlice<u32>,
        down: CudaSlice<u32>,
    }

    let alloc_mat = |rows: usize, cols: usize| -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let words = (rows / 16) * (cols / 16) * wpt;
        Ok(stream.alloc_zeros::<u32>(words)?)
    };

    let mut layers = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        layers.push(LayerWeights {
            q: alloc_mat(H, H)?,
            k: alloc_mat(KV, H)?,
            v: alloc_mat(KV, H)?,
            o: alloc_mat(H, H)?,
            gate: alloc_mat(FFN, H)?,
            up: alloc_mat(FFN, H)?,
            down: alloc_mat(H, FFN)?,
        });
    }
    let total_mb = (LAYERS * (H * H + 2 * KV * H + H * H + 3 * FFN * H) * PAIR_BITS as usize / 2 / 8) as f64 / 1e6;
    println!("Allocated 32 distinct Llama 3.1 8B PLT1 HYB Q=5 (2.0 bpw) layers in VRAM: {:.1} MB", total_mb);

    let d_signs: CudaSlice<f32> = stream.clone_htod(&vec![1.0f32; FFN])?;
    let d_x: CudaSlice<f32> = stream.alloc_zeros(16 * H)?;
    let mut d_q: CudaSlice<f32> = stream.alloc_zeros(16 * H)?;
    let mut d_k: CudaSlice<f32> = stream.alloc_zeros(16 * KV)?;
    let mut d_v: CudaSlice<f32> = stream.alloc_zeros(16 * KV)?;
    let mut d_o: CudaSlice<f32> = stream.alloc_zeros(16 * H)?;
    let mut d_gate: CudaSlice<f32> = stream.alloc_zeros(16 * FFN)?;
    let mut d_up: CudaSlice<f32> = stream.alloc_zeros(16 * FFN)?;
    let mut d_down: CudaSlice<f32> = stream.alloc_zeros(16 * H)?;

    let lut_ptr = d_lut.device_ptr(&stream).0;
    let signs_ptr = d_signs.device_ptr(&stream).0;
    let x_ptr = d_x.device_ptr(&stream).0;
    let q_ptr = d_q.device_ptr(&stream).0;
    let k_ptr = d_k.device_ptr(&stream).0;
    let v_ptr = d_v.device_ptr(&stream).0;
    let o_ptr = d_o.device_ptr(&stream).0;
    let gate_ptr = d_gate.device_ptr(&stream).0;
    let up_ptr = d_up.device_ptr(&stream).0;
    let down_ptr = d_down.device_ptr(&stream).0;
    let scale = 0.05f32 / 16.0;

    for tokens in [4usize, 6, 8, 16] {
        let tokens_i = tokens as i32;
        let run_32_layers = || -> Result<(), Box<dyn std::error::Error>> {
            for l in &layers {
                // 1. QKV fused (1 launch, skip_in=1, skip_out=0)
                {
                    let rb_total = (H + KV + KV) / 256;
                    let chunks = H / 256;
                    let splits = 340usize.div_ceil(rb_total).clamp(1, chunks);
                    let cps = chunks.div_ceil(splits);
                    let splits = chunks.div_ceil(cps);
                    let (t0, t1, t2) = (
                        l.q.device_ptr(&stream).0,
                        l.k.device_ptr(&stream).0,
                        l.v.device_ptr(&stream).0,
                    );
                    let (r0, r1, r2) = (H as i32, KV as i32, KV as i32);
                    let (rb0, rb1) = ((H / 256) as i32, (KV / 256) as i32);
                    let cols_i = H as i32;
                    let cps_i = cps as i32;
                    let zero_u64 = 0u64;
                    let mut b = stream.launch_builder(&fn_in_folded);
                    b.arg(&t0).arg(&t1).arg(&t2);
                    b.arg(&lut_ptr).arg(&lut_ptr).arg(&lut_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&q_ptr).arg(&k_ptr).arg(&v_ptr);
                    b.arg(&scale).arg(&scale).arg(&scale);
                    b.arg(&r0).arg(&r1).arg(&r2);
                    b.arg(&rb0).arg(&rb1).arg(&x_ptr).arg(&zero_u64);
                    b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (rb_total as u32, splits as u32, 1),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    }
                }
                // 2. O_proj (1 launch, skip_in=0, skip_out=1)
                {
                    let rb_total = H / 256;
                    let chunks = H / 256;
                    let splits = 340usize.div_ceil(rb_total).clamp(1, chunks);
                    let cps = chunks.div_ceil(splits);
                    let splits = chunks.div_ceil(cps);
                    let t0 = l.o.device_ptr(&stream).0;
                    let r0 = H as i32;
                    let rb0 = rb_total as i32;
                    let cols_i = H as i32;
                    let cps_i = cps as i32;
                    let zero_u64 = 0u64;
                    let mut b = stream.launch_builder(&fn_out_folded);
                    b.arg(&t0).arg(&t0).arg(&t0);
                    b.arg(&lut_ptr).arg(&lut_ptr).arg(&lut_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&o_ptr).arg(&o_ptr).arg(&o_ptr);
                    b.arg(&scale).arg(&scale).arg(&scale);
                    b.arg(&r0).arg(&r0).arg(&r0);
                    b.arg(&rb0).arg(&rb0).arg(&q_ptr).arg(&zero_u64);
                    b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (rb_total as u32, splits as u32, 1),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    }
                }
                // 3. Gate+Up fused (1 launch, skip_in=1, skip_out=0)
                {
                    let rb_total = (2 * FFN) / 256;
                    let chunks = H / 256;
                    let splits = 340usize.div_ceil(rb_total).clamp(1, chunks);
                    let cps = chunks.div_ceil(splits);
                    let splits = chunks.div_ceil(cps);
                    let (t0, t1) = (l.gate.device_ptr(&stream).0, l.up.device_ptr(&stream).0);
                    let r0 = FFN as i32;
                    let rb0 = (FFN / 256) as i32;
                    let cols_i = H as i32;
                    let cps_i = cps as i32;
                    let zero_u64 = 0u64;
                    let mut b = stream.launch_builder(&fn_in_folded);
                    b.arg(&t0).arg(&t1).arg(&t1);
                    b.arg(&lut_ptr).arg(&lut_ptr).arg(&lut_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&gate_ptr).arg(&up_ptr).arg(&up_ptr);
                    b.arg(&scale).arg(&scale).arg(&scale);
                    b.arg(&r0).arg(&r0).arg(&r0);
                    b.arg(&rb0).arg(&rb0).arg(&x_ptr).arg(&zero_u64);
                    b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (rb_total as u32, splits as u32, 1),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    }
                }
                // 4. Down+SwiGLU fused (1 launch, skip_in=0, skip_out=1)
                {
                    let rb_total = H / 256;
                    let chunks = FFN / 256;
                    let splits = 340usize.div_ceil(rb_total).clamp(1, chunks);
                    let cps = chunks.div_ceil(splits);
                    let splits = chunks.div_ceil(cps);
                    let t0 = l.down.device_ptr(&stream).0;
                    let r0 = H as i32;
                    let rb0 = rb_total as i32;
                    let cols_i = FFN as i32;
                    let cps_i = cps as i32;
                    let mut b = stream.launch_builder(&fn_down_swiglu);
                    b.arg(&t0).arg(&t0).arg(&t0);
                    b.arg(&lut_ptr).arg(&lut_ptr).arg(&lut_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&signs_ptr).arg(&signs_ptr).arg(&signs_ptr);
                    b.arg(&down_ptr).arg(&down_ptr).arg(&down_ptr);
                    b.arg(&scale).arg(&scale).arg(&scale);
                    b.arg(&r0).arg(&r0).arg(&r0);
                    b.arg(&rb0).arg(&rb0).arg(&gate_ptr).arg(&up_ptr);
                    b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (rb_total as u32, splits as u32, 1),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    }
                }
            }
            Ok(())
        };

        for _ in 0..3 {
            run_32_layers()?;
        }
        stream.synchronize()?;
        let mut samples = Vec::new();
        for _ in 0..7 {
            let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for _ in 0..5 {
                run_32_layers()?;
            }
            let stop = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            stop.synchronize()?;
            samples.push(f64::from(start.elapsed_ms(&stop)?) / 5.0);
        }
        samples.sort_by(f64::total_cmp);
        let ms = samples[samples.len() / 2];
        let bw = (total_mb * 1e6) / (ms * 1e6);
        println!(
            "k={tokens:2} | 32-layer 8B projections: {ms:.2} ms ({bw:.0} GB/s) | w/ tau=3.0 -> {:.0} tok/s | w/ tau=4.5 -> {:.0} tok/s",
            3000.0 / (ms + 0.6),
            4500.0 / (ms + 1.4),
        );
    }
    let _ = (&mut d_q, &mut d_k, &mut d_v, &mut d_o, &mut d_gate, &mut d_up, &mut d_down);
    Ok(())
}
