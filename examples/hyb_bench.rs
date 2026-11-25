//! Benchmark & CPU-vs-GPU correctness check for QTIP HYB (L=16, Q=9, V=2)
//! 32x bank-replicated SMEM LUT decode (1.5 / 2.0 / 2.5 / 3.0 bpw) and
//! multi-vector verification (k = 4, 8, 16 tokens) on RTX 5090 (SM_120).
//!
//!     cargo run --release --features cuda,qtip-research --example hyb-bench

#[cfg(not(all(feature = "cuda", feature = "qtip-research")))]
fn main() {
    eprintln!("hyb-bench requires the cuda and qtip-research features");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", feature = "qtip-research"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench::run()
}

#[cfg(all(feature = "cuda", feature = "qtip-research"))]
mod bench {
    use clap::Parser;
    use cudarc::{
        driver::{
            CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
            sys::{CUevent_flags, CUfunction_attribute_enum},
        },
        nvrtc::{CompileOptions, compile_ptx_with_opts},
    };
    use half::f16;
    use palito::qtip::hyb::{HybCodebook, decode_hyb_matrix, decode_hyb_matrix_mma, hyb_tile_words};
    use std::sync::Arc;

    type Error = Box<dyn std::error::Error>;
    const STATIC_SHARED_LIMIT: u32 = 48 * 1024;
    const COLD_BYTES: usize = 384 << 20;

    #[derive(Parser)]
    struct Args {
        #[arg(long, default_value = "4096x4096,16384x4096,4096x16384")]
        shapes: String,
        /// Bits per pair (3=1.5bpw, 4=2.0bpw, 5=2.5bpw, 6=3.0bpw).
        #[arg(long, default_value_t = 4)]
        pair_bits: u32,
        /// LUT index bits Q (5=32 entries in warp registers/0 SMEM, 6=64 entries/8KB SMEM, 9=512 entries/64KB SMEM).
        #[arg(long, default_value_t = 5)]
        lut_bits: u32,
        #[arg(long, default_value_t = 4)]
        unroll: u32,
        #[arg(long, default_value_t = 340)]
        fused_blocks: usize,
        #[arg(long, default_value_t = 100)]
        launches: usize,
        /// Skip input/output RHT (measures offline-folded QuaRot/SpinQuant projections).
        #[arg(long)]
        skip_rht: bool,
        /// Multi-vector batch size k for verification kernel (e.g. 4, 8, 16).
        #[arg(long, default_value_t = 8)]
        vec_k: u32,
        /// Reuse one weight copy (warm L2), matching llama.cpp test-backend-ops perf.
        #[arg(long)]
        warm: bool,
    }

    fn random_u32(count: usize, seed: u64) -> Vec<u32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) as u32
            })
            .collect()
    }

    fn uniform(count: usize, seed: u64) -> Vec<f32> {
        random_u32(count, seed)
            .iter()
            .map(|&v| v as f32 / u32::MAX as f32 * 2.0 - 1.0)
            .collect()
    }

    fn parse_shapes(text: &str) -> Result<Vec<(usize, usize)>, Error> {
        text.split(',')
            .map(|s| {
                let (r, c) = s.split_once('x').ok_or("shape must be ROWSxCOLS")?;
                Ok((r.trim().parse()?, c.trim().parse()?))
            })
            .collect()
    }

    fn opt_in(function: &CudaFunction, bytes: u32) -> Result<(), Error> {
        if bytes > STATIC_SHARED_LIMIT {
            function.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes as i32,
            )?;
        }
        Ok(())
    }

    fn fwht256_cpu(v: &mut [f32]) {
        for chunk in v.chunks_exact_mut(256) {
            let mut span = 1;
            while span < 256 {
                for base in (0..256).step_by(span * 2) {
                    for i in 0..span {
                        let a = chunk[base + i];
                        let b = chunk[base + i + span];
                        chunk[base + i] = a + b;
                        chunk[base + i + span] = a - b;
                    }
                }
                span <<= 1;
            }
        }
    }

    fn time(
        stream: &Arc<CudaStream>,
        count: usize,
        mut launch: impl FnMut(usize) -> Result<(), Error>,
    ) -> Result<f64, Error> {
        for i in 0..10 {
            launch(i)?;
        }
        stream.synchronize()?;
        let mut samples = Vec::with_capacity(7);
        for _ in 0..7 {
            let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for i in 0..count {
                launch(i)?;
            }
            let stop = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            stop.synchronize()?;
            let ms = start.elapsed_ms(&stop)?;
            samples.push(f64::from(ms) * 1000.0 / count as f64);
        }
        samples.sort_by(f64::total_cmp);
        Ok(samples[samples.len() / 2])
    }

    pub fn run() -> Result<(), Error> {
        let args = Args::parse();
        let shapes = parse_shapes(&args.shapes)?;
        let skip = if args.skip_rht { 1 } else { 0 };
        let source = format!(
            "#define HYB_B {}\n#define HYB_Q {}\n#define HYB_UNROLL {}\n#define HYB_SKIP_IN_RHT {}\n#define HYB_SKIP_OUT_RHT {}\n#define HYB_VEC_K {}\n{}",
            args.pair_bits,
            args.lut_bits,
            args.unroll,
            skip,
            skip,
            args.vec_k,
            include_str!("../src/qtip/hyb_matvec.cu")
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )?;
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_module(ptx)?;
        let stream = ctx.default_stream();
        let fused_fn = module.load_function("hyb_matvec_fused")?;
        let multivec_fn = module.load_function("hyb_matvec_multivec")?;
        let mma_fn = module.load_function("hyb_matvec_mma")?;

        let cb = HybCodebook::gaussian_for_q(args.lut_bits);
        let d_lut: CudaSlice<u32> = stream.clone_htod(&cb.lut_f16x2)?;
        let lut_smem_bytes: usize = if args.lut_bits == 5 {
            0
        } else {
            (1usize << args.lut_bits) * 32 * 4
        };

        // 1. CPU-vs-GPU correctness check on 512x512.
        {
            let (rows, cols) = (512usize, 512usize);
            let wpt = hyb_tile_words(args.pair_bits);
            let tile_words = random_u32((rows / 16) * (cols / 16) * wpt, 42);
            let x = uniform(cols, 100);
            let col_signs: Vec<f32> = random_u32(cols, 200)
                .iter()
                .map(|&v| if v & 1 == 0 { 1.0 } else { -1.0 })
                .collect();
            let row_signs: Vec<f32> = random_u32(rows, 300)
                .iter()
                .map(|&v| if v & 1 == 0 { 1.0 } else { -1.0 })
                .collect();
            let scale = if args.skip_rht { 0.05f32 } else { 0.05f32 / 256.0 };

            let d_tiles = stream.clone_htod(&tile_words)?;
            let d_x = stream.clone_htod(&x)?;
            let d_cs = stream.clone_htod(&col_signs)?;
            let d_rs = stream.clone_htod(&row_signs)?;
            let mut d_y: CudaSlice<f32> = stream.alloc_zeros(rows)?;

            let col_tiles = cols / 16;
            let splits = 2usize;
            let tps = col_tiles.div_ceil(splits);
            let smem = (lut_smem_bytes + (tps * 16 + 256) * 4) as u32;
            opt_in(&fused_fn, smem)?;
            let cfg = LaunchConfig {
                grid_dim: ((rows / 256) as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: smem,
            };
            unsafe {
                let mut b = stream.launch_builder(&fused_fn);
                b.arg(&d_tiles);
                b.arg(&d_lut);
                b.arg(&d_x);
                b.arg(&d_cs);
                b.arg(&d_rs);
                b.arg(&mut d_y);
                let cols_i = cols as i32;
                let tps_i = tps as i32;
                b.arg(&cols_i);
                b.arg(&tps_i);
                b.arg(&scale);
                b.launch(cfg)?;
            }
            let gpu_y = stream.clone_dtoh(&d_y)?;

            // CPU reference.
            let w_dense = decode_hyb_matrix(&tile_words, rows, cols, args.pair_bits, &cb, 1.0);
            let mut x_rot = x.clone();
            if !args.skip_rht {
                for i in 0..cols {
                    x_rot[i] *= col_signs[i];
                }
                fwht256_cpu(&mut x_rot);
            }
            let mut cpu_y = vec![0.0f32; rows];
            for r in 0..rows {
                let mut s = 0.0f32;
                for c in 0..cols {
                    s += w_dense[r * cols + c] * x_rot[c];
                }
                cpu_y[r] = s;
            }
            if !args.skip_rht {
                fwht256_cpu(&mut cpu_y);
                for r in 0..rows {
                    cpu_y[r] *= row_signs[r] * scale;
                }
            } else {
                for r in 0..rows {
                    cpu_y[r] *= scale;
                }
            }
            let max_err = gpu_y
                .iter()
                .zip(&cpu_y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_ref = cpu_y.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
            let rel_err = max_err / max_ref;
            assert!(rel_err < 1e-5, "HYB fused CPU/GPU mismatch: rel_err={rel_err:.2e}");

            // Check multivec correctness on k tokens.
            let k_vec = args.vec_k as usize;
            let x_mv = uniform(k_vec * cols, 777);
            let d_x_mv = stream.clone_htod(&x_mv)?;
            let mut d_y_mv: CudaSlice<f32> = stream.alloc_zeros(k_vec * rows)?;
            let smem_mv = lut_smem_bytes as u32;
            opt_in(&multivec_fn, smem_mv)?;
            let cfg_mv = LaunchConfig {
                grid_dim: ((rows / 256) as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: smem_mv,
            };
            unsafe {
                let mut b = stream.launch_builder(&multivec_fn);
                b.arg(&d_tiles);
                b.arg(&d_lut);
                b.arg(&d_x_mv);
                b.arg(&mut d_y_mv);
                let rows_i = rows as i32;
                let cols_i = cols as i32;
                let tps_i = tps as i32;
                b.arg(&rows_i);
                b.arg(&cols_i);
                b.arg(&tps_i);
                b.arg(&scale);
                b.launch(cfg_mv)?;
            }
            let gpu_mv = stream.clone_dtoh(&d_y_mv)?;
            let mut max_mv_err = 0.0f32;
            let mut max_mv_ref = 1e-6f32;
            for t in 0..k_vec {
                for r in 0..rows {
                    let mut s = 0.0f32;
                    for c in 0..cols {
                        s += w_dense[r * cols + c] * x_mv[t * cols + c];
                    }
                    let expected = s * scale;
                    max_mv_err = max_mv_err.max((gpu_mv[t * rows + r] - expected).abs());
                    max_mv_ref = max_mv_ref.max(expected.abs());
                }
            }
            let mv_rel = max_mv_err / max_mv_ref;
            assert!(mv_rel < 1e-5, "HYB multivec CPU/GPU mismatch: rel_err={mv_rel:.2e}");

            // Check Tensor-Core MMA multivec correctness (k multiple of 8).
            let mut mma_rel = 0.0f32;
            if k_vec % 8 == 0 {
                let x_h2: Vec<u32> = x_mv
                    .chunks_exact(2)
                    .map(|p| {
                        u32::from(f16::from_f32(p[0]).to_bits())
                            | (u32::from(f16::from_f32(p[1]).to_bits()) << 16)
                    })
                    .collect();
                let x_f16_ref: Vec<f32> = x_mv.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
                let d_x_h2 = stream.clone_htod(&x_h2)?;
                let mut d_y_mma: CudaSlice<f32> = stream.alloc_zeros(k_vec * rows)?;
                opt_in(&mma_fn, smem_mv)?;
                unsafe {
                    let mut b = stream.launch_builder(&mma_fn);
                    b.arg(&d_tiles);
                    b.arg(&d_lut);
                    b.arg(&d_x_h2);
                    b.arg(&mut d_y_mma);
                    let rows_i = rows as i32;
                    let cols_i = cols as i32;
                    let tps_i = tps as i32;
                    b.arg(&rows_i);
                    b.arg(&cols_i);
                    b.arg(&tps_i);
                    b.arg(&scale);
                    b.launch(cfg_mv)?;
                }
                let gpu_mma = stream.clone_dtoh(&d_y_mma)?;
                let w_mma = decode_hyb_matrix_mma(&tile_words, rows, cols, args.pair_bits, &cb, 1.0);
                let mut max_e = 0.0f32;
                let mut max_r = 1e-6f32;
                for t in 0..k_vec {
                    for r in 0..rows {
                        let mut s = 0.0f32;
                        for c in 0..cols {
                            s += w_mma[r * cols + c] * x_f16_ref[t * cols + c];
                        }
                        let expected = s * scale;
                        max_e = max_e.max((gpu_mma[t * rows + r] - expected).abs());
                        max_r = max_r.max(expected.abs());
                    }
                }
                mma_rel = max_e / max_r;
                assert!(mma_rel < 1e-5, "HYB MMA CPU/GPU mismatch: rel_err={mma_rel:.2e}");
            }

            println!(
                "HYB L16 Q={} B={} ({:.1} bpw), skip_rht={}, vec_k={} | CPU/GPU PASS (fused={:.1e}, scalar_mv={:.1e}, mma={:.1e})",
                args.lut_bits,
                args.pair_bits,
                args.pair_bits as f32 / 2.0,
                args.skip_rht,
                args.vec_k,
                rel_err,
                mv_rel,
                mma_rel
            );
        }

        println!(
            "shape | bytes (MB) | B=1 fused us (GB/s) | k={} MMA us (us/tok, eff GB/s, vs B=1)",
            args.vec_k
        );

        let wpt = hyb_tile_words(args.pair_bits);
        for (rows, cols) in shapes {
            let num_tiles = (rows / 16) * (cols / 16);
            let tile_bytes = num_tiles * wpt * 4;
            let copies = if args.warm {
                1
            } else {
                (COLD_BYTES / tile_bytes).max(1)
            };
            let one_copy = random_u32(num_tiles * wpt, 999);
            let mut d_copies = Vec::with_capacity(copies);
            for _ in 0..copies {
                d_copies.push(stream.clone_htod(&one_copy)?);
            }
            let d_x = stream.clone_htod(&uniform(cols, 11))?;
            let d_cs = stream.clone_htod(&vec![1.0f32; cols])?;
            let d_rs = stream.clone_htod(&vec![1.0f32; rows])?;
            let mut d_y: CudaSlice<f32> = stream.alloc_zeros(rows)?;

            let row_blocks = rows / 256;
            let col_tiles = cols / 16;
            let splits = (args.fused_blocks / row_blocks).max(1).min(col_tiles);
            let tps = col_tiles.div_ceil(splits);
            let max_chunks = (tps * 16).div_ceil(256) + 1;
            let smem = (lut_smem_bytes + max_chunks * 256 * 4) as u32;
            opt_in(&fused_fn, smem)?;
            let cfg = LaunchConfig {
                grid_dim: (row_blocks as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: smem,
            };
            let cols_i = cols as i32;
            let tps_i = tps as i32;
            let scale = 1.0f32 / 256.0;

            let fused_us = time(&stream, args.launches, |i| {
                let tiles = &d_copies[i % copies];
                unsafe {
                    let mut b = stream.launch_builder(&fused_fn);
                    b.arg(tiles);
                    b.arg(&d_lut);
                    b.arg(&d_x);
                    b.arg(&d_cs);
                    b.arg(&d_rs);
                    b.arg(&mut d_y);
                    b.arg(&cols_i);
                    b.arg(&tps_i);
                    b.arg(&scale);
                    b.launch(cfg)?;
                }
                Ok(())
            })?;

            // Tensor-Core MMA multi-vector verification benchmark (k tokens).
            let k_vec = args.vec_k as usize;
            let x_h2_bench: Vec<u32> = random_u32(k_vec * (cols / 2), 22);
            let d_x_h2 = stream.clone_htod(&x_h2_bench)?;
            let mut d_y_mv: CudaSlice<f32> = stream.alloc_zeros(k_vec * rows)?;
            let smem_mv = lut_smem_bytes as u32;
            opt_in(&mma_fn, smem_mv)?;
            let cfg_mv = LaunchConfig {
                grid_dim: (row_blocks as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: smem_mv,
            };
            let rows_i = rows as i32;
            let mv_us = time(&stream, args.launches, |i| {
                let tiles = &d_copies[i % copies];
                unsafe {
                    let mut b = stream.launch_builder(&mma_fn);
                    b.arg(tiles);
                    b.arg(&d_lut);
                    b.arg(&d_x_h2);
                    b.arg(&mut d_y_mv);
                    b.arg(&rows_i);
                    b.arg(&cols_i);
                    b.arg(&tps_i);
                    b.arg(&scale);
                    b.launch(cfg_mv)?;
                }
                Ok(())
            })?;

            let bw_fused = (tile_bytes as f64) / (fused_us * 1e3);
            let bw_mv = (tile_bytes as f64) / (mv_us * 1e3);
            println!(
                "{rows}x{cols} | {:.2} | {:.2} us ({:.0} GB/s) | {:.2} us ({:.2} us/tok, {:.0} GB/s, {:.2}x)",
                tile_bytes as f64 / 1e6,
                fused_us,
                bw_fused,
                mv_us,
                mv_us / k_vec as f64,
                bw_mv,
                mv_us / fused_us
            );
        }
        Ok(())
    }
}
