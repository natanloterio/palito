//! QTIP prototype: times the batch-1 product with packed trellis tiles
//! (src/qtip/matvec.cu) against an F16 matvec of the same shape, checks it
//! against the CPU decoder, and times the Hadamard transforms around it.
//! The tiles hold random bits: decoding cost does not depend on them, and the
//! CPU check covers the layout (src/qtip/packed.rs ties it to the encoder).
//!
//!     cargo run --release --features cuda,qtip-research --example qtip-matvec-bench

#[cfg(not(all(feature = "cuda", feature = "qtip-research")))]
fn main() {
    eprintln!("qtip-matvec-bench requires the cuda and qtip-research features");
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
    use palito::qtip::packed::{PackedMatrix, tile_words};
    use std::sync::Arc;

    type Error = Box<dyn std::error::Error>;
    const BYTE_SUM_STD: f32 = 147.800_54;
    const STATIC_SHARED_LIMIT: u32 = 48 * 1024;
    /// Rotate over weight copies totalling at least this much (4x the 96 MB L2),
    /// so every launch reads weights from DRAM as a real decode step does.
    const COLD_BYTES: usize = 384 << 20;

    #[derive(Parser)]
    struct Args {
        /// rows x cols, comma-separated (powers of two, multiples of 256).
        #[arg(
            long,
            default_value = "2048x2048,512x2048,8192x2048,2048x8192,4096x4096,16384x4096,4096x16384"
        )]
        shapes: String,
        #[arg(long, default_value_t = 12)]
        state_bits: u32,
        #[arg(long, default_value_t = 3)]
        symbol_bits: u32,
        /// Warps per matvec block.
        #[arg(long, default_value_t = 8)]
        warps: u32,
        /// Aim for about this many matvec blocks (split-K over column tiles).
        #[arg(long, default_value_t = 680)]
        target_blocks: usize,
        /// Aim for about this many fused blocks (16 warps each).
        #[arg(long, default_value_t = 340)]
        fused_blocks: usize,
        #[arg(long, default_value_t = 100)]
        launches: usize,
        /// Reuse one weight copy (warm L2), as llama.cpp's test-backend-ops perf does.
        #[arg(long)]
        warm: bool,
        /// Tiles each warp loads before decoding (QTIP_UNROLL).
        #[arg(long, default_value_t = 4)]
        unroll: u32,
    }

    struct Gpu {
        stream: Arc<CudaStream>,
        matvec: CudaFunction,
        fwht: CudaFunction,
        fwht_fast: CudaFunction,
        fused: CudaFunction,
        f16: CudaFunction,
    }

    /// Deterministic values (SplitMix64).
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
            .map(|shape| {
                let (rows, cols) = shape.split_once('x').ok_or("shape must be ROWSxCOLS")?;
                let (rows, cols): (usize, usize) = (rows.trim().parse()?, cols.trim().parse()?);
                if !rows.is_power_of_two() || !cols.is_power_of_two() || rows < 256 || cols < 256 {
                    return Err(format!("{shape}: need powers of two of at least 256").into());
                }
                Ok((rows, cols))
            })
            .collect()
    }

    fn compile(args: &Args) -> Result<Gpu, Error> {
        let source = format!(
            "#define QTIP_L {}\n#define QTIP_K {}\n#define QTIP_UNROLL {}\n{}",
            args.state_bits,
            args.symbol_bits,
            args.unroll,
            include_str!("../src/qtip/matvec.cu")
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )?;
        let context = CudaContext::new(0)?;
        let module = context.load_module(ptx)?;
        Ok(Gpu {
            stream: context.default_stream(),
            matvec: module.load_function("qtip_matvec")?,
            fwht: module.load_function("qtip_fwht")?,
            fwht_fast: module.load_function("qtip_fwht_fast")?,
            fused: module.load_function("qtip_matvec_fused")?,
            f16: module.load_function("f16_matvec")?,
        })
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

    /// Median microseconds per call of `launch` over 7 rounds of `count` calls.
    fn time(
        stream: &Arc<CudaStream>,
        count: usize,
        mut launch: impl FnMut() -> Result<(), Error>,
    ) -> Result<f64, Error> {
        launch()?;
        stream.synchronize()?;
        let mut rounds = Vec::new();
        for _ in 0..7 {
            let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for _ in 0..count {
                launch()?;
            }
            let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            stream.synchronize()?;
            rounds.push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / count as f64);
        }
        rounds.sort_by(f64::total_cmp);
        Ok(rounds[3])
    }

    fn fwht_cpu(values: &mut [f32]) {
        let n = values.len();
        let mut span = 1;
        while span < n {
            for start in (0..n).step_by(2 * span) {
                for i in start..start + span {
                    let (a, b) = (values[i], values[i + span]);
                    values[i] = a + b;
                    values[i + span] = a - b;
                }
            }
            span *= 2;
        }
        let norm = 1.0 / (n as f32).sqrt();
        values.iter_mut().for_each(|v| *v *= norm);
    }

    fn max_relative(actual: &[f32], expected: &[f64]) -> f64 {
        let peak = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        actual
            .iter()
            .zip(expected)
            .map(|(&a, &e)| (f64::from(a) - e).abs())
            .fold(0.0, f64::max)
            / peak
    }

    struct Timings {
        f16_us: f64,
        qtip_us: f64,
        fwht_in_us: f64,
        fwht_out_us: f64,
        pipeline_us: f64,
        fused_us: f64,
        fused_error: f64,
        qtip_error: f64,
        fwht_error: f64,
    }

    fn launch_fwht(
        gpu: &Gpu,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        n: usize,
        signs_first: bool,
        scale: f32,
    ) -> Result<(), Error> {
        let fast = (1024..=32768).contains(&n);
        let shared = if fast { n + n / 32 } else { n } * 4;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (n, first) = (n as i32, i32::from(signs_first));
        let mut builder = gpu
            .stream
            .launch_builder(if fast { &gpu.fwht_fast } else { &gpu.fwht });
        builder
            .arg(input)
            .arg(output)
            .arg(signs)
            .arg(&n)
            .arg(&first)
            .arg(&scale);
        // One block of 1024 threads; shared holds the n-float vector.
        unsafe { builder.launch(config) }?;
        Ok(())
    }

    fn measure(gpu: &Gpu, args: &Args, rows: usize, cols: usize) -> Result<Timings, Error> {
        let stream = &gpu.stream;
        let words_per_tile = tile_words(args.state_bits, args.symbol_bits);
        let sigma = 0.02f32;
        let packed = PackedMatrix {
            rows,
            cols,
            state_bits: args.state_bits,
            symbol_bits: args.symbol_bits,
            sigma,
            words: random_u32(
                rows / 16 * cols / 16 * words_per_tile,
                (rows * 31 + cols) as u64,
            ),
        };
        let x = uniform(cols, 7);
        let dense = packed.decode();
        let expected: Vec<f64> = dense
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&x)
                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                    .sum()
            })
            .collect();

        let copies = |bytes: usize| {
            if args.warm {
                1
            } else {
                COLD_BYTES.div_ceil(bytes).max(2)
            }
        };
        let tile_copies = (0..copies(packed.words.len() * 4))
            .map(|_| stream.clone_htod(&packed.words))
            .collect::<Result<Vec<_>, _>>()?;
        let next = std::cell::Cell::new(0usize);
        let x_device = stream.clone_htod(&x)?;
        let mut y = stream.alloc_zeros::<f32>(rows)?;
        let col_tiles = cols / 16;
        let groups = rows / 16;
        let splits = args.target_blocks.div_ceil(groups).clamp(1, col_tiles);
        let tiles_per_split = col_tiles.div_ceil(splits);
        let splits = col_tiles.div_ceil(tiles_per_split);
        let shared = (tiles_per_split * 16 * 4) as u32;
        opt_in(&gpu.matvec, shared)?;
        let config = LaunchConfig {
            grid_dim: (groups as u32, splits as u32, 1),
            block_dim: (32 * args.warps, 1, 1),
            shared_mem_bytes: shared,
        };
        let (cols_i, tps_i, scale) = (cols as i32, tiles_per_split as i32, sigma / BYTE_SUM_STD);
        let matvec = |stream: &Arc<CudaStream>,
                      x: &CudaSlice<f32>,
                      y: &mut CudaSlice<f32>|
         -> Result<(), Error> {
            let tiles = &tile_copies[next.get() % tile_copies.len()];
            next.set(next.get() + 1);
            stream.memset_zeros(y)?;
            let mut builder = stream.launch_builder(&gpu.matvec);
            builder
                .arg(tiles)
                .arg(x)
                .arg(&mut *y)
                .arg(&cols_i)
                .arg(&tps_i)
                .arg(&scale);
            // Grid and shared size follow the kernel's contract in matvec.cu.
            unsafe { builder.launch(config) }?;
            Ok(())
        };
        matvec(stream, &x_device, &mut y)?;
        let qtip_error = max_relative(&stream.clone_dtoh(&y)?, &expected);
        let qtip_us = time(stream, args.launches, || matvec(stream, &x_device, &mut y))?;

        let halves: Vec<u16> = dense
            .iter()
            .map(|&v| f16::from_f32(v * 50.0).to_bits())
            .collect();
        let w16_copies = (0..copies(halves.len() * 2))
            .map(|_| stream.clone_htod(&halves))
            .collect::<Result<Vec<_>, _>>()?;
        let mut f16_next = 0usize;
        let (rows_i, cols16) = (rows as i32, cols as i32);
        let f16_config = LaunchConfig {
            grid_dim: ((rows as u32).div_ceil(8), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let f16_us = time(stream, args.launches, || {
            let w16 = &w16_copies[f16_next % w16_copies.len()];
            f16_next += 1;
            let mut builder = stream.launch_builder(&gpu.f16);
            builder
                .arg(w16)
                .arg(&x_device)
                .arg(&mut y)
                .arg(&rows_i)
                .arg(&cols16);
            // 8 rows per 256-thread block; cols is a multiple of 256.
            unsafe { builder.launch(f16_config) }?;
            Ok(())
        })?;

        let signs_in = stream.clone_htod(
            &uniform(cols, 3)
                .iter()
                .map(|v| v.signum())
                .collect::<Vec<f32>>(),
        )?;
        let signs_out = stream.clone_htod(
            &uniform(rows, 5)
                .iter()
                .map(|v| v.signum())
                .collect::<Vec<f32>>(),
        )?;
        opt_in(&gpu.fwht, (cols.max(rows) * 4) as u32)?;
        opt_in(&gpu.fwht_fast, (cols.max(rows) * 33 / 32 * 4) as u32)?;
        let mut rotated = stream.alloc_zeros::<f32>(cols)?;
        launch_fwht(gpu, &x_device, &mut rotated, &signs_in, cols, true, 1.0)?;
        let mut cpu = x
            .iter()
            .zip(&stream.clone_dtoh(&signs_in)?)
            .map(|(a, s)| a * s)
            .collect::<Vec<f32>>();
        fwht_cpu(&mut cpu);
        let mut fwht_error = max_relative(
            &stream.clone_dtoh(&rotated)?,
            &cpu.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        );
        let fwht_in_us = time(stream, args.launches, || {
            launch_fwht(gpu, &x_device, &mut rotated, &signs_in, cols, true, 1.0)
        })?;
        let mut out = stream.alloc_zeros::<f32>(rows)?;
        launch_fwht(gpu, &y, &mut out, &signs_out, rows, false, 1.0)?;
        let mut cpu_out = stream.clone_dtoh(&y)?;
        fwht_cpu(&mut cpu_out);
        let cpu_out: Vec<f64> = cpu_out
            .iter()
            .zip(&stream.clone_dtoh(&signs_out)?)
            .map(|(v, s)| f64::from(v * s))
            .collect();
        fwht_error = fwht_error.max(max_relative(&stream.clone_dtoh(&out)?, &cpu_out));
        let fwht_out_us = time(stream, args.launches, || {
            launch_fwht(gpu, &y, &mut out, &signs_out, rows, false, 1.0)
        })?;
        let pipeline_us = time(stream, args.launches, || {
            launch_fwht(gpu, &x_device, &mut rotated, &signs_in, cols, true, 1.0)?;
            matvec(stream, &rotated, &mut y)?;
            launch_fwht(gpu, &y, &mut out, &signs_out, rows, false, 1.0)
        })?;
        // Fused: block-diagonal Hadamard of 256 on both sides, inside the kernel.
        let host_signs_in: Vec<f32> = uniform(cols, 3).iter().map(|v| v.signum()).collect();
        let host_signs_out: Vec<f32> = uniform(rows, 5).iter().map(|v| v.signum()).collect();
        let mut x_rot: Vec<f32> = x.iter().zip(&host_signs_in).map(|(a, s)| a * s).collect();
        x_rot.chunks_exact_mut(256).for_each(fwht_cpu);
        let mut y_rot: Vec<f32> = dense
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&x_rot)
                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                    .sum::<f64>() as f32
            })
            .collect();
        y_rot.chunks_exact_mut(256).for_each(fwht_cpu);
        let fused_expected: Vec<f64> = y_rot
            .iter()
            .zip(&host_signs_out)
            .map(|(v, s)| f64::from(v * s))
            .collect();
        let fused_tiles = fused_split(col_tiles, (rows / 256).max(1), args.fused_blocks);
        let fused_splits = col_tiles.div_ceil(fused_tiles);
        let chunks = (fused_tiles * 16).div_ceil(256);
        let fused_shared = (chunks * 256 * 4) as u32;
        opt_in(&gpu.fused, fused_shared)?;
        let fused_config = LaunchConfig {
            grid_dim: ((rows / 256) as u32, fused_splits as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: fused_shared,
        };
        let (fused_tiles_i, fused_scale) = (fused_tiles as i32, sigma / BYTE_SUM_STD / 256.0);
        let fused = |y: &mut CudaSlice<f32>| -> Result<(), Error> {
            let tiles = &tile_copies[next.get() % tile_copies.len()];
            next.set(next.get() + 1);
            stream.memset_zeros(y)?;
            let mut builder = stream.launch_builder(&gpu.fused);
            builder
                .arg(tiles)
                .arg(&x_device)
                .arg(&signs_in)
                .arg(&signs_out)
                .arg(&mut *y)
                .arg(&cols_i)
                .arg(&fused_tiles_i)
                .arg(&fused_scale);
            // 256 rows per 512-thread block; the split covers whole 256-chunks.
            unsafe { builder.launch(fused_config) }?;
            Ok(())
        };
        fused(&mut y)?;
        let fused_error = max_relative(&stream.clone_dtoh(&y)?, &fused_expected);
        let fused_us = time(stream, args.launches, || fused(&mut y))?;
        Ok(Timings {
            fused_us,
            fused_error,
            f16_us,
            qtip_us,
            fwht_in_us,
            fwht_out_us,
            pipeline_us,
            qtip_error,
            fwht_error,
        })
    }

    /// Column tiles per fused split: about `target` blocks in all, and the
    /// split's columns divide 256 or are a multiple of it.
    fn fused_split(col_tiles: usize, row_blocks: usize, target: usize) -> usize {
        let wanted = col_tiles.div_ceil(target.div_ceil(row_blocks).clamp(1, col_tiles));
        if wanted >= 16 {
            wanted.div_ceil(16) * 16
        } else {
            wanted.next_power_of_two()
        }
    }

    pub fn run() -> Result<(), Error> {
        let args = Args::parse();
        if args.warps == 0
            || args.warps > 32
            || args.launches == 0
            || !(1..=8).contains(&args.unroll)
        {
            return Err("--warps must be 1..=32, --unroll 1..=8 and --launches positive".into());
        }
        let shapes = parse_shapes(&args.shapes)?;
        let gpu = compile(&args)?;
        let bits = tile_words(args.state_bits, args.symbol_bits) as f64 * 32.0 / 256.0;
        println!(
            "L{} K{}: {bits:.3} bits/weight stored; {} warps/block, unroll {}; {}",
            args.state_bits,
            args.symbol_bits,
            args.warps,
            args.unroll,
            if args.warm {
                "one weight copy (warm L2)".to_string()
            } else {
                format!("weights rotated over >= {} MB (cold L2)", COLD_BYTES >> 20)
            }
        );
        println!(
            "shape | f16 us (GB/s) | qtip us (GB/s) | speedup vs f16 | fwht in / out us | in+matvec+out us | fused us (vs f16) | max rel err qtip / fwht / fused"
        );
        for (rows, cols) in shapes {
            let t = measure(&gpu, &args, rows, cols)?;
            let weights = (rows * cols) as f64;
            let gbs = |bytes: f64, us: f64| bytes / us / 1e3;
            println!(
                "{rows}x{cols} | {:.2} ({:.0}) | {:.2} ({:.0}) | {:.2}x | {:.2} / {:.2} | {:.2} | {:.2} ({:.2}x) | {:.1e} / {:.1e} / {:.1e}",
                t.f16_us,
                gbs(weights * 2.0, t.f16_us),
                t.qtip_us,
                gbs(weights * bits / 8.0, t.qtip_us),
                t.f16_us / t.qtip_us,
                t.fwht_in_us,
                t.fwht_out_us,
                t.pipeline_us,
                t.fused_us,
                t.f16_us / t.fused_us,
                t.qtip_error,
                t.fwht_error,
                t.fused_error
            );
            if t.qtip_error > 1e-4 || t.fwht_error > 1e-5 || t.fused_error > 1e-4 {
                return Err(
                    format!("{rows}x{cols}: GPU result differs from the CPU reference").into(),
                );
            }
        }
        Ok(())
    }
}
