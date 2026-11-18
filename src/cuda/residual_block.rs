//! Test-only block-affine arithmetic. No inference dispatch uses this module.
use crate::cuda::*;

fn planes(coefficient: i32) -> (i8, i8) {
    (
        coefficient.rem_euclid(128) as i8,
        coefficient.div_euclid(128) as i8,
    )
}

fn packed_coefficient(p: &[u8], kind: u32, i: usize) -> (i32, i32) {
    if kind == 12 {
        let g = i / 32;
        let m = &p[4..16];
        let scale = if g < 4 {
            m[g] & 63
        } else {
            (m[g + 4] & 15) | ((m[g - 4] >> 6) << 4)
        };
        let minimum = if g < 4 {
            m[g + 4] & 63
        } else {
            (m[g + 4] >> 4) | ((m[g] >> 6) << 4)
        };
        let code = (p[16 + i / 64 * 32 + i % 32] >> (i / 32 % 2 * 4)) & 15;
        (i32::from(scale) * i32::from(code), i32::from(minimum))
    } else {
        assert_eq!(kind, 14);
        let h = i / 128;
        let quarter = i % 128 / 32;
        let low = (p[h * 64 + i % 32 + quarter % 2 * 32] >> (quarter / 2 * 4)) & 15;
        let high = (p[128 + h * 32 + i % 32] >> (quarter * 2)) & 3;
        let code = i32::from(low | high << 4) - 32;
        (i32::from(p[192 + i / 16] as i8) * code, 0)
    }
}

#[test]
fn block_planes_exhaustive_coefficients_and_accumulator_bounds() {
    for (scales, codes) in [(0..=63, 0..=15), (-128..=127, -32..=31)] {
        for scale in scales {
            for code in codes.clone() {
                let coefficient = scale * code;
                let (low, high) = planes(coefficient);
                assert_eq!(i32::from(low) + 128 * i32::from(high), coefficient);
                assert!((0..=127).contains(&low));
                assert!((-32..=32).contains(&high));
            }
        }
    }
    // Reset each integer accumulator at the GGUF block, never at the row end.
    for (coefficient, expected) in [(127i64, 4_129_024), (4096, 133_169_152), (63, 2_048_256)] {
        let bound = coefficient * 127 * 256;
        assert_eq!(bound, expected);
        assert!(bound < i64::from(i32::MAX));
    }
}

#[test]
fn block_planes_packed_affine_matches_independent_tensor_decoder() -> Result<()> {
    for kind in [12, 14] {
        let mut tensor = crate::cuda::tests::packed_fixture(kind, 256, 257, 1);
        let Storage::Packed { bytes, .. } = &mut tensor.storage else {
            unreachable!()
        };
        let size = if kind == 12 { 144 } else { 210 };
        for (b, p) in bytes.chunks_exact_mut(size).enumerate() {
            if kind == 12 {
                p[4..16].fill(b as u8);
            } else {
                for (i, scale) in p[192..208].iter_mut().enumerate() {
                    *scale = (b + i) as u8;
                }
            }
        }
        let Storage::Packed { bytes, .. } = &tensor.storage else {
            unreachable!()
        };
        let mut decoded = [0.0; 256];
        for (b, p) in bytes.chunks_exact(size).enumerate() {
            tensor.row(b, &mut decoded);
            let factor = |offset| {
                f64::from(
                    half::f16::from_bits(u16::from_le_bytes([p[offset], p[offset + 1]])).to_f32(),
                )
            };
            let d = factor(if kind == 12 { 0 } else { 208 });
            let minimum = if kind == 12 { factor(2) } else { 0.0 };
            let mut dots = [0i64; 3];
            let mut reference = 0.0f64;
            for (i, &weight) in decoded.iter().enumerate() {
                let (coefficient, correction) = packed_coefficient(p, kind, i);
                let (low, high) = planes(coefficient);
                assert_eq!(
                    d * f64::from(coefficient) - minimum * f64::from(correction),
                    f64::from(weight)
                );
                let q = (i as i32 * 73 + b as i32 * 17) % 255 - 127;
                for (dot, value) in
                    dots.iter_mut()
                        .zip([i32::from(low), i32::from(high), correction])
                {
                    *dot += i64::from(value) * i64::from(q);
                }
                reference += f64::from(weight) * f64::from(q);
            }
            let combined = dots[0] + 128 * dots[1];
            assert!(combined.abs() <= 133_169_152);
            assert_eq!(d * combined as f64 - minimum * dots[2] as f64, reference);
        }
    }
    Ok(())
}

#[test]
fn block_three_component_reconstruction_and_errors() -> Result<()> {
    use crate::cuda::residual_precise::quantize_groups;
    for input in [
        vec![],
        vec![0.0; 255],
        vec![f32::NAN; 256],
        vec![f32::INFINITY; 256],
        vec![f32::MAX; 256],
    ] {
        assert!(quantize_groups(&input, 256).is_err());
    }
    assert!(quantize_groups(&[0.0; 256], 64).is_err());
    for source in [
        vec![0.0; 256],
        (0..256).map(|i| f32::from_bits(i % 32)).collect(),
        (0..256)
            .map(|i| {
                if i == 0 {
                    1000.0
                } else {
                    (i as f32 * 0.371).sin() * 0.001
                }
            })
            .collect(),
        crate::cuda::prefill::inputs(768, 3),
    ] {
        let q = quantize_groups(&source, 256)?;
        for (block, values) in source.chunks_exact(256).enumerate() {
            let maximum = values
                .iter()
                .map(|x| f64::from(x.abs()))
                .fold(0.0, f64::max);
            for (lane, &expected) in values.iter().enumerate() {
                let i = block * 256 + lane;
                let actual: f64 = (0..3)
                    .map(|c| {
                        f64::from(q.scales[block * 3 + c])
                            * f64::from(q.values[c * source.len() + i])
                    })
                    .sum();
                assert!((actual - f64::from(expected)).abs() <= maximum * 8e-8 + 3e-45);
            }
        }
        for (group, sums) in q.sums.chunks_exact(3).enumerate() {
            for (c, &sum) in sums.iter().enumerate() {
                assert_eq!(
                    sum,
                    q.values[c * source.len() + group * 32..][..32]
                        .iter()
                        .map(|&v| i32::from(v))
                        .sum::<i32>()
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "serial CUDA, 256-value three-component quantizer CPU-bit oracle and guards"]
fn block_three_component_gpu_oracle() -> Result<()> {
    let harness = crate::cuda::prefill::Harness::new()?;
    for input in [
        vec![f32::NAN; 256],
        vec![f32::INFINITY; 256],
        vec![f32::NEG_INFINITY; 256],
        vec![f32::MAX; 256],
        vec![0.0; 256],
        (0..256).map(|i| f32::from_bits(i % 32)).collect(),
        crate::cuda::prefill::inputs(768, 3),
    ] {
        println!(
            "BLOCK_QUANT {}",
            crate::cuda::residual_precise::check_gpu_groups(&harness, &input, 256)?
        );
    }
    Ok(())
}

#[test]
#[ignore = "serial CUDA block-affine Q4/Q6, sparse routing, layouts, row/token tails and CPU oracle"]
fn block_three_component_mma_matrices() -> Result<()> {
    crate::cuda::residual_precise::check_mma_matrices(256)
}
