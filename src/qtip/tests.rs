use super::{
    codebook::one_mad,
    scalar::{LLOYD_MAX_2BIT_MSE, lloyd_max_2bit},
    trellis::Trellis,
};

const BLOCK: usize = 256;
const BLOCKS: usize = 16;

/// Deterministic N(0, 1) samples (SplitMix64 and Box-Muller).
fn gaussian(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let radius = (-2.0 * (1.0 - uniform()).ln()).sqrt();
            (radius * (std::f64::consts::TAU * uniform()).cos()) as f32
        })
        .collect()
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    let total: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    total / a.len() as f64
}

#[test]
fn one_mad_codebook_is_roughly_standard_normal() {
    let values: Vec<f64> = (0..1u32 << 16).map(|s| f64::from(one_mad(s))).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
}

#[test]
fn scalar_baseline_matches_the_known_gaussian_distortion() {
    let x = gaussian(BLOCK * BLOCKS * 4, 7);
    let quantized: Vec<f32> = x.iter().map(|&v| lloyd_max_2bit(v)).collect();
    let distortion = mse(&x, &quantized);
    assert!((distortion - LLOYD_MAX_2BIT_MSE).abs() < 0.005, "scalar MSE {distortion}");
}

#[test]
fn decode_reproduces_what_quantize_chose() {
    let trellis = Trellis::new(12, 2, 0.9).unwrap();
    let x = gaussian(BLOCK, 1);
    let encoded = trellis.quantize(&x);
    assert_eq!(encoded.symbols.len(), BLOCK - 1);
    assert!(encoded.symbols.iter().all(|&s| s < 4));
    let decoded = trellis.decode(&encoded);
    assert_eq!(decoded.len(), BLOCK);
    assert!((mse(&x, &decoded) - encoded.distortion / BLOCK as f64).abs() < 1e-6);
}

#[test]
fn viterbi_is_optimal_on_a_brute_forceable_trellis() {
    // L=4, K=2, T=4: 16 start states x 4^3 paths, small enough to enumerate.
    let trellis = Trellis::new(4, 2, 1.0).unwrap();
    let x = gaussian(4, 3);
    let best = trellis.quantize(&x).distortion;
    let mut exhaustive = f64::INFINITY;
    for start in 0..16u32 {
        for path in 0..64u32 {
            let symbols = [(path & 3) as u8, ((path >> 2) & 3) as u8, (path >> 4) as u8];
            let candidate = trellis.decode(&super::trellis::Encoded {
                start,
                symbols: symbols.to_vec(),
                distortion: 0.0,
            });
            exhaustive = exhaustive.min(mse(&x, &candidate) * 4.0);
        }
    }
    assert!((best - exhaustive).abs() < 1e-9, "viterbi {best}, brute force {exhaustive}");
}

#[test]
fn trellis_l12_k2_beats_scalar_2bit_on_gaussian_blocks() {
    let x = gaussian(BLOCK * BLOCKS, 11);
    let best = [0.85f32, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        .into_iter()
        .map(|scale| {
            let trellis = Trellis::new(12, 2, scale).unwrap();
            let total: f64 = x
                .chunks_exact(BLOCK)
                .map(|block| trellis.quantize(block).distortion)
                .sum();
            (total / x.len() as f64, scale)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    println!(
        "QTIP trellis L=12 K=2: MSE {:.4} at scale {} (scalar 2-bit {:.4})",
        best.0, best.1, LLOYD_MAX_2BIT_MSE
    );
    assert!(best.0 < LLOYD_MAX_2BIT_MSE * 0.85, "trellis MSE {:.4}", best.0);
}

#[test]
fn rejects_invalid_trellis_shapes() {
    assert!(Trellis::new(0, 2, 1.0).is_err());
    assert!(Trellis::new(12, 0, 1.0).is_err());
    assert!(Trellis::new(2, 2, 1.0).is_err());
    assert!(Trellis::new(25, 2, 1.0).is_err());
    assert!(Trellis::new(12, 2, f32::NAN).is_err());
}
