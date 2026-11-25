//! Step 2 on real weights. Set PALITO_QTIP_GGUF to a Llama GGUF with F16 weights.

use super::{
    incoherence::Incoherence,
    matrix::{distortion, excess_kurtosis, scalar_reconstruct, trellis_reconstruct},
    trellis::Trellis,
};
use crate::gguf::Gguf;
use std::path::Path;

const BLOCK: usize = 256;
const SEED: u64 = 0x5EED;
const TENSORS: [&str; 4] = [
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.7.ffn_down.weight",
    "blk.15.ffn_gate.weight",
];

fn load(model: &Gguf, name: &str) -> (Vec<f32>, usize, usize) {
    super::model::matrix(&model.tensors[name])
}

#[test]
#[ignore = "requires PALITO_QTIP_GGUF (Llama with F16 weights); run with --release"]
fn incoherence_makes_trellis_quantization_of_real_layers_better() {
    let path = std::env::var_os("PALITO_QTIP_GGUF").expect("set PALITO_QTIP_GGUF");
    let model = Gguf::open(Path::new(&path), false).unwrap();
    let trellis = Trellis::new(12, 2, 1.0).unwrap();
    println!("tensor | kurtosis raw/RHT | scalar raw | scalar RHT | trellis raw | trellis RHT");
    for name in TENSORS {
        let (w, rows, cols) = load(&model, name);
        let rht = Incoherence::new(rows, cols, SEED).unwrap();
        let rotated = rht.forward(&w);
        // The RHT is orthogonal, so error measured after the inverse equals
        // error in the rotated domain; check that on the real data too.
        let trellis_rht = rht.inverse(&trellis_reconstruct(&rotated, &trellis, BLOCK));
        let in_rotated = distortion(&rotated, &rht.forward(&trellis_rht)).relative;
        let results = [
            distortion(&w, &scalar_reconstruct(&w)).relative,
            distortion(&w, &rht.inverse(&scalar_reconstruct(&rotated))).relative,
            distortion(&w, &trellis_reconstruct(&w, &trellis, BLOCK)).relative,
            distortion(&w, &trellis_rht).relative,
        ];
        let kurtosis = (excess_kurtosis(&w), excess_kurtosis(&rotated));
        println!(
            "{name} {rows}x{cols} | {:.2} / {:.2} | {:.4} | {:.4} | {:.4} | {:.4}",
            kurtosis.0, kurtosis.1, results[0], results[1], results[2], results[3]
        );
        assert!((in_rotated - results[3]).abs() < 1e-3, "orthogonality");
        assert!(
            kurtosis.1.abs() < kurtosis.0.abs(),
            "RHT should Gaussianize {name}"
        );
        assert!(
            results[3] < results[2],
            "RHT should help the trellis on {name}"
        );
        assert!(
            results[3] < results[1],
            "trellis should beat scalar after RHT on {name}"
        );
    }
}

/// Step 4 on real weights: also set PALITO_QTIP_HESSIANS to a directory from
/// `scripts/qtip/collect_hessians.py` for the same model.
#[test]
#[ignore = "requires PALITO_QTIP_GGUF and PALITO_QTIP_HESSIANS; run with --release"]
fn ldlq_lowers_the_proxy_loss_of_real_layers() {
    use super::{
        hessian::Hessians,
        ldlq::{TILE, block_feedback, cholesky_upper, ldlq, proxy_error},
        matrix::rms,
    };
    let path = std::env::var_os("PALITO_QTIP_GGUF").expect("set PALITO_QTIP_GGUF");
    let dir = std::env::var_os("PALITO_QTIP_HESSIANS").expect("set PALITO_QTIP_HESSIANS");
    let model = Gguf::open(Path::new(&path), false).unwrap();
    let hessians = Hessians::open(Path::new(&dir)).unwrap();
    let trellis = Trellis::new(12, 2, 1.0).unwrap();
    println!("tensor | relative: rows / tiles / ldlq | proxy: rows / tiles / ldlq | ldlq seconds");
    for name in TENSORS {
        let (w, rows, cols) = load(&model, name);
        let h = hessians.load(name, cols, 0.01).unwrap();
        let rht = Incoherence::new(rows, cols, SEED).unwrap();
        let rotated = rht.forward(&w);
        let sigma = rms(&rotated);
        let tile = |t: &[f32]| -> Vec<f32> {
            let normalized: Vec<f32> = t.iter().map(|&v| v / sigma).collect();
            trellis
                .decode(&trellis.quantize(&normalized))
                .into_iter()
                .map(|v| v * sigma)
                .collect()
        };
        let identity: Vec<f64> = (0..cols * cols)
            .map(|e| f64::from(u8::from(e / cols == e % cols)))
            .collect();
        let started = std::time::Instant::now();
        let u = block_feedback(
            &cholesky_upper(&rht.transform_hessian(&h), cols).unwrap(),
            cols,
        )
        .unwrap();
        let fed_back = rht.inverse(&ldlq(&rotated, rows, cols, &u, tile).unwrap());
        let seconds = started.elapsed().as_secs_f64();
        let plain_u = block_feedback(&identity, cols).unwrap();
        let tiles = rht.inverse(&ldlq(&rotated, rows, cols, &plain_u, tile).unwrap());
        let by_rows = rht.inverse(&trellis_reconstruct(&rotated, &trellis, BLOCK));
        let candidates = [&by_rows, &tiles, &fed_back];
        let relative: Vec<f64> = candidates
            .iter()
            .map(|c| distortion(&w, c).relative)
            .collect();
        let proxy: Vec<f64> = candidates
            .iter()
            .map(|c| proxy_error(&w, c, rows, cols, &h))
            .collect();
        println!(
            "{name} {rows}x{cols} | {:.4} / {:.4} / {:.4} | {:.4} / {:.4} / {:.4} | {seconds:.1}",
            relative[0], relative[1], relative[2], proxy[0], proxy[1], proxy[2]
        );
        assert_eq!(rows % TILE, 0);
        assert!(
            proxy[2] < proxy[1],
            "LDLQ should lower the proxy loss on {name}"
        );
    }
}
