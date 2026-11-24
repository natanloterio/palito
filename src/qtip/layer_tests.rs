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
        assert!(kurtosis.1.abs() < kurtosis.0.abs(), "RHT should Gaussianize {name}");
        assert!(results[3] < results[2], "RHT should help the trellis on {name}");
        assert!(results[3] < results[1], "trellis should beat scalar after RHT on {name}");
    }
}
