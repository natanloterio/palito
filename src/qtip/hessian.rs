//! Input Hessians written by `scripts/qtip/collect_hessians.py` or
//! `examples/collect_hessians_cuda.rs`: a directory with `manifest.json` and
//! one row-major little-endian f32 `dim x dim` file (or packed
//! `experts x dim x dim` file for MoE `ffn_down_exps`) of `X^T X / n` per group
//! of layers that read the same input.

use crate::{Error, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Entry {
    file: String,
    dim: usize,
    #[serde(default)]
    experts: Option<usize>,
}

#[derive(Deserialize)]
struct Manifest {
    tokens: u64,
    tensors: BTreeMap<String, Entry>,
}

pub struct Hessians {
    dir: PathBuf,
    pub tokens: u64,
    tensors: BTreeMap<String, Entry>,
}

fn decode_and_damp(bytes: &[u8], dim: usize, damping: f64, label: &str) -> Result<Vec<f64>> {
    let expected = dim * dim * 4;
    if bytes.len() != expected {
        return Err(Error::Invalid(format!(
            "{label} has {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    let mut h: Vec<f64> = bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    if h.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid(format!("{label} has non-finite entries")));
    }
    let mean_diagonal = (0..dim).map(|i| h[i * dim + i]).sum::<f64>() / dim as f64;
    if mean_diagonal.is_nan() || mean_diagonal <= 0.0 {
        return Err(Error::Invalid(format!(
            "{label} has a non-positive diagonal"
        )));
    }
    for i in 0..dim {
        h[i * dim + i] += damping * mean_diagonal;
    }
    Ok(h)
}

impl Hessians {
    /// An empty Hessian store (causes exporters to fall back to damped identity `H = I`).
    pub fn empty() -> Self {
        Self {
            dir: PathBuf::new(),
            tokens: 0,
            tensors: BTreeMap::new(),
        }
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let text = fs::read_to_string(dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        if manifest.tokens == 0 {
            return Err(Error::Invalid("Hessian manifest has zero tokens".into()));
        }
        for (name, entry) in &manifest.tensors {
            if entry.file.contains(['/', '\\'])
                || entry.file.starts_with('.')
                || entry.dim == 0
                || entry.experts == Some(0)
            {
                return Err(Error::Invalid(format!("bad Hessian entry for {name}")));
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            tokens: manifest.tokens,
            tensors: manifest.tensors,
        })
    }

    pub fn contains(&self, tensor: &str) -> bool {
        self.tensors.contains_key(tensor)
    }

    /// Number of packed per-expert Hessians if `tensor` has a 3D expert entry.
    pub fn expert_count(&self, tensor: &str) -> Option<usize> {
        self.tensors.get(tensor).and_then(|e| e.experts)
    }

    /// The `dim x dim` Hessian of `tensor`'s input, as f64, with
    /// `damping * mean(diag)` added to the diagonal.
    pub fn load(&self, tensor: &str, dim: usize, damping: f64) -> Result<Vec<f64>> {
        let entry = self
            .tensors
            .get(tensor)
            .ok_or_else(|| Error::Invalid(format!("no Hessian for {tensor}")))?;
        if entry.experts.is_some() {
            return Err(Error::Invalid(format!(
                "Hessian for {tensor} is packed per-expert; use load_expert"
            )));
        }
        if entry.dim != dim {
            return Err(Error::Invalid(format!(
                "Hessian for {tensor} is {0} x {0}, the matrix has {dim} columns",
                entry.dim
            )));
        }
        let bytes = fs::read(self.dir.join(&entry.file))?;
        decode_and_damp(&bytes, dim, damping, &entry.file)
    }

    /// The `dim x dim` Hessian for one expert `expert` of a packed 3D MoE tensor
    /// (`experts x dim x dim` little-endian f32), as f64, with
    /// `damping * mean(diag)` added to the diagonal.
    pub fn load_expert(
        &self,
        tensor: &str,
        expert: usize,
        n_experts: usize,
        dim: usize,
        damping: f64,
    ) -> Result<Vec<f64>> {
        let entry = self
            .tensors
            .get(tensor)
            .ok_or_else(|| Error::Invalid(format!("no Hessian for {tensor}")))?;
        if entry.experts != Some(n_experts) {
            return Err(Error::Invalid(format!(
                "Hessian for {tensor} has {:?} experts, expected {n_experts}",
                entry.experts
            )));
        }
        if entry.dim != dim {
            return Err(Error::Invalid(format!(
                "Hessian for {tensor} is {0} x {0}, the matrix has {dim} columns",
                entry.dim
            )));
        }
        if expert >= n_experts {
            return Err(Error::Invalid(format!(
                "expert index {expert} out of range for {tensor} ({n_experts} experts)"
            )));
        }
        let slice_bytes = dim * dim * 4;
        let expected_total = (n_experts as u64) * (slice_bytes as u64);
        let mut file = File::open(self.dir.join(&entry.file))?;
        let file_len = file.metadata()?.len();
        if file_len != expected_total {
            return Err(Error::Invalid(format!(
                "{} has {file_len} bytes, expected {expected_total}",
                entry.file
            )));
        }
        file.seek(SeekFrom::Start((expert as u64) * (slice_bytes as u64)))?;
        let mut bytes = vec![0u8; slice_bytes];
        file.read_exact(&mut bytes)?;
        decode_and_damp(
            &bytes,
            dim,
            damping,
            &format!("{}[expert {expert}]", entry.file),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_damps_a_hessian() {
        let dir = std::env::temp_dir().join(format!("palito-hessian-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let values = [2.0f32, 1.0, 1.0, 4.0];
        fs::write(
            dir.join("blk.0.x.f32"),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        fs::write(
            dir.join("manifest.json"),
            r#"{"tokens": 8, "tensors": {"blk.0.attn_q.weight": {"file": "blk.0.x.f32", "dim": 2}}}"#,
        )
        .unwrap();
        let hessians = Hessians::open(&dir).unwrap();
        assert!(hessians.contains("blk.0.attn_q.weight"));
        assert_eq!(hessians.expert_count("blk.0.attn_q.weight"), None);
        assert_eq!(
            hessians.load("blk.0.attn_q.weight", 2, 0.1).unwrap(),
            vec![2.3, 1.0, 1.0, 4.3]
        );
        assert!(
            hessians.load("blk.0.attn_q.weight", 4, 0.1).is_err(),
            "dimension mismatch"
        );
        assert!(
            hessians.load("blk.1.attn_q.weight", 2, 0.1).is_err(),
            "missing tensor"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loads_packed_per_expert_hessians() {
        let dir = std::env::temp_dir().join(format!(
            "palito-hessian-experts-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let packed = [
            2.0f32, 0.5, 0.5, 4.0, // expert 0: mean_diag = 3.0
            6.0f32, 1.0, 1.0, 2.0, // expert 1: mean_diag = 4.0
        ];
        fs::write(
            dir.join("blk.0.ffn_down_exps.f32"),
            packed
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        fs::write(
            dir.join("manifest.json"),
            r#"{"tokens": 16, "tensors": {"blk.0.ffn_down_exps.weight": {"file": "blk.0.ffn_down_exps.f32", "dim": 2, "experts": 2}}}"#,
        )
        .unwrap();
        let hessians = Hessians::open(&dir).unwrap();
        assert_eq!(hessians.expert_count("blk.0.ffn_down_exps.weight"), Some(2));
        assert_eq!(
            hessians
                .load_expert("blk.0.ffn_down_exps.weight", 0, 2, 2, 0.1)
                .unwrap(),
            vec![2.3, 0.5, 0.5, 4.3]
        );
        assert_eq!(
            hessians
                .load_expert("blk.0.ffn_down_exps.weight", 1, 2, 2, 0.1)
                .unwrap(),
            vec![6.4, 1.0, 1.0, 2.4]
        );
        assert!(
            hessians
                .load_expert("blk.0.ffn_down_exps.weight", 2, 2, 2, 0.1)
                .is_err()
        );
        assert!(
            hessians
                .load_expert("blk.0.ffn_down_exps.weight", 0, 3, 2, 0.1)
                .is_err()
        );
        assert!(hessians.load("blk.0.ffn_down_exps.weight", 2, 0.1).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
