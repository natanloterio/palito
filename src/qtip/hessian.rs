//! Input Hessians written by `scripts/qtip/collect_hessians.py`: a directory
//! with `manifest.json` and one row-major little-endian f32 `dim x dim` file
//! of `X^T X / n` per group of layers that read the same input.

use crate::{Error, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Entry {
    file: String,
    dim: usize,
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

impl Hessians {
    pub fn open(dir: &Path) -> Result<Self> {
        let text = fs::read_to_string(dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        if manifest.tokens == 0 {
            return Err(Error::Invalid("Hessian manifest has zero tokens".into()));
        }
        for (name, entry) in &manifest.tensors {
            if entry.file.contains(['/', '\\']) || entry.file.starts_with('.') || entry.dim == 0 {
                return Err(Error::Invalid(format!("bad Hessian entry for {name}")));
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            tokens: manifest.tokens,
            tensors: manifest.tensors,
        })
    }

    /// The `dim x dim` Hessian of `tensor`'s input, as f64, with
    /// `damping * mean(diag)` added to the diagonal.
    pub fn load(&self, tensor: &str, dim: usize, damping: f64) -> Result<Vec<f64>> {
        let entry = self
            .tensors
            .get(tensor)
            .ok_or_else(|| Error::Invalid(format!("no Hessian for {tensor}")))?;
        if entry.dim != dim {
            return Err(Error::Invalid(format!(
                "Hessian for {tensor} is {0} x {0}, the matrix has {dim} columns",
                entry.dim
            )));
        }
        let bytes = fs::read(self.dir.join(&entry.file))?;
        if bytes.len() != dim * dim * 4 {
            return Err(Error::Invalid(format!(
                "{} has {} bytes, expected {}",
                entry.file,
                bytes.len(),
                dim * dim * 4
            )));
        }
        let mut h: Vec<f64> = bytes
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect();
        if h.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(format!(
                "{} has non-finite entries",
                entry.file
            )));
        }
        let mean_diagonal = (0..dim).map(|i| h[i * dim + i]).sum::<f64>() / dim as f64;
        if mean_diagonal.is_nan() || mean_diagonal <= 0.0 {
            return Err(Error::Invalid(format!(
                "{} has a non-positive diagonal",
                entry.file
            )));
        }
        for i in 0..dim {
            h[i * dim + i] += damping * mean_diagonal;
        }
        Ok(h)
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
}
