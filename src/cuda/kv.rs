//! Per-layer K/V cache buffers: F32 on every profile but `int8-fast`, which
//! stores FP16 (half the bytes; its prefill attention already rounds K/V to FP16)
//! and reads them through the `_f16` kernel variants (F32 accumulation).

use super::{CudaMatmul, failure};
use crate::Result;
use cudarc::driver::{CudaSlice, CudaStream, LaunchArgs, PushKernelArg};
use std::sync::Arc;

/// Kernels that read or write the K/V cache and have an `_f16` variant with the
/// same arguments. On an FP16 cache the variant is registered under the base name,
/// so launch sites stay profile-agnostic.
pub(super) const F16_KERNELS: &[(&str, &str)] = &[
    ("cache_kv", "cache_kv_f16"),
    ("cache_kv_decode", "cache_kv_decode_f16"),
    ("qkv_head_norm_rope_cache", "qkv_head_norm_rope_cache_f16"),
    (
        "qkv_head_norm_rope_cache_decode",
        "qkv_head_norm_rope_cache_decode_f16",
    ),
    ("attention_scores", "attention_scores_f16"),
    ("attention_values", "attention_values_f16"),
    ("gemma_attention_scores", "gemma_attention_scores_f16"),
    ("gemma_attention_values", "gemma_attention_values_f16"),
    ("gemma_attention_scores_decode", "gemma_attention_scores_decode_f16"),
    ("gemma_attention_values_decode", "gemma_attention_values_decode_f16"),
    ("attention_fused", "attention_fused_f16"),
    ("attention_fused_decode", "attention_fused_decode_f16"),
    ("attention_split", "attention_split_f16"),
    ("attention_split_decode", "attention_split_decode_f16"),
    ("attention_split_gqa", "attention_split_gqa_f16"),
    (
        "attention_split_gqa_decode",
        "attention_split_gqa_decode_f16",
    ),
    ("attention_split_head", "attention_split_head_f16"),
    (
        "attention_split_head_decode",
        "attention_split_head_decode_f16",
    ),
    ("attention_split_group", "attention_split_group_f16"),
    (
        "attention_split_group_decode",
        "attention_split_group_decode_f16",
    ),
    ("attention_prefill_flash4", "attention_prefill_flash4_f16"),
];

impl CudaMatmul {
    /// int8-fast keeps an FP16 K/V cache; the strict profiles keep F32.
    pub(super) fn kv_f16(self) -> bool {
        self == Self::Int8Fast
    }

    pub(super) fn kv_element_bytes(self) -> u64 {
        if self.kv_f16() { 2 } else { 4 }
    }
}

/// One layer's keys or values: `[position][kv_head][dim]` elements.
pub(super) enum KvBuffer {
    F32(CudaSlice<f32>),
    /// FP16 bits.
    F16(CudaSlice<u16>),
}

impl KvBuffer {
    pub(super) fn zeros(
        stream: &Arc<CudaStream>,
        elements: usize,
        matmul: CudaMatmul,
    ) -> Result<Self> {
        Ok(if matmul.kv_f16() {
            Self::F16(stream.alloc_zeros(elements).map_err(failure)?)
        } else {
            Self::F32(stream.alloc_zeros(elements).map_err(failure)?)
        })
    }
}

// Launch arguments: the device pointer of whichever buffer the profile holds; the
// kernel registered under the launched name takes that element type.
unsafe impl<'a, 'b: 'a> PushKernelArg<&'b KvBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b KvBuffer) -> &mut Self {
        match arg {
            KvBuffer::F32(slice) => self.arg(slice),
            KvBuffer::F16(slice) => self.arg(slice),
        }
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b mut KvBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut KvBuffer) -> &mut Self {
        match arg {
            KvBuffer::F32(slice) => self.arg(slice),
            KvBuffer::F16(slice) => self.arg(slice),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_int8_fast_stores_f16() {
        for matmul in [
            CudaMatmul::F32,
            CudaMatmul::F32Tiled,
            CudaMatmul::Int8Prefill,
            CudaMatmul::ResidualMma3Research,
        ] {
            assert!(!matmul.kv_f16(), "{matmul:?}");
            assert_eq!(matmul.kv_element_bytes(), 4);
        }
        assert!(CudaMatmul::Int8Fast.kv_f16());
        assert_eq!(CudaMatmul::Int8Fast.kv_element_bytes(), 2);
    }

    #[test]
    fn f16_variants_are_unique_and_suffixed() {
        let mut names: Vec<_> = F16_KERNELS.iter().map(|(name, _)| *name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), F16_KERNELS.len());
        for (name, entry) in F16_KERNELS {
            assert_eq!(*entry, format!("{name}_f16"));
        }
    }
}
