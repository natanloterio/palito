use crate::gguf::{Storage, Tensor, decode, layout};
use rayon::prelude::*;

pub(crate) fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn packed_row(
    kind: u32,
    bytes: &[u8],
    input: &[f32],
    width: usize,
    sums: &mut [f32],
    accelerated: bool,
) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = accelerated;
    let (block, size, _) = layout(kind).expect("validated tensor type");
    let mut weights = [0.0; 256];
    for (index, packed) in bytes.chunks_exact(size).enumerate() {
        #[cfg(target_arch = "x86_64")]
        if accelerated {
            // Block sizes were validated by the GGUF reader.
            unsafe { decode_avx2(kind, packed, &mut weights[..block]) };
        } else {
            decode(kind, packed, &mut weights[..block]);
        }
        #[cfg(not(target_arch = "x86_64"))]
        decode(kind, packed, &mut weights[..block]);
        let offset = index * block;
        for (sum, token) in sums.iter_mut().zip(input.chunks_exact(width)) {
            #[cfg(target_arch = "x86_64")]
            if accelerated {
                // The caller has checked AVX2; both slices contain exactly one validated block.
                *sum += unsafe { dot_avx2(&weights[..block], &token[offset..offset + block]) };
                continue;
            }
            for (&w, &x) in weights[..block].iter().zip(&token[offset..offset + block]) {
                *sum += w * x;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn packed_row_avx2(kind: u32, bytes: &[u8], input: &[f32], width: usize, sums: &mut [f32]) {
    packed_row(kind, bytes, input, width, sums, true);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2(kind: u32, bytes: &[u8], output: &mut [f32]) {
    use crate::gguf::half;
    use std::arch::x86_64::*;
    if !matches!(kind, 6 | 8 | 12 | 14) {
        decode(kind, bytes, output);
        return;
    }
    let scale = half(bytes, if kind == 14 { 208 } else { 0 });
    let high_bits = if kind == 6 {
        u32::from_le_bytes(bytes[2..6].try_into().expect("Q5 high bits"))
    } else {
        0
    };
    // Every load/store covers eight bytes/floats within a validated quantization block.
    unsafe {
        for i in (0..output.len()).step_by(8) {
            let load = |offset: usize| {
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(offset).cast()))
            };
            let (quant, multiplier, minimum) = match kind {
                6 => {
                    let mut low = load(6 + i % 16);
                    if i >= 16 {
                        low = _mm256_srli_epi32::<4>(low);
                    }
                    low = _mm256_and_si256(low, _mm256_set1_epi32(15));
                    let shifts = _mm256_setr_epi32(
                        i as i32,
                        i as i32 + 1,
                        i as i32 + 2,
                        i as i32 + 3,
                        i as i32 + 4,
                        i as i32 + 5,
                        i as i32 + 6,
                        i as i32 + 7,
                    );
                    let high = _mm256_and_si256(
                        _mm256_srlv_epi32(_mm256_set1_epi32(high_bits as i32), shifts),
                        _mm256_set1_epi32(1),
                    );
                    (
                        _mm256_sub_epi32(
                            _mm256_or_si256(low, _mm256_slli_epi32::<4>(high)),
                            _mm256_set1_epi32(16),
                        ),
                        scale,
                        0.0,
                    )
                }
                8 => (
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(2 + i).cast())),
                    scale,
                    0.0,
                ),
                12 => {
                    let group = i / 32;
                    let scales = &bytes[4..16];
                    let (s, m) = if group < 4 {
                        (scales[group] & 63, scales[group + 4] & 63)
                    } else {
                        (
                            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                        )
                    };
                    let mut quant = load(16 + (i / 64) * 32 + i % 32);
                    if group % 2 == 1 {
                        quant = _mm256_srli_epi32::<4>(quant);
                    }
                    (
                        _mm256_and_si256(quant, _mm256_set1_epi32(15)),
                        scale * f32::from(s),
                        half(bytes, 2) * f32::from(m),
                    )
                }
                14 => {
                    let half_block = i / 128;
                    let quarter = (i % 128) / 32;
                    let mut low = load(half_block * 64 + i % 32 + (quarter % 2) * 32);
                    if quarter >= 2 {
                        low = _mm256_srli_epi32::<4>(low);
                    }
                    let high = _mm256_srlv_epi32(
                        load(128 + half_block * 32 + i % 32),
                        _mm256_set1_epi32((2 * quarter) as i32),
                    );
                    let quant = _mm256_or_si256(
                        _mm256_and_si256(low, _mm256_set1_epi32(15)),
                        _mm256_slli_epi32::<4>(_mm256_and_si256(high, _mm256_set1_epi32(3))),
                    );
                    (
                        _mm256_sub_epi32(quant, _mm256_set1_epi32(32)),
                        scale * (bytes[192 + i / 16] as i8) as f32,
                        0.0,
                    )
                }
                _ => unreachable!(),
            };
            let values = _mm256_mul_ps(_mm256_cvtepi32_ps(quant), _mm256_set1_ps(multiplier));
            let values = if kind == 12 {
                _mm256_sub_ps(values, _mm256_set1_ps(minimum))
            } else {
                values
            };
            _mm256_storeu_ps(output.as_mut_ptr().add(i), values);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(left.len(), right.len());
    let mut accumulator = _mm256_setzero_ps();
    let mut a = left.chunks_exact(8);
    let mut b = right.chunks_exact(8);
    for (a, b) in a.by_ref().zip(b.by_ref()) {
        // chunks_exact guarantees eight readable floats for each unaligned load.
        let product =
            unsafe { _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr())) };
        accumulator = _mm256_add_ps(accumulator, product);
    }
    let mut lanes = [0.0; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    let mut sum = lanes.iter().sum::<f32>();
    for (&a, &b) in a.remainder().iter().zip(b.remainder()) {
        sum += a * b;
    }
    sum
}

impl Tensor {
    pub fn allocated_bytes(&self) -> u64 {
        match &self.storage {
            Storage::F32(values) => values.capacity() as u64 * 4,
            Storage::Packed { bytes, .. } => bytes.capacity() as u64,
        }
    }

    pub fn floats(&self) -> &[f32] {
        match &self.storage {
            Storage::F32(values) => values,
            _ => unreachable!("one-dimensional tensors are expanded during validation"),
        }
    }

    pub fn row(&self, index: usize, output: &mut [f32]) {
        let width = self.dims[0];
        assert_eq!(output.len(), width);
        match &self.storage {
            Storage::F32(values) => {
                output.copy_from_slice(&values[index * width..(index + 1) * width])
            }
            Storage::Packed { kind, bytes } => {
                let (block, size, _) = layout(*kind).expect("validated tensor type");
                let row_bytes = width / block * size;
                for (input, output) in bytes[index * row_bytes..(index + 1) * row_bytes]
                    .chunks_exact(size)
                    .zip(output.chunks_exact_mut(block))
                {
                    decode(*kind, input, output);
                }
            }
        }
    }

    /// Weight rows are partitioned between workers. Each packed block is decoded
    /// once and reused across the input batch; activations remain F32.
    pub fn multiply(&self, input: &[f32], count: usize, output: &mut [f32], scratch: &mut [f32]) {
        let width = self.dims[0];
        let rows = self.dims[1];
        assert_eq!(input.len(), count * width);
        assert_eq!(output.len(), count * rows);
        let columns = if count == 1 {
            &mut *output
        } else {
            &mut scratch[..count * rows]
        };
        self.multiply_columns(input, count, columns);
        if count > 1 {
            output
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(token, output)| {
                    for (row, value) in output.iter_mut().enumerate() {
                        *value = scratch[row * count + token];
                    }
                });
        }
    }

    pub fn multiply_columns(&self, input: &[f32], count: usize, columns: &mut [f32]) {
        self.multiply_columns_at(input, count, columns, 0);
    }

    pub fn multiply_expert(&self, expert: usize, input: &[f32], output: &mut [f32]) {
        assert_eq!(self.dims.len(), 3);
        assert!(expert < self.dims[2]);
        self.multiply_columns_at(input, 1, output, expert * self.dims[1]);
    }

    fn multiply_columns_at(
        &self,
        input: &[f32],
        count: usize,
        columns: &mut [f32],
        first_row: usize,
    ) {
        let width = self.dims[0];
        debug_assert!(self.block_elements > 0);
        assert_eq!(input.len(), count * width);
        assert_eq!(columns.len(), count * self.dims[1]);
        columns
            .par_chunks_mut(count)
            .enumerate()
            .for_each(|(row, sums)| {
                let row = first_row + row;
                sums.fill(0.0);
                match &self.storage {
                    Storage::F32(values) => {
                        let weights = &values[row * width..(row + 1) * width];
                        for (sum, token) in sums.iter_mut().zip(input.chunks_exact(width)) {
                            #[cfg(target_arch = "x86_64")]
                            if self.accelerated {
                                for (weights, input) in weights
                                    .chunks(self.block_elements)
                                    .zip(token.chunks(self.block_elements))
                                {
                                    // Tensor construction guards runtime AVX2 support.
                                    *sum += unsafe { dot_avx2(weights, input) };
                                }
                                continue;
                            }
                            *sum = weights.iter().zip(token).map(|(w, x)| w * x).sum();
                        }
                    }
                    Storage::Packed { kind, bytes } => {
                        let (block, size, _) = layout(*kind).expect("validated tensor type");
                        let row_bytes = width / block * size;
                        let bytes = &bytes[row * row_bytes..(row + 1) * row_bytes];
                        #[cfg(target_arch = "x86_64")]
                        if self.accelerated {
                            // Set only after runtime AVX2 detection.
                            unsafe { packed_row_avx2(*kind, bytes, input, width, sums) };
                            return;
                        }
                        packed_row(*kind, bytes, input, width, sums, false);
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn packed_matmul_matches_expansion_across_formats_batches_and_rows() {
        for kind in [1, 6, 8, 12, 13, 14] {
            let (block, size, _) = layout(kind).unwrap();
            let width = block * 2;
            let mut bytes = vec![0; size * 6];
            for (index, packed) in bytes.chunks_exact_mut(size).enumerate() {
                for (i, byte) in packed.iter_mut().enumerate() {
                    *byte = ((i * 13 + index * 7) % 251) as u8;
                }
                let scale_offset = if kind == 14 { 208 } else { 0 };
                packed[scale_offset..scale_offset + 2]
                    .copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
                if matches!(kind, 12 | 13) {
                    packed[2..4].copy_from_slice(&f16::from_f32(0.0625).to_bits().to_le_bytes());
                }
            }
            let mut packed = Tensor {
                dims: vec![width, 3],
                storage: Storage::Packed { kind, bytes },
                accelerated: avx2_available(),
                block_elements: block,
            };
            let mut values = vec![0.0; width * 3];
            for (row, output) in values.chunks_mut(width).enumerate() {
                packed.row(row, output);
            }
            let mut expanded = Tensor {
                dims: vec![width, 3],
                storage: Storage::F32(values),
                accelerated: avx2_available(),
                block_elements: block,
            };
            assert!(packed.allocated_bytes() < expanded.allocated_bytes());
            for (count, accelerated) in [
                (1, false),
                (3, false),
                (1, avx2_available()),
                (3, avx2_available()),
            ] {
                packed.accelerated = accelerated;
                expanded.accelerated = accelerated;
                let input: Vec<_> = (0..width * count)
                    .map(|i| (i % 23) as f32 / 32.0 - 0.25)
                    .collect();
                let mut actual = vec![0.0; count * 3];
                let mut expected = actual.clone();
                let mut scratch = actual.clone();
                packed.multiply(&input, count, &mut actual, &mut scratch);
                expanded.multiply(&input, count, &mut expected, &mut scratch);
                assert_eq!(actual, expected, "kind={kind}, count={count}");
            }
        }
    }
}
