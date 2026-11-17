use crate::{Error, Result};
use half::f16;
use std::{collections::HashMap, fs, path::Path};

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(format!("GGUF: {}", message.into()))
}

#[derive(Debug)]
pub(crate) enum Value {
    Text(String),
    Integer(u64),
    Float(f32),
    Bool(bool),
    Strings(Vec<String>),
    Integers(Vec<u64>),
}

pub(crate) struct Tensor {
    pub dims: Vec<usize>,
    pub storage: Storage,
    pub accelerated: bool,
    pub block_elements: usize,
}

pub(crate) enum Storage {
    F32(Vec<f32>),
    Packed { kind: u32, bytes: Vec<u8> },
}

pub(crate) struct Gguf {
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
    pub tensor_types: Vec<String>,
    pub file_bytes: u64,
    pub parameters: u64,
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| invalid("offset overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated file"))?;
        self.position = end;
        Ok(value)
    }

    fn integer(&mut self, count: usize) -> Result<u64> {
        let mut bytes = [0; 8];
        bytes[..count].copy_from_slice(self.take(count)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn count(&mut self) -> Result<usize> {
        usize::try_from(self.integer(8)?).map_err(|_| invalid("count exceeds address space"))
    }

    fn string(&mut self) -> Result<String> {
        let size = self.count()?;
        String::from_utf8(self.take(size)?.to_vec()).map_err(|_| invalid("invalid UTF-8 string"))
    }

    fn value(&mut self, kind: u64, depth: usize) -> Result<Option<Value>> {
        if depth > 4 {
            return Err(invalid("metadata nesting exceeds limit"));
        }
        Ok(match kind {
            0 | 1 => Some(Value::Integer(self.integer(1)?)),
            7 => Some(Value::Bool(match self.integer(1)? {
                0 => false,
                1 => true,
                _ => return Err(invalid("invalid boolean")),
            })),
            2 | 3 => Some(Value::Integer(self.integer(2)?)),
            4 | 5 => Some(Value::Integer(self.integer(4)?)),
            6 => Some(Value::Float(f32::from_bits(self.integer(4)? as u32))),
            8 => Some(Value::Text(self.string()?)),
            9 => {
                let element = self.integer(4)?;
                let count = self.count()?;
                if count > self.bytes.len() {
                    return Err(invalid("metadata array count exceeds file size"));
                }
                if element == 8 {
                    if count > (self.bytes.len() - self.position) / 8 {
                        return Err(invalid("truncated string array"));
                    }
                    return Ok(Some(Value::Strings(
                        (0..count).map(|_| self.string()).collect::<Result<_>>()?,
                    )));
                }
                if element == 4 || element == 5 {
                    if count > (self.bytes.len() - self.position) / 4 {
                        return Err(invalid("truncated integer array"));
                    }
                    return Ok(Some(Value::Integers(
                        (0..count).map(|_| self.integer(4)).collect::<Result<_>>()?,
                    )));
                }
                for _ in 0..count {
                    self.value(element, depth + 1)?;
                }
                None
            }
            10 | 11 => Some(Value::Integer(self.integer(8)?)),
            12 => Some(Value::Float(f64::from_bits(self.integer(8)?) as f32)),
            _ => return Err(invalid(format!("unsupported metadata type {kind}"))),
        })
    }

    fn header(&mut self) -> Result<(usize, HashMap<String, Value>)> {
        if self.take(4)? != b"GGUF" || self.integer(4)? != 3 {
            return Err(invalid("require little-endian GGUF v3"));
        }
        let tensor_count = self.count()?;
        let metadata_count = self.count()?;
        if tensor_count > self.bytes.len() / 24 || metadata_count > self.bytes.len() / 12 {
            return Err(invalid("impossible directory counts"));
        }
        let mut metadata = HashMap::new();
        let mut keys = std::collections::HashSet::new();
        for _ in 0..metadata_count {
            let key = self.string()?;
            if !keys.insert(key.clone()) {
                return Err(invalid("duplicate metadata key"));
            }
            let kind = self.integer(4)?;
            if let Some(value) = self.value(kind, 0)? {
                metadata.insert(key, value);
            }
        }
        Ok((tensor_count, metadata))
    }
}

pub(crate) fn layout(kind: u32) -> Result<(usize, usize, &'static str)> {
    match kind {
        0 => Ok((1, 4, "F32")),
        1 => Ok((1, 2, "F16")),
        6 => Ok((32, 22, "Q5_0")),
        8 => Ok((32, 34, "Q8_0")),
        12 => Ok((256, 144, "Q4_K")),
        14 => Ok((256, 210, "Q6_K")),
        _ => Err(invalid(format!("unsupported tensor type {kind}"))),
    }
}

#[inline(always)]
pub(crate) fn half(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn finite_packed(kind: u32, bytes: &[u8]) -> bool {
    // Integer quants/scales are bounded: finite F16 scales cannot overflow F32 products.
    half(bytes, if kind == 14 { 208 } else { 0 }).is_finite()
        && (kind != 12 || half(bytes, 2).is_finite())
}

#[inline(always)]
pub(crate) fn decode(kind: u32, bytes: &[u8], output: &mut [f32]) {
    match kind {
        0 => output[0] = f32::from_le_bytes(bytes.try_into().expect("validated F32 block")),
        1 => output[0] = half(bytes, 0),
        6 => {
            let scale = half(bytes, 0);
            let high = u32::from_le_bytes(bytes[2..6].try_into().expect("Q5 high bits"));
            for (i, value) in output.iter_mut().enumerate() {
                let low = (bytes[6 + i % 16] >> (4 * (i / 16))) & 15;
                let quant = i32::from(low) + (((high >> i) & 1) as i32) * 16 - 16;
                *value = scale * quant as f32;
            }
        }
        8 => {
            let scale = half(bytes, 0);
            for (value, &quant) in output.iter_mut().zip(&bytes[2..]) {
                *value = scale * (quant as i8) as f32;
            }
        }
        12 => {
            let scale = half(bytes, 0);
            let minimum = half(bytes, 2);
            let scales = &bytes[4..16];
            for (i, value) in output.iter_mut().enumerate() {
                let group = i / 32;
                let (s, m) = if group < 4 {
                    (scales[group] & 63, scales[group + 4] & 63)
                } else {
                    (
                        (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                        (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                    )
                };
                let quant = (bytes[16 + (i / 64) * 32 + i % 32] >> (4 * (group % 2))) & 15;
                *value = scale * f32::from(s) * f32::from(quant) - minimum * f32::from(m);
            }
        }
        14 => {
            let scale = half(bytes, 208);
            for (i, value) in output.iter_mut().enumerate() {
                let half_block = i / 128;
                let lane = i % 32;
                let quarter = (i % 128) / 32;
                let low = (bytes[half_block * 64 + lane + (quarter % 2) * 32]
                    >> (4 * (quarter / 2)))
                    & 15;
                let high = (bytes[128 + half_block * 32 + lane] >> (2 * quarter)) & 3;
                let quant = i32::from(low | (high << 4)) - 32;
                *value = scale * (bytes[192 + i / 16] as i8) as f32 * quant as f32;
            }
        }
        _ => unreachable!("tensor type checked before decoding"),
    }
}

impl Gguf {
    pub fn open(path: &Path, expand: bool) -> Result<Self> {
        Self::parse(&fs::read(path)?, expand)
    }

    pub fn read_metadata(path: &Path) -> Result<HashMap<String, Value>> {
        let bytes = fs::read(path)?;
        let mut reader = Reader {
            bytes: &bytes,
            position: 0,
        };
        Ok(reader.header()?.1)
    }

    fn parse(bytes: &[u8], expand: bool) -> Result<Self> {
        let mut reader = Reader { bytes, position: 0 };
        let (tensor_count, metadata) = reader.header()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::Integer(value)) => {
                usize::try_from(*value).map_err(|_| invalid("alignment exceeds address space"))?
            }
            None => 32,
            _ => return Err(invalid("invalid alignment type")),
        };
        if !alignment.is_power_of_two() {
            return Err(invalid("alignment must be a positive power of two"));
        }
        let mut entries = Vec::new();
        let mut names = std::collections::HashSet::new();
        for _ in 0..tensor_count {
            let name = reader.string()?;
            if !names.insert(name.clone()) {
                return Err(invalid("duplicate tensor name"));
            }
            let rank = reader.integer(4)?;
            if !(1..=3).contains(&rank) {
                return Err(invalid(
                    "only vectors, matrices, and expert stacks are supported",
                ));
            }
            let mut dims = Vec::new();
            let mut elements = 1usize;
            for _ in 0..rank {
                let dim = reader.count()?;
                if dim == 0 {
                    return Err(invalid("zero tensor dimension"));
                }
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| invalid("tensor size overflow"))?;
                dims.push(dim);
            }
            let kind = reader.integer(4)? as u32;
            let offset = reader.count()?;
            let (block, size, label) = layout(kind)?;
            if dims[0] % block != 0 || offset % alignment != 0 {
                return Err(invalid("unaligned tensor or quantization block"));
            }
            let length = (elements / block)
                .checked_mul(size)
                .ok_or_else(|| invalid("tensor size overflow"))?;
            entries.push((name, dims, elements, kind, offset, length, label));
        }
        let start = reader
            .position
            .checked_add(alignment - 1)
            .ok_or_else(|| invalid("alignment overflow"))?
            & !(alignment - 1);
        let mut ranges = Vec::new();
        let mut parameters = 0u64;
        for (_, _, elements, _, offset, length, _) in &entries {
            let begin = start
                .checked_add(*offset)
                .ok_or_else(|| invalid("tensor offset overflow"))?;
            let end = begin
                .checked_add(*length)
                .ok_or_else(|| invalid("tensor end overflow"))?;
            if end > bytes.len() {
                return Err(invalid("tensor extends beyond file"));
            }
            ranges.push((begin, end));
            parameters = parameters
                .checked_add(*elements as u64)
                .ok_or_else(|| invalid("parameter count overflow"))?;
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid("overlapping tensor data"));
        }
        let mut tensors = HashMap::new();
        let mut tensor_types = Vec::new();
        for (name, dims, elements, kind, offset, length, label) in entries {
            let (block, size, _) = layout(kind)?;
            let input = &bytes[start + offset..start + offset + length];
            let storage = if expand || kind == 0 || dims.len() == 1 {
                let mut values = vec![0.0; elements];
                for (input, output) in input.chunks_exact(size).zip(values.chunks_exact_mut(block))
                {
                    decode(kind, input, output);
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid(format!("non-finite weights in {name}")));
                }
                Storage::F32(values)
            } else {
                for input in input.chunks_exact(size) {
                    if !finite_packed(kind, input) {
                        return Err(invalid(format!("non-finite weights in {name}")));
                    }
                }
                Storage::Packed {
                    kind,
                    bytes: input.to_vec(),
                }
            };
            tensor_types.push(label.to_owned());
            tensors.insert(
                name,
                Tensor {
                    dims,
                    storage,
                    accelerated: crate::kernels::avx2_available(),
                    block_elements: if kind == 0 { elements } else { block },
                },
            );
        }
        tensor_types.sort();
        tensor_types.dedup();
        Ok(Self {
            metadata,
            tensors,
            tensor_types,
            file_bytes: bytes.len() as u64,
            parameters,
        })
    }

    pub fn integer(&self, key: &str) -> Result<usize> {
        match self.metadata.get(key) {
            Some(Value::Integer(value)) => {
                usize::try_from(*value).map_err(|_| invalid(format!("{key} exceeds address space")))
            }
            _ => Err(invalid(format!("missing integer {key}"))),
        }
    }

    pub fn float(&self, key: &str) -> Result<f32> {
        match self.metadata.get(key) {
            Some(Value::Float(value)) if value.is_finite() && *value > 0.0 => Ok(*value),
            _ => Err(invalid(format!("missing positive float {key}"))),
        }
    }

    pub fn text(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(Value::Text(value)) => Ok(value),
            _ => Err(invalid(format!("missing text {key}"))),
        }
    }

    pub fn tensor(&mut self, name: &str, dims: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        if tensor.dims != dims {
            return Err(invalid(format!(
                "{name}: expected shape {dims:?}, got {:?}",
                tensor.dims
            )));
        }
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_validation_rejects_nonfinite_scales_and_accepts_extreme_finite_products() {
        for kind in [1, 6, 8, 12, 14] {
            let (block, size, _) = layout(kind).unwrap();
            let mut bytes = vec![255; size];
            let offset = if kind == 14 { 208 } else { 0 };
            bytes[offset..offset + 2].copy_from_slice(&f16::MAX.to_bits().to_le_bytes());
            if kind == 12 {
                bytes[2..4].copy_from_slice(&f16::MAX.to_bits().to_le_bytes());
            }
            assert!(finite_packed(kind, &bytes));
            let mut values = vec![0.0; block];
            decode(kind, &bytes, &mut values);
            assert!(values.iter().all(|v| v.is_finite()));
            for value in [f16::INFINITY, f16::NAN] {
                bytes[offset..offset + 2].copy_from_slice(&value.to_bits().to_le_bytes());
                assert!(!finite_packed(kind, &bytes));
            }
            if kind == 12 {
                bytes[..2].copy_from_slice(&f16::MAX.to_bits().to_le_bytes());
                bytes[2..4].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
                assert!(!finite_packed(kind, &bytes));
            }
        }
    }

    #[test]
    fn rejects_truncated_tokenizer_arrays_and_invalid_boolean_metadata() {
        for element in [4u32, 8] {
            let mut bytes = element.to_le_bytes().to_vec();
            bytes.extend_from_slice(&3u64.to_le_bytes());
            bytes.extend_from_slice(&[0; 5]);
            let mut reader = Reader {
                bytes: &bytes,
                position: 0,
            };
            assert!(reader.value(9, 0).is_err());
        }
        let mut reader = Reader {
            bytes: &[2],
            position: 0,
        };
        assert!(reader.value(7, 0).is_err());
    }

    #[test]
    fn decodes_signed_q5_high_bits_and_q8_scale() {
        let mut q5 = [0; 22];
        q5[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        q5[2..6].copy_from_slice(&0x8000_0001u32.to_le_bytes());
        q5[6] = 0x31;
        q5[21] = 0x70;
        let mut output = [0.0; 32];
        decode(6, &q5, &mut output);
        assert_eq!((output[0], output[16], output[31]), (0.5, -6.5, 3.5));
        let mut q8 = [0; 34];
        q8[..2].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        q8[2] = 128;
        q8[33] = 127;
        decode(8, &q8, &mut output);
        assert_eq!((output[0], output[31]), (-256.0, 254.0));
    }

    #[test]
    fn decodes_k_quant_subgroup_scales_and_high_bits() {
        let mut q4 = [0; 144];
        q4[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        q4[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        q4[4] = 0x81;
        q4[8] = 0x43;
        q4[12] = 0x25;
        q4[80] = 7;
        let mut output = [0.0; 256];
        decode(12, &q4, &mut output);
        assert_eq!(output[0], -0.75);
        assert_eq!(output[128], 0.5 * 37.0 * 7.0 - 0.25 * 18.0);
        let mut q6 = [0; 210];
        q6[208..].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        q6[0] = 0x71;
        q6[128] = 0b10_00_00_01;
        q6[192] = (-2i8) as u8;
        q6[196] = 3;
        q6[198] = 4;
        decode(14, &q6, &mut output);
        assert_eq!(output[0], 15.0);
        assert_eq!(output[64], -37.5);
        assert_eq!(output[96], 0.0);
    }

    #[test]
    fn rejects_truncated_and_impossible_gguf_directories() {
        for bytes in [
            vec![],
            b"GGUF\x03\0\0\0".to_vec(),
            [
                b"GGUF\x03\0\0\0".as_slice(),
                &u64::MAX.to_le_bytes(),
                &[0; 8],
            ]
            .concat(),
        ] {
            assert!(Gguf::parse(&bytes, false).is_err());
        }
    }
}
