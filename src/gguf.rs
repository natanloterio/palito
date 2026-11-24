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
    Bools(Vec<bool>),
    Strings(Vec<String>),
    Integers(Vec<u64>),
    Floats(Vec<f32>),
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

impl Tensor {
    pub fn split_vector(mut self, count: usize) -> Result<(Self, Self)> {
        let [total] = self.dims[..] else {
            return Err(invalid("vector split requires one dimension"));
        };
        if count == 0 || count >= total {
            return Err(invalid("invalid vector split"));
        }
        let Storage::F32(values) = &mut self.storage else {
            return Err(invalid("vector split requires F32 storage"));
        };
        let tail = values.split_off(count);
        values.shrink_to_fit();
        self.dims[0] = count;
        let right = Self {
            dims: vec![total - count],
            storage: Storage::F32(tail),
            accelerated: self.accelerated,
            block_elements: self.block_elements,
        };
        Ok((self, right))
    }

    /// Partition a matrix along its output dimension without unpacking or requantizing rows.
    pub fn split_rows(mut self, rows: usize) -> Result<(Self, Self)> {
        let [_, total] = self.dims[..] else {
            return Err(invalid("row split requires a matrix"));
        };
        if rows == 0 || rows >= total {
            return Err(invalid("invalid matrix row split"));
        }
        let tail = match &mut self.storage {
            Storage::F32(values) => {
                let tail = values.split_off(values.len() / total * rows);
                values.shrink_to_fit();
                Storage::F32(tail)
            }
            Storage::Packed { kind, bytes } => {
                let tail = bytes.split_off(bytes.len() / total * rows);
                bytes.shrink_to_fit();
                Storage::Packed {
                    kind: *kind,
                    bytes: tail,
                }
            }
        };
        let right = Self {
            dims: vec![self.dims[0], total - rows],
            storage: tail,
            accelerated: self.accelerated,
            block_elements: self.block_elements,
        };
        self.dims[1] = rows;
        Ok((self, right))
    }
}

pub(crate) struct Gguf {
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
    pub tensor_types: Vec<String>,
    pub file_bytes: u64,
    pub parameters: u64,
    /// Absolute file offset and GGUF type of each tensor's data.
    #[cfg_attr(not(feature = "qtip-research"), allow(dead_code))]
    pub data_locations: HashMap<String, (usize, u32)>,
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

    fn value(&mut self, kind: u64, depth: usize, keep_float_array: bool) -> Result<Option<Value>> {
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
                if element == 6 && keep_float_array {
                    if count > (self.bytes.len() - self.position) / 4 {
                        return Err(invalid("truncated tokenizer scores"));
                    }
                    return Ok(Some(Value::Floats(
                        (0..count)
                            .map(|_| self.integer(4).map(|bits| f32::from_bits(bits as u32)))
                            .collect::<Result<_>>()?,
                    )));
                }
                if element == 7 {
                    let values = self.take(count)?;
                    if values.iter().any(|&value| value > 1) {
                        return Err(invalid("invalid boolean array"));
                    }
                    return Ok(Some(Value::Bools(
                        values.iter().map(|&value| value != 0).collect(),
                    )));
                }
                for _ in 0..count {
                    self.value(element, depth + 1, false)?;
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
            if let Some(value) = self.value(kind, 0, key == "tokenizer.ggml.scores")? {
                metadata.insert(key, value);
            } else if key == "gemma3.attention.sliding_window_pattern" {
                return Err(invalid("unsupported Gemma 3 window pattern array"));
            }
        }
        Ok((tensor_count, metadata))
    }
}

/// QTIP3 research tensor (feature `qtip-research`): 16 x 16 trellis tiles
/// (L = 12, K = 3, 25 words each, layout in qtip::packed) in row-group order,
/// for a rotated matrix with 256-blocks of Hadamard on both sides. Its bytes
/// are followed in memory by its companion `<name>.qtip` F32 vector
/// `[sigma, row signs.., column signs..]`, which `parse` folds into it.
pub(crate) const QTIP3_KIND: u32 = 200;
pub(crate) const QTIP3_TILE_BYTES: usize = 100;
/// Suffix of the companion vector of a QTIP3 tensor.
pub(crate) const QTIP3_COMPANION: &str = ".qtip";

/// Projection matrices that may be QTIP3.
fn qtip3_allowed(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return false;
    };
    layer.parse::<usize>().is_ok()
        && matches!(
            suffix,
            "attn_q.weight"
                | "attn_k.weight"
                | "attn_v.weight"
                | "attn_output.weight"
                | "ffn_gate.weight"
                | "ffn_up.weight"
                | "ffn_down.weight"
        )
}

pub(crate) fn layout(kind: u32) -> Result<(usize, usize, &'static str)> {
    match kind {
        QTIP3_KIND if cfg!(feature = "qtip-research") => Ok((256, QTIP3_TILE_BYTES, "QTIP3")),
        0 => Ok((1, 4, "F32")),
        1 => Ok((1, 2, "F16")),
        2 => Ok((32, 18, "Q4_0")),
        3 => Ok((32, 20, "Q4_1")),
        6 => Ok((32, 22, "Q5_0")),
        7 => Ok((32, 24, "Q5_1")),
        8 => Ok((32, 34, "Q8_0")),
        10 => Ok((256, 84, "Q2_K")),
        11 => Ok((256, 110, "Q3_K")),
        12 => Ok((256, 144, "Q4_K")),
        13 => Ok((256, 176, "Q5_K")),
        14 => Ok((256, 210, "Q6_K")),
        20 => Ok((32, 18, "IQ4_NL")),
        30 => Ok((1, 2, "BF16")),
        _ => Err(invalid(format!("unsupported tensor type {kind}"))),
    }
}

#[inline(always)]
pub(crate) fn half(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn finite_packed(kind: u32, bytes: &[u8]) -> bool {
    if kind == QTIP3_KIND {
        // Trellis bits only; the scale is checked with the companion vector.
        return true;
    }
    if kind == 30 {
        return bf16(bytes).is_finite();
    }
    // Integer quants/scales are bounded: finite F16 scales cannot overflow F32 products.
    half(
        bytes,
        match kind {
            10 => 80,
            11 => 108,
            14 => 208,
            _ => 0,
        },
    )
    .is_finite()
        && (!matches!(kind, 3 | 7 | 12 | 13) || half(bytes, 2).is_finite())
        && (kind != 10 || half(bytes, 82).is_finite())
}

#[inline(always)]
fn bf16(bytes: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
}

#[inline(always)]
pub(crate) fn decode(kind: u32, bytes: &[u8], output: &mut [f32]) {
    match kind {
        0 => output[0] = f32::from_le_bytes(bytes.try_into().expect("validated F32 block")),
        1 => output[0] = half(bytes, 0),
        30 => output[0] = bf16(bytes),
        2 => {
            let scale = half(bytes, 0);
            for (i, value) in output.iter_mut().enumerate() {
                let quant = (bytes[2 + i % 16] >> (4 * (i / 16))) & 15;
                *value = scale * (i32::from(quant) - 8) as f32;
            }
        }
        3 | 7 => {
            let scale = half(bytes, 0);
            let offset = half(bytes, 2);
            let (high, low_offset) = if kind == 7 {
                (
                    u32::from_le_bytes(bytes[4..8].try_into().expect("Q5 high bits")),
                    8,
                )
            } else {
                (0, 4)
            };
            for (i, value) in output.iter_mut().enumerate() {
                let low = (bytes[low_offset + i % 16] >> (4 * (i / 16))) & 15;
                let quant = u32::from(low) | (((high >> i) & 1) << 4);
                *value = scale * quant as f32 + offset;
            }
        }
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
        10 => {
            let scale = half(bytes, 80);
            let minimum = half(bytes, 82);
            for (i, value) in output.iter_mut().enumerate() {
                let group = bytes[i / 16];
                let shift = 2 * ((i % 128) / 32);
                let quant = (bytes[16 + (i / 128) * 32 + i % 32] >> shift) & 3;
                *value = scale * f32::from(group & 15) * f32::from(quant)
                    - minimum * f32::from(group >> 4);
            }
        }
        11 => {
            let scale = half(bytes, 108);
            for (i, value) in output.iter_mut().enumerate() {
                let group = i / 16;
                let low_scale = (bytes[96 + group % 8] >> (4 * (group / 8))) & 15;
                let high_scale = (bytes[104 + group % 4] >> (2 * (group / 4))) & 3;
                let multiplier = i32::from(low_scale | (high_scale << 4)) - 32;
                let shift = i / 32;
                let low = (bytes[32 + (i / 128) * 32 + i % 32] >> (2 * (shift % 4))) & 3;
                let subtract = if bytes[i % 32] & (1 << shift) == 0 {
                    4
                } else {
                    0
                };
                *value = scale * multiplier as f32 * (i32::from(low) - subtract) as f32;
            }
        }
        12 | 13 => {
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
                let low_offset = if kind == 13 { 48 } else { 16 };
                let low = (bytes[low_offset + (i / 64) * 32 + i % 32] >> (4 * (group % 2))) & 15;
                let high = if kind == 13 {
                    ((bytes[16 + i % 32] >> group) & 1) << 4
                } else {
                    0
                };
                let quant = low | high;
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
        20 => {
            const CODEBOOK: [i8; 16] = [
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
            ];
            let scale = half(bytes, 0);
            for (i, value) in output.iter_mut().enumerate() {
                let code = (bytes[2 + i % 16] >> (4 * (i / 16))) & 15;
                *value = scale * f32::from(CODEBOOK[usize::from(code)]);
            }
        }
        _ => unreachable!("tensor type checked before decoding"),
    }
}

/// Appends each QTIP3 tensor's companion `[sigma, row signs, column signs]`
/// (little-endian f32) to its bytes and drops the companion entry.
fn fold_qtip3_companions(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    let names: Vec<String> = tensors
        .iter()
        .filter(|(_, t)| {
            matches!(
                t.storage,
                Storage::Packed {
                    kind: QTIP3_KIND,
                    ..
                }
            )
        })
        .map(|(name, _)| name.clone())
        .collect();
    for name in names {
        let companion = tensors
            .remove(&format!("{name}{QTIP3_COMPANION}"))
            .ok_or_else(|| invalid(format!("{name}: missing QTIP3 companion vector")))?;
        let tensor = tensors.get_mut(&name).expect("listed above");
        let (cols, rows) = (tensor.dims[0], tensor.dims[1]);
        let Storage::F32(values) = companion.storage else {
            return Err(invalid(format!("{name}: QTIP3 companion must be F32")));
        };
        let sigma = values.first().copied().unwrap_or(0.0);
        if values.len() != 1 + rows + cols
            || !(sigma.is_finite() && sigma > 0.0)
            || values[1..].iter().any(|&v| v != 1.0 && v != -1.0)
        {
            return Err(invalid(format!("{name}: invalid QTIP3 companion vector")));
        }
        let Storage::Packed { bytes, .. } = &mut tensor.storage else {
            unreachable!("filtered on packed QTIP3");
        };
        // Exact capacity: device budgets count Vec capacity.
        bytes.reserve_exact(4 * values.len());
        bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }
    if let Some(orphan) = tensors.keys().find(|n| n.ends_with(QTIP3_COMPANION)) {
        return Err(invalid(format!(
            "{orphan}: QTIP3 companion without its tensor"
        )));
    }
    Ok(())
}

/// A tensor's entry in a GGUF directory, with its absolute data range.
#[cfg(feature = "qtip-research")]
pub(crate) struct DirectoryEntry {
    pub name: String,
    pub dims: Vec<usize>,
    pub kind: u32,
    pub data: std::ops::Range<usize>,
}

/// The raw layout of a GGUF file, so a tool can rewrite its tensors and keep
/// the metadata bytes verbatim.
#[cfg(feature = "qtip-research")]
pub(crate) struct Directory {
    pub metadata_count: u64,
    /// Byte range of the metadata key-value section.
    pub metadata: std::ops::Range<usize>,
    pub alignment: usize,
    pub tensors: Vec<DirectoryEntry>,
}

#[cfg(feature = "qtip-research")]
impl Directory {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader { bytes, position: 0 };
        let (tensor_count, metadata) = reader.header()?;
        let metadata_count = u64::from_le_bytes(bytes[16..24].try_into().expect("header read"));
        let metadata_range = 24..reader.position;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::Integer(value)) => *value as usize,
            _ => 32,
        };
        if !alignment.is_power_of_two() {
            return Err(invalid("alignment must be a positive power of two"));
        }
        let mut raw = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let rank = reader.integer(4)?;
            if !(1..=3).contains(&rank) {
                return Err(invalid(
                    "only vectors, matrices, and expert stacks are supported",
                ));
            }
            let dims = (0..rank)
                .map(|_| reader.count())
                .collect::<Result<Vec<_>>>()?;
            let kind = reader.integer(4)? as u32;
            let offset = reader.count()?;
            raw.push((name, dims, kind, offset));
        }
        let start = (reader.position + alignment - 1) & !(alignment - 1);
        let tensors = raw
            .into_iter()
            .map(|(name, dims, kind, offset)| {
                let (block, size, _) = layout(kind)?;
                let elements: usize = dims.iter().product();
                let begin = start + offset;
                let end = begin + elements / block * size;
                if end > bytes.len() {
                    return Err(invalid(format!("{name} extends beyond file")));
                }
                Ok(DirectoryEntry {
                    name,
                    dims,
                    kind,
                    data: begin..end,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            metadata_count,
            metadata: metadata_range,
            alignment,
            tensors,
        })
    }
}

/// Writes a GGUF v3 file with `original`'s metadata bytes and the given
/// tensors (name, dims, type, data) in order.
#[cfg(feature = "qtip-research")]
pub(crate) fn write_gguf(
    path: &Path,
    original: &[u8],
    directory: &Directory,
    tensors: &[(String, Vec<usize>, u32, &[u8])],
) -> Result<()> {
    use std::io::Write;
    let align = |n: usize| (n + directory.alignment - 1) & !(directory.alignment - 1);
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((tensors.len() as u64).to_le_bytes());
    header.extend(directory.metadata_count.to_le_bytes());
    header.extend(&original[directory.metadata.clone()]);
    let mut offset = 0usize;
    for (name, dims, kind, data) in tensors {
        let (block, size, _) = layout(*kind)?;
        if dims.iter().product::<usize>() / block * size != data.len() {
            return Err(invalid(format!(
                "{name}: data size does not match its shape"
            )));
        }
        header.extend((name.len() as u64).to_le_bytes());
        header.extend(name.as_bytes());
        header.extend((dims.len() as u32).to_le_bytes());
        dims.iter()
            .for_each(|&d| header.extend((d as u64).to_le_bytes()));
        header.extend(kind.to_le_bytes());
        header.extend((offset as u64).to_le_bytes());
        offset = align(offset + data.len());
    }
    header.resize(align(header.len()), 0);
    let mut file = std::io::BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    );
    file.write_all(&header)?;
    for (_, _, _, data) in tensors {
        file.write_all(data)?;
        file.write_all(&vec![0u8; align(data.len()) - data.len()])?;
    }
    file.flush()?;
    Ok(())
}

impl Gguf {
    /// Whether any tensor is QTIP3, which only the CUDA backend executes.
    pub fn has_qtip3(&self) -> bool {
        self.tensors.values().any(|t| {
            matches!(
                t.storage,
                Storage::Packed {
                    kind: QTIP3_KIND,
                    ..
                }
            )
        })
    }

    pub fn open(path: &Path, expand: bool) -> Result<Self> {
        Self::parse(&fs::read(path)?, expand)
    }

    /// Header metadata only: reads a growing file prefix (16 MiB first) instead of
    /// the whole model, retrying with more bytes while the header is truncated.
    pub fn read_metadata(path: &Path) -> Result<HashMap<String, Value>> {
        Ok(Self::read_prefix(path, |reader| Ok(reader.header()?.1))?)
    }

    /// Header metadata and tensor names, without reading tensor data.
    pub fn read_layout(path: &Path) -> Result<(HashMap<String, Value>, Vec<String>)> {
        Self::read_prefix(path, |reader| {
            let (tensor_count, metadata) = reader.header()?;
            let mut names = Vec::new();
            for _ in 0..tensor_count {
                names.push(reader.string()?);
                let rank = reader.integer(4)?;
                if !(1..=3).contains(&rank) {
                    return Err(invalid(
                        "only vectors, matrices, and expert stacks are supported",
                    ));
                }
                for _ in 0..rank {
                    reader.count()?;
                }
                reader.integer(4)?;
                reader.count()?;
            }
            Ok((metadata, names))
        })
    }

    fn read_prefix<T>(
        path: &Path,
        read: impl Fn(&mut Reader<'_>) -> Result<T>,
    ) -> Result<T> {
        use std::io::Read;
        let length = fs::metadata(path)?.len();
        let mut size: u64 = 16 << 20;
        loop {
            let take = size.min(length);
            let mut bytes = Vec::with_capacity(take as usize);
            fs::File::open(path)?.take(take).read_to_end(&mut bytes)?;
            let mut reader = Reader {
                bytes: &bytes,
                position: 0,
            };
            match read(&mut reader) {
                Err(Error::Invalid(message)) if message.contains("truncated") && take < length => {
                    size = size.saturating_mul(4);
                }
                result => return result,
            }
        }
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
            if kind == QTIP3_KIND
                && (expand || dims.len() != 2 || dims[1] % 256 != 0 || !qtip3_allowed(&name))
            {
                return Err(invalid(format!(
                    "{name}: QTIP3 needs a 256-aligned projection matrix and packed weights"
                )));
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
        let mut data_locations = HashMap::new();
        for (name, dims, elements, kind, offset, length, label) in entries {
            data_locations.insert(name.clone(), (start + offset, kind));
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
        fold_qtip3_companions(&mut tensors)?;
        tensor_types.sort();
        tensor_types.dedup();
        Ok(Self {
            metadata,
            tensors,
            tensor_types,
            file_bytes: bytes.len() as u64,
            parameters,
            data_locations,
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
    fn metadata_prefix_grows_past_large_headers_and_rejects_truncation() {
        fn text(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        text(&mut bytes, "general.architecture");
        bytes.extend(8u32.to_le_bytes());
        text(&mut bytes, "llama");
        // Larger than the first 16 MiB read, before the key that matters.
        text(&mut bytes, "test.blob");
        bytes.extend(8u32.to_le_bytes());
        text(&mut bytes, &"x".repeat(20 << 20));
        text(&mut bytes, "llama.context_length");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(8192u32.to_le_bytes());
        let path = std::env::temp_dir().join(format!("li-metadata-{}.gguf", std::process::id()));
        fs::write(&path, &bytes).unwrap();
        let metadata = Gguf::read_metadata(&path);
        assert_eq!(crate::model_context_tokens(&path).unwrap(), 8192);
        fs::write(&path, &bytes[..bytes.len() - 2]).unwrap();
        let truncated = Gguf::read_metadata(&path);
        fs::remove_file(&path).unwrap();
        assert!(matches!(
            metadata.unwrap().get("llama.context_length"),
            Some(Value::Integer(8192))
        ));
        assert!(truncated.is_err());
    }

    #[cfg(feature = "qtip-research")]
    fn qtip3_file(name: &str, companion: &[f32]) -> Result<Gguf> {
        let empty = Directory {
            metadata_count: 0,
            metadata: 24..24,
            alignment: 32,
            tensors: Vec::new(),
        };
        let tiles = vec![7u8; 256 * 512 / 256 * QTIP3_TILE_BYTES];
        let companion: Vec<u8> = companion.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = std::env::temp_dir().join(format!(
            "li-qtip3-{}-{}.gguf",
            std::process::id(),
            name.len() + companion.len()
        ));
        let _ = fs::remove_file(&path);
        write_gguf(
            &path,
            &[0; 24],
            &empty,
            &[
                (name.into(), vec![256, 512], QTIP3_KIND, &tiles),
                (
                    format!("{name}{QTIP3_COMPANION}"),
                    vec![companion.len() / 4],
                    0,
                    &companion,
                ),
            ],
        )?;
        let parsed = Gguf::open(&path, false);
        fs::remove_file(&path)?;
        parsed
    }

    #[cfg(feature = "qtip-research")]
    #[test]
    fn qtip3_folds_its_companion_and_rejects_bad_ones() {
        let mut companion = vec![0.5f32];
        companion.extend((0..512 + 256).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }));
        let model = qtip3_file("blk.0.attn_q.weight", &companion).unwrap();
        assert!(model.has_qtip3());
        assert_eq!(model.tensors.len(), 1);
        let Storage::Packed { kind, bytes } = &model.tensors["blk.0.attn_q.weight"].storage else {
            panic!("QTIP3 stays packed");
        };
        assert_eq!(*kind, QTIP3_KIND);
        let tiles = 256 * 512 / 256 * QTIP3_TILE_BYTES;
        assert_eq!(bytes.len(), tiles + 4 * companion.len());
        assert_eq!(bytes[tiles..tiles + 4], 0.5f32.to_le_bytes());
        // Wrong length, a non-sign entry, a bad scale, a non-projection name.
        let mut bad = companion.clone();
        bad.pop();
        assert!(qtip3_file("blk.0.attn_q.weight", &bad).is_err());
        let mut bad = companion.clone();
        bad[3] = 0.5;
        assert!(qtip3_file("blk.0.attn_q.weight", &bad).is_err());
        let mut bad = companion.clone();
        bad[0] = f32::NAN;
        assert!(qtip3_file("blk.0.attn_q.weight", &bad).is_err());
        assert!(qtip3_file("token_embd.weight", &companion).is_err());
    }

    #[test]
    fn split_packed_matrix_preserves_whole_rows() {
        let tensor = Tensor {
            dims: vec![256, 3],
            storage: Storage::Packed {
                kind: 12,
                bytes: [1, 2, 3].into_iter().flat_map(|b| [b; 144]).collect(),
            },
            accelerated: true,
            block_elements: 256,
        };
        let (first, tail) = tensor.split_rows(1).unwrap();
        let (second, third) = tail.split_rows(1).unwrap();
        for (tensor, byte) in [first, second, third].into_iter().zip([1, 2, 3]) {
            assert_eq!(tensor.dims, [256, 1]);
            assert!(tensor.accelerated);
            assert_eq!(tensor.block_elements, 256);
            assert!(
                matches!(tensor.storage, Storage::Packed { kind: 12, bytes } if bytes == [byte; 144])
            );
        }
    }

    #[test]
    fn q5_k_uses_one_high_bit_per_group_and_lane() {
        let mut packed = [0; 176];
        packed[..2].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        packed[2..4].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        packed[4..8].fill(2);
        packed[8..12].fill(3);
        packed[12..16].fill(0x32);
        packed[16..24]
            .iter_mut()
            .enumerate()
            .for_each(|(group, byte)| *byte = 1 << group);
        packed[48..].fill(0x41);
        let mut output = [0.0; 256];
        decode(13, &packed, &mut output);
        for (group, values) in output.chunks_exact(32).enumerate() {
            let baseline = if group % 2 == 0 { -1.0 } else { 5.0 };
            for (lane, &value) in values.iter().enumerate() {
                assert_eq!(value, baseline + if lane == group { 32.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn packed_validation_rejects_nonfinite_scales_and_accepts_extreme_finite_products() {
        for kind in [1, 6, 8, 12, 13, 14] {
            let (block, size, _) = layout(kind).unwrap();
            let mut bytes = vec![255; size];
            let offset = if kind == 14 { 208 } else { 0 };
            bytes[offset..offset + 2].copy_from_slice(&f16::MAX.to_bits().to_le_bytes());
            if matches!(kind, 12 | 13) {
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
            if matches!(kind, 12 | 13) {
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
            assert!(reader.value(9, 0, false).is_err());
        }
        let mut reader = Reader {
            bytes: &[2],
            position: 0,
        };
        assert!(reader.value(7, 0, false).is_err());
    }

    #[test]
    fn bf16_preserves_subnormals_and_rejects_nonfinite_weights() {
        for bits in [0x8000u16, 0x0001, 0x7f7f] {
            let block = bits.to_le_bytes();
            let mut output = [0.0];
            assert!(finite_packed(30, &block));
            decode(30, &block, &mut output);
            assert_eq!(output[0].to_bits(), u32::from(bits) << 16);
        }
        for bits in [0x7f80u16, 0xff81] {
            assert!(!finite_packed(30, &bits.to_le_bytes()));
        }
    }

    #[test]
    fn decodes_q2_k_planes_and_subgroup_minima() {
        let mut block = [0; 84];
        block[0] = 0x21;
        block[2] = 0x43;
        block[8] = 0x65;
        block[16] = 0x0b;
        block[48] = 0x02;
        block[80..82].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[82..84].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        assert!(finite_packed(10, &block));
        let mut output = [0.0; 256];
        decode(10, &block, &mut output);
        assert_eq!((output[0], output[32], output[128]), (1.0, 2.0, 3.5));
        block[82..84].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(!finite_packed(10, &block));
    }

    #[test]
    fn decodes_q3_k_planes_and_signed_subgroup_scales() {
        let mut block = [0; 110];
        block[0] = 0x11;
        block[32] = 3;
        block[64] = 2;
        block[96] = 0x61;
        block[104] = 0x12;
        block[108..].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        let mut output = [0.0; 256];
        decode(11, &block, &mut output);
        assert_eq!(
            (output[0], output[1], output[128], output[129]),
            (1.5, -2.0, -10.0, 20.0)
        );
        block[108..].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(!finite_packed(11, &block));
    }

    #[test]
    fn decodes_iq4_nl_nonuniform_nibbles() {
        let mut block = [0; 18];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2] = 0xf0;
        block[17] = 0x87;
        let mut output = [0.0; 32];
        decode(20, &block, &mut output);
        assert_eq!(
            (output[0], output[15], output[16], output[31]),
            (-63.5, -5.0, 56.5, 0.5)
        );
    }

    #[test]
    fn decodes_q4_0_split_nibbles_with_signed_scale() {
        let mut block = [0; 18];
        block[..2].copy_from_slice(&f16::from_f32(-0.5).to_bits().to_le_bytes());
        block[2] = 0xf1;
        block[17] = 0x28;
        let mut output = [0.0; 32];
        decode(2, &block, &mut output);
        assert_eq!(
            (output[0], output[15], output[16], output[31]),
            (3.5, 0.0, -3.5, 3.0)
        );
    }

    #[test]
    fn decodes_q4_1_split_nibbles_with_affine_offset() {
        let mut block = [0; 20];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(-3.0).to_bits().to_le_bytes());
        block[4] = 0xf1;
        block[19] = 0x28;
        let mut output = [0.0; 32];
        assert!(finite_packed(3, &block));
        decode(3, &block, &mut output);
        assert_eq!(
            (output[0], output[15], output[16], output[31]),
            (-2.5, 1.0, 4.5, -2.0)
        );
        block[2..4].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(!finite_packed(3, &block));
    }

    #[test]
    fn decodes_q5_1_high_bits_with_affine_offset() {
        let mut block = [0; 24];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(-3.0).to_bits().to_le_bytes());
        block[4..8].copy_from_slice(&0x8000_0001u32.to_le_bytes());
        block[8] = 0xf1;
        block[23] = 0x28;
        let mut output = [0.0; 32];
        assert!(finite_packed(7, &block));
        decode(7, &block, &mut output);
        assert_eq!(
            (output[0], output[15], output[16], output[31]),
            (5.5, 1.0, 4.5, 6.0)
        );
        block[2..4].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(!finite_packed(7, &block));
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
