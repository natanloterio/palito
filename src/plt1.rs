//! Palito GPU-native binary model container (`PLT1` / `.palito`).
//!
//! Designed specifically for Blackwell SM_120 (RTX 5090) zero-copy loading and
//! Tensor-Core / LUT-native execution:
//! - 4 KiB-aligned header & 64B-aligned section directory (direct `O_DIRECT` / `mmap`)
//! - 128B-aligned tensor payloads (exact GDDR7 cache line & TMA swizzle alignment)
//! - Native metadata for QTIP HYB (`Q=5` warp-register LUT or `Q=6..9` SMEM LUT,
//!   1.5 / 2.0 / 2.5 / 3.0 bpw circular tiles pre-swizzled for `mma.m16n8k16`),
//!   Blackwell `NVFP4_BLK16` (`128x4` scales), offline-folded RHT invariance flags,
//!   frequency-ordered LM head (`VOCAB_FREQ_MAP`), and optional embedded
//!   `EAGLE3_SPEC_HEAD` + `STATIC_SUFFIX_TRIE`.

use crate::{Error, Result};
use std::io::{Read, Seek, SeekFrom, Write};

pub const PLT1_MAGIC: [u8; 8] = *b"PALITO1\0";
pub const PLT1_VERSION: u32 = 1;
pub const HEADER_ALIGN: usize = 4096;
pub const PAYLOAD_ALIGN: usize = 128;

/// Global model-level feature flags in `Plt1Header::flags`.
pub mod flags {
    /// RMSNorm weights folded offline into following projections.
    pub const FOLDED_RMSNORM: u32 = 1 << 0;
    /// Global residual-stream Hadamard rotation R_1 folded offline (zero residual RHT).
    pub const FOLDED_RESIDUAL_RHT: u32 = 1 << 1;
    /// Per-head V->O Hadamard rotation R_2 folded offline.
    pub const FOLDED_VO_RHT: u32 = 1 << 2;
    /// `TARGET_LM_HEAD` rows are sorted in descending corpus frequency order.
    pub const FREQ_ORDERED_LM_HEAD: u32 = 1 << 3;
    /// File bundles an embedded `EAGLE3_SPEC_HEAD` section.
    pub const HAS_EAGLE3: u32 = 1 << 4;
    /// File bundles a `STATIC_SUFFIX_TRIE` section.
    pub const HAS_SUFFIX_TRIE: u32 = 1 << 5;
    /// Weight tiles are pre-swizzled offline for `mma.sync.aligned.m16n8k16`.
    pub const MMA_M16N8K16_SWIZZLED: u32 = 1 << 6;
}

/// Typed section kinds in the PLT1 Section Directory.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionKind {
    MetadataJson = 1,
    TensorDirectory = 2,
    TargetEmbedTok = 3,
    LayerSlabs = 4,
    TargetLmHead = 5,
    VocabFreqMap = 6,
    Eagle3SpecHead = 7,
    StaticSuffixTrie = 8,
    /// Embedded skeleton GGUF (metadata, tokenizer, embeddings, LM head, norms).
    BaseGguf = 9,
}

impl SectionKind {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            1 => Ok(Self::MetadataJson),
            2 => Ok(Self::TensorDirectory),
            3 => Ok(Self::TargetEmbedTok),
            4 => Ok(Self::LayerSlabs),
            5 => Ok(Self::TargetLmHead),
            6 => Ok(Self::VocabFreqMap),
            7 => Ok(Self::Eagle3SpecHead),
            8 => Ok(Self::StaticSuffixTrie),
            9 => Ok(Self::BaseGguf),
            _ => Err(Error::Invalid(format!("unknown PLT1 section kind {v}"))),
        }
    }
}

/// Returns true if `path` starts with the `PALITO1\0` magic header.
pub fn is_plt1_file(path: &std::path::Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).is_ok() && magic == PLT1_MAGIC
}

/// Returns `(offset, size)` of the embedded `BaseGguf` section in a `.palito` file.
pub fn base_gguf_range(path: &std::path::Path) -> Result<(u64, u64)> {
    let mut f = std::fs::File::open(path)?;
    let index = Plt1Index::read_from(&mut f)?;
    let sec = index
        .sections
        .iter()
        .find(|s| s.kind == SectionKind::BaseGguf)
        .ok_or_else(|| Error::Invalid("PLT1 file has no BaseGguf section".into()))?;
    Ok((sec.offset, sec.size))
}

/// Weight encoding codecs supported by PLT1.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    F16 = 0,
    Q8_0 = 1,
    Q6K = 2,
    Q4K = 3,
    /// QTIP HYB (`L=16, V=2`) with warp-register `Q=5` LUT (0 KB SMEM) & circular tiles.
    QtipHybQ5 = 10,
    /// QTIP HYB (`L=16, V=2`) with bank-replicated SMEM LUT (`Q=6..9`).
    QtipHybSmem = 11,
    /// Blackwell SM_120 native NVFP4 (E2M1 + UE4M3 block-16 scales in `128x4` layout).
    NvFp4Blk16 = 20,
    /// Blackwell SM_120 native MXFP4 (E2M1 + UE8M0 block-32 scales).
    MxFp4Blk32 = 21,
}

impl Codec {
    fn from_u16(v: u16) -> Result<Self> {
        match v {
            0 => Ok(Self::F16),
            1 => Ok(Self::Q8_0),
            2 => Ok(Self::Q6K),
            3 => Ok(Self::Q4K),
            10 => Ok(Self::QtipHybQ5),
            11 => Ok(Self::QtipHybSmem),
            20 => Ok(Self::NvFp4Blk16),
            21 => Ok(Self::MxFp4Blk32),
            _ => Err(Error::Invalid(format!("unknown PLT1 codec {v}"))),
        }
    }
}

/// Per-tensor runtime flags (e.g. which RHTs are already folded offline).
pub mod tensor_flags {
    pub const SKIP_IN_RHT: u16 = 1 << 0;
    pub const SKIP_OUT_RHT: u16 = 1 << 1;
    pub const MMA_SWIZZLED: u16 = 1 << 2;
    pub const COL_MAJOR_SPARSE: u16 = 1 << 3;
}

/// Fixed 64-byte file header (padded to 4 KiB on disk).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plt1Header {
    pub version: u32,
    pub target_sm: u32,
    pub flags: u32,
    pub num_sections: u32,
    pub section_dir_offset: u64,
    pub n_layers: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub vocab_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionEntry {
    pub kind: SectionKind,
    pub flags: u32,
    pub offset: u64,
    pub size: u64,
}

/// Compact descriptor for one tensor inside `LayerSlabs`, `TargetEmbedTok`,
/// `TargetLmHead`, or `Eagle3SpecHead`.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorDesc {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub codec: Codec,
    pub flags: u16,
    /// For HYB: bits per pair (`3, 4, 5, 6` -> `1.5, 2.0, 2.5, 3.0` bpw).
    pub pair_bits: u8,
    /// For HYB: LUT index bits `Q` (`5..=9`).
    pub lut_bits: u8,
    pub scale: f32,
    /// Optional TEAL dynamic activation sparsity thresholds (`[t_25, t_40, t_50, t_60]`).
    pub teal_thresholds: [f32; 4],
    /// Byte offset relative to the start of the file (128B-aligned).
    pub data_offset: u64,
    pub data_size: u64,
    /// Optional byte offset of the 2^Q `u32` HYB codebook (0 if none).
    pub lut_offset: u64,
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

fn write_padding(w: &mut impl Write, current_pos: usize, align: usize) -> Result<usize> {
    let target = align_up(current_pos, align);
    if target > current_pos {
        let zeros = vec![0u8; target - current_pos];
        w.write_all(&zeros)?;
    }
    Ok(target)
}

/// Builder for writing a valid 4KiB/128B-aligned `.palito` (`PLT1`) file.
pub struct Plt1Writer {
    header: Plt1Header,
    sections: Vec<(SectionKind, u32, Vec<u8>)>,
    tensors: Vec<(TensorDesc, Vec<u8>, Option<Vec<u32>>)>,
}

impl Plt1Writer {
    pub fn new(header: Plt1Header) -> Self {
        Self {
            header,
            sections: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn add_section(&mut self, kind: SectionKind, flags: u32, data: Vec<u8>) {
        self.sections.push((kind, flags, data));
    }

    pub fn add_tensor(&mut self, desc: TensorDesc, data: Vec<u8>, lut: Option<Vec<u32>>) {
        self.tensors.push((desc, data, lut));
    }

    /// Serializes the full PLT1 container to `out`, filling exact 128B-aligned
    /// offsets for every section, tensor payload, and HYB LUT.
    pub fn write_to<W: Write + Seek>(mut self, out: &mut W) -> Result<Plt1Header> {
        // Reserve 4 KiB for header + section directory (written at the end).
        out.write_all(&vec![0u8; HEADER_ALIGN])?;
        let mut cursor = HEADER_ALIGN;

        let mut entries = Vec::new();

        // 1. Write raw sections (metadata, vocab map, EAGLE-3, suffix trie, etc.).
        for (kind, sflags, data) in self.sections {
            cursor = write_padding(out, cursor, PAYLOAD_ALIGN)?;
            let offset = cursor as u64;
            out.write_all(&data)?;
            cursor += data.len();
            entries.push(SectionEntry {
                kind,
                flags: sflags,
                offset,
                size: data.len() as u64,
            });
        }

        // 2. Write all tensor payloads + optional HYB LUTs into a contiguous LayerSlabs region.
        if !self.tensors.is_empty() {
            cursor = write_padding(out, cursor, PAYLOAD_ALIGN)?;
            let slabs_start = cursor as u64;
            let mut finalized_descs = Vec::with_capacity(self.tensors.len());
            for (mut desc, data, lut) in self.tensors {
                cursor = write_padding(out, cursor, PAYLOAD_ALIGN)?;
                desc.data_offset = cursor as u64;
                desc.data_size = data.len() as u64;
                out.write_all(&data)?;
                cursor += data.len();

                if let Some(lut_words) = lut {
                    cursor = write_padding(out, cursor, PAYLOAD_ALIGN)?;
                    desc.lut_offset = cursor as u64;
                    for w in lut_words {
                        out.write_all(&w.to_le_bytes())?;
                        cursor += 4;
                    }
                } else {
                    desc.lut_offset = 0;
                }
                finalized_descs.push(desc);
            }
            let slabs_size = cursor as u64 - slabs_start;
            entries.push(SectionEntry {
                kind: SectionKind::LayerSlabs,
                flags: 0,
                offset: slabs_start,
                size: slabs_size,
            });

            // Serialize TensorDirectory section.
            let mut dir_bytes = Vec::new();
            dir_bytes.extend_from_slice(&(finalized_descs.len() as u32).to_le_bytes());
            for d in &finalized_descs {
                let name_bytes = d.name.as_bytes();
                dir_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                dir_bytes.extend_from_slice(name_bytes);
                dir_bytes.extend_from_slice(&d.rows.to_le_bytes());
                dir_bytes.extend_from_slice(&d.cols.to_le_bytes());
                dir_bytes.extend_from_slice(&(d.codec as u16).to_le_bytes());
                dir_bytes.extend_from_slice(&d.flags.to_le_bytes());
                dir_bytes.push(d.pair_bits);
                dir_bytes.push(d.lut_bits);
                dir_bytes.extend_from_slice(&d.scale.to_le_bytes());
                for t in d.teal_thresholds {
                    dir_bytes.extend_from_slice(&t.to_le_bytes());
                }
                dir_bytes.extend_from_slice(&d.data_offset.to_le_bytes());
                dir_bytes.extend_from_slice(&d.data_size.to_le_bytes());
                dir_bytes.extend_from_slice(&d.lut_offset.to_le_bytes());
            }
            cursor = write_padding(out, cursor, PAYLOAD_ALIGN)?;
            let dir_off = cursor as u64;
            out.write_all(&dir_bytes)?;
            entries.push(SectionEntry {
                kind: SectionKind::TensorDirectory,
                flags: 0,
                offset: dir_off,
                size: dir_bytes.len() as u64,
            });
        }

        // 3. Rewind and write Header (64 bytes) + Section Directory (at offset 64).
        self.header.version = PLT1_VERSION;
        self.header.num_sections = entries.len() as u32;
        self.header.section_dir_offset = 64;

        let dir_bytes_len = entries.len() * 24;
        if 64 + dir_bytes_len > HEADER_ALIGN {
            return Err(Error::Invalid("too many PLT1 sections for 4 KiB header".into()));
        }

        out.seek(SeekFrom::Start(0))?;
        out.write_all(&PLT1_MAGIC)?;
        out.write_all(&self.header.version.to_le_bytes())?;
        out.write_all(&self.header.target_sm.to_le_bytes())?;
        out.write_all(&self.header.flags.to_le_bytes())?;
        out.write_all(&self.header.num_sections.to_le_bytes())?;
        out.write_all(&self.header.section_dir_offset.to_le_bytes())?;
        out.write_all(&self.header.n_layers.to_le_bytes())?;
        out.write_all(&self.header.d_model.to_le_bytes())?;
        out.write_all(&self.header.d_ff.to_le_bytes())?;
        out.write_all(&self.header.vocab_size.to_le_bytes())?;
        // Pad header to 64 bytes.
        out.write_all(&[0u8; 16])?;

        for e in &entries {
            out.write_all(&(e.kind as u32).to_le_bytes())?;
            out.write_all(&e.flags.to_le_bytes())?;
            out.write_all(&e.offset.to_le_bytes())?;
            out.write_all(&e.size.to_le_bytes())?;
        }
        Ok(self.header)
    }
}

/// Parsed PLT1 container index (zero-copy header + directory + tensor descriptors).
#[derive(Clone, Debug)]
pub struct Plt1Index {
    pub header: Plt1Header,
    pub sections: Vec<SectionEntry>,
    pub tensors: Vec<TensorDesc>,
}

impl Plt1Index {
    pub fn read_from<R: Read + Seek>(r: &mut R) -> Result<Self> {
        r.seek(SeekFrom::Start(0))?;
        let mut hdr = [0u8; 64];
        r.read_exact(&mut hdr)?;
        if hdr[0..8] != PLT1_MAGIC {
            return Err(Error::Invalid("not a PLT1 (.palito) file".into()));
        }
        let u32_at = |off: usize| u32::from_le_bytes(hdr[off..off + 4].try_into().unwrap());
        let u64_at = |off: usize| u64::from_le_bytes(hdr[off..off + 8].try_into().unwrap());

        let header = Plt1Header {
            version: u32_at(8),
            target_sm: u32_at(12),
            flags: u32_at(16),
            num_sections: u32_at(20),
            section_dir_offset: u64_at(24),
            n_layers: u32_at(32),
            d_model: u32_at(36),
            d_ff: u32_at(40),
            vocab_size: u32_at(44),
        };
        if header.version != PLT1_VERSION {
            return Err(Error::Invalid(format!(
                "unsupported PLT1 version {}",
                header.version
            )));
        }

        r.seek(SeekFrom::Start(header.section_dir_offset))?;
        let mut sections = Vec::with_capacity(header.num_sections as usize);
        for _ in 0..header.num_sections {
            let mut buf = [0u8; 24];
            r.read_exact(&mut buf)?;
            sections.push(SectionEntry {
                kind: SectionKind::from_u32(u32::from_le_bytes(buf[0..4].try_into().unwrap()))?,
                flags: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
                offset: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
                size: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            });
        }

        let mut tensors = Vec::new();
        if let Some(dir) = sections
            .iter()
            .find(|s| s.kind == SectionKind::TensorDirectory)
        {
            r.seek(SeekFrom::Start(dir.offset))?;
            let mut buf = vec![0u8; dir.size as usize];
            r.read_exact(&mut buf)?;
            let mut p = 0usize;
            let count = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            for _ in 0..count {
                let nlen = u16::from_le_bytes(buf[p..p + 2].try_into().unwrap()) as usize;
                p += 2;
                let name = String::from_utf8(buf[p..p + nlen].to_vec())
                    .map_err(|e| Error::Invalid(e.to_string()))?;
                p += nlen;
                let rows = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
                p += 4;
                let cols = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
                p += 4;
                let codec = Codec::from_u16(u16::from_le_bytes(buf[p..p + 2].try_into().unwrap()))?;
                p += 2;
                let tflags = u16::from_le_bytes(buf[p..p + 2].try_into().unwrap());
                p += 2;
                let pair_bits = buf[p];
                let lut_bits = buf[p + 1];
                p += 2;
                let scale = f32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
                p += 4;
                let mut teal_thresholds = [0.0f32; 4];
                for slot in &mut teal_thresholds {
                    *slot = f32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
                    p += 4;
                }
                let data_offset = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap());
                p += 8;
                let data_size = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap());
                p += 8;
                let lut_offset = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap());
                p += 8;
                tensors.push(TensorDesc {
                    name,
                    rows,
                    cols,
                    codec,
                    flags: tflags,
                    pair_bits,
                    lut_bits,
                    scale,
                    teal_thresholds,
                    data_offset,
                    data_size,
                    lut_offset,
                });
            }
        }

        Ok(Self {
            header,
            sections,
            tensors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn plt1_round_trip_preserves_alignment_tensors_and_sections() {
        let hdr = Plt1Header {
            version: PLT1_VERSION,
            target_sm: 1200,
            flags: flags::FOLDED_RMSNORM
                | flags::FOLDED_RESIDUAL_RHT
                | flags::MMA_M16N8K16_SWIZZLED
                | flags::HAS_EAGLE3,
            num_sections: 0,
            section_dir_offset: 0,
            n_layers: 32,
            d_model: 4096,
            d_ff: 14336,
            vocab_size: 128256,
        };
        let mut writer = Plt1Writer::new(hdr.clone());
        writer.add_section(
            SectionKind::MetadataJson,
            0,
            br#"{"arch":"llama3","quant":"hyb-q5"}"#.to_vec(),
        );
        writer.add_section(SectionKind::Eagle3SpecHead, 0, vec![0xAB; 512]);

        let lut: Vec<u32> = (0..32u32).map(|i| 0x3C00_3C00u32.wrapping_add(i)).collect();
        writer.add_tensor(
            TensorDesc {
                name: "blk.0.qkv_super".into(),
                rows: 6144,
                cols: 4096,
                codec: Codec::QtipHybQ5,
                flags: tensor_flags::SKIP_IN_RHT | tensor_flags::MMA_SWIZZLED,
                pair_bits: 4,
                lut_bits: 5,
                scale: 0.042,
                teal_thresholds: [0.05, 0.12, 0.21, 0.35],
                data_offset: 0,
                data_size: 0,
                lut_offset: 0,
            },
            vec![0x5Au8; 1024],
            Some(lut.clone()),
        );

        let mut buf = Cursor::new(Vec::new());
        writer.write_to(&mut buf).unwrap();
        let parsed = Plt1Index::read_from(&mut buf).unwrap();

        assert_eq!(parsed.header.target_sm, 1200);
        assert_eq!(parsed.header.n_layers, 32);
        assert_eq!(parsed.header.flags & flags::FOLDED_RESIDUAL_RHT, flags::FOLDED_RESIDUAL_RHT);
        assert_eq!(parsed.tensors.len(), 1);

        let t = &parsed.tensors[0];
        assert_eq!(t.name, "blk.0.qkv_super");
        assert_eq!(t.codec, Codec::QtipHybQ5);
        assert_eq!(t.pair_bits, 4);
        assert_eq!(t.lut_bits, 5);
        assert_eq!(t.data_offset % PAYLOAD_ALIGN as u64, 0);
        assert_eq!(t.lut_offset % PAYLOAD_ALIGN as u64, 0);
        assert_eq!(t.data_size, 1024);

        // Verify LUT bytes on disk match the original codebook bit-for-bit.
        let raw = buf.into_inner();
        let lut_off = t.lut_offset as usize;
        let read_lut: Vec<u32> = raw[lut_off..lut_off + lut.len() * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(read_lut, lut);
    }
}
