//! Bundles a base GGUF (tokenizer, metadata, embeddings, LM head, norms),
//! a PLT1 HYB Q=5 layers file, and an optional PLT1 EAGLE-3 draft head into
//! ONE self-contained `.palito` file that `palito chat`, `palito serve`, and
//! `palito generate` load directly for 1,000+ tok/s out of the box!

#[cfg(not(feature = "qtip-research"))]
fn main() {
    eprintln!("pack-palito requires --features qtip-research");
    std::process::exit(2);
}

#[cfg(feature = "qtip-research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use palito::plt1::{Plt1Index, Plt1Writer, SectionKind, flags};
    use std::{
        fs::File,
        io::{BufWriter, Read, Seek, SeekFrom},
        path::PathBuf,
    };

    #[derive(Parser)]
    struct Args {
        /// Base GGUF providing tokenizer, metadata, embed_tokens, output.weight, norms.
        #[arg(long)]
        base_gguf: PathBuf,
        /// PLT1 (.palito) file containing the quantized HYB Q=5 layer projections.
        #[arg(long)]
        plt1_layers: PathBuf,
        /// Optional PLT1 (.palito) file containing the EAGLE-3 draft head.
        #[arg(long)]
        eagle3: Option<PathBuf>,
        /// Output self-contained `.palito` bundle path.
        #[arg(long)]
        output: PathBuf,
    }

    let args = Args::parse();

    // 1. Strip the 224 layer projections (+ .qtip companions) from base_gguf into a
    //    compact skeleton GGUF (~850 MB for 8B Q6_K embed+head).
    let skeleton_tmp = args.output.with_extension("skeleton.gguf.tmp");
    palito::qtip::plt1_export::strip_projections_to_skeleton_gguf(&args.base_gguf, &skeleton_tmp)?;
    let skeleton_bytes = std::fs::read(&skeleton_tmp)?;
    let _ = std::fs::remove_file(&skeleton_tmp);
    println!(
        "Skeleton GGUF (tokenizer + metadata + embed + lm_head + norms): {:.2} MB",
        skeleton_bytes.len() as f64 / 1e6
    );

    // 2. Read PLT1 layers index and start unified Plt1Writer.
    let mut layers_file = File::open(&args.plt1_layers)?;
    let layers_idx = Plt1Index::read_from(&mut layers_file)?;
    let mut header = layers_idx.header.clone();
    if args.eagle3.is_some() {
        header.flags |= flags::HAS_EAGLE3;
    }

    let mut writer = Plt1Writer::new(header);
    writer.add_section(SectionKind::BaseGguf, 0, skeleton_bytes);

    let read_slice = |f: &mut File, off: u64, size: u64| -> std::io::Result<Vec<u8>> {
        f.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; size as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    };

    // Copy MetadataJson from layers file if present.
    if let Some(meta) = layers_idx
        .sections
        .iter()
        .find(|s| s.kind == SectionKind::MetadataJson)
    {
        let data = read_slice(&mut layers_file, meta.offset, meta.size)?;
        writer.add_section(SectionKind::MetadataJson, meta.flags, data);
    }

    // Copy all HYB layer tensors + LUTs + sign vectors.
    for desc in &layers_idx.tensors {
        let data = read_slice(&mut layers_file, desc.data_offset, desc.data_size)?;
        let lut = if desc.lut_offset != 0 && desc.lut_bits > 0 {
            let lut_bytes = read_slice(&mut layers_file, desc.lut_offset, (1u64 << desc.lut_bits) * 4)?;
            Some(
                lut_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        } else {
            None
        };
        writer.add_tensor(desc.clone(), data, lut);
    }

    // 3. Optionally embed the EAGLE-3 section + EAGLE-3 tensors.
    if let Some(eagle_path) = &args.eagle3 {
        let mut ef = File::open(eagle_path)?;
        let eidx = Plt1Index::read_from(&mut ef)?;
        let esec = eidx
            .sections
            .iter()
            .find(|s| s.kind == SectionKind::Eagle3SpecHead)
            .ok_or("eagle3 file has no Eagle3SpecHead section")?;
        let edata = read_slice(&mut ef, esec.offset, esec.size)?;
        writer.add_section(SectionKind::Eagle3SpecHead, esec.flags, edata);
        for desc in &eidx.tensors {
            let data = read_slice(&mut ef, desc.data_offset, desc.data_size)?;
            writer.add_tensor(desc.clone(), data, None);
        }
        println!("Embedded EAGLE-3 draft head ({} tensors)", eidx.tensors.len());
    }

    let partial = args.output.with_extension("palito.partial");
    {
        let out_file = File::create(&partial)?;
        let mut bw = BufWriter::new(out_file);
        writer.write_to(&mut bw)?;
    }
    std::fs::rename(&partial, &args.output)?;
    let total_mb = std::fs::metadata(&args.output)?.len() as f64 / 1e6;
    println!(
        "Wrote self-contained bundle: {} ({:.2} MB)",
        args.output.display(),
        total_mb
    );
    Ok(())
}
