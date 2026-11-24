use super::*;

/// Runs `$fallback` unless `$t` is QTIP3, which takes the batched tensor-core
/// path (`qtip::QtipKernel::matmul`) instead.
macro_rules! qtip_or {
    ($self:ident, $t:expr, $x:expr, $y:expr, $n:expr, $fallback:block) => {{
        #[cfg(feature = "qtip-research")]
        if $t.is_qtip() {
            $self
                .qtip
                .as_mut()
                .ok_or_else(|| failure("QTIP3 kernels are not loaded"))?
                .matmul(&$self.stream, $t, $x, $y, $n as usize)?;
        } else $fallback
        #[cfg(not(feature = "qtip-research"))]
        $fallback
    }};
}

fn reference_dense_shape(
    matmul: CudaMatmul,
    kind: i32,
    width: i32,
    rows: i32,
    tokens: i32,
) -> bool {
    matmul.f32_tiled()
        && tokens >= 256
        && matches!(kind, 12 | 14)
        && matches!((width, rows), (2048, 4096) | (4096, 2048) | (2048, 512))
}

/// Register-blocked packed GEMM (gemm.cu): tiled profile, K-quant rows, enough
/// tokens to fill 64-token tiles.
fn packed_gemm_shape(matmul: CudaMatmul, tensor: &DeviceTensor, tokens: i32) -> bool {
    matmul.f32_tiled()
        && tokens >= 64
        && matches!(tensor.kind, 12 | 14)
        && tensor.width % 256 == 0
}

fn residual_projection(matmul: CudaMatmul, tensor: &DeviceTensor, tokens: i32) -> bool {
    matmul.residual_components() > 0
        && tokens >= matmul.residual_min_tokens() as i32
        && (matches!(tensor.kind, 12 | 14)
            || (matches!(matmul, CudaMatmul::Int8Prefill | CudaMatmul::Int8Fast)
                && tensor.kind == 13))
        && tensor.width % 256 == 0
}

fn dense_mma3_splits(width: i32, rows: i32, tokens: i32, scratch_floats: usize) -> u32 {
    if width <= 0 || rows <= 0 || tokens <= 0 {
        return 1;
    }
    let base_ctas = (rows as u32).div_ceil(32) * (tokens as u32).div_ceil(32);
    let mut splits = 1u32;
    while (width as u32) % (splits * 2 * 128) == 0
        && base_ctas * (splits * 2) <= 1024
        && (splits as usize * 2)
            .saturating_mul(tokens as usize)
            .saturating_mul(rows as usize)
            <= scratch_floats
    {
        splits *= 2;
    }
    splits
}

// int8-prefill quantizes attention-projection inputs as the S16 pair and FFN
// inputs (expert gate/up/down) as three components: the pair's ~16-bit
// activations flip Qwen3-30B expert routing on a 2.9k prompt (logit max_abs 0.73,
// first token changed), three components on the FFN keep F32 tokens.
macro_rules! prepare_residual {
    ($backend:ident,$input:expr,$count:expr) => {
        prepare_residual!($backend, $input, $count, false)
    };
    ($backend:ident,$input:expr,$count:expr,$pair:expr) => {{
        let count: i32 = $count;
        if $backend.matmul.integer_mma() {
            let format = match $backend.matmul {
                CudaMatmul::Int8Fast => ResidualFormat::Single,
                CudaMatmul::Int8Prefill if $pair => ResidualFormat::Pair,
                _ => ResidualFormat::Three,
            };
            let work = $backend
                .work
                .residual_mma3
                .as_mut()
                .ok_or_else(|| failure("residual MMA3 workspace missing"))?;
            work.format = format;
            launch!(
                $backend,
                match format {
                    ResidualFormat::Single => "quantize32_q8",
                    ResidualFormat::Pair => "quantize32_residual_pair",
                    ResidualFormat::Three => "residual_quantize32_precise3",
                },
                ((count as u32).div_ceil(256), 1, 1),
                $input,
                &mut work.values,
                &mut work.scales,
                &mut work.sums,
                &count,
                &mut $backend.work.status
            );
        } else {
            let work = $backend
                .work
                .residual_int8
                .as_mut()
                .ok_or_else(|| failure("residual INT8 workspace missing"))?;
            launch!(
                $backend,
                "residual_quantize32",
                ((count as u32).div_ceil(256), 1, 1),
                $input,
                &mut work.q0,
                &mut work.q1,
                &mut work.scales,
                &mut work.sums,
                &count,
                &mut $backend.work.status
            );
        }
    }};
}

macro_rules! project_residual {
    ($backend:ident,$tensor:expr,$input:expr,$output:expr,$tokens:expr,$grouped:expr,$major:expr,$used:expr,$slots:expr,$scratch:expr) => {{
        let tensor = $tensor;
        let tokens:i32 = $tokens;
        let grouped = i32::from($grouped);
        let major:i32 = $major;
        let used:i32 = $used;
        let mut scratch: Option<&mut CudaSlice<f32>> = $scratch;
        let groups = $backend.work.groups.as_ref().unwrap_or(&$backend.work.ids);
        let grid = ((tensor.rows as u32).div_ceil(32),
            if grouped!=0 {$slots}
            else {(tokens as u32).div_ceil(32)},1);
        #[cfg(test)]
        let fp16 = $backend.functions.contains_key("projection_fp16_packed");
        #[cfg(not(test))]
        let fp16 = false;
        if fp16 {
            launch!($backend,"projection_fp16_packed",grid;128,
                &tensor.bytes,&tensor.kind,$input,$output,&tensor.width,&tensor.rows,
                &tokens,groups,&used,&grouped,&major,&mut $backend.work.status);
        } else if let (true, Some(mix)) = (grouped != 0, $backend.work.group_mix) {
            // int8-fast mixed expert tiles: one MMQ per descriptor region.
            let work = $backend.work.residual_mma3.as_ref()
                .ok_or_else(||failure("residual MMA3 workspace missing"))?;
            let row_tiles = (tensor.rows as u32).div_ceil(128);
            for (kernel, (slots, offset)) in
                ["mmq_c1_pipe128", "mmq_c1_pipe64", "mmq_c1_pipe32"].into_iter().zip(mix)
            {
                if slots == 0 {
                    continue;
                }
                let region = groups.slice(offset..);
                launch_shared!($backend,kernel,(row_tiles, slots, 1);256,
                    &tensor.bytes,&tensor.kind,&work.values,&work.scales,&work.sums,
                    $output,&tensor.width,&tensor.rows,&tokens,&region,&used,&grouped,&major,
                    &mut $backend.work.status);
            }
        } else if $backend.matmul.integer_mma() {
            let work = $backend.work.residual_mma3.as_ref()
                .ok_or_else(||failure("residual MMA3 workspace missing"))?;
            let splits = if cfg!(feature = "cuda-mma3-splitk-research") && grouped == 0 {
                dense_mma3_splits(
                    tensor.width,
                    tensor.rows,
                    tokens,
                    scratch.as_ref().map_or(0, |buf| buf.len()),
                )
            } else {
                1
            };
            if splits > 1 {
                let partials = scratch.as_deref_mut()
                    .ok_or_else(||failure("residual MMA3 split-k scratch missing"))?;
                let splits_i32 = splits as i32;
                let count = (tokens as usize) * (tensor.rows as usize);
                let count_i32 = count as i32;
                launch!($backend,"residual_matmul_mma3_splitk",(grid.0, grid.1, splits);128,
                    &tensor.bytes,&tensor.kind,&work.values,&work.scales,&work.sums,
                    &mut *partials,&tensor.width,&tensor.rows,&tokens,&splits_i32,
                    &mut $backend.work.status);
                launch!($backend,"residual_splitk_reduce",((count as u32).div_ceil(256), 1, 1);256,
                    &*partials,$output,&count_i32,&splits_i32,&mut $backend.work.status);
            } else {
                let (kernel, grid, threads) = if $backend.matmul.tensor_prefill() {
                    let row_tiles = (tensor.rows as u32).div_ceil(128);
                    let format = $backend
                        .work
                        .residual_mma3
                        .as_ref()
                        .map_or(ResidualFormat::Three, |w| w.format);
                    let tiles = |bn: u32| (tokens as u32).div_ceil(bn);
                    // Measured on Qwen3-30B (2.9k prompt). int8-prefill: 64-token tiles
                    // for wide projections, 32 for few-row ones (K/V, attention output).
                    // int8-fast (cp.async-pipelined Q4_K tiles): 128-token tiles once
                    // there are >= 100 64-token blocks (Q/K/V/O 48 -> 38 ms), else 32;
                    // experts in 32-assignment tiles unless group_mix is set.
                    let (name, columns) = match format {
                        ResidualFormat::Single if grouped != 0 => ("mmq_c1_pipe32", $slots),
                        ResidualFormat::Single if row_tiles * tiles(64) >= 100 => {
                            ("mmq_c1_pipe128", tiles(128))
                        }
                        ResidualFormat::Single => ("mmq_c1_pipe32", tiles(32)),
                        ResidualFormat::Pair if grouped != 0 => ("mmq_s16_grouped", $slots),
                        ResidualFormat::Three if grouped != 0 => ("mmq_c3_grouped", $slots),
                        ResidualFormat::Pair if row_tiles * tiles(64) >= 256 => ("mmq_s16", tiles(64)),
                        ResidualFormat::Three if row_tiles * tiles(64) >= 256 => ("mmq_c3", tiles(64)),
                        ResidualFormat::Pair => ("mmq_s16_rows32", tiles(32)),
                        ResidualFormat::Three => ("mmq_c3_rows32", tiles(32)),
                    };
                    (name, (row_tiles, columns, 1), 256)
                } else if cfg!(feature = "cuda-mma3-wide-rows-research")
                    && tokens == 512
                    && (grouped != 0 || cfg!(feature = "cuda-mma3-dense-wide-rows-research"))
                {
                    ("residual_matmul_mma3_rows64", ((tensor.rows as u32).div_ceil(64), grid.1, 1), 256)
                } else {
                    ("residual_matmul_mma3", grid, 128)
                };
                let scratch_floats = scratch.as_ref().map_or(0, |buf| buf.len());
                let splits = match kernel {
                    "mmq_c1_pipe32" if grouped == 0 => fast_splits(
                        tensor.width,
                        tensor.rows,
                        tokens,
                        grid.0 * grid.1,
                        scratch_floats,
                    ),
                    "mmq_c1_pipe128" if grouped == 0 => wave_splits(
                        tensor.width,
                        tensor.rows,
                        tokens,
                        grid.0 * grid.1,
                        scratch_floats,
                    ),
                    _ => 1,
                };
                let split_kernel = if kernel == "mmq_c1_pipe128" {
                    "mmq_c1_pipe128_splitk"
                } else {
                    "mmq_c1_pipe32_splitk"
                };
                if splits > 1 {
                    let partials = scratch.as_deref_mut()
                        .ok_or_else(||failure("MMQ split-k scratch missing"))?;
                    let splits_i32 = splits as i32;
                    let count = (tokens as usize) * (tensor.rows as usize);
                    let count_i32 = count as i32;
                    launch_shared!($backend,split_kernel,(grid.0, grid.1, splits);threads,
                        &tensor.bytes,&tensor.kind,&work.values,&work.scales,&work.sums,
                        &mut *partials,&tensor.width,&tensor.rows,&tokens,groups,&used,&grouped,
                        &major,&mut $backend.work.status);
                    launch!($backend,"mmq_splitk_reduce",((count as u32).div_ceil(256), 1, 1);256,
                        &*partials,$output,&count_i32,&splits_i32,&mut $backend.work.status);
                } else {
                    launch_shared!($backend,kernel,grid;threads,
                        &tensor.bytes,&tensor.kind,&work.values,&work.scales,&work.sums,
                        $output,&tensor.width,&tensor.rows,&tokens,groups,&used,&grouped,&major,
                        &mut $backend.work.status);
                }
            }
        } else {
            let work = $backend.work.residual_int8.as_ref()
                .ok_or_else(||failure("residual INT8 workspace missing"))?;
            launch!($backend,"residual_matmul32",grid;128,
                &tensor.bytes,&tensor.kind,&work.q0,&work.q1,&work.scales,&work.sums,
                $output,&tensor.width,&tensor.rows,&tokens,groups,&used,&grouped,&major,
                &mut $backend.work.status);
        }
    }};
}

macro_rules! capture_projection {
    ($backend:ident, $layer:expr, $position:expr, $tokens:expr, $stage:literal, $source:expr, $width:expr) => {
        if let Some(capture) = &mut $backend.diagnostic {
            capture.projection(
                &$backend.stream,
                ($layer, $position, $tokens),
                $stage,
                $source,
                $width,
            )?;
        }
    };
}

impl CudaBackend {
    pub(super) fn batched_prefill(
        &mut self,
        tokens: &[i32],
        position: usize,
        output_logits: bool,
    ) -> Result<()> {
        let s = &self.weights.spec;
        let n = tokens.len() as i32;
        let (projection, grouped_projection, row_tile, token_tile, group_row_tile) = match self
            .matmul
        {
            CudaMatmul::F32 => ("prefill_matmul", "grouped_matmul", 1u32, 4u32, 1u32),
            CudaMatmul::Tf32 => ("prefill_matmul_tf32", "grouped_matmul_tf32", 128, 8, 128),
            CudaMatmul::Tf32x3 => (
                "prefill_matmul_tf32x3",
                "grouped_matmul_tf32x3",
                128,
                8,
                128,
            ),
            CudaMatmul::F32Tiled
            | CudaMatmul::ResidualInt8Research
            | CudaMatmul::ResidualMma3Research
            | CudaMatmul::Int8Prefill
            | CudaMatmul::Int8Fast => {
                ("prefill_matmul", "grouped_matmul_tiled", 1, 4, 32)
            }
            CudaMatmul::Tf32x3Tiled => ("prefill_matmul", "grouped_matmul_tf32x3_tiled", 1, 4, 128),
        };
        let matmul = self.matmul;
        let dense_projection = |t: &DeviceTensor| {
            if packed_gemm_shape(matmul, t, n) {
                ("gemm_packed_dense", 64, 64)
            } else if reference_dense_shape(matmul, t.kind, t.width, t.rows, n) {
                ("prefill_matmul_ref2x16", 2, 16)
            } else {
                (projection, row_tile, token_tile)
            }
        };
        let w = s.width as i32;
        let qw = (s.heads * s.dim) as i32;
        let kw = (s.kv_heads * s.dim) as i32;
        let dim = s.dim as i32;
        let heads = s.heads as i32;
        let kv_heads = s.kv_heads as i32;
        let context = s.info.context_tokens as i32;
        let pos = position as i32;
        let count = pos + 1;
        let neox = i32::from(s.neox);
        let total_width = w * n;
        let used = s.used as i32;
        let experts = s.experts as i32;
        let activated = s.hidden as i32 * used * n;
        let assignments = tokens.len() * s.used;
        let vector = |size: i32| ((size as u32).div_ceil(256), 1, 1);
        let ids = self
            .work
            .tokens
            .as_mut()
            .ok_or_else(|| failure("batched prefill token workspace missing"))?;
        self.stream.memcpy_htod(tokens, ids).map_err(failure)?;
        let t = &self.weights.embedding;
        launch!(
            self,
            "prefill_embedding",
            ((w as u32).div_ceil(256), n as u32, 1),
            &t.bytes,
            &t.kind,
            self.work.tokens.as_ref().unwrap(),
            &mut self.work.x,
            &w
        );
        let embedding_norm = self.weights.embedding_norm.as_ref();
        if let Some([weight, bias]) = embedding_norm {
            transformer_norm!(
                self,
                &self.work.x,
                Some(weight),
                Some(bias),
                n,
                w,
                s.epsilon
            );
            std::mem::swap(&mut self.work.x, &mut self.work.normalized);
        }
        let position_embedding = self.weights.position_embedding.as_ref();
        if let Some(t) = position_embedding {
            launch!(
                self,
                "add_position_embedding",
                ((w as u32).div_ceil(256), n as u32, 1),
                &mut self.work.x,
                &t.bytes,
                &t.kind,
                &w,
                &pos
            );
        }
        let gemma = self.weights.gemma.as_ref();
        if gemma.is_some() {
            let scale = (w as f32).sqrt();
            launch!(
                self,
                "gemma_embedding_scale",
                vector(total_width),
                &mut self.work.x,
                &total_width,
                &scale
            );
        }
        for (index, l) in self.weights.layers.iter().enumerate() {
            let gate = l.gate.as_ref();
            let affine = l.affine.as_ref();
            let gemma_attention = gemma.map(|g| g.attention(s, index));
            let rope_base = gemma_attention.map_or(s.rope_base, |(_, base, _)| base);
            let integer_groups = s.experts > 0
                && gate.is_some_and(|gate| {
                    [gate, &l.up, &l.down]
                        .iter()
                        .all(|t| residual_projection(matmul, t, n))
                });
            let register_groups = register_group_shape(s, matmul, tokens.len())
                && gate.is_some_and(|gate| {
                    [gate, &l.up, &l.down]
                        .iter()
                        .all(|t| matches!(t.kind, 12 | 14))
                });
            let (grouped_projection, group_row_tile, group_tile, group_threads) =
                if register_groups {
                    ("gemm_packed_grouped", 64, 32, 128)
                } else if register_groups || integer_groups {
                    ("grouped_matmul_unroll8", 32, 32, 128)
                } else {
                    (grouped_projection, group_row_tile, matmul.group_tile(), 256)
                };
            // int8-fast: mixed 128/64/32 expert tiles (group_experts_mixed).
            let group_mix = (integer_groups && matmul == CudaMatmul::Int8Fast && mixed_expert_tiles(s, n))
                .then(|| super::mixed_group_regions(assignments, s.experts, MIX_SMALL, MIX_LARGE).0);
            let group_slots = if s.experts > 0 {
                assignments.min(assignments.div_ceil(group_tile) + s.experts - 1) as u32
            } else {
                0
            };
            let olmo2 = s.info.architecture == "olmo2";
            let olmoe = s.info.architecture == "olmoe";
            if olmo2 {
                self.stream
                    .memcpy_dtod(
                        &self.work.x.slice(..total_width as usize),
                        &mut self.work.normalized.slice_mut(..total_width as usize),
                    )
                    .map_err(failure)?;
            } else {
                transformer_norm!(
                    self,
                    &self.work.x,
                    l.norm.as_ref(),
                    affine.map(|b| &b.norm),
                    n,
                    w,
                    s.epsilon
                );
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "attn_norm",
                &self.work.normalized,
                s.width
            );
            if [&l.q, &l.k, &l.v]
                .iter()
                .any(|t| residual_projection(matmul, t, n))
            {
                prepare_residual!(self, &self.work.normalized, n * w, true);
            }
            for (component, (tensor, output, stage)) in [
                (&l.q, &mut self.work.q, "Qproj"),
                (&l.k, &mut self.work.k, "Kproj"),
                (&l.v, &mut self.work.v, "Vproj"),
            ]
            .into_iter()
            .enumerate()
            {
                let (projection, row_tile, token_tile) = dense_projection(tensor);
                if residual_projection(matmul, tensor, n) {
                    project_residual!(
                        self,
                        tensor,
                        &self.work.normalized,
                        &mut *output,
                        n,
                        false,
                        0i32,
                        used,
                        group_slots,
                        Some(&mut self.work.down)
                    );
                } else {
                    qtip_or!(self, tensor, &self.work.normalized, &mut *output, n, {
                        launch!(
                            self,
                            projection,
                            (
                                (tensor.rows as u32).div_ceil(row_tile),
                                (n as u32).div_ceil(token_tile),
                                1
                            ),
                            &tensor.bytes,
                            &tensor.kind,
                            &self.work.normalized,
                            &mut *output,
                            &tensor.width,
                            &tensor.rows,
                            &n
                        );
                    });
                }
                if let Some(biases) = &l.qkv_bias {
                    let bias = &biases[component];
                    launch!(
                        self,
                        "add_row_bias",
                        ((bias.width as u32).div_ceil(256), n as u32, 1),
                        &mut *output,
                        &bias.bytes,
                        &bias.width
                    );
                }
                if let Some(capture) = &mut self.diagnostic {
                    capture.projection(
                        &self.stream,
                        (index, position, tokens.len()),
                        stage,
                        output,
                        tensor.rows as usize,
                    )?;
                }
            }
            for (norm, values, size) in [
                (&l.q_norm, &mut self.work.q, qw),
                (&l.k_norm, &mut self.work.k, kw),
            ] {
                if let (Some(norm), true) = (
                    norm,
                    matmul == CudaMatmul::Int8Fast
                        && !(olmo2 || olmoe)
                        && position_embedding.is_none()
                        && embedding_norm.is_none()
                        && s.rotary_dim == s.dim
                        && self.weights.rope_factors.is_none()
                        && dim % 64 == 0
                        && dim <= 2048,
                ) {
                    launch!(
                        self,
                        "head_norm_rotary",
                        ((size / dim * n) as u32, 1, 1); (dim / 2) as u32,
                        &mut *values,
                        &norm.bytes,
                        &dim,
                        &s.epsilon,
                        &size,
                        &pos,
                        &rope_base,
                        &neox
                    );
                    continue;
                }
                if let Some(norm) = norm {
                    let norm_width = if olmo2 || olmoe { size } else { dim };
                    launch!(
                        self,
                        "head_norm",
                        ((size / norm_width * n) as u32, 1, 1),
                        &mut *values,
                        &norm.bytes,
                        &norm_width,
                        &s.epsilon
                    );
                }
                if position_embedding.is_some() || embedding_norm.is_some() {
                    continue;
                }
                let grid = (((size / 2) as u32).div_ceil(256), n as u32, 1);
                if s.rotary_dim != s.dim {
                    let rotary_dim = s.rotary_dim as i32;
                    launch!(self, "rotary_partial_neox", grid;
                        256, values, &size, &dim, &pos, &rope_base, &rotary_dim);
                } else if let Some(factors) = &self.weights.rope_factors {
                    launch!(self, "rotary_factored", grid;
                        256, values, &size, &dim, &pos, &rope_base, &neox, &factors.bytes, &s.rope_attn_factor);
                } else {
                    launch!(
                        self, "rotary", grid, values, &size, &dim, &pos, &rope_base, &neox
                    );
                }
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "Qcur",
                &self.work.q,
                qw as usize
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "Kcur",
                &self.work.k,
                kw as usize
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "Vcur",
                &self.work.v,
                kw as usize
            );
            launch!(
                self,
                "cache_kv",
                ((kw as u32).div_ceil(256), n as u32, 1),
                &self.work.k,
                &self.work.v,
                &mut self.keys[index],
                &mut self.values[index],
                &kw,
                &pos
            );
            // Score rows exist per prefill token only without flash attention
            // (flash_prefill_attention sizes the workspace).
            if (embedding_norm.is_some() || gemma_attention.is_some())
                && self.work.scores.len() < heads as usize * context as usize * n as usize
            {
                return Err(invalid("prefill score workspace too small for this chunk"));
            }
            // Future KV rows are already written; each query can only see its prefix.
            if embedding_norm.is_some() {
                alibi_attention!(self, index, heads, kv_heads, dim, context, count, n);
            } else if let Some((window, _, scale)) = gemma_attention {
                gemma_attention!(
                    self,
                    index,
                    heads,
                    kv_heads,
                    dim,
                    context,
                    count,
                    n,
                    window,
                    scale,
                    gemma.unwrap().attention_softcap
                );
            } else if cfg!(feature = "cuda-attention-query-pair-research")
                && self.matmul == CudaMatmul::ResidualMma3Research
                && fused_attention_supported(s)
                && dim == 128
                && (n == 512
                    || (cfg!(feature = "cuda-attention-query-pair-even-research")
                        && n >= 32
                        && n % 2 == 0))
            {
                launch!(
                    self,
                    "attention_fused_query_pair",
                    (heads as u32, n as u32 / 2, 1),
                    &self.work.q,
                    &self.keys[index],
                    &self.values[index],
                    &mut self.work.attended,
                    &heads,
                    &kv_heads,
                    &count
                );
            } else if let (Some(halves), true) = (
                self.work.attention_half.as_mut(),
                self.matmul == CudaMatmul::Int8Prefill && dim == 128 && heads == 8 * kv_heads,
            ) {
                // int8-prefill: split K/V into FP16 hi/lo planes once per layer, then
                // FP16 tensor-core attention, 16 tokens x 8 heads per block.
                let positions = count + n;
                let stride = attention_half_stride(s) as i32;
                launch!(
                    self,
                    "attention_half_kv",
                    ((positions as u32).div_ceil(32), kv_heads as u32, 1); 256,
                    &self.keys[index],
                    &self.values[index],
                    &mut *halves,
                    &kv_heads,
                    &positions,
                    &stride
                );
                // 8 warps (16 tokens) per block: 4 warps measured equal, 2 slower.
                launch!(
                    self,
                    "attention_prefill_mma_half",
                    (kv_heads as u32, (n as u32).div_ceil(16), 1); 256,
                    &self.work.q,
                    &*halves,
                    &mut self.work.attended,
                    &heads,
                    &kv_heads,
                    &n,
                    &count,
                    &stride
                );
            } else if self.matmul.f32_tiled()
                && dim <= 128
                && dim % 4 == 0
                && kv_heads > 0
                && heads % kv_heads == 0
                && heads / kv_heads <= 8
            {
                if dim == 128 {
                    let (name, grid_y, threads) = if self.matmul.kv_f16() && heads == 8 * kv_heads {
                        // int8-fast: single FP16 attention straight from the FP16 K/V
                        // cache, 16 tokens x 8 heads per block; 8 tokens while that leaves
                        // fewer than ATTENTION_FAST_BLOCKS blocks.
                        if kv_heads as u32 * (n as u32).div_ceil(16) >= ATTENTION_FAST_BLOCKS {
                            ("attention_prefill_mma_fast_f16", (n as u32).div_ceil(16), 256)
                        } else {
                            ("attention_prefill_mma_fast4_f16", (n as u32).div_ceil(8), 128)
                        }
                    } else if self.matmul.kv_f16() && heads == kv_heads {
                        // MHA (heads == kv_heads): 64 tokens per block on FP16 tensor cores.
                        ("attention_prefill_mma_fast_mha_f16", (n as u32).div_ceil(64), 128)
                    } else if self.matmul.tensor_prefill() && heads == 8 * kv_heads {
                        ("attention_prefill_mma", (n as u32).div_ceil(8), 128)
                    } else {
                        ("attention_prefill_flash4", (n as u32).div_ceil(4), 512)
                    };
                    launch!(
                        self,
                        name,
                        (kv_heads as u32, grid_y, 1); threads,
                        &self.work.q,
                        &self.keys[index],
                        &self.values[index],
                        &mut self.work.attended,
                        &heads,
                        &kv_heads,
                        &n,
                        &count
                    );
                } else if self.matmul.kv_f16() && dim == 96 && heads == kv_heads {
                    launch!(
                        self,
                        "attention_prefill_mma_fast_mha96_f16",
                        (kv_heads as u32, (n as u32).div_ceil(64), 1); 128,
                        &self.work.q,
                        &self.keys[index],
                        &self.values[index],
                        &mut self.work.attended,
                        &heads,
                        &kv_heads,
                        &n,
                        &count
                    );
                } else {
                    launch!(
                        self,
                        "attention_prefill_flash4_dim",
                        (kv_heads as u32, (n as u32).div_ceil(4), 1); 512,
                        &self.work.q,
                        &self.keys[index],
                        &self.values[index],
                        &mut self.work.attended,
                        &dim,
                        &heads,
                        &kv_heads,
                        &n,
                        &count
                    );
                }
            } else if fused_attention_supported(s) {
                launch!(
                    self,
                    "attention_fused",
                    (heads as u32, n as u32, 1),
                    &self.work.q,
                    &self.keys[index],
                    &self.values[index],
                    &mut self.work.attended,
                    &dim,
                    &heads,
                    &kv_heads,
                    &count
                );
            } else {
                launch!(
                    self,
                    "attention_scores",
                    (heads as u32, (pos + n) as u32, n as u32),
                    &self.work.q,
                    &self.keys[index],
                    &mut self.work.scores,
                    &dim,
                    &heads,
                    &kv_heads,
                    &context,
                    &count
                );
                launch!(
                    self,
                    "attention_values",
                    (heads as u32, n as u32, 1),
                    &mut self.work.scores,
                    &self.values[index],
                    &mut self.work.attended,
                    &dim,
                    &heads,
                    &kv_heads,
                    &context,
                    &count
                );
            }
            let t = &l.output;
            let (output_projection, output_rows, output_tokens) = dense_projection(t);
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "attn_attended",
                &self.work.attended,
                t.width as usize
            );
            if residual_projection(matmul, t, n) {
                prepare_residual!(self, &self.work.attended, n * t.width, true);
                project_residual!(
                    self,
                    t,
                    &self.work.attended,
                    &mut self.work.projected,
                    n,
                    false,
                    0i32,
                    used,
                    group_slots,
                    Some(&mut self.work.down)
                );
            } else {
                qtip_or!(self, t, &self.work.attended, &mut self.work.projected, n, {
                    launch!(
                        self,
                        output_projection,
                        (
                            (t.rows as u32).div_ceil(output_rows),
                            (n as u32).div_ceil(output_tokens),
                            1
                        ),
                        &t.bytes,
                        &t.kind,
                        &self.work.attended,
                        &mut self.work.projected,
                        &t.width,
                        &t.rows,
                        &n
                    );
                });
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "attn_output",
                &self.work.projected,
                t.rows as usize
            );
            if let Some(b) = affine {
                launch!(
                    self,
                    "add_row_bias",
                    ((w as u32).div_ceil(256), n as u32, 1),
                    &mut self.work.projected,
                    &b.attention.bytes,
                    &w
                );
            }
            if let Some([post_attn, _]) = &l.post_norms {
                gemma_post_norm!(self, projected, post_attn, n, w, s.epsilon);
            } else if !s.parallel_residual {
                launch!(
                    self,
                    "residual",
                    vector(total_width),
                    &mut self.work.x,
                    &self.work.projected,
                    &total_width
                );
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_inp",
                &self.work.x,
                s.width
            );
            if olmo2 {
                self.stream
                    .memcpy_dtod(
                        &self.work.x.slice(..total_width as usize),
                        &mut self.work.normalized.slice_mut(..total_width as usize),
                    )
                    .map_err(failure)?;
            } else {
                transformer_norm!(
                    self,
                    &self.work.x,
                    l.ffn_norm.as_ref(),
                    affine.map(|b| &b.ffn_norm),
                    n,
                    w,
                    s.epsilon
                );
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_norm",
                &self.work.normalized,
                s.width
            );
            if let Some(b) = affine {
                affine_ffn!(self, l, b, n, w, s.parallel_residual);
                capture_projection!(
                    self,
                    index,
                    position,
                    tokens.len(),
                    "l_out",
                    &self.work.x,
                    s.width
                );
                continue;
            }
            let gate = gate.expect("gated CUDA layer");
            if let Some(t) = &l.router {
                // int8-fast routes in TF32 tensor cores as llama.cpp (cuBLAS TF32).
                // F32 router weights: 64x16 cp.async tiles, Qwen3-30B per prefill 1.4k
                // 5.3 -> ~0.5 ms, 0.55k (F32 before) 3.5 -> ~0.3 ms; F32 is faster
                // below ~64 tokens. Other kinds: one warp (16 rows) per block.
                let tiled = t.kind == 0 && t.width % 32 == 0;
                let wide = (t.rows as u32).div_ceil(64) * (n as u32).div_ceil(16);
                let (name, grid, threads) = if self.matmul == CudaMatmul::Int8Fast && n >= 64 && tiled {
                    // Few 64x16 tiles leave SMs idle: 32x8 tiles with a deeper pipeline
                    // (same per-output K order, so identical results; 0.5k 33 -> 21 us).
                    if wide >= ROUTER_WIDE_BLOCKS {
                        ("router_tf32_tiled", ((t.rows as u32).div_ceil(64), (n as u32).div_ceil(16), 1), 256)
                    } else {
                        ("router_tf32_small", ((t.rows as u32).div_ceil(32), (n as u32).div_ceil(8), 1), 64)
                    }
                } else if self.matmul == CudaMatmul::Int8Fast && n >= 1024 {
                    ("prefill_matmul_tf32", ((t.rows as u32).div_ceil(16), (n as u32).div_ceil(8), 1), 32)
                } else {
                    ("prefill_matmul", (t.rows as u32, (n as u32).div_ceil(4), 1), 256)
                };
                launch!(
                    self,
                    name,
                    grid; threads,
                    &t.bytes,
                    &t.kind,
                    &self.work.normalized,
                    &mut self.work.router,
                    &t.width,
                    &t.rows,
                    &n
                );
                capture_projection!(
                    self,
                    index,
                    position,
                    tokens.len(),
                    "ffn_moe_logits",
                    &self.work.router,
                    s.experts
                );
                if cfg!(feature = "cuda-decode-fusion-research")
                    && matmul == CudaMatmul::ResidualMma3Research
                    && experts == 128
                    && used <= 32
                {
                    launch!(
                        self,
                        "router_warp128",
                        (n as u32, 1, 1); 32,
                        &self.work.router,
                        &used,
                        &mut self.work.ids,
                        &mut self.work.probabilities,
                        &mut self.work.status
                    );
                } else {
                    launch!(
                        self,
                        if olmoe {
                            "router_full_softmax"
                        } else {
                            "router"
                        },
                        (n as u32, 1, 1),
                        &self.work.router,
                        &experts,
                        &used,
                        &mut self.work.ids,
                        &mut self.work.probabilities,
                        &mut self.work.status
                    );
                }
                capture_projection!(
                    self,
                    index,
                    position,
                    tokens.len(),
                    "ffn_moe_weights_norm",
                    &self.work.probabilities,
                    s.used
                );
                if let Some(capture) = &mut self.diagnostic {
                    capture.routing(
                        &self.stream,
                        (index, position, tokens.len()),
                        s.experts,
                        s.used,
                        &self.work.ids,
                        &self.work.router,
                    )?;
                }
                let groups = self
                    .work
                    .groups
                    .as_mut()
                    .ok_or_else(|| failure("expert group workspace missing"))?;
                let slots = group_slots as i32;
                self.work.group_mix = group_mix;
                if cfg!(feature = "cuda-parallel-grouping-research")
                    && integer_groups
                    && matmul == CudaMatmul::ResidualMma3Research
                {
                    let counts = self
                        .work
                        .group_counts
                        .as_mut()
                        .ok_or_else(|| failure("parallel expert count workspace missing"))?;
                    let assignments = assignments as i32;
                    launch!(self, "group_parallel_count32", (experts as u32, 1, 1);32,
                        &self.work.ids, &mut *counts, &assignments);
                    launch!(self, "group_parallel_emit32", (experts as u32 + 1, 1, 1);32,
                        &self.work.ids, &*counts, &mut *groups, &assignments, &experts, &slots);
                } else if let Some(mix) = group_mix {
                    // GROUP_FAST_EXPERTS in gemm.cu bounds the parallel groupings.
                    let (small, large) = (MIX_SMALL as i32, MIX_LARGE as i32);
                    let (s128, s64, s32) = (mix[0].0 as i32, mix[1].0 as i32, mix[2].0 as i32);
                    launch!(self, "group_experts_mixed", (1, 1, 1); 1024,
                        &self.work.ids, &mut *groups, &experts, &used, &n,
                        &s128, &s64, &s32, &small, &large);
                } else {
                    let (name, threads) = match group_tile {
                        32 if experts <= 256 => ("group_experts_fast32", 1024),
                        32 => ("group_experts_reg32", 256),
                        8 => ("group_experts_tiled", 256),
                        _ => ("group_experts", 256),
                    };
                    launch!(
                        self,
                        name,
                        (1, 1, 1); threads,
                        &self.work.ids,
                        &mut *groups,
                        &experts,
                        &used,
                        &n,
                        &slots
                    );
                }
            }
            if integer_groups
                || (s.experts == 0
                    && [gate, &l.up]
                        .iter()
                        .any(|t| residual_projection(matmul, t, n)))
            {
                prepare_residual!(self, &self.work.normalized, n * w);
            }
            for (t, out) in [(gate, &mut self.work.gate), (&l.up, &mut self.work.up)] {
                if integer_groups || (s.experts == 0 && residual_projection(matmul, t, n)) {
                    project_residual!(
                        self,
                        t,
                        &self.work.normalized,
                        &mut *out,
                        n,
                        integer_groups,
                        0i32,
                        used,
                        group_slots,
                        if integer_groups {
                            None
                        } else {
                            Some(&mut self.work.down)
                        }
                    );
                } else if let Some(groups) = &self.work.groups {
                    launch!(
                        self,
                        grouped_projection,
                        ((t.rows as u32).div_ceil(group_row_tile), group_slots, 1); group_threads,
                        &t.bytes,
                        &t.kind,
                        &self.work.normalized,
                        out,
                        &t.width,
                        &t.rows,
                        groups,
                        &used,
                        &0i32
                    );
                } else {
                    let (projection, row_tile, token_tile) = dense_projection(t);
                    qtip_or!(self, t, &self.work.normalized, &mut *out, n, {
                        launch!(
                            self,
                            projection,
                            (
                                (t.rows as u32).div_ceil(row_tile),
                                (n as u32).div_ceil(token_tile),
                                1
                            ),
                            &t.bytes,
                            &t.kind,
                            &self.work.normalized,
                            out,
                            &t.width,
                            &t.rows,
                            &n
                        );
                    });
                }
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_gate",
                &self.work.gate,
                s.hidden * s.used
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_up",
                &self.work.up,
                s.hidden * s.used
            );
            let t = &l.down;
            let quantized_down = integer_groups || (s.experts == 0 && residual_projection(matmul, t, n));
            // int8-fast: SwiGLU writes the q8 down-projection input directly.
            let fused_down = quantized_down
                && matmul == CudaMatmul::Int8Fast
                && gemma.is_none()
                && self.diagnostic.is_none()
                && activated % 32 == 0;
            if fused_down {
                let work = self
                    .work
                    .residual_mma3
                    .as_mut()
                    .ok_or_else(|| failure("residual MMA3 workspace missing"))?;
                work.format = ResidualFormat::Single;
                launch!(
                    self,
                    "swiglu_quantize32_q8",
                    vector(activated),
                    &self.work.gate,
                    &self.work.up,
                    &mut work.values,
                    &mut work.scales,
                    &mut work.sums,
                    &activated,
                    &mut self.work.status
                );
            } else {
            launch!(
                self,
                if gemma.is_some() {
                    "gemma_geglu"
                } else {
                    "swiglu"
                },
                vector(activated),
                &mut self.work.gate,
                &self.work.up,
                &activated
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_down_inp",
                &self.work.gate,
                s.hidden * s.used
            );
            }
            if quantized_down {
                if !fused_down {
                    prepare_residual!(
                        self,
                        &self.work.gate,
                        n * t.width * if integer_groups { used } else { 1 }
                    );
                }
                project_residual!(
                    self,
                    t,
                    &self.work.gate,
                    &mut self.work.down,
                    n,
                    integer_groups,
                    1i32,
                    used,
                    group_slots,
                    if integer_groups {
                        None
                    } else {
                        Some(&mut self.work.projected)
                    }
                );
            } else if let Some(groups) = &self.work.groups {
                launch!(
                    self,
                    grouped_projection,
                    ((t.rows as u32).div_ceil(group_row_tile), group_slots, 1); group_threads,
                    &t.bytes,
                    &t.kind,
                    &self.work.gate,
                    &mut self.work.down,
                    &t.width,
                    &t.rows,
                    groups,
                    &used,
                    &1i32
                );
            } else {
                let (projection, row_tile, token_tile) = dense_projection(t);
                qtip_or!(self, t, &self.work.gate, &mut self.work.down, n, {
                    launch!(
                        self,
                        projection,
                        (
                            (t.rows as u32).div_ceil(row_tile),
                            (n as u32).div_ceil(token_tile),
                            1
                        ),
                        &t.bytes,
                        &t.kind,
                        &self.work.gate,
                        &mut self.work.down,
                        &t.width,
                        &t.rows,
                        &n
                    );
                });
            }

            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_down",
                &self.work.down,
                s.width * s.used
            );
            if let Some([_, post_ffn]) = &l.post_norms {
                gemma_post_norm!(self, down, post_ffn, n, w, s.epsilon);
            } else {
                launch!(
                    self,
                    "combine_batch",
                    ((w as u32).div_ceil(256), n as u32, 1),
                    &mut self.work.x,
                    &self.work.down,
                    &self.work.probabilities,
                    &w,
                    &used
                );
            }
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "l_out",
                &self.work.x,
                s.width
            );
        }
        launch!(
            self,
            "check_finite",
            vector(total_width),
            &self.work.x,
            &total_width,
            &mut self.work.status
        );
        if output_logits {
            self.project_logits((tokens.len() - 1) * s.width)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_dense_dispatch_keeps_unmeasured_modes_formats_shapes_and_small_chunks() {
        let selected = CudaMatmul::F32Tiled;
        for kind in [12, 14] {
            for (width, rows) in [(2048, 4096), (4096, 2048), (2048, 512)] {
                assert!(!reference_dense_shape(selected, kind, width, rows, 255));
                assert!(reference_dense_shape(selected, kind, width, rows, 256));
                assert!(reference_dense_shape(selected, kind, width, rows, 512));
                for mode in [
                    CudaMatmul::F32,
                    CudaMatmul::Tf32,
                    CudaMatmul::Tf32x3,
                    CudaMatmul::Tf32x3Tiled,
                ] {
                    assert!(!reference_dense_shape(mode, kind, width, rows, 512));
                }
            }
        }
        for kind in [0, 1, 6, 8] {
            assert!(!reference_dense_shape(selected, kind, 2048, 4096, 512));
        }
        for (width, rows) in [(2048, 128), (2048, 768), (768, 2048), (576, 1536)] {
            assert!(!reference_dense_shape(selected, 12, width, rows, 512));
        }
    }
}

/// int8-fast expert remainder tiles (group_experts_mixed): after whole 128-tiles, an
/// expert's last r <= MIX_SMALL assignments take a 32-tile, r <= 64 a 64-tile, r <=
/// MIX_LARGE a 64- and a 32-tile, more another 128-tile. Qwen3-30B (pipelined MMQ)
/// within 1 ms for small 16..32, large 80..96; 0.55k / 1.4k / 2.9k prompt 44 / 90 /
/// 162 ms against 64- or 128-tiles alone 49 / 102 / 178 ms (before pipelining).
const MIX_SMALL: usize = 32;
const MIX_LARGE: usize = 96;

/// int8-fast router: 64x16-tile blocks below which router_tf32_small is used.
const ROUTER_WIDE_BLOCKS: u32 = 512;

/// int8-fast prefill attention: 16-token blocks below which 8-token blocks are used
/// (two waves of 170 SMs).
const ATTENTION_FAST_BLOCKS: u32 = 340;

/// Mixed expert tiles once experts average MIX_MIN_AVERAGE assignments; below,
/// 32-tiles (128-wide Q4_K K steps). Qwen3-30B TTFT, 32-tiles / mixed: 0.5k 37 / 44,
/// 1k 63 / 65, 1.5k 87 / 88, 2.9k 169 / 158 ms. GROUP_FAST_EXPERTS bounds
/// group_experts_mixed.
const MIX_MIN_AVERAGE: usize = 128;

fn mixed_expert_tiles(s: &Spec, tokens: i32) -> bool {
    s.experts <= 256 && tokens as usize * s.used / s.experts.max(1) >= MIX_MIN_AVERAGE
}

/// K splits for int8-fast few-block dense MMQ (short prompts): double while the grid
/// stays under two waves of 170 SMs, each split keeps whole 256-value Q4_K/Q6_K
/// blocks, and the partials fit the scratch buffer.
fn fast_splits(width: i32, rows: i32, tokens: i32, blocks: u32, scratch_floats: usize) -> u32 {
    let max_blocks: u32 = 340;
    let mut splits = 1u32;
    while width > 0
        && (width as u32) % (splits * 2 * 256) == 0
        && blocks * splits * 2 <= max_blocks
        && (splits as usize * 2) * tokens.max(0) as usize * rows.max(0) as usize <= scratch_floats
    {
        splits *= 2;
    }
    splits
}

/// K splits that fill the last wave of 1-block/SM 128-token MMQ tiles (RTX 5090:
/// 170 SMs; the 2.9k-prompt O projection has 368 blocks = 2.2 waves): the split
/// count minimizing waves / splits, ties to fewer splits, partials in scratch.
fn wave_splits(width: i32, rows: i32, tokens: i32, blocks: u32, scratch_floats: usize) -> u32 {
    let sms: u32 = 170;
    let mut best = (blocks.div_ceil(sms) as f32, 1u32);
    for splits in [2u32, 4] {
        if width <= 0
            || (width as u32) % (splits * 256) != 0
            || splits as usize * tokens.max(0) as usize * rows.max(0) as usize > scratch_floats
        {
            continue;
        }
        let cost = (blocks * splits).div_ceil(sms) as f32 / splits as f32 * 1.05;
        if cost < best.0 {
            best = (cost, splits);
        }
    }
    best.1
}

