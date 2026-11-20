use super::*;

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

fn residual_projection(
    matmul: CudaMatmul,
    tensor: &DeviceTensor,
    tokens: i32,
    packed_llama: bool,
) -> bool {
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    if tokens >= 32 && tensor.fp16.is_some() {
        return false;
    }
    (packed_llama && tokens >= 32 && tensor.kind == 8 && tensor.width % 32 == 0)
        || (matmul.residual_components() > 0
            && tokens >= matmul.residual_min_tokens() as i32
            && matches!(tensor.kind, 12 | 14)
            && tensor.width % 256 == 0)
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

macro_rules! prepare_residual {
    ($backend:ident,$input:expr,$count:expr) => {{
        let count: i32 = $count;
        if $backend.work.residual_mma3.is_some() {
            let work = $backend
                .work
                .residual_mma3
                .as_mut()
                .ok_or_else(|| failure("residual MMA3 workspace missing"))?;
            launch!(
                $backend,
                "residual_quantize32_precise3",
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
        } else if $backend.work.residual_mma3.is_some() {
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
            if tensor.kind == 8 && grouped == 0 {
                launch!($backend,"dense_packed_replay_mma3",grid;128,
                    &tensor.bytes,&tensor.kind,&work.values,&work.scales,&work.sums,
                    $output,&tensor.width,&tensor.rows,&tokens,&mut $backend.work.status);
            } else if splits > 1 {
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
                let (kernel, grid, threads) = if cfg!(feature = "cuda-mma3-wide-rows-research")
                    && tokens == 512
                    && (grouped != 0 || cfg!(feature = "cuda-mma3-dense-wide-rows-research"))
                {
                    ("residual_matmul_mma3_rows64", ((tensor.rows as u32).div_ceil(64), grid.1, 1), 256)
                } else {
                    ("residual_matmul_mma3", grid, 128)
                };
                launch!($backend,kernel,grid;threads,
                    &tensor.bytes,&tensor.kind,&work.values,&work.scales,&work.sums,
                    $output,&tensor.width,&tensor.rows,&tokens,groups,&used,&grouped,&major,
                    &mut $backend.work.status);
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

macro_rules! try_corrected_fp16 {
    ($backend:ident, $t:expr, $x:ident, $y:expr, $n:expr) => {{
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let done = if let Some(fp16) = &mut $backend.fp16 {
            fp16.project($t, &$backend.work.$x, $y, $n)?
        } else {
            false
        };
        #[cfg(not(feature = "cuda-fp16-corrected-prefill-research"))]
        let done = false;
        done
    }};
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
            | CudaMatmul::ResidualMma3Research => {
                ("prefill_matmul", "grouped_matmul_tiled", 1, 4, 32)
            }
            CudaMatmul::Tf32x3Tiled => ("prefill_matmul", "grouped_matmul_tf32x3_tiled", 1, 4, 128),
        };
        let matmul = self.matmul;
        let packed_llama = cfg!(feature = "cuda-packed-prefill-research")
            && self.work.residual_mma3.is_some()
            && self.functions.contains_key("dense_packed_replay_mma3");
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let packed_llama = packed_llama && self.fp16.is_none();
        let dense_projection = |t: &DeviceTensor| {
            let qwen3_two_rows = cfg!(feature = "cuda-qwen3-prefill-2x16-research")
                && matmul == CudaMatmul::F32
                && s.info.architecture == "qwen3"
                && n >= 128
                && t.width % 256 == 0
                && matches!(t.kind, 12 | 14);
            if qwen3_two_rows || reference_dense_shape(matmul, t.kind, t.width, t.rows, n) {
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
                        .all(|t| residual_projection(matmul, t, n, false))
                });
            let register_groups = register_group_shape(s, matmul, tokens.len())
                && gate.is_some_and(|gate| {
                    [gate, &l.up, &l.down]
                        .iter()
                        .all(|t| matches!(t.kind, 12 | 14))
                });
            let (grouped_projection, group_row_tile, group_tile, group_threads) =
                if register_groups || integer_groups {
                    ("grouped_matmul_unroll8", 32, 32, 128)
                } else {
                    (grouped_projection, group_row_tile, matmul.group_tile(), 256)
                };
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
            #[cfg(feature = "cuda-llama-attn-mmq-joined-research")]
            let mmq_qkv = position == 0
                && (384..=512).contains(&n)
                && self.ffn_mmq.is_some()
                && [(&l.q, 2048), (&l.k, 512), (&l.v, 512)]
                    .iter()
                    .all(|(weight, rows)| ffn_mmq::supports_attention(weight, *rows));
            #[cfg(not(feature = "cuda-llama-attn-mmq-joined-research"))]
            let mmq_qkv = false;
            #[cfg(feature = "cuda-llama-attn-mmq-joined-research")]
            if let Some(mmq) = self.ffn_mmq.as_mut().filter(|_| mmq_qkv) {
                mmq.pack(&self.stream, &self.work.normalized, l.q.width, n)?;
            }
            if !mmq_qkv
                && [&l.q, &l.k, &l.v]
                    .iter()
                    .any(|t| residual_projection(matmul, t, n, packed_llama))
            {
                prepare_residual!(self, &self.work.normalized, n * w);
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
                let mmq_component = mmq_qkv
                    && !(component == 0 && cfg!(feature = "cuda-llama-prefill-qo-single-term-research"));
                #[cfg(feature = "cuda-llama-attn-mmq-joined-research")]
                if let Some(mmq) = self.ffn_mmq.as_mut().filter(|_| mmq_component) {
                    mmq.project(&self.stream, tensor, &mut *output, n)?;
                }
                #[cfg(feature = "cuda-qwen3-prefill-kv-weight-only-research")]
                let qkv_weight_only = if !mmq_qkv
                    && position == 0
                    && (384..=512).contains(&n)
                    && n % 128 == 0
                    && (component != 0
                        || (cfg!(feature = "cuda-qwen3-prefill-q-weight-only-research")
                            && s.info.architecture == "qwen3"
                            && tensor.rows == 2048))
                    && let Some(fp16) = self.fp16.as_mut()
                {
                    fp16.project_qwen3_attention_weights(
                        tensor, &self.work.normalized, &mut *output, 1024, n,
                    )?
                } else {
                    false
                };
                #[cfg(not(feature = "cuda-qwen3-prefill-kv-weight-only-research"))]
                let qkv_weight_only = false;
                if mmq_component || qkv_weight_only
                    || try_corrected_fp16!(self, tensor, normalized, &mut *output, n)
                {
                } else if residual_projection(matmul, tensor, n, packed_llama) {
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
            let fused_head_rope = if cfg!(feature = "cuda-qwen3-prefill-fused-head-rope-research")
                && n >= 384
                && position == 0
                && dim == 128
                && s.rotary_dim == 128
                && neox == 1
                && position_embedding.is_none()
                && embedding_norm.is_none()
                && self.weights.rope_factors.is_none()
                && self.functions.contains_key("prefill_qkv_head_norm_rope_cache")
                && let (Some(q_norm), Some(k_norm)) = (&l.q_norm, &l.k_norm)
            {
                launch!(self, "prefill_qkv_head_norm_rope_cache",
                    ((heads + kv_heads) as u32, n as u32, 1),
                    &mut self.work.q, &q_norm.bytes, &heads,
                    &mut self.work.k, &k_norm.bytes, &self.work.v,
                    &mut self.keys[index], &mut self.values[index], &kv_heads,
                    &s.epsilon, &pos, &rope_base);
                true
            } else {
                false
            };
            for (norm, values, size) in [
                (&l.q_norm, &mut self.work.q, qw),
                (&l.k_norm, &mut self.work.k, kw),
            ] {
                if fused_head_rope {
                    break;
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
            if !fused_head_rope {
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
            } else if cfg!(any(
                feature = "cuda-qwen3-batched-attention-research",
                feature = "cuda-llama-batched-attention-research"
            )) && position == 0
                && (256..=512).contains(&n)
                && n % 2 == 0
                && (s.info.architecture != "llama"
                    || cfg!(feature = "cuda-llama-batched-attention-256-research")
                    || n >= 384)
                && self.functions.contains_key("qwen3_batched_causal_softmax")
            {
                #[cfg(any(
                    feature = "cuda-qwen3-batched-attention-research",
                    feature = "cuda-llama-batched-attention-research"
                ))]
                {
                    let llama_grouped = (heads, dim, kv_heads) == (32, 64, 8)
                        && (n >= 384 || cfg!(feature = "cuda-llama-grouped-prefill-256-research"))
                        && self.functions.contains_key("llama_grouped_causal_softmax");
                    let qwen3_grouped = n >= 384 && (heads, dim, kv_heads) == (16, 128, 8)
                        && self.functions.contains_key("qwen3_grouped_causal_softmax");
                    let grouped = llama_grouped || qwen3_grouped;
                    let softmax = if grouped {
                        launch!(self, "llama_grouped_q_pack", vector(n * qw),
                            &self.work.q, &mut self.work.attended, &n);
                        if llama_grouped { "llama_grouped_causal_softmax" } else { "qwen3_grouped_causal_softmax" }
                    } else {
                        "qwen3_batched_causal_softmax"
                    };
                    qwen3_batched_attention(
                        &self.stream,
                        self.attention_blas
                            .as_ref()
                            .ok_or_else(|| invalid("missing Qwen3 attention BLAS handle"))?,
                        &self.functions[softmax],
                        &mut self.work,
                        &self.keys[index],
                        &self.values[index],
                        n,
                        heads,
                        dim,
                        grouped,
                        match (s.info.architecture.as_str(), heads, dim) {
                            ("llama", 32, 64) => cfg!(feature = "cuda-llama-batched-attention-tf32-research")
                                && (n >= 384 || cfg!(feature = "cuda-llama-batched-attention-256-research")),
                            ("qwen3", 16, 128) => n >= 384 && cfg!(feature = "cuda-qwen3-batched-attention-tf32-research"),
                            _ => false,
                        },
                    )?;
                    if grouped {
                        launch!(self, "llama_grouped_out_unpack", vector(n * qw),
                            &self.work.q, &mut self.work.attended, &n);
                    }
                }
            } else if fused_attention_supported(s)
                && ((dim == 64
                    && n >= 32
                    && n % 2 == 0
                    && self.functions.contains_key("attention_fused_query_pair_64"))
                    || (cfg!(feature = "cuda-attention-query-pair-research")
                        && self.matmul == CudaMatmul::ResidualMma3Research
                        && dim == 128
                        && (n == 512
                            || (cfg!(feature = "cuda-attention-query-pair-even-research")
                                && n >= 32
                                && n % 2 == 0))))
            {
                launch!(
                    self,
                    if dim == 64 {
                        "attention_fused_query_pair_64"
                    } else {
                        "attention_fused_query_pair"
                    },
                    (heads as u32, n as u32 / 2, 1),
                    &self.work.q,
                    &self.keys[index],
                    &self.values[index],
                    &mut self.work.attended,
                    &heads,
                    &kv_heads,
                    &count
                );
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

            #[cfg(any(
                feature = "cuda-qwen3-batched-attention-research",
                feature = "cuda-llama-batched-attention-research"
            ))]
            fn qwen3_batched_attention(
                stream: &Arc<CudaStream>,
                blas: &cudarc::cublas::CudaBlas,
                softmax: &CudaFunction,
                work: &mut Workspace,
                keys: &CudaSlice<f32>,
                values: &CudaSlice<f32>,
                n: i32,
                heads: i32,
                dim: i32,
                grouped: bool,
                tf32_values: bool,
            ) -> Result<()> {
                use cudarc::{
                    cublas::{result, sys},
                    driver::{DevicePtr, DevicePtrMut},
                };
                const KV: i32 = 8;
                let ratio = heads / KV;
                let qw = heads * dim;
                let kw = KV * dim;
                let scores = (heads * n * n) as usize;
                if work.scores.len() < scores
                    || work.q.len() < (n * qw) as usize
                    || work.attended.len() < (n * qw) as usize
                    || keys.len().min(values.len()) < (n * kw) as usize
                {
                    return Err(invalid("batched attention workspace too small"));
                }
                let ty = sys::cudaDataType::CUDA_R_32F;
                let nn = sys::cublasOperation_t::CUBLAS_OP_N;
                let tn = sys::cublasOperation_t::CUBLAS_OP_T;
                let value_compute = if tf32_values {
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
                } else {
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
                };
                let zero = 0.0f32;
                let mm = |op, m, inner, alpha: f32, a, lda, sa, b, ldb, sb, c, ldc, sc, compute| {
                    unsafe {
                        result::gemm_strided_batched_ex(
                            *blas.handle(),
                            op,
                            nn,
                            m,
                            if grouped { ratio * n } else { n },
                            inner,
                            (&alpha as *const f32).cast(),
                            a as *const _,
                            ty,
                            lda,
                            sa,
                            b as *const _,
                            ty,
                            ldb,
                            sb,
                            (&zero as *const f32).cast(),
                            c as *mut _,
                            ty,
                            ldc,
                            sc,
                            KV,
                            compute,
                            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                        )
                    }
                    .map_err(failure)
                };
                stream.context().bind_to_thread().map_err(failure)?;
                {
                    let (q, _q) = if grouped { &work.attended } else { &work.q }.device_ptr(stream);
                    let (k, _k) = keys.device_ptr(stream);
                    let (s, _s) = work.scores.device_ptr_mut(stream);
                    for slot in 0..if grouped { 1 } else { ratio as u64 } {
                        mm(
                            tn,
                            n,
                            dim,
                            1.0 / (dim as f32).sqrt(),
                            k,
                            kw,
                            dim.into(),
                            q + slot * dim as u64 * 4,
                            if grouped { dim } else { qw },
                            (if grouped { n * ratio * dim } else { ratio * dim }).into(),
                            s + slot * (n * n) as u64 * 4,
                            n,
                            (ratio * n * n).into(),
                            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                        )?;
                    }
                }
                let mut args = stream.launch_builder(softmax);
                args.arg(&mut work.scores).arg(&n);
                unsafe {
                    args.launch(LaunchConfig {
                        grid_dim: (
                            if cfg!(feature = "cuda-batched-attention-warp-softmax-research") {
                                (n as u32).div_ceil(8)
                            } else {
                                n as u32
                            },
                            heads as u32,
                            1,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
                {
                    let (v, _v) = values.device_ptr(stream);
                    let (s, _s) = work.scores.device_ptr(stream);
                    let (y, _y) = if grouped { &mut work.q } else { &mut work.attended }.device_ptr_mut(stream);
                    for slot in 0..if grouped { 1 } else { ratio as u64 } {
                        mm(
                            nn,
                            dim,
                            n,
                            1.0,
                            v,
                            kw,
                            dim.into(),
                            s + slot * (n * n) as u64 * 4,
                            n,
                            (ratio * n * n).into(),
                            y + slot * dim as u64 * 4,
                            if grouped { dim } else { qw },
                            (if grouped { n * ratio * dim } else { ratio * dim }).into(),
                            value_compute,
                        )?;
                    }
                }
                Ok(())
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
            #[cfg(feature = "cuda-llama-attn-mmq-joined-research")]
            let mmq_output = if let Some(mmq) = self
                .ffn_mmq
                .as_mut()
                .filter(|_| !cfg!(feature = "cuda-llama-prefill-qo-single-term-research")
                    && position == 0 && (384..=512).contains(&n) && ffn_mmq::supports_attention(t, 2048))
            {
                mmq.pack(&self.stream, &self.work.attended, t.width, n)?;
                mmq.project(&self.stream, t, &mut self.work.projected, n)?;
                true
            } else {
                false
            };
            #[cfg(not(feature = "cuda-llama-attn-mmq-joined-research"))]
            let mmq_output = false;
            #[cfg(feature = "cuda-qwen3-prefill-output-weight-only-research")]
            let weight_only_output = if !mmq_output && position == 0 && (384..=512).contains(&n)
                && n % 128 == 0
                && let Some(fp16) = self.fp16.as_mut()
            {
                fp16.project_qwen3_attention_weights(
                    t, &self.work.attended, &mut self.work.projected, 2048, n,
                )?
            } else {
                false
            };
            #[cfg(not(feature = "cuda-qwen3-prefill-output-weight-only-research"))]
            let weight_only_output = false;
            if mmq_output || weight_only_output
                || try_corrected_fp16!(self, t, attended, &mut self.work.projected, n)
            {
            } else if residual_projection(matmul, t, n, packed_llama) {
                prepare_residual!(self, &self.work.attended, n * t.width);
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
            #[cfg(feature = "cuda-llama-prefill-last-ffn-research")]
            if output_logits
                && position == 0
                && n == 512
                && index + 1 == self.weights.layers.len()
                && self.diagnostic.is_none()
                && s.info.architecture == "llama"
                && s.experts == 0
                && s.used == 1
                && !s.parallel_residual
                && affine.is_none()
                && l.post_norms.is_none()
                && let Some(gate) = gate
                && (gate.kind, gate.width, gate.rows) == (8, 2048, 8192)
                && (l.up.kind, l.up.width, l.up.rows) == (8, 2048, 8192)
                && (l.down.kind, l.down.width, l.down.rows) == (8, 8192, 2048)
                && self.functions.contains_key("llama_q8_dp4a_swiglu")
                && self.functions.contains_key("llama_q8_dp4a")
                && self.work.residual_mma3.is_some()
            {
                // This layer's KV is already populated. Only the last FFN row can affect logits.
                let offset = (tokens.len() - 1) * s.width;
                let epsilon = s.epsilon;
                let last = self.work.x.slice(offset..offset + s.width);
                transformer_norm!(self, &last, l.ffn_norm.as_ref(), None::<&DeviceTensor>, 1, w, epsilon);
                let scratch = self.work.residual_mma3.as_mut().unwrap();
                if 3 * s.width > scratch.values.len()
                    || 3 * (s.width / 32) > scratch.scales.len().min(scratch.sums.len())
                {
                    return Err(invalid("last-row Q8 FFN input exceeds workspace"));
                }
                launch!(self, "residual_quantize32_precise3", vector(w),
                    &self.work.normalized, &mut scratch.values, &mut scratch.scales,
                    &mut scratch.sums, &w, &mut self.work.status);
                let (kernel, rows_per_block) = if self.functions.contains_key("llama_q8_dp4a_swiglu_pairs") {
                    ("llama_q8_dp4a_swiglu_pairs", 8)
                } else {
                    ("llama_q8_dp4a_swiglu", 4)
                };
                launch!(self, kernel, ((gate.rows as u32).div_ceil(rows_per_block), 1, 1); 128,
                    &gate.bytes, &l.up.bytes, &scratch.values, &scratch.scales,
                    &mut self.work.gate, &w, &gate.rows);
                if !q8_dp4a!(self, &l.down, gate, &mut self.work.down) {
                    return Err(invalid("last-row Q8 FFN down kernel unavailable"));
                }
                let mut last = self.work.x.slice_mut(offset..offset + s.width);
                launch!(self, "combine", vector(w), &mut last, &self.work.down,
                    &self.work.probabilities, &w, &used);
                continue;
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
                launch!(
                    self,
                    "prefill_matmul",
                    (t.rows as u32, (n as u32).div_ceil(4), 1),
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
                } else {
                    launch!(
                        self,
                        match group_tile {
                            32 => "group_experts_reg32",
                            8 => "group_experts_tiled",
                            _ => "group_experts",
                        },
                        (1, 1, 1),
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
                        .any(|t| residual_projection(matmul, t, n, packed_llama)))
            {
                prepare_residual!(self, &self.work.normalized, n * w);
            }
            #[cfg(feature = "cuda-llama-ffn-mmq-research")]
            let use_mmq = position == 0
                && ffn_mmq::supports_chunk(n)
                && [
                    (gate, 2048, 8192),
                    (&l.up, 2048, 8192),
                    (&l.down, 8192, 2048),
                ]
                .iter()
                .all(|(t, width, rows)| (t.kind, t.width, t.rows) == (8, *width, *rows));
            #[cfg(feature = "cuda-llama-ffn-mmq-research")]
            let projected_gate_up = if let Some(mmq) = self.ffn_mmq.as_mut().filter(|_| use_mmq) {
                if cfg!(feature = "cuda-llama-prefill-fp16-gate-up-research")
                    && (n >= 384
                        || (n >= 128 && cfg!(feature = "cuda-llama-prefill-fp16-gate-up-256-research"))
                        || (n >= 32 && cfg!(feature = "cuda-llama-prefill-fp16-gate-up-46-research")))
                {
                    #[cfg(feature = "cuda-llama-prefill-fp16-gate-up-research")]
                    self.fp16
                        .as_mut()
                        .ok_or_else(|| invalid("missing FP16 prefill workspace"))?
                        .project_q8_gate_up(
                            &self.work.normalized,
                            gate,
                            &l.up,
                            &mut self.work.gate,
                            &mut self.work.up,
                            n,
                        )?;
                } else {
                    mmq.pack(&self.stream, &self.work.normalized, gate.width, n)?;
                    mmq.project(&self.stream, gate, &mut self.work.gate, n)?;
                    mmq.project(&self.stream, &l.up, &mut self.work.up, n)?;
                }
                true
            } else {
                false
            };
            for (t, out) in [(gate, &mut self.work.gate), (&l.up, &mut self.work.up)] {
                #[cfg(feature = "cuda-llama-ffn-mmq-research")]
                if projected_gate_up {
                    break;
                }
                if try_corrected_fp16!(self, t, normalized, &mut *out, n) {
                } else if integer_groups
                    || (s.experts == 0 && residual_projection(matmul, t, n, packed_llama))
                {
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
            let t = &l.down;
            #[cfg(feature = "cuda-llama-ffn-mmq-research")]
            let mmq_down = if let Some(mmq) = self.ffn_mmq.as_mut().filter(|_| use_mmq) {
                mmq.pack(&self.stream, &self.work.gate, t.width, n)?;
                mmq.project(&self.stream, t, &mut self.work.down, n)?;
                true
            } else {
                false
            };
            #[cfg(not(feature = "cuda-llama-ffn-mmq-research"))]
            let mmq_down = false;
            if mmq_down || try_corrected_fp16!(self, t, gate, &mut self.work.down, n) {
            } else if integer_groups
                || (s.experts == 0 && residual_projection(matmul, t, n, packed_llama))
            {
                prepare_residual!(
                    self,
                    &self.work.gate,
                    n * t.width * if integer_groups { used } else { 1 }
                );
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
