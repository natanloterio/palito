use super::*;

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
            CudaMatmul::F32Tiled => ("prefill_matmul", "grouped_matmul_tiled", 1, 4, 32),
            CudaMatmul::Tf32x3Tiled => ("prefill_matmul", "grouped_matmul_tf32x3_tiled", 1, 4, 128),
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
        let group_slots = if s.experts > 0 {
            assignments.min(assignments.div_ceil(self.matmul.group_tile()) + s.experts - 1) as u32
        } else {
            0
        };
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
        for (index, l) in self.weights.layers.iter().enumerate() {
            launch!(
                self,
                "normalize",
                (n as u32, 1, 1),
                &self.work.x,
                &l.norm.bytes,
                &mut self.work.normalized,
                &w,
                &s.epsilon
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "attn_norm",
                &self.work.normalized,
                s.width
            );
            for (tensor, output, stage) in [
                (&l.q, &mut self.work.q, "Qproj"),
                (&l.k, &mut self.work.k, "Kproj"),
                (&l.v, &mut self.work.v, "Vproj"),
            ] {
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
                if let Some(norm) = norm {
                    launch!(
                        self,
                        "head_norm",
                        ((size / dim * n) as u32, 1, 1),
                        &mut *values,
                        &norm.bytes,
                        &dim,
                        &s.epsilon
                    );
                }
                launch!(
                    self,
                    "rotary",
                    (((size / 2) as u32).div_ceil(256), n as u32, 1),
                    values,
                    &size,
                    &dim,
                    &pos,
                    &s.rope_base,
                    &neox
                );
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
            // Future KV rows are already written; each query can only see its prefix.
            if fused_attention_supported(s) {
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
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "attn_attended",
                &self.work.attended,
                t.width as usize
            );
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
                &self.work.attended,
                &mut self.work.projected,
                &t.width,
                &t.rows,
                &n
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "attn_output",
                &self.work.projected,
                t.rows as usize
            );
            launch!(
                self,
                "residual",
                vector(total_width),
                &mut self.work.x,
                &self.work.projected,
                &total_width
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_inp",
                &self.work.x,
                s.width
            );
            launch!(
                self,
                "normalize",
                (n as u32, 1, 1),
                &self.work.x,
                &l.ffn_norm.bytes,
                &mut self.work.normalized,
                &w,
                &s.epsilon
            );
            capture_projection!(
                self,
                index,
                position,
                tokens.len(),
                "ffn_norm",
                &self.work.normalized,
                s.width
            );
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
                launch!(
                    self,
                    "router",
                    (n as u32, 1, 1),
                    &self.work.router,
                    &experts,
                    &used,
                    &mut self.work.ids,
                    &mut self.work.probabilities,
                    &mut self.work.status
                );
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
                launch!(
                    self,
                    if self.matmul.group_tile() == 8 {
                        "group_experts_tiled"
                    } else {
                        "group_experts"
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
            for (t, out) in [(&l.gate, &mut self.work.gate), (&l.up, &mut self.work.up)] {
                if let Some(groups) = &self.work.groups {
                    launch!(
                        self,
                        grouped_projection,
                        ((t.rows as u32).div_ceil(group_row_tile), group_slots, 1),
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
                "swiglu",
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
            if let Some(groups) = &self.work.groups {
                launch!(
                    self,
                    grouped_projection,
                    ((t.rows as u32).div_ceil(group_row_tile), group_slots, 1),
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
