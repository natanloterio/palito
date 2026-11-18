//! Opt-in, bounded batched-prefill diagnostics. No weights are copied.
//!
//! Enable after reset, evaluate only batches whose internal chunks contain at least
//! two tokens, then save once. Decode graphs are unsupported. Reset cancels capture.
//! JSONL projection stages use the CPU reference names where applicable. Expert
//! projection rows contain all selected experts in routing-slot order.
//! Routing records cover every token of each batch in the selected layers,
//! independently of the projection-position filter. Counts split expert-major
//! assignment IDs; each assignment ID is `token * used + routing_slot`.
//! `max_bytes` bounds retained numeric payload and record storage, not JSON text
//! size or driver staging memory. Every device read is admitted before allocation.
use super::*;
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    mem::size_of,
};

#[derive(Clone, Debug)]
pub struct CaptureOptions {
    pub layers: Vec<usize>,
    /// Empty selects all projection stages. Routing is always retained.
    pub stages: Vec<String>,
    /// Global, zero-based prompt positions, not positions within an evaluate call.
    pub positions: Vec<usize>,
    pub max_bytes: usize,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Record {
    Projection {
        position: usize,
        layer: usize,
        stage: &'static str,
        values: Vec<f32>,
    },
    #[serde(rename = "projection")]
    ExpertIds {
        position: usize,
        layer: usize,
        stage: &'static str,
        values: Vec<i32>,
    },
    Routing {
        version: u32,
        layer: usize,
        position: usize,
        tokens: usize,
        experts: usize,
        used: usize,
        ids: Vec<i32>,
        counts: Vec<usize>,
        assignment_ids: Vec<usize>,
        logits: Vec<f32>,
    },
}

pub(super) struct Capture {
    options: CaptureOptions,
    bytes: usize,
    records: Vec<Record>,
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("capture size overflow"))
}

fn charge(bytes: &mut usize, limit: usize, amount: usize) -> Result<()> {
    let next = bytes
        .checked_add(amount)
        .ok_or_else(|| invalid("capture byte budget overflow"))?;
    if next > limit {
        return Err(invalid(format!(
            "capture requires {next} bytes; byte budget is {limit}"
        )));
    }
    *bytes = next;
    Ok(())
}

impl Capture {
    fn new(mut options: CaptureOptions, layers: usize, context: usize) -> Result<Self> {
        options.layers.sort_unstable();
        options.positions.sort_unstable();
        options.stages.sort_unstable();
        if options.max_bytes == 0
            || options.layers.is_empty()
            || options.positions.is_empty()
            || options.layers.last().is_some_and(|&n| n >= layers)
            || options.positions.last().is_some_and(|&n| n >= context)
            || options.layers.windows(2).any(|p| p[0] == p[1])
            || options.positions.windows(2).any(|p| p[0] == p[1])
            || options.stages.windows(2).any(|p| p[0] == p[1])
            || options.stages.iter().any(|stage| {
                !matches!(
                    stage.as_str(),
                    "attn_norm"
                        | "Qproj"
                        | "Kproj"
                        | "Vproj"
                        | "Qcur"
                        | "Kcur"
                        | "Vcur"
                        | "attn_attended"
                        | "attn_output"
                        | "ffn_inp"
                        | "ffn_norm"
                        | "ffn_moe_logits"
                        | "ffn_moe_weights_norm"
                        | "ffn_gate"
                        | "ffn_up"
                        | "ffn_down_inp"
                        | "ffn_down"
                        | "l_out"
                        | "ffn_moe_topk"
                )
            })
        {
            return Err(invalid("invalid capture layers, positions, or byte budget"));
        }
        let mut bytes = product(
            options
                .layers
                .capacity()
                .checked_add(options.positions.capacity())
                .ok_or_else(|| invalid("capture selection size overflow"))?,
            size_of::<usize>(),
        )?;
        charge(
            &mut bytes,
            options.max_bytes,
            product(options.stages.capacity(), size_of::<String>())?,
        )?;
        for stage in &options.stages {
            charge(&mut bytes, options.max_bytes, stage.capacity())?;
        }
        if bytes > options.max_bytes {
            return Err(invalid("capture selections exceed byte budget"));
        }
        Ok(Self {
            options,
            bytes,
            records: Vec::new(),
        })
    }

    pub(super) fn validate_batch(&self, tokens: usize, capacity: usize) -> Result<()> {
        if capacity < 2 || tokens < 2 || tokens % capacity == 1 {
            return Err(invalid(
                "capture supports batched prefill only; singleton chunks are unsupported",
            ));
        }
        Ok(())
    }

    fn selected(&self, layer: usize) -> bool {
        self.options.layers.binary_search(&layer).is_ok()
    }

    fn selected_stage(&self, stage: &str) -> bool {
        self.options.stages.is_empty()
            || self
                .options
                .stages
                .binary_search_by(|name| name.as_str().cmp(stage))
                .is_ok()
    }

    fn reserve_record(&mut self, payload: usize) -> Result<()> {
        // Grow exactly once per record rather than allowing Vec's geometric spare
        // capacity to escape the diagnostic budget.
        charge(
            &mut self.bytes,
            self.options.max_bytes,
            payload
                .checked_add(size_of::<Record>())
                .ok_or_else(|| invalid("capture record size overflow"))?,
        )?;
        self.records.try_reserve_exact(1).map_err(failure)?;
        Ok(())
    }

    pub(super) fn projection(
        &mut self,
        stream: &Arc<CudaStream>,
        (layer, position, tokens): (usize, usize, usize),
        stage: &'static str,
        source: &CudaSlice<f32>,
        width: usize,
    ) -> Result<()> {
        if !self.selected(layer) || !self.selected_stage(stage) {
            return Ok(());
        }
        let end = position
            .checked_add(tokens)
            .ok_or_else(|| invalid("capture position overflow"))?;
        for i in 0..self.options.positions.len() {
            let selected = self.options.positions[i];
            if selected < position || selected >= end {
                continue;
            }
            let start = product(selected - position, width)?;
            let stop = start
                .checked_add(width)
                .ok_or_else(|| invalid("capture projection size overflow"))?;
            if stop > source.len() {
                return Err(invalid("capture projection exceeds workspace"));
            }
            self.reserve_record(product(width, size_of::<f32>())?)?;
            let mut values = vec![0.0; width];
            stream
                .memcpy_dtoh(&source.slice(start..stop), &mut values)
                .map_err(failure)?;
            stream.synchronize().map_err(failure)?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(failure("nonfinite captured projection; request poisoned"));
            }
            self.records.push(Record::Projection {
                position: selected,
                layer,
                stage,
                values,
            });
        }
        Ok(())
    }

    pub(super) fn routing(
        &mut self,
        stream: &Arc<CudaStream>,
        (layer, position, tokens): (usize, usize, usize),
        experts: usize,
        used: usize,
        device_ids: &CudaSlice<i32>,
        device_logits: &CudaSlice<f32>,
    ) -> Result<()> {
        if !self.selected(layer) {
            return Ok(());
        }
        let assignments = product(tokens, used)?;
        let scores = product(tokens, experts)?;
        let payload = product(assignments, size_of::<i32>())?
            .checked_add(product(scores, size_of::<f32>())?)
            .and_then(|n| n.checked_add(product(experts, size_of::<usize>()).ok()?))
            .and_then(|n| n.checked_add(product(assignments, size_of::<usize>()).ok()?))
            .ok_or_else(|| invalid("capture routing size overflow"))?;
        if assignments > device_ids.len() || scores > device_logits.len() {
            return Err(invalid("capture routing exceeds workspace"));
        }
        self.reserve_record(payload)?;
        let mut ids = vec![0; assignments];
        let mut logits = vec![0.0; scores];
        stream
            .memcpy_dtoh(&device_ids.slice(..assignments), &mut ids)
            .map_err(failure)?;
        stream
            .memcpy_dtoh(&device_logits.slice(..scores), &mut logits)
            .map_err(failure)?;
        stream.synchronize().map_err(failure)?;
        if logits.iter().any(|v| !v.is_finite())
            || ids.iter().any(|&id| id < 0 || id as usize >= experts)
        {
            return Err(failure("invalid captured routing; request poisoned"));
        }
        let (counts, assignment_ids) = assignments_by_expert(&ids, experts, used)?;
        let routing_index = self.records.len();
        self.records.push(Record::Routing {
            version: 1,
            layer,
            position,
            tokens,
            experts,
            used,
            ids,
            counts,
            assignment_ids,
            logits,
        });
        if !self.selected_stage("ffn_moe_topk") {
            return Ok(());
        }
        for i in 0..self.options.positions.len() {
            let selected = self.options.positions[i];
            if selected >= position && selected - position < tokens {
                self.reserve_record(product(used, size_of::<i32>())?)?;
                let offset = product(selected - position, used)?;
                let Record::Routing { ids, .. } = &self.records[routing_index] else {
                    unreachable!();
                };
                let values = ids[offset..offset + used].to_vec();
                self.records.push(Record::ExpertIds {
                    position: selected,
                    layer,
                    stage: "ffn_moe_topk",
                    values,
                });
            }
        }
        Ok(())
    }
}

fn assignments_by_expert(
    ids: &[i32],
    experts: usize,
    used: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if experts == 0 || used == 0 || used > experts || ids.len() % used != 0 {
        return Err(invalid("invalid capture routing shape"));
    }
    for selected in ids.chunks_exact(used) {
        for (slot, &id) in selected.iter().enumerate() {
            if id < 0 || id as usize >= experts || selected[..slot].contains(&id) {
                return Err(invalid("invalid or duplicate captured expert ID"));
            }
        }
    }
    let mut counts = vec![0usize; experts];
    let mut assignment_ids = Vec::with_capacity(ids.len());
    for (expert, count) in counts.iter_mut().enumerate() {
        for (assignment, &id) in ids.iter().enumerate() {
            if id as usize == expert {
                *count += 1;
                assignment_ids.push(assignment);
            }
        }
    }
    Ok((counts, assignment_ids))
}

impl CudaBackend {
    /// Enable one bounded capture on a fresh request, with decode graphs disabled.
    /// All subsequent evaluate calls must consist exclusively of batched-prefill
    /// chunks (at least two tokens each). Reset cancels an unfinished capture.
    pub fn enable_capture(&mut self, options: CaptureOptions) -> Result<()> {
        if self.decode.is_some()
            || self.diagnostic.is_some()
            || self.state.failed
            || self.state.position != 0
        {
            return Err(invalid(
                "capture requires a fresh, unpoisoned request with graphs disabled",
            ));
        }
        self.diagnostic = Some(Capture::new(
            options,
            self.weights.layers.len(),
            self.weights.spec.info.context_tokens as usize,
        )?);
        Ok(())
    }

    /// Complete capture and write a new immutable JSONL file; never overwrite.
    /// Failed requests and incomplete selected positions cannot be saved.
    /// A successful save disables capture. An IO error retains it for retry at
    /// another new path (the failed path may contain a partial file).
    pub fn save_capture(&mut self, path: &Path) -> Result<()> {
        let capture = self
            .diagnostic
            .as_ref()
            .ok_or_else(|| invalid("capture is not enabled"))?;
        if self.state.failed
            || !self.state.valid
            || capture
                .options
                .positions
                .last()
                .is_some_and(|&p| p >= self.state.position)
            || capture.records.is_empty()
        {
            return Err(invalid("capture request is failed or incomplete"));
        }
        let mut writer =
            BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "kind": "metadata", "version": 1, "matmul": self.matmul,
                "layers": capture.options.layers, "positions": capture.options.positions,
                "stages": capture.options.stages,
                "payload_bytes": capture.bytes, "max_bytes": capture.options.max_bytes,
                "layout": "token-major; expert projections routing-slot-major within token",
            }),
        )?;
        writer.write_all(b"\n")?;
        for record in &capture.records {
            serde_json::to_writer(&mut writer, record)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        self.diagnostic = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> CaptureOptions {
        CaptureOptions {
            layers: vec![0],
            stages: vec![],
            positions: vec![0],
            max_bytes: 4096,
        }
    }

    #[test]
    fn validates_selections_and_singleton_chunks() {
        let capture = Capture::new(options(), 2, 128).unwrap();
        assert!(capture.validate_batch(128, 128).is_ok());
        assert!(capture.validate_batch(130, 128).is_ok());
        assert!(capture.validate_batch(129, 128).is_err());
        assert!(capture.validate_batch(1, 128).is_err());
        assert!(capture.validate_batch(2, 1).is_err());
        for (layers, positions, budget) in [
            (vec![], vec![0], 1024),
            (vec![2], vec![0], 1024),
            (vec![0, 0], vec![0], 1024),
            (vec![0], vec![128], 1024),
            (vec![0], vec![0, 0], 1024),
            (vec![0], vec![0], 0),
        ] {
            assert!(
                Capture::new(
                    CaptureOptions {
                        layers,
                        stages: vec![],
                        positions,
                        max_bytes: budget,
                    },
                    2,
                    128
                )
                .is_err()
            );
        }
    }

    #[test]
    fn stage_filters_are_explicit_validated_and_budgeted() {
        let all = Capture::new(options(), 2, 128).unwrap();
        assert!(all.selected_stage("ffn_gate"));
        let mut selected = options();
        selected.stages = vec!["Qproj".into()];
        let filtered = Capture::new(selected.clone(), 2, 128).unwrap();
        assert!(filtered.selected_stage("Qproj"));
        assert!(!filtered.selected_stage("ffn_gate"));
        assert!(filtered.bytes > all.bytes);
        selected.max_bytes = all.bytes;
        assert!(Capture::new(selected.clone(), 2, 128).is_err());
        selected.max_bytes = 4096;
        selected.stages = vec!["unknown".into()];
        assert!(Capture::new(selected.clone(), 2, 128).is_err());
        selected.stages = vec!["Qproj".into(), "Qproj".into()];
        assert!(Capture::new(selected, 2, 128).is_err());
    }

    #[test]
    fn budget_rejects_before_allocating_record() {
        let mut capture = Capture::new(options(), 2, 128).unwrap();
        let bytes = capture.bytes;
        assert!(capture.reserve_record(4096).is_err());
        assert_eq!(capture.bytes, bytes);
        assert_eq!(capture.records.capacity(), 0);
        assert!(capture.reserve_record(usize::MAX).is_err());
        assert!(product(usize::MAX, 2).is_err());
        let mut exact = 0;
        charge(&mut exact, 16, 16).unwrap();
        assert!(charge(&mut exact, 16, 1).is_err());
        assert_eq!(exact, 16);
        let mut bytes = usize::MAX;
        assert!(charge(&mut bytes, usize::MAX, 1).is_err());
    }

    #[test]
    fn routing_assignments_preserve_token_slot_order() {
        let (counts, ids) = assignments_by_expert(&[2, 0, 1, 2, 2, 0], 4, 2).unwrap();
        assert_eq!(counts, [2, 1, 3, 0]);
        assert_eq!(ids, [1, 5, 2, 0, 3, 4]);
        assert!(assignments_by_expert(&[0, 0], 2, 2).is_err());
        assert!(assignments_by_expert(&[-1], 2, 1).is_err());
        assert!(assignments_by_expert(&[2], 2, 1).is_err());
        assert!(assignments_by_expert(&[0], 2, 2).is_err());
    }

    #[test]
    fn routing_json_roundtrip_exposes_replay_schema() {
        let ids = vec![2, 0, 1, 2];
        let (counts, assignment_ids) = assignments_by_expert(&ids, 3, 2).unwrap();
        let record = Record::Routing {
            version: 1,
            layer: 3,
            position: 128,
            tokens: 2,
            experts: 3,
            used: 2,
            ids,
            counts,
            assignment_ids,
            logits: vec![0.0; 6],
        };
        let json = serde_json::to_string(&record).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["kind"], "routing");
        assert_eq!(value["ids"], serde_json::json!([2, 0, 1, 2]));
        assert_eq!(value["counts"], serde_json::json!([1, 1, 2]));
        assert_eq!(value["assignment_ids"], serde_json::json!([1, 2, 0, 3]));
    }

    #[test]
    #[ignore = "requires CUDA, CUDA_CAPTURE_MODEL and CUDA_CAPTURE_IDS (JSON token array); writes under target"]
    fn real_model_capture_roundtrip() -> Result<()> {
        use crate::backend::Backend;
        let model = std::env::var("CUDA_CAPTURE_MODEL").map_err(failure)?;
        let ids_path = std::env::var("CUDA_CAPTURE_IDS").map_err(failure)?;
        let ids: Vec<i32> = serde_json::from_slice(&std::fs::read(ids_path)?)?;
        if ids.len() < 2 || ids.len() > 128 {
            return Err(invalid("roundtrip fixture needs 2..=128 prompt IDs"));
        }
        let config = Config {
            context_tokens: 128,
            batch_tokens: 128,
            prefill_chunk: 128,
            threads: 2,
        };
        let mut backend = CudaBackend::open(Path::new(&model), &config, CudaOptions::default())?;
        backend.evaluate(&ids, 0)?;
        let expected = backend.logits()?.to_vec();
        backend.reset()?;
        backend.enable_capture(CaptureOptions {
            layers: vec![0],
            stages: vec![],
            positions: vec![0, ids.len() - 1],
            max_bytes: 64 * 1024 * 1024,
        })?;
        backend.evaluate(&ids, 0)?;
        assert_eq!(backend.logits()?, expected);
        std::fs::create_dir_all("target")?;
        let path = PathBuf::from(format!("target/cuda-capture-{}.jsonl", std::process::id()));
        backend.save_capture(&path)?;
        let contents = std::fs::read_to_string(&path)?;
        backend.reset()?;
        backend.enable_capture(CaptureOptions {
            layers: vec![0],
            stages: vec!["Qproj".into()],
            positions: vec![0],
            max_bytes: 64 * 1024 * 1024,
        })?;
        backend.evaluate(&ids, 0)?;
        assert!(backend.save_capture(&path).is_err());
        assert_eq!(std::fs::read_to_string(&path)?, contents);
        assert_eq!(backend.logits()?, expected);
        let filtered_path = path.with_extension("filtered.jsonl");
        backend.save_capture(&filtered_path)?;
        let filtered: Vec<serde_json::Value> = std::fs::read_to_string(&filtered_path)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(filtered[0]["stages"], serde_json::json!(["Qproj"]));
        assert!(filtered.iter().any(|r| r["stage"] == "Qproj"));
        assert!(
            filtered
                .iter()
                .filter(|r| r["kind"] == "projection")
                .all(|r| r["stage"] == "Qproj")
        );
        std::fs::remove_file(filtered_path)?;
        let rows: Vec<serde_json::Value> = contents
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert!(
            rows.iter()
                .any(|r| r["stage"] == "Qproj" && r["position"] == 0)
        );
        assert!(
            rows.iter()
                .any(|r| r["stage"] == "l_out" && r["position"] == ids.len() - 1)
        );
        std::fs::remove_file(&path)?;
        backend.reset()?;
        backend.enable_capture(CaptureOptions {
            layers: vec![0],
            stages: vec![],
            positions: vec![0],
            max_bytes: 2 * size_of::<usize>(),
        })?;
        assert!(backend.evaluate(&ids, 0).is_err());
        assert!(backend.logits().is_err());
        assert!(backend.save_capture(&path).is_err());
        assert!(!path.try_exists()?);
        Ok(())
    }
}
