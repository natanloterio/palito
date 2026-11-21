use super::*;
use cudarc::driver::{CudaGraph, PinnedHostSlice};

pub(super) struct DecodeState {
    graph: Option<CudaGraph>,
    pub control: CudaSlice<i32>,
    scores: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    tokens: CudaSlice<i32>,
    /// Pinned copies of `tokens` taken after each pipelined replay (see
    /// `sample_decode_tokens`); one more slot than replays allowed in flight.
    ring: Vec<PinnedHostSlice<i32>>,
    pub sampled: Option<i32>,
}

/// Replays the host lets run ahead of the tokens it has read back.
const REPLAYS_IN_FLIGHT: usize = 2;

impl DecodeState {
    pub fn bytes(vocabulary: usize) -> u64 {
        (12 + 2 * vocabulary.div_ceil(256)) as u64 * 4
    }

    pub fn new(stream: &Arc<CudaStream>, vocabulary: usize) -> Result<Self> {
        Ok(Self {
            graph: None,
            control: stream.alloc_zeros(3).map_err(failure)?,
            scores: stream
                .alloc_zeros(vocabulary.div_ceil(256))
                .map_err(failure)?,
            ids: stream
                .alloc_zeros(vocabulary.div_ceil(256))
                .map_err(failure)?,
            tokens: stream.alloc_zeros(9).map_err(failure)?,
            ring: (0..=REPLAYS_IN_FLIGHT)
                .map(|_| {
                    // Safety: every slot is written by a device copy before it is read.
                    unsafe { stream.context().alloc_pinned::<i32>(9) }.map_err(failure)
                })
                .collect::<Result<_>>()?,
            sampled: None,
        })
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Graphs refer to raw addresses; finish replay before field destruction.
        self.stream.context().record_err(self.stream.synchronize());
    }
}

impl CudaBackend {
    pub(super) fn capture_decode_graph(&mut self) -> Result<()> {
        self.stream.synchronize().map_err(failure)?;
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(failure)?;
        let submitted = self
            .step(0, 0, true, true)
            .and_then(|()| self.device_sample());
        // End capture even if submission failed, so teardown never retains capture state.
        let captured = self.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        submitted?;
        let graph = captured
            .map_err(failure)?
            .ok_or_else(|| failure("decode graph capture was empty"))?;
        self.decode
            .as_mut()
            .ok_or_else(|| failure("decode state missing"))?
            .graph = Some(graph);
        Ok(())
    }
}

impl CudaBackend {
    pub(super) fn prepare_decode(&mut self, token: i32, position: usize, slot: i32) -> Result<()> {
        let decode = self
            .decode
            .as_mut()
            .ok_or_else(|| failure("decode state missing"))?;
        self.stream
            .memcpy_htod(&[token, position as i32, slot], &mut decode.control)
            .map_err(failure)
    }

    pub(super) fn replay_decode(&mut self, token: i32, position: usize) -> Result<()> {
        self.prepare_decode(token, position, 0)?;
        self.launch_decode()
    }

    fn launch_decode(&self) -> Result<()> {
        self.decode
            .as_ref()
            .and_then(|d| d.graph.as_ref())
            .ok_or_else(|| failure("decode graph missing"))?
            .launch()
            .map_err(failure)
    }

    pub(super) fn device_sample(&mut self) -> Result<()> {
        let count = self.weights.spec.info.vocab_tokens as i32;
        let decode = self
            .decode
            .as_mut()
            .ok_or_else(|| failure("decode state missing"))?;
        let blocks = (count as u32).div_ceil(256) as i32;
        launch!(
            self,
            "greedy_partials",
            (blocks as u32, 1, 1),
            &self.work.logits,
            &count,
            &mut decode.scores,
            &mut decode.ids,
            &mut self.work.status
        );
        launch!(
            self,
            "greedy_finish",
            (1, 1, 1),
            &decode.scores,
            &decode.ids,
            &blocks,
            &mut decode.control,
            &mut decode.tokens,
            &mut self.work.status
        );
        Ok(())
    }

    pub(super) fn read_decode_tokens(&self, count: usize) -> Result<Vec<i32>> {
        let decode = self
            .decode
            .as_ref()
            .ok_or_else(|| failure("decode state missing"))?;
        let mut result = [0i32; 9];
        self.stream
            .memcpy_dtoh(&decode.tokens, &mut result)
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)?;
        if result[8] != 0
            || result[..count]
                .iter()
                .any(|&id| id < 0 || id as u32 >= self.weights.spec.info.vocab_tokens)
        {
            return Err(failure(
                "nonfinite router scores/logits or invalid device sample; request poisoned",
            ));
        }
        Ok(result[..count].to_vec())
    }

    pub(super) fn sample_decode_tokens(
        &mut self,
        maximum: usize,
        mode: crate::GenerationMode,
    ) -> Result<Vec<i32>> {
        if maximum == 0 {
            return Err(invalid("sample maximum must be positive"));
        }
        let first = self.sample_greedy()?;
        let until_eog = mode == crate::GenerationMode::UntilEog;
        if self.decode.is_none() || (until_eog && self.is_eog(first)) {
            return Ok(vec![first]);
        }
        let count = maximum
            .min(self.weights.spec.info.context_tokens as usize - self.state.position + 1);
        if count == 1 {
            return Ok(vec![first]);
        }
        self.state.valid = false;
        self.state.failed = true;
        self.prepare_decode(first, self.state.position, 1)?;
        let decode = self.decode.as_mut().unwrap();
        self.stream
            .memcpy_htod(&[first], &mut decode.tokens.slice_mut(..1))
            .map_err(failure)?;
        // Replay i writes its token to tokens[i % 8] (the slot counter is rewound every
        // 8 replays) and is followed by a copy of `tokens` into ring[i % ring.len()].
        // The host keeps at most REPLAYS_IN_FLIGHT replays unread, so the GPU never
        // waits on a full-stream sync, and stops issuing at the first end-of-generation
        // token: at most REPLAYS_IN_FLIGHT speculative steps are discarded.
        let mut tokens = vec![first];
        let mut launched = 1;
        while tokens.len() < count {
            if launched < count && launched - tokens.len() < REPLAYS_IN_FLIGHT {
                let decode = self.decode.as_mut().unwrap();
                if launched % 8 == 0 {
                    self.stream
                        .memcpy_htod(&[0], &mut decode.control.slice_mut(2..3))
                        .map_err(failure)?;
                }
                self.launch_decode()?;
                let decode = self.decode.as_mut().unwrap();
                let slot = launched % decode.ring.len();
                self.stream
                    .memcpy_dtoh(&decode.tokens, &mut decode.ring[slot])
                    .map_err(failure)?;
                launched += 1;
                continue;
            }
            let index = tokens.len();
            let decode = self.decode.as_ref().unwrap();
            let copy = decode.ring[index % decode.ring.len()]
                .as_slice()
                .map_err(failure)?;
            let token = copy[index % 8];
            if copy[8] != 0 || token < 0 || token as u32 >= self.weights.spec.info.vocab_tokens {
                return Err(failure(
                    "nonfinite router scores/logits or invalid device sample; request poisoned",
                ));
            }
            tokens.push(token);
            if until_eog && self.is_eog(token) {
                break;
            }
        }
        // Only the kept steps count as live context; replays issued past an
        // end-of-generation token leave no usable next-token state.
        self.state.finish(tokens.len() - 1);
        let decode = self.decode.as_mut().unwrap();
        if launched > tokens.len() {
            self.state.valid = false;
            decode.sampled = None;
        } else {
            decode.sampled = tokens.last().copied();
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA 13; run serially in release mode"]
    fn warp_decode_matvec_is_bitwise_and_measures_representative_shapes() -> Result<()> {
        let context = CudaContext::new(0).map_err(failure)?;
        let module = context
            .load_module(
                compile_ptx_with_opts(
                    concat!(
                        "#define LOCAL_DECODE_BENCH\n",
                        include_str!("kernels.cu"),
                        "\n",
                        include_str!("decode.cu")
                    ),
                    CompileOptions {
                        fmad: Some(false),
                        ..Default::default()
                    },
                )
                .map_err(failure)?,
            )
            .map_err(failure)?;
        struct Harness {
            stream: Arc<CudaStream>,
            functions: HashMap<&'static str, CudaFunction>,
        }
        let harness = Harness {
            stream: context.default_stream(),
            functions: ["matvec", "matvec_decode"]
                .into_iter()
                .map(|name| Ok((name, module.load_function(name).map_err(failure)?)))
                .collect::<Result<_>>()?,
        };
        for kind in [0, 1, 6, 8, 12, 14] {
            for width in [256usize, 768, 2048] {
                let rows = 257;
                let t = DeviceTensor::upload(
                    super::super::tests::packed_fixture(kind, width, rows, 3),
                    &harness.stream,
                )?;
                let x = harness
                    .stream
                    .clone_htod(
                        &(0..width * 2)
                            .map(|i| (i % 19) as f32 / 19.0 - 0.3)
                            .collect::<Vec<_>>(),
                    )
                    .map_err(failure)?;
                let ids = harness.stream.clone_htod(&[2i32, 0]).map_err(failure)?;
                for batched in [0i32, 1] {
                    let mut expected = harness
                        .stream
                        .alloc_zeros::<f32>(rows * 2)
                        .map_err(failure)?;
                    let mut actual = harness
                        .stream
                        .clone_htod(&vec![12345.0f32; rows * 2 + 8])
                        .map_err(failure)?;
                    launch!(
                        harness,
                        "matvec",
                        (rows as u32, 2, 1),
                        &t.bytes,
                        &t.kind,
                        &x,
                        &mut expected,
                        &t.width,
                        &t.rows,
                        &ids,
                        &1i32,
                        &batched
                    );
                    let mut output = actual.slice_mut(..rows * 2);
                    launch!(
                        harness,
                        "matvec_decode",
                        ((rows as u32).div_ceil(8), 2, 1),
                        &t.bytes,
                        &t.kind,
                        &x,
                        &mut output,
                        &t.width,
                        &t.rows,
                        &ids,
                        &1i32,
                        &batched
                    );
                    let expected = harness.stream.clone_dtoh(&expected).map_err(failure)?;
                    let actual = harness.stream.clone_dtoh(&actual).map_err(failure)?;
                    assert_eq!(
                        &actual[..rows * 2],
                        expected,
                        "kind={kind},width={width},batched={batched}"
                    );
                    assert!(actual[rows * 2..].iter().all(|&x| x == 12345.0));
                }
            }
        }
        for kind in [12, 14] {
            for (width, rows, batched) in [(2048, 768, 0i32), (768, 2048, 1)] {
                let t = DeviceTensor::upload(
                    super::super::tests::packed_fixture(kind, width, rows, 8),
                    &harness.stream,
                )?;
                let x = harness
                    .stream
                    .clone_htod(&vec![0.125f32; width * 8])
                    .map_err(failure)?;
                let ids = harness
                    .stream
                    .clone_htod(&(0..8).collect::<Vec<i32>>())
                    .map_err(failure)?;
                let mut out = harness
                    .stream
                    .alloc_zeros::<f32>(rows * 8)
                    .map_err(failure)?;
                for (name, tile) in [("matvec", 1), ("matvec_decode", 8)] {
                    let mut timings = Vec::new();
                    for iteration in 0..10 {
                        let start = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        for _ in 0..20 {
                            launch!(
                                harness,
                                name,
                                ((rows as u32).div_ceil(tile), 8, 1),
                                &t.bytes,
                                &t.kind,
                                &x,
                                &mut out,
                                &t.width,
                                &t.rows,
                                &ids,
                                &1i32,
                                &batched
                            );
                        }
                        let end = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        harness.stream.synchronize().map_err(failure)?;
                        if iteration >= 3 {
                            timings.push(start.elapsed_ms(&end).map_err(failure)? / 20.0);
                        }
                    }
                    timings.sort_by(f32::total_cmp);
                    println!(
                        "decode_product kind={kind} width={width} rows={rows} kernel={name} median_ms={}",
                        timings[3]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA 13; run serially in release mode"]
    fn device_greedy_preserves_ties_rejects_all_nonfinites_and_handles_vocab_tails() -> Result<()> {
        let context = CudaContext::new(0).map_err(failure)?;
        let stream = context.default_stream();
        let module = context
            .load_module(
                compile_ptx_with_opts(
                    concat!(include_str!("kernels.cu"), "\n", include_str!("decode.cu")),
                    CompileOptions::default(),
                )
                .map_err(failure)?,
            )
            .map_err(failure)?;
        struct Harness {
            stream: Arc<CudaStream>,
            functions: HashMap<&'static str, CudaFunction>,
        }
        let harness = Harness {
            stream,
            functions: ["greedy_partials", "greedy_finish"]
                .into_iter()
                .map(|name| Ok((name, module.load_function(name).map_err(failure)?)))
                .collect::<Result<_>>()?,
        };
        for count in [1usize, 257, 1025, 151_936] {
            for nonfinite in [
                None,
                Some(f32::NAN),
                Some(f32::INFINITY),
                Some(f32::NEG_INFINITY),
            ] {
                let mut values = vec![-0.0f32; count];
                if count > 256 {
                    values[255] = 4.0;
                    values[count - 1] = 4.0;
                }
                if let Some(value) = nonfinite {
                    values[0] = value;
                }
                let input = harness.stream.clone_htod(&values).map_err(failure)?;
                let blocks = count.div_ceil(256) as i32;
                let mut state = DecodeState::new(&harness.stream, count)?;
                harness
                    .stream
                    .memcpy_htod(&[0, 7, 0], &mut state.control)
                    .map_err(failure)?;
                let mut status = harness.stream.alloc_zeros::<i32>(1).map_err(failure)?;
                let n = count as i32;
                launch!(
                    harness,
                    "greedy_partials",
                    (blocks as u32, 1, 1),
                    &input,
                    &n,
                    &mut state.scores,
                    &mut state.ids,
                    &mut status
                );
                launch!(
                    harness,
                    "greedy_finish",
                    (1, 1, 1),
                    &state.scores,
                    &state.ids,
                    &blocks,
                    &mut state.control,
                    &mut state.tokens,
                    &mut status
                );
                let actual = harness.stream.clone_dtoh(&state.tokens).map_err(failure)?;
                if nonfinite.is_some() {
                    assert_eq!(actual[8], 1);
                } else {
                    assert_eq!(actual[8], 0);
                    assert_eq!(actual[0], greedy(&values)?);
                    assert_eq!(
                        harness.stream.clone_dtoh(&state.control).map_err(failure)?,
                        [actual[0], 8, 1]
                    );
                }
                assert!(actual[1..8].iter().all(|&id| id == 0));
            }
        }
        Ok(())
    }
}
