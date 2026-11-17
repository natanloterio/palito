use super::{
    prefill::{Harness, inputs},
    tests::packed_fixture,
    *,
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, io::Read, time::Instant};

#[test]
#[ignore = "grouped GPU correctness/microbenchmark; run serially with --nocapture"]
fn benchmark_grouped_activation_reuse() -> Result<()> {
    benchmark_grouped_modes(
        &[("grouped_matmul", 1, 4), ("grouped_matmul_rows2", 2, 4)],
        &[128],
        true,
    )
}

#[test]
#[ignore = "grouped tiled F32 GPU microbenchmark; run serially with --nocapture"]
fn benchmark_grouped_tiled_f32() -> Result<()> {
    benchmark_grouped_modes(
        &[("grouped_matmul", 1, 4), ("grouped_matmul_tiled", 32, 8)],
        &[64, 128, 256, 512],
        false,
    )
}

#[test]
#[ignore = "grouped tiled TF32x3 GPU microbenchmark; run serially with --nocapture"]
fn benchmark_grouped_tiled_tf32x3() -> Result<()> {
    benchmark_grouped_modes(
        &[
            ("grouped_matmul_tf32x3", 128, 4),
            ("grouped_matmul_tf32x3_tiled", 128, 8),
        ],
        &[64, 128, 256, 512],
        false,
    )
}

fn benchmark_grouped_modes(
    modes: &[(&'static str, u32, usize)],
    token_counts: &[usize],
    exact: bool,
) -> Result<()> {
    let harness = Harness::new()?;
    let mut shapes = vec![(256usize, 5usize, 7usize, 3usize, 2usize)];
    for &tokens in token_counts {
        shapes.push((2048, 768, tokens, 128, 8));
        shapes.push((768, 2048, tokens, 128, 8));
    }
    for kind in [0, 1, 6, 8, 12, 14] {
        for &(width, rows, tokens, experts, used) in &shapes {
            if rows > 5 && !matches!(kind, 12 | 14) {
                continue;
            }
            let ids: Vec<_> = (0..tokens)
                .flat_map(|t| (0..used).map(move |s| ((t * 5 + s) % experts) as i32))
                .collect();
            let ids = harness.stream.clone_htod(&ids).map_err(failure)?;
            let (ne, nu, nt) = (experts as i32, used as i32, tokens as i32);
            let mut descriptors = Vec::new();
            for tile in [4usize, 8] {
                let slots = (tokens * used).min((tokens * used).div_ceil(tile) + experts - 1);
                let mut groups = harness
                    .stream
                    .alloc_zeros::<i32>(slots * (tile + 1))
                    .map_err(failure)?;
                let ns = slots as i32;
                launch!(
                    harness,
                    if tile == 4 {
                        "group_experts"
                    } else {
                        "group_experts_tiled"
                    },
                    (1, 1, 1),
                    &ids,
                    &mut groups,
                    &ne,
                    &nu,
                    &nt,
                    &ns
                );
                descriptors.push((slots, groups));
            }
            let tensor =
                DeviceTensor::upload(packed_fixture(kind, width, rows, experts), &harness.stream)?;
            for batched in [0i32, 1] {
                let count = tokens * if batched == 1 { used } else { 1 };
                let input = harness
                    .stream
                    .clone_htod(&inputs(width, count))
                    .map_err(failure)?;
                let length = tokens * used * rows;
                let mut output = harness
                    .stream
                    .clone_htod(&vec![1337.0f32; length + 16])
                    .map_err(failure)?;
                macro_rules! project {
                    ($mode:expr) => {{
                        let (name, row_tile, tile) = $mode;
                        let (slots, groups) = &descriptors[if tile == 4 { 0 } else { 1 }];
                        let mut y = output.slice_mut(8..8 + length);
                        launch!(
                            harness,
                            name,
                            ((rows as u32).div_ceil(row_tile), *slots as u32, 1),
                            &tensor.bytes,
                            &tensor.kind,
                            &input,
                            &mut y,
                            &tensor.width,
                            &tensor.rows,
                            groups,
                            &nu,
                            &batched
                        );
                    }};
                }
                let mut expected = None;
                for &mode in modes {
                    let (name, _, _) = mode;
                    output = harness
                        .stream
                        .clone_htod(&vec![1337.0f32; length + 16])
                        .map_err(failure)?;
                    project!(mode);
                    let actual = harness.stream.clone_dtoh(&output).map_err(failure)?;
                    assert!(
                        actual[..8]
                            .iter()
                            .chain(&actual[8 + length..])
                            .all(|&x| x == 1337.0)
                    );
                    if let Some(expected) = &expected {
                        for (i, (&a, &b)) in actual[8..8 + length].iter().zip(expected).enumerate()
                        {
                            assert!(
                                a.is_finite()
                                    && if exact {
                                        a.to_bits() == f32::to_bits(b)
                                    } else {
                                        (a - b).abs() <= 0.005 + 0.003 * b.abs()
                                    },
                                "{name} kind={kind} output={i}: {a} != {b}"
                            );
                        }
                    } else {
                        expected = Some(actual[8..8 + length].to_vec());
                    }
                }
                for _ in 0..3 {
                    for &mode in modes {
                        project!(mode);
                    }
                }
                for repetition in 0..7 {
                    for slot in 0..modes.len() {
                        let mode = modes[(slot + repetition) % modes.len()];
                        let (name, _, group_tile) = mode;
                        let start = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        for _ in 0..8 {
                            project!(mode);
                        }
                        let end = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        end.synchronize().map_err(failure)?;
                        println!(
                            "GROUPED_BENCH {}",
                            json!({
                                "kind":kind,"width":width,"rows":rows,"tokens":tokens,"experts":experts,
                                "used":used,"batched":batched,"mode":name,"group_tile":group_tile,
                                "repetition":repetition,
                                "cuda_event_ms":start.elapsed_ms(&end).map_err(failure)? / 8.0,
                            })
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

// M8_ROUTING_FIXTURE accepts a local JSON object, array, {"records":[...]}
// manifest, or diagnostics JSONL (metadata/projection records are skipped).
// Routing records use token-major ids[token * used + slot]. Captured activations
// and weights are not replayed here: projection operands remain packed_fixture.
#[derive(Debug, Deserialize)]
struct RoutingFixture {
    #[serde(default)]
    name: String,
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    layer: Option<usize>,
    #[serde(default)]
    position: Option<usize>,
    experts: usize,
    used: usize,
    tokens: usize,
    ids: Vec<i32>,
    #[serde(default)]
    counts: Option<Vec<usize>>,
    #[serde(default)]
    assignment_ids: Option<Vec<usize>>,
}

const MAX_FIXTURE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ROUTING_RECORDS: usize = 128;
const MAX_ROUTING_EXPERTS: usize = 4096;
const MAX_ROUTING_ASSIGNMENTS: usize = 65535;

impl RoutingFixture {
    fn validate(&self) -> Result<Vec<usize>> {
        let assignments = self
            .tokens
            .checked_mul(self.used)
            .ok_or_else(|| failure("routing tokens * used overflow"))?;
        if self.version.is_some_and(|version| version != 1) {
            return Err(failure("unsupported routing fixture version"));
        }
        if self.experts == 0
            || self.experts > MAX_ROUTING_EXPERTS
            || self.used == 0
            || self.used > self.experts
            || self.tokens == 0
            || assignments > MAX_ROUTING_ASSIGNMENTS
            || self.ids.len() != assignments
        {
            return Err(failure("invalid/bounded routing dimensions or ids length"));
        }
        let mut counts = vec![0; self.experts];
        let mut seen = vec![usize::MAX; self.experts];
        for (assignment, &id) in self.ids.iter().enumerate() {
            let expert = usize::try_from(id).map_err(|_| failure("negative routing expert ID"))?;
            if expert >= self.experts {
                return Err(failure("routing expert ID out of range"));
            }
            let token = assignment / self.used;
            if seen[expert] == token {
                return Err(failure("duplicate routing expert per token"));
            }
            seen[expert] = token;
            counts[expert] += 1;
        }
        if self
            .counts
            .as_ref()
            .is_some_and(|expected| expected != &counts)
        {
            return Err(failure("routing counts disagree with ids"));
        }
        if let Some(expected) = &self.assignment_ids {
            let actual: Vec<_> = (0..self.experts)
                .flat_map(|expert| {
                    self.ids
                        .iter()
                        .enumerate()
                        .filter_map(move |(assignment, &id)| {
                            (id as usize == expert).then_some(assignment)
                        })
                })
                .collect();
            if *expected != actual {
                return Err(failure("routing assignment_ids disagree with ids"));
            }
        }
        Ok(counts)
    }

    fn slots(&self, tile: usize) -> usize {
        self.ids
            .len()
            .min(self.ids.len().div_ceil(tile) + self.experts - 1)
    }

    fn descriptors(&self, tile: usize) -> Vec<i32> {
        let mut result = Vec::new();
        for expert in 0..self.experts {
            let assignments: Vec<_> = self
                .ids
                .iter()
                .enumerate()
                .filter_map(|(a, &id)| (id as usize == expert).then_some(a as i32))
                .collect();
            for chunk in assignments.chunks(tile) {
                result.push(expert as i32);
                result.extend_from_slice(chunk);
                result.resize(result.len() + tile - chunk.len(), -1);
            }
        }
        result.resize(self.slots(tile) * (tile + 1), -1);
        result
    }

    fn statistics(
        &self,
        counts: &[usize],
        tile: usize,
        rows: usize,
        row_tile: usize,
    ) -> serde_json::Value {
        let mut sorted = counts.to_vec();
        sorted.sort_unstable();
        let quantile = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
        let mut histogram = BTreeMap::<usize, usize>::new();
        for &count in counts {
            *histogram.entry(count).or_default() += 1;
        }
        let active = counts.iter().map(|n| n.div_ceil(tile)).sum::<usize>();
        let capacity = self.slots(tile);
        let row_tiles = rows.div_ceil(row_tile);
        json!({
            "counts": counts, "count_histogram": histogram,
            "empty_experts": counts.iter().filter(|&&n| n == 0).count(),
            "tiny_experts_1_to_4": counts.iter().filter(|&&n| (1..=4).contains(&n)).count(),
            "count_min": sorted[0], "count_p50": quantile(50),
            "count_p90": quantile(90), "count_p95": quantile(95),
            "count_max": sorted[sorted.len() - 1],
            "assignments": self.ids.len(), "group_tile": tile,
            "descriptor_capacity": capacity, "active_descriptors": active,
            "inactive_descriptors": capacity - active,
            "active_assignment_lanes": active * tile,
            "assignment_padding": active * tile - self.ids.len(),
            "reserved_assignment_lanes": capacity * tile,
            "useful_output_values": self.ids.len() * rows,
            "active_padded_output_values": active * tile * row_tiles * row_tile,
            "active_row_tile_work": active * row_tiles,
            "launched_row_tile_work": capacity * row_tiles,
        })
    }
}

fn collect_routing(value: serde_json::Value, fixtures: &mut Vec<RoutingFixture>) -> Result<()> {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_routing(value, fixtures)?;
            }
        }
        serde_json::Value::Object(mut object) => {
            if let Some(records) = object.remove("records") {
                if !records.is_array() {
                    return Err(failure("routing manifest records must be an array"));
                }
                collect_routing(records, fixtures)?;
            } else {
                match object.get("kind") {
                    Some(serde_json::Value::String(kind))
                        if matches!(kind.as_str(), "metadata" | "projection") =>
                    {
                        return Ok(());
                    }
                    None => {}
                    Some(serde_json::Value::String(kind)) if kind == "routing" => {}
                    _ => return Err(failure("unknown routing fixture record kind")),
                }
                if fixtures.len() == MAX_ROUTING_RECORDS {
                    return Err(failure("too many routing fixture records"));
                }
                let fixture: RoutingFixture =
                    serde_json::from_value(object.into()).map_err(failure)?;
                fixture.validate()?;
                fixtures.push(fixture);
            }
        }
        _ => return Err(failure("routing fixture must contain objects")),
    }
    Ok(())
}

fn parse_routing(text: &str) -> Result<Vec<RoutingFixture>> {
    let mut fixtures = Vec::new();
    if let Ok(value) = serde_json::from_str(text) {
        collect_routing(value, &mut fixtures)?;
    } else {
        for (line, text) in text
            .lines()
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
        {
            let value = serde_json::from_str(text)
                .map_err(|error| failure(format!("routing JSONL line {}: {error}", line + 1)))?;
            collect_routing(value, &mut fixtures)?;
        }
    }
    if fixtures.is_empty() {
        return Err(failure("routing fixture contains no routing records"));
    }
    Ok(fixtures)
}

fn load_routing(path: &Path) -> Result<Vec<RoutingFixture>> {
    let file = std::fs::File::open(path)
        .map_err(|error| failure(format!("routing fixture {}: {error}", path.display())))?;
    let mut bytes = Vec::new();
    file.take(MAX_FIXTURE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(failure)?;
    if bytes.len() as u64 > MAX_FIXTURE_BYTES {
        return Err(failure("routing fixture exceeds 64 MiB replay limit"));
    }
    parse_routing(std::str::from_utf8(&bytes).map_err(failure)?)
}

fn synthetic_routing() -> Vec<RoutingFixture> {
    [
        (
            "skewed_empty",
            11,
            2,
            17,
            (0..17).flat_map(|t| [0, 1 + t % 3]).collect(),
        ),
        ("tiny_empty", 17, 2, 3, (0..6).collect()),
        (
            "expert_round_tail",
            257,
            2,
            7,
            (0..7).flat_map(|t| [256, t]).collect(),
        ),
        ("dense_tail", 1, 1, 9, vec![0; 9]),
        (
            "group32_tail",
            5,
            2,
            65,
            (0..65).flat_map(|t| [0, 1 + t % 3]).collect(),
        ),
    ]
    .into_iter()
    .map(|(name, experts, used, tokens, ids)| RoutingFixture {
        name: name.into(),
        version: Some(1),
        layer: None,
        position: None,
        experts,
        used,
        tokens,
        ids,
        counts: None,
        assignment_ids: None,
    })
    .collect()
}

#[test]
fn routing_fixture_validation_and_capture_jsonl() -> Result<()> {
    let record = json!({
        "kind":"routing", "version":1, "layer":3, "position":64,
        "experts":4, "used":2, "tokens":2, "ids":[0,3,3,1],
        "counts":[1,1,0,2], "assignment_ids":[0,3,1,2], "logits":[0.0]
    });
    let text = format!(
        "{{\"kind\":\"metadata\"}}\n{record}\n\
        {{\"kind\":\"projection\",\"values\":[1.0]}}\n"
    );
    let fixtures = parse_routing(&text)?;
    assert_eq!(fixtures.len(), 1);
    assert_eq!(fixtures[0].validate()?, vec![1, 1, 0, 2]);
    assert_eq!(fixtures[0].layer, Some(3));
    assert_eq!(fixtures[0].position, Some(64));
    assert_eq!(
        parse_routing(&json!({"records":[record.clone()]}).to_string())?.len(),
        1
    );
    assert_eq!(
        parse_routing(&json!([record.clone()]).to_string())?.len(),
        1
    );
    for (field, bad) in [
        ("ids", json!([-1, 3, 3, 1])),
        ("ids", json!([4, 3, 3, 1])),
        ("ids", json!([0, 0, 3, 1])),
        ("ids", json!([0, 3, 1])),
        ("experts", json!(0)),
        ("used", json!(0)),
        ("used", json!(5)),
        ("tokens", json!(0)),
        ("tokens", json!(usize::MAX)),
        ("tokens", json!(MAX_ROUTING_ASSIGNMENTS)),
        ("experts", json!(MAX_ROUTING_EXPERTS + 1)),
        ("version", json!(2)),
        ("counts", json!([1, 1, 0, 1])),
        ("assignment_ids", json!([0, 1, 2, 3])),
    ] {
        let mut bad_record = record.clone();
        bad_record[field] = bad;
        assert!(parse_routing(&bad_record.to_string()).is_err(), "{field}");
    }
    assert!(parse_routing("").is_err());
    assert!(parse_routing("{\"kind\":\"metadata\"}").is_err());
    assert!(parse_routing(&format!("{record}\ninvalid")).is_err());
    assert!(parse_routing("{\"records\":{}}").is_err());
    assert!(
        parse_routing(&json!({"records":vec![record; MAX_ROUTING_RECORDS + 1]}).to_string())
            .is_err()
    );
    Ok(())
}

#[test]
fn routing_statistics_and_descriptor_padding() -> Result<()> {
    let fixture = synthetic_routing().remove(0);
    let counts = fixture.validate()?;
    assert_eq!(counts, vec![17, 6, 6, 5, 0, 0, 0, 0, 0, 0, 0]);
    for (tile, active, padding) in [(4, 11, 10), (8, 6, 14), (32, 4, 94)] {
        let stats = fixture.statistics(&counts, tile, 35, 32);
        assert_eq!(stats["empty_experts"], 7);
        assert_eq!(stats["count_p50"], 0);
        assert_eq!(stats["count_p95"], 17);
        assert_eq!(stats["active_descriptors"], active);
        assert_eq!(stats["assignment_padding"], padding);
        assert_eq!(stats["active_row_tile_work"], active * 2);
        assert_eq!(stats["useful_output_values"], 34 * 35);
        let descriptors = fixture.descriptors(tile);
        assert_eq!(descriptors.len(), fixture.slots(tile) * (tile + 1));
        let mut seen = vec![false; fixture.ids.len()];
        let mut active_count = 0;
        for chunk in descriptors.chunks_exact(tile + 1) {
            if chunk[0] == -1 {
                assert!(chunk.iter().all(|&x| x == -1));
                continue;
            }
            active_count += 1;
            let mut padded = false;
            for &assignment in &chunk[1..] {
                if assignment == -1 {
                    padded = true;
                } else {
                    assert!(!padded);
                    let a = assignment as usize;
                    assert_eq!(fixture.ids[a], chunk[0]);
                    assert!(!seen[a]);
                    seen[a] = true;
                }
            }
        }
        assert_eq!(active_count, active);
        assert!(seen.into_iter().all(|x| x));
    }
    for fixture in synthetic_routing() {
        let counts = fixture.validate()?;
        assert_eq!(counts.iter().sum::<usize>(), fixture.tokens * fixture.used);
        for tile in [4, 8, 32] {
            let descriptors = fixture.descriptors(tile);
            assert_eq!(descriptors.len(), fixture.slots(tile) * (tile + 1));
            let assignments: Vec<_> = descriptors
                .chunks_exact(tile + 1)
                .flat_map(|chunk| chunk[1..].iter().copied())
                .filter(|&assignment| assignment >= 0)
                .collect();
            let mut sorted = assignments;
            sorted.sort_unstable();
            assert_eq!(sorted, (0..fixture.ids.len() as i32).collect::<Vec<_>>());
        }
    }
    let dense = synthetic_routing().remove(3);
    assert_eq!(
        dense.descriptors(4),
        vec![0, 0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, -1, -1, -1]
    );
    let tail32 = synthetic_routing().remove(4);
    let counts = tail32.validate()?;
    assert_eq!(counts, vec![65, 22, 22, 21, 0]);
    let stats = tail32.statistics(&counts, 32, 35, 32);
    assert_eq!(stats["active_descriptors"], 6);
    assert_eq!(stats["assignment_padding"], 62);
    let descriptors = tail32.descriptors(32);
    assert_eq!(descriptors[2 * 33], 0);
    assert_eq!(descriptors[2 * 33 + 1], 128);
    assert!(descriptors[2 * 33 + 2..3 * 33].iter().all(|&a| a == -1));
    for tokens in [64, 128, 256, 512] {
        let fixture = RoutingFixture {
            name: "balanced".into(),
            version: None,
            layer: None,
            position: None,
            experts: 128,
            used: 8,
            tokens,
            ids: (0..tokens)
                .flat_map(|t| (0..8).map(move |s| ((t * 5 + s) % 128) as i32))
                .collect(),
            counts: None,
            assignment_ids: None,
        };
        let counts = fixture.validate()?;
        for tile in [4, 8, 32] {
            let active = counts.iter().map(|n| n.div_ceil(tile)).sum::<usize>();
            assert!(active <= fixture.slots(tile));
        }
    }
    Ok(())
}

#[test]
#[ignore = "synthetic M8 routing GPU correctness; run serially with --nocapture"]
fn grouped_routing_synthetic_matches_cpu_and_legacy() -> Result<()> {
    let harness = Harness::new()?;
    for fixture in synthetic_routing() {
        replay_routing(&harness, &fixture, 256, 35, &[0, 1, 6, 8, 12, 14], false)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires explicit M8_ROUTING_FIXTURE local JSON/JSONL; run serially with --nocapture"]
fn benchmark_grouped_routing_replay() -> Result<()> {
    let path = std::env::var_os("M8_ROUTING_FIXTURE")
        .ok_or_else(|| failure("set M8_ROUTING_FIXTURE to an explicit local routing fixture"))?;
    // Load and validate every record before creating a CUDA context.
    let fixtures = load_routing(Path::new(&path))?;
    let harness = Harness::new()?;
    for (index, mut fixture) in fixtures.into_iter().enumerate() {
        if fixture.name.is_empty() {
            fixture.name = format!("record_{index}");
        }
        replay_routing(&harness, &fixture, 256, 35, &[0, 1, 6, 8, 12, 14], false)?;
        for (width, rows) in [(2048, 768), (768, 2048)] {
            replay_routing(&harness, &fixture, width, rows, &[12, 14], true)?;
        }
    }
    Ok(())
}

fn replay_routing(
    harness: &Harness,
    fixture: &RoutingFixture,
    width: usize,
    rows: usize,
    kinds: &[u32],
    timed: bool,
) -> Result<()> {
    let counts = fixture.validate()?;
    let (ne, nu, nt) = (
        fixture.experts as i32,
        fixture.used as i32,
        fixture.tokens as i32,
    );
    let ids = harness.stream.clone_htod(&fixture.ids).map_err(failure)?;
    let mut descriptors = Vec::new();
    for tile in [4, 8, 32] {
        let slots = fixture.slots(tile);
        let length = slots * (tile + 1);
        let mut groups = harness
            .stream
            .clone_htod(&vec![1337i32; length + 16])
            .map_err(failure)?;
        let ns = slots as i32;
        macro_rules! group_replay {
            () => {{
                let mut view = groups.slice_mut(8..8 + length);
                launch!(
                    harness,
                    match tile {
                        4 => "group_experts",
                        8 => "group_experts_tiled",
                        32 => "group_experts_reg32",
                        _ => unreachable!(),
                    },
                    (1, 1, 1),
                    &ids,
                    &mut view,
                    &ne,
                    &nu,
                    &nt,
                    &ns
                );
            }};
        }
        let start = harness
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(failure)?;
        group_replay!();
        let end = harness
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(failure)?;
        end.synchronize().map_err(failure)?;
        let actual = harness.stream.clone_dtoh(&groups).map_err(failure)?;
        assert!(
            actual[..8]
                .iter()
                .chain(&actual[8 + length..])
                .all(|&v| v == 1337)
        );
        assert_eq!(
            &actual[8..8 + length],
            fixture.descriptors(tile),
            "routing descriptor mismatch: {}",
            fixture.name
        );
        println!(
            "GROUPED_ROUTING {}",
            json!({
                "name":fixture.name, "layer":fixture.layer, "position":fixture.position,
                "tokens":fixture.tokens, "experts":fixture.experts, "used":fixture.used,
                "width":width, "rows":rows, "statistics":fixture.statistics(&counts,tile,rows,1),
                "grouping_cuda_event_ms":start.elapsed_ms(&end).map_err(failure)?,
                "timing_scope":"single diagnostic grouping launch; excludes validation/readback",
            })
        );
        if timed {
            for _ in 0..3 {
                group_replay!();
            }
            for repetition in 0..7 {
                let start = harness
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                group_replay!();
                let end = harness
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                end.synchronize().map_err(failure)?;
                println!(
                    "GROUPED_ROUTING_BENCH {}",
                    json!({
                        "name":fixture.name, "layer":fixture.layer, "position":fixture.position,
                        "width":width, "rows":rows, "tokens":fixture.tokens,
                        "experts":fixture.experts, "used":fixture.used, "group_tile":tile,
                        "repetition":repetition, "iterations":1,
                        "grouping_cuda_event_ms":start.elapsed_ms(&end).map_err(failure)?,
                        "timing_scope":"grouping only; resident IDs/descriptors; no diagnostic copies",
                    })
                );
            }
        }
        descriptors.push(groups);
    }
    for &kind in kinds {
        let mut modes = vec![
            ("grouped_matmul", 1u32, 4usize, 256u32),
            ("grouped_matmul_tiled", 32, 8, 256),
            ("grouped_matmul_tf32x3", 128, 4, 256),
            ("grouped_matmul_tf32x3_tiled", 128, 8, 256),
        ];
        if matches!(kind, 12 | 14) {
            modes.push(("grouped_matmul_reg32x32", 32, 32, 128));
        }
        let host_tensor = (!timed).then(|| packed_fixture(kind, width, rows, fixture.experts));
        let tensor = DeviceTensor::upload(
            packed_fixture(kind, width, rows, fixture.experts),
            &harness.stream,
        )?;
        for batched in [0i32, 1] {
            let input_count = if batched == 1 {
                fixture.ids.len()
            } else {
                fixture.tokens
            };
            let host_input = inputs(width, input_count);
            let input = harness.stream.clone_htod(&host_input).map_err(failure)?;
            let length = fixture.ids.len() * rows;
            let mut reference = harness.stream.alloc_zeros::<f32>(length).map_err(failure)?;
            let legacy_start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            for token in 0..fixture.tokens {
                let stride = if batched == 1 { fixture.used } else { 1 };
                let x = input.slice(token * stride * width..(token + 1) * stride * width);
                let selected = ids.slice(token * fixture.used..(token + 1) * fixture.used);
                let mut y = reference
                    .slice_mut(token * fixture.used * rows..(token + 1) * fixture.used * rows);
                launch!(
                    harness,
                    "matvec",
                    (rows as u32, fixture.used as u32, 1),
                    &tensor.bytes,
                    &tensor.kind,
                    &x,
                    &mut y,
                    &tensor.width,
                    &tensor.rows,
                    &selected,
                    &1i32,
                    &batched
                );
            }
            let legacy_end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            legacy_end.synchronize().map_err(failure)?;
            let expected = harness.stream.clone_dtoh(&reference).map_err(failure)?;
            let cpu_start = Instant::now();
            // Full small-shape CPU oracle; representative timings use the same
            // routing plus legacy GPU oracle without billions of host dot products.
            if !timed {
                let mut decoded = vec![0.0f32; width];
                for (assignment, &expert) in fixture.ids.iter().enumerate() {
                    let x = if batched == 1 {
                        assignment
                    } else {
                        assignment / fixture.used
                    };
                    for row in 0..rows {
                        host_tensor
                            .as_ref()
                            .unwrap()
                            .row(expert as usize * rows + row, &mut decoded);
                        let cpu = decoded
                            .iter()
                            .zip(&host_input[x * width..(x + 1) * width])
                            .map(|(&w, &v)| f64::from(w) * f64::from(v))
                            .sum::<f64>() as f32;
                        let gpu = expected[assignment * rows + row];
                        assert!(
                            gpu.is_finite() && (gpu - cpu).abs() <= 0.0001 + 1e-5 * cpu.abs(),
                            "legacy/CPU kind={kind} assignment={assignment} row={row}: {gpu} != {cpu}"
                        );
                    }
                }
            }
            let cpu_reference_ms = (!timed).then(|| cpu_start.elapsed().as_secs_f64() * 1000.0);
            let mut output = harness
                .stream
                .clone_htod(&vec![1337.0f32; length + 16])
                .map_err(failure)?;
            macro_rules! project_replay {
                ($mode:expr) => {{
                    let (name, row_tile, tile, threads) = $mode;
                    let groups = &descriptors[match tile {
                        4 => 0,
                        8 => 1,
                        32 => 2,
                        _ => unreachable!(),
                    }];
                    let groups = groups.slice(8..groups.len() - 8);
                    let mut y = output.slice_mut(8..8 + length);
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (rows as u32).div_ceil(row_tile),
                            fixture.slots(tile) as u32,
                            1,
                        ),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut args = harness.stream.launch_builder(&harness.functions[name]);
                    args.arg(&tensor.bytes)
                        .arg(&tensor.kind)
                        .arg(&input)
                        .arg(&mut y)
                        .arg(&tensor.width)
                        .arg(&tensor.rows)
                        .arg(&groups)
                        .arg(&nu)
                        .arg(&batched);
                    // The register-tiled prototype uses 128 threads; legacy
                    // grouped reductions and tensor-core modes require 256.
                    unsafe { args.launch(cfg) }.map_err(failure)?;
                }};
            }
            for &mode in &modes {
                output = harness
                    .stream
                    .clone_htod(&vec![1337.0f32; length + 16])
                    .map_err(failure)?;
                project_replay!(mode);
                let actual = harness.stream.clone_dtoh(&output).map_err(failure)?;
                assert!(
                    actual[..8]
                        .iter()
                        .chain(&actual[8 + length..])
                        .all(|&v| v == 1337.0)
                );
                for (i, (&a, &b)) in actual[8..8 + length].iter().zip(&expected).enumerate() {
                    assert!(
                        a.is_finite()
                            && if mode.0 == "grouped_matmul" {
                                a.to_bits() == b.to_bits()
                            } else {
                                (a - b).abs() <= 0.005 + 0.003 * b.abs()
                            },
                        "{} kind={kind} routing={} batched={batched} output={i}: {a} != {b}",
                        mode.0,
                        fixture.name
                    );
                }
            }
            println!(
                "GROUPED_REPLAY_CHECK {}",
                json!({
                    "name":fixture.name, "layer":fixture.layer, "position":fixture.position,
                    "kind":kind, "width":width, "rows":rows, "tokens":fixture.tokens,
                    "experts":fixture.experts, "used":fixture.used, "batched":batched,
                    "cpu_reference_ms":cpu_reference_ms,
                    "legacy_projection_cuda_event_ms":legacy_start.elapsed_ms(&legacy_end).map_err(failure)?,
                    "guards_ok":true, "legacy_comparison_ok":true, "cpu_comparison":!timed,
                    "checked_modes":modes.iter().map(|mode| mode.0).collect::<Vec<_>>(),
                    "operands":"deterministic packed_fixture weights and inputs; captured routing only",
                })
            );
            if !timed {
                continue;
            }
            for _ in 0..3 {
                for &mode in &modes {
                    project_replay!(mode);
                }
            }
            for repetition in 0..7 {
                for index in 0..modes.len() {
                    let mode = modes[(index + repetition) % modes.len()];
                    let start = harness
                        .stream
                        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .map_err(failure)?;
                    project_replay!(mode);
                    let end = harness
                        .stream
                        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .map_err(failure)?;
                    end.synchronize().map_err(failure)?;
                    println!(
                        "GROUPED_REPLAY_BENCH {}",
                        json!({
                            "name":fixture.name, "layer":fixture.layer, "position":fixture.position,
                            "kind":kind, "width":width, "rows":rows, "tokens":fixture.tokens,
                            "experts":fixture.experts, "used":fixture.used, "batched":batched,
                            "mode":mode.0, "row_tile":mode.1, "group_tile":mode.2,
                            "threads":mode.3,
                            "repetition":repetition, "iterations":1,
                            "projection_cuda_event_ms":start.elapsed_ms(&end).map_err(failure)?,
                            "statistics":fixture.statistics(&counts, mode.2, rows, mode.1 as usize),
                            "timing_scope":"projection only; resident operands/descriptors; no diagnostic copies",
                        })
                    );
                }
            }
        }
    }
    Ok(())
}
