import unittest
from copy import deepcopy

from benchmark_rust import summarize, validate
from benchmark_cuda_paired import validate_controls
from validate_rust_logits import compare_logits


class IndependentBenchmarkTests(unittest.TestCase):
    def test_paired_cuda_baseline_profile_is_explicit_and_stable(self):
        labels = ["a1", "b1", "b2", "a2"]
        report = {"controls": {"batch": 512, "chunk": 512, "matmul": "f32-tiled",
                               "decode_graph": True},
                  "results": {"case": {"memory": {"numerical_reservation_bytes": 100}}}}
        reports = {label: deepcopy(report) for label in labels}
        provenance = {label: {"models": {}, "affinity": [0]} for label in labels}
        options = dict(candidate_decode_graph=True, candidate_batch=512, candidate_chunk=512)
        with self.assertRaises(ValueError):
            validate_controls(reports, provenance, "f32-tiled", **options)
        options.update(baseline_matmul="f32-tiled", baseline_decode_graph=True)
        validate_controls(reports, provenance, "f32-tiled", **options)
        for field, value in [("matmul", "f32"), ("decode_graph", False), ("batch", 128)]:
            changed = deepcopy(reports)
            changed["a2"]["controls"][field] = value
            with self.assertRaises(ValueError):
                validate_controls(changed, provenance, "f32-tiled", **options)

    def test_graph_and_batch_changes_require_explicit_bounded_paired_controls(self):
        labels = ["a1", "b1", "b2", "a2"]
        report = {"controls": {"batch": 128, "chunk": 128, "matmul": "f32"},
                  "results": {"case": {"memory": {"numerical_reservation_bytes": 100}}}}
        reports = {label: deepcopy(report) for label in labels}
        provenance = {label: {"models": {}, "affinity": [0]} for label in labels}
        for label in ["b1", "b2"]:
            reports[label]["controls"].update(decode_graph=True, batch=256, chunk=256)
            reports[label]["results"]["case"]["memory"]["numerical_reservation_bytes"] = 110
        with self.assertRaises(ValueError):
            validate_controls(reports, provenance, "f32")
        options = dict(candidate_decode_graph=True, candidate_batch=256, candidate_chunk=256)
        with self.assertRaises(ValueError):
            validate_controls(reports, provenance, "f32", **options)
        validate_controls(reports, provenance, "f32", allowed_reservation_increase=10, **options)
        reports["b2"]["results"]["case"]["memory"]["numerical_reservation_bytes"] = 111
        with self.assertRaises(ValueError):
            validate_controls(reports, provenance, "f32", allowed_reservation_increase=10, **options)

    def test_paired_cuda_comparisons_reject_control_and_memory_drift(self):
        labels = ["a1", "b1", "b2", "a2"]
        report = {"controls": {"warmups": 3, "repetitions": 7, "matmul": "f32"},
                  "results": {"case": {"memory": {"numerical_reservation_bytes": 100}}}}
        reports = {label: deepcopy(report) for label in labels}
        provenance = {label: {"models": {"model": "same-hash"}, "affinity": [0, 1]}
                      for label in labels}
        validate_controls(reports, provenance, "f32")
        for field, value in [("warmups", 1), ("repetitions", 3), ("matmul", "tf32x3")]:
            changed = deepcopy(reports)
            changed["b1"]["controls"][field] = value
            with self.assertRaises(ValueError):
                validate_controls(changed, provenance, "f32")
        for field, value in [("models", {"model": "different-hash"}), ("affinity", [2, 3])]:
            changed = deepcopy(provenance)
            changed["b2"][field] = value
            with self.assertRaises(ValueError):
                validate_controls(reports, changed, "f32")
        reports["b1"]["results"]["case"]["memory"]["numerical_reservation_bytes"] = 101
        with self.assertRaises(ValueError):
            validate_controls(reports, provenance, "f32")

    def test_prefix_reuse_requires_opt_in_and_cannot_exceed_prompt(self):
        record = {"prompt_tokens": 1, "prompt_token_ids": [1], "samples": [{
            "generated_tokens": [2], "decode_steps": 0, "prefill_ns": 10,
            "ttft_ns": 12, "decode_ns": 0, "cached_prompt_tokens": 1,
        }]}
        with self.assertRaises(ValueError):
            validate(record, [1], 1, 1)
        validate(record, [1], 1, 1, allow_cached=True)
        for invalid in [-1, 2, True]:
            record["samples"][0]["cached_prompt_tokens"] = invalid
            with self.assertRaises(ValueError):
                validate(record, [1], 1, 1, allow_cached=True)

    def test_logit_comparison_detects_misaligned_vocab_and_nonfinite_results(self):
        result = compare_logits([1.0, 3.0, -2.0], [1.0, 3.001, -2.0])
        self.assertGreater(result["cosine_similarity"], 0.9999)
        self.assertLess(result["max_absolute_error"], 0.002)
        changed = compare_logits([3.0, 1.0, -2.0], [1.0, 3.0, -2.0])
        self.assertNotEqual(changed["rust_top_token"], changed["reference_top_token"])
        for actual, expected in [([1.0], [1.0, 2.0]), ([float("nan")], [1.0]), ([0.0], [0.0])]:
            with self.assertRaises(ValueError):
                compare_logits(actual, expected)

    def test_requires_complete_repeatable_input_and_generation(self):
        sample = {"generated_tokens": [4, 5], "decode_steps": 1,
                  "prefill_ns": 10, "ttft_ns": 11, "decode_ns": 20}
        record = {"prompt_tokens": 2, "prompt_token_ids": [1, 2],
                  "samples": [sample, dict(sample)]}
        validate(record, [1, 2], 2, 2)
        with self.assertRaises(ValueError):
            validate(record, [1, 3], 2, 2)
        record["samples"][1]["generated_tokens"] = [4, 6]
        with self.assertRaises(ValueError):
            validate(record, [1, 2], 2, 2)

    def test_single_generated_token_has_no_decode_rate(self):
        record = {"prompt_tokens": 1, "prompt_token_ids": [1], "samples": [{
            "generated_tokens": [2], "decode_steps": 0, "prefill_ns": 10,
            "ttft_ns": 12, "decode_ns": 0,
        }]}
        validate(record, [1], 1, 1)
        self.assertIsNone(summarize(record)["decode_tokens_per_second"])
        record["samples"][0]["decode_ns"] = 1
        with self.assertRaises(ValueError):
            validate(record, [1], 1, 1)
