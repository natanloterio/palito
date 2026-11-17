import copy
import unittest

from benchmark import summarize, validate_records


def records():
    sample = {
        "generated_tokens": [4, 5], "decode_steps": 1,
        "prefill_ns": 100, "ttft_ns": 120, "decode_ns": 50,
    }
    return [
        {"schema_version": 1, "runner": runner, "prompt_tokens": 4, "prompt_token_ids": [1, 2, 3, 4],
         "samples": [copy.deepcopy(sample)]}
        for runner in ("cpp-reference", "rust")
    ]


class BenchmarkTests(unittest.TestCase):
    def test_detects_mismatch_and_incomplete_generation(self):
        for change in (
            lambda record: record["samples"][0]["generated_tokens"].__setitem__(1, 6),
            lambda record: record["samples"].clear(),
            lambda record: record.__setitem__("prompt_tokens", 5),
            lambda record: record["prompt_token_ids"].__setitem__(0, 7),
            lambda record: record["samples"][0].__setitem__("decode_steps", 2),
        ):
            data = records()
            change(data[1])
            with self.assertRaises(ValueError):
                validate_records(data, 1, 2)

    def test_detects_zero_decode_and_invalid_ttft(self):
        for field, value in (("decode_ns", 0), ("ttft_ns", 99), ("prefill_ns", -1)):
            data = records()
            data[1]["samples"][0][field] = value
            with self.assertRaises(ValueError):
                validate_records(data, 1, 2)

    def test_ratio_direction_and_decode_denominator(self):
        data = records()
        data[1]["samples"][0]["decode_ns"] = 100
        validate_records(data, 1, 2)
        summary = summarize(data)
        self.assertEqual(summary["rust_over_cpp_latency_ratio"]["decode"], 2)
        self.assertEqual(summary["rust"]["decode_median_tokens_per_second"], 1e7)

    def test_single_generated_token_has_no_decode_phase(self):
        data = records()
        for record in data:
            record["samples"][0].update(generated_tokens=[4], decode_steps=0, decode_ns=0)
        validate_records(data, 1, 1)
        self.assertNotIn("decode", summarize(data)["rust_over_cpp_latency_ratio"])


if __name__ == "__main__":
    unittest.main()
