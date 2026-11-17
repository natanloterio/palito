from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from benchmark_moe import server, summarize, validate_llama_sample


class MoeBenchmarkTests(unittest.TestCase):
    def test_failed_parent_still_terminates_its_owned_worker_group(self):
        with patch("benchmark_moe.subprocess.Popen") as popen, \
                patch("benchmark_moe.os.killpg") as killpg, \
                tempfile.TemporaryDirectory() as directory:
            process = popen.return_value
            process.pid = 123
            process.poll.return_value = 1
            with self.assertRaises(RuntimeError):
                with server(["unused"], "http://127.0.0.1:1", Path(directory) / "server.log", {}):
                    self.fail("An exited parent must not become ready")
            killpg.assert_called_once_with(123, signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=20)

    def test_rejects_cached_truncated_and_incomplete_workloads(self):
        response = {
            "tokens": [4, 5], "stop_type": "limit", "truncated": False,
            "tokens_evaluated": 3, "tokens_cached": 4,
            "timings": {"prompt_n": 3, "cache_n": 0},
        }
        validate_llama_sample(response, [1, 2, 3], 2)
        for field, value in (("tokens", [4]), ("stop_type", "eos"),
                             ("truncated", True), ("tokens_evaluated", 2),
                             ("timings", {"prompt_n": 3, "cache_n": 1}),
                             ("timings", {"prompt_n": 2, "cache_n": 0})):
            with self.assertRaises(ValueError):
                validate_llama_sample(dict(response, **{field: value}), [1, 2, 3], 2)

    def test_reports_request_rate_not_decode_rate(self):
        result = summarize([{"elapsed_ns": 1_000_000_000}, {"elapsed_ns": 3_000_000_000}], 32)
        self.assertEqual(result["median_ms"], 2000)
        self.assertEqual(result["generated_tokens_per_second_including_prompt"], (32 + 32 / 3) / 2)
        with self.assertRaises(ValueError):
            summarize([{"elapsed_ns": 0}], 32)


if __name__ == "__main__":
    unittest.main()
