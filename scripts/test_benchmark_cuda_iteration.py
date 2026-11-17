import unittest
import signal
import subprocess
from unittest.mock import patch

from benchmark_cuda_iteration import capture_process, distribution, summarize


class CudaIterationTests(unittest.TestCase):
    def test_timeout_cannot_wait_forever_on_a_kernel_blocked_owned_child(self):
        with patch("benchmark_cuda_iteration.subprocess.Popen") as popen, \
                patch("benchmark_cuda_iteration.os.killpg") as killpg:
            process = popen.return_value
            process.pid = 12345
            process.poll.return_value = None
            process.communicate.side_effect = [
                subprocess.TimeoutExpired(["owned"], 1, output=b"partial", stderr=b"failure")
            ] * 3
            result = capture_process(["owned"], {}, timeout=1, cleanup_timeout=.01)
            self.assertTrue(result["timed_out"])
            self.assertTrue(result["cleanup_incomplete"])
            self.assertEqual(result["stdout"], "partial")
            self.assertEqual(result["stderr"], "failure")
            self.assertEqual([call.args for call in killpg.call_args_list],
                             [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
            process.stdout.close.assert_called_once()
            process.stderr.close.assert_called_once()

    def test_descriptive_percentiles_validate_and_do_not_invent_tail_samples(self):
        self.assertEqual(distribution([4, 1, 2, 3])["median"], 2.5)
        self.assertEqual(distribution([7])["p99_descriptive"], 7)
        self.assertAlmostEqual(distribution([1, 2, 3])["p95_descriptive"], 2.9)
        for values in [[], [-1], [float("inf")], [float("nan")]]:
            with self.assertRaises(ValueError):
                distribution(values)

    def test_phase_summary_distinguishes_request_and_mean_token_latency(self):
        record = {"prompt_tokens": 10, "load_ns": 2000000, "memory": {},
                  "samples": [{"prefill_ns": 1000000, "ttft_ns": 1100000,
                               "decode_ns": 2000000, "decode_steps": 2}]}
        result = summarize(record)
        self.assertEqual(result["synchronized_request_ms"]["median"], 3.1)
        self.assertEqual(result["mean_inter_token_ms_per_request"]["median"], 1)
        self.assertEqual(result["decode_tokens_per_second"]["median"], 1000)
        record["samples"][0].update(decode_ns=0, decode_steps=0)
        self.assertIsNone(summarize(record)["decode_tokens_per_second"])
        self.assertIsNone(summarize(record)["mean_inter_token_ms_per_request"])


if __name__ == "__main__":
    unittest.main()
