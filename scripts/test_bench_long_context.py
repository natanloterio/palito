import unittest

from bench_long_context import (
    flatten,
    medians,
    palito_input,
    parse_args,
    parse_palito_stats,
    parse_sse_metrics,
    table,
)


class BenchLongContextTests(unittest.TestCase):
    def test_flatten_removes_newlines(self):
        self.assertEqual(flatten("a\n\n b\tc \n"), "a b c")

    def test_palito_input_resets_between_requests(self):
        self.assertEqual(palito_input("hi", 2), "hi\n/reset\nhi\n/reset\n")

    def test_parses_chat_stats_lines(self):
        stderr = "(conversation cleared)\n[256 tokens · first token 6390 ms · 103.2 tok/s]\nnoise\n" \
                 "[12 tokens · first token 41 ms · 335.0 tok/s]\n"
        self.assertEqual(parse_palito_stats(stderr), [
            {"generated": 256, "ttft_ms": 6390.0, "decode_tps": 103.2},
            {"generated": 12, "ttft_ms": 41.0, "decode_tps": 335.0},
        ])

    def test_missing_stats_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_palito_stats("palito: out of memory\n")

    def test_medians_and_table(self):
        samples = [{"generated": 256, "ttft_ms": t, "decode_tps": r} for t, r in ((10, 100), (30, 90), (20, 300))]
        self.assertEqual(medians(samples), {"ttft_ms": 20, "decode_tps": 100, "generated": 256})
        text = table({"A": {500: samples}, "B": {}}, {500: 512})
        self.assertIn("| 512 | 20 | 100.0 | – | – |", text)

    def test_defaults_cover_acceptance_sizes(self):
        args = parse_args([])
        self.assertEqual(args.sizes, [500, 2900, 10000, 40000])
        self.assertEqual(args.runs, 3)
        self.assertEqual(args.palito_mode, "serve")
        self.assertIn("fa-on-q8kv", args.llama_configs)

    def test_parse_sse_metrics(self):
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
            'data: {"choices":[{"delta":{"content":" there"}}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":512,"completion_tokens":101}}\n',
            'data: [DONE]\n',
        ]
        timestamps = [10.01, 10.10, 10.60, 11.10, 11.10]
        got = parse_sse_metrics(lines, started=10.00, timestamps=timestamps)
        self.assertEqual(got, {
            "generated": 101,
            "ttft_ms": 100.0,
            "decode_tps": 100.0,
            "prompt_tokens": 512,
        })


if __name__ == "__main__":
    unittest.main()
