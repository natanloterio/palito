"""Run: PALITO_TEST_MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf python3 tests/test_palito.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import palito

MODEL = os.environ.get("PALITO_TEST_MODEL")
HELLO = [{"role": "user", "content": "Say hello."}]


@unittest.skipUnless(MODEL, "set PALITO_TEST_MODEL to a small GGUF model")
class PalitoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = palito.Palito(MODEL, device=palito.DEVICE_CPU, context_tokens=512)

    @classmethod
    def tearDownClass(cls):
        cls.llm.close()

    def test_streamed_text_equals_the_reply(self):
        pieces = []
        reply = self.llm.chat(HELLO, max_tokens=24, on_text=pieces.append)
        self.assertEqual("".join(pieces), reply.text)
        self.assertTrue(0 < reply.generated_tokens <= 24)
        self.assertEqual("".join(self.llm.stream_chat(HELLO, max_tokens=24)), reply.text)

    def test_stop_sampling_and_cancel(self):
        cut = self.llm.complete("1, 2, 3, 4,", max_tokens=24, stop=[" 7"])
        self.assertEqual(cut.finish_reason, "stop")
        self.assertNotIn(" 7", cut.text)
        a = self.llm.chat(HELLO, max_tokens=16, temperature=0.9, seed=4)
        b = self.llm.chat(HELLO, max_tokens=16, temperature=0.9, seed=4)
        self.assertEqual(a.text, b.text)
        cancelled = self.llm.complete("Once upon a time", on_text=lambda _: False)
        self.assertEqual(cancelled.finish_reason, "cancelled")
        stream = self.llm.stream_complete("Once upon a time", max_tokens=200)
        next(stream)
        stream.close()  # stops generation; the engine is usable again
        self.assertEqual(self.llm.complete("Hi", max_tokens=2).generated_tokens, 2)

    def test_errors(self):
        with self.assertRaises(palito.PalitoError):
            palito.Palito("/nonexistent.gguf")
        with self.assertRaises(palito.PalitoError):
            self.llm.chat(HELLO, temperature=-1)
        with self.assertRaises(ZeroDivisionError):
            self.llm.complete("Hi", on_text=lambda _: 1 / 0)


if __name__ == "__main__":
    unittest.main()
