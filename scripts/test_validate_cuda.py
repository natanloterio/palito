import unittest

from validate_cuda import compare_logits


class CudaValidationTests(unittest.TestCase):
    def test_requires_full_finite_vectors(self):
        for expected, actual in [([], []), ([1.0], [1.0, 2.0]),
                                 ([float("nan")], [1.0]), ([1.0], [float("inf")])]:
            with self.assertRaises(ValueError):
                compare_logits(expected, actual)

    def test_enforces_absolute_error_cosine_and_greedy_parity_independently(self):
        self.assertAlmostEqual(compare_logits([1.0, 0.0], [1.009, 0.0])["maximum_absolute_error"], 0.009)
        for expected, actual in [
            ([1.0, 0.0], [1.011, 0.0]),
            ([0.001, 0.0], [0.001, 0.001]),
            ([1.0, 1.00001], [1.00001, 1.0]),
        ]:
            with self.assertRaises(ValueError):
                compare_logits(expected, actual)


if __name__ == "__main__":
    unittest.main()
