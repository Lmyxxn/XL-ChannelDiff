import unittest
import numpy as np
from evaluate_baseline import reference_completions


class ReferenceCompletionTests(unittest.TestCase):
    def test_observations_are_preserved(self):
        channels = np.full((2, 4, 4), 2+3j, dtype=np.complex64)
        masks = np.zeros(channels.shape, dtype=bool)
        masks[:, :, ::2] = True
        full, partial, random = reference_completions(channels, masks, 0.1, np.random.RandomState(42))
        np.testing.assert_array_equal(partial[masks], full[masks])
        np.testing.assert_array_equal(random[masks], full[masks])
        np.testing.assert_array_equal(partial[~masks], 0)

    def test_random_is_reproducible(self):
        channels = np.ones((2, 4, 4), dtype=np.complex64)
        masks = np.zeros(channels.shape, dtype=bool)
        a = reference_completions(channels, masks, 0.1, np.random.RandomState(42))[2]
        b = reference_completions(channels, masks, 0.1, np.random.RandomState(42))[2]
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
