import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cs_baselines import omp_dft2_completion, dft2_synthesis_dictionary
from plot_results import load_series


class ResultScriptsTest(unittest.TestCase):
    def test_omp_recovers_sparse_channel(self):
        dictionary = dft2_synthesis_dictionary((4, 4))
        channel = dictionary[:, 3].reshape(4, 4)
        mask = np.ones((4, 4), dtype=bool)
        mask[0] = False
        estimate = omp_dft2_completion(channel * mask, mask, sparsity=1)
        np.testing.assert_allclose(estimate, channel, atol=1e-6)

    def test_csv_and_structured_npy_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [dict(mask_ratio=0.2, avg_nmse=0.01), dict(mask_ratio=0.4, avg_nmse=0.1)]
            with (path / 'result.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.save(path / 'result.npy', np.array([(0.2, 0.01), (0.4, 0.1)],
                    dtype=[('mask_ratio', 'f8'), ('avg_nmse', 'f8')]))
            first = load_series(path / 'result.csv')
            second = load_series(path / 'result.npy')
            np.testing.assert_allclose(first, second)
            np.testing.assert_allclose(first[1], [-20, -10])

    def test_sample_npy_averages_before_db_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.npy'
            np.save(path, [[0.01, 0.09], [0.1, 0.3]])
            _, values = load_series(path, mask_ratios=[0.2, 0.4])
            np.testing.assert_allclose(values, 10*np.log10([0.05, 0.2]))

    def test_numeric_npy_requires_explicit_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'values.npy'
            np.save(path, [0.01, 0.1])
            with self.assertRaises(ValueError):
                load_series(path)


if __name__ == '__main__':
    unittest.main()
