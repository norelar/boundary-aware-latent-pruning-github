from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from latent_pruning.config import PruningConfig
from latent_pruning.selection import allocate_exact_counts, select_features, target_count


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = PruningConfig(
            n_clusters=5,
            n_init=10,
            nu=0.05,
            boundary_quantile=0.20,
            n_margin_strata=5,
            scaler="standard",
            pca_components=8,
        )

    def test_exact_allocation_respects_capacity(self):
        allocated = allocate_exact_counts([11, 7, 2], 9)
        self.assertEqual(int(allocated.sum()), 9)
        self.assertTrue(np.all(allocated <= np.asarray([11, 7, 2])))
        self.assertTrue(np.all(allocated >= 1))

    def test_proposed_is_deterministic_and_exact(self):
        features, _ = make_blobs(n_samples=101, centers=5, n_features=12, random_state=4)
        first = select_features(features, 0.5, self.config, seed=42, mode="proposed")
        second = select_features(features, 0.5, self.config, seed=42, mode="proposed")
        expected = target_count(101, 0.5)
        self.assertEqual(len(first.selected_indices), expected)
        np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
        self.assertEqual(sum(metric.selected_size for metric in first.cluster_metrics), expected)
        self.assertTrue(first.boundary_mask.any())
        self.assertTrue(first.support_mask.any())

    def test_random_and_clustered_random_are_exact(self):
        features, _ = make_blobs(n_samples=43, centers=4, n_features=5, random_state=9)
        for mode in ("random", "clustered_random"):
            with self.subTest(mode=mode):
                result = select_features(features, 0.25, self.config, seed=7, mode=mode)
                self.assertEqual(len(result.selected_indices), target_count(43, 0.25))

    def test_small_class_fallback(self):
        features = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        result = select_features(features, 0.5, self.config, seed=1, mode="proposed")
        self.assertEqual(len(result.selected_indices), 1)
        self.assertFalse(result.cluster_metrics[0].used_ocsvm)


if __name__ == "__main__":
    unittest.main()
