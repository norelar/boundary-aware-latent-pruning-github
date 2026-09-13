from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latent_pruning.analysis.legacy_logs import parse_performance_log, parse_pruning_log


PRUNING_LOG = """
=== Pruning run started ===
Seed: 42
Pruning classes: 100%|████| 10/10 [00:01, Normal=10 / 10] features_vectors shape is: (10, 32)
___________________________________pruning ratio is 50
Cluster 0 - Sélection initiale:
Cluster size: 10
Selected size: 5
Border points: 2 (20.00%)
--- Entire Cluster ---
Mean |distance|: 0.100000
Std  |distance|: 0.020000
--- Selected Samples ---
Mean |distance|: 0.050000
Std  |distance|: 0.010000
Cluster Margin Compactness (CMC): 0.5000
Boundary Focus Score (BFS): 0.5000
Stratification Score (CV): 0.2000
  ✓ 5 images copiées vers /tmp/50%/Normal/cluster_0
"""


class LegacyLogTests(unittest.TestCase):
    def test_pruning_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "COVID_prune_log_seed_42.txt"
            path.write_text(PRUNING_LOG, encoding="utf-8")
            frame = parse_pruning_log(path)
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            self.assertEqual(row["dataset"], "COVID19")
            self.assertEqual(row["class"], "Normal")
            self.assertEqual(row["retention_percent"], 50)
            self.assertAlmostEqual(row["bfs"], 0.5)
            self.assertAlmostEqual(row["completion_fraction"], 0.6)

    def test_performance_parser_takes_last_test_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resnet50_train_20260312_052521_COVID_50.log"
            path.write_text(
                "TEST | Loss: 1.0 | Acc: 0.80 | Precision: 0.81 | Recall: 0.80 | F1-score: 0.79\n"
                "TEST | Loss: 0.5 | Acc: 0.90 | Precision: 0.91 | Recall: 0.90 | F1-score: 0.89\n",
                encoding="utf-8",
            )
            result = parse_performance_log(path)
            self.assertEqual(result["dataset"], "COVID19")
            self.assertEqual(result["retention_percent"], 50)
            self.assertAlmostEqual(result["accuracy"], 0.90)
            self.assertAlmostEqual(result["f1"], 0.89)


if __name__ == "__main__":
    unittest.main()
