from __future__ import annotations

import unittest
from pathlib import Path

from latent_pruning.config import (
    DatasetConfig,
    ExperimentConfig,
    PruningConfig,
    TrainingConfig,
    VQVAEConfig,
)
from latent_pruning.data import ImageSample
from latent_pruning.pruning import prune_samples


class PruningTests(unittest.TestCase):
    def test_random_policy_preserves_unlisted_class(self):
        samples = [ImageSample(Path(f"/tmp/major_{index}.png"), "major", 0) for index in range(10)]
        samples += [ImageSample(Path(f"/tmp/minor_{index}.png"), "minor", 1) for index in range(3)]
        config = ExperimentConfig(
            project_name="test",
            output_dir=Path("outputs"),
            dataset=DatasetConfig(
                name="DEMO",
                train_dir=Path("train"),
                test_dir=Path("test"),
                pruned_classes=["major"],
                retention={"major": 0.4},
            ),
            vqvae=VQVAEConfig(checkpoint=Path("unused.pth")),
            pruning=PruningConfig(n_clusters=3),
            training=TrainingConfig(),
        )
        config.validate()
        artifact = prune_samples(samples, config, seed=42, selector="random", feature_mode="none")
        self.assertEqual(len(artifact.selected_samples), 7)
        selected_counts = dict(
            zip(artifact.class_summary["class"], artifact.class_summary["selected_count"])
        )
        self.assertEqual(selected_counts, {"major": 4, "minor": 3})
        minor = artifact.selection_manifest[artifact.selection_manifest["class"] == "minor"]
        self.assertTrue(minor["selected"].all())
        self.assertTrue((minor["selection_reason"] == "preserved_full_class").all())


if __name__ == "__main__":
    unittest.main()
