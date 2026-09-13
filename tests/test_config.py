from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latent_pruning.config import ConfigurationError, load_config


VALID_CONFIG = """
project_name: test
output_dir: outputs/test
dataset:
  name: DEMO
  train_dir: data/train
  test_dir: data/test
  pruned_classes: [major]
  retention: {major: 0.5}
  retention_sweep: [0.25, 0.5]
vqvae:
  checkpoint: checkpoints/demo.pth
  embedding_dim: 16
  num_embeddings: 32
  hidden_dims: [8, 16]
pruning:
  n_clusters: 3
training:
  seeds: [7, 42]
"""


class ConfigTests(unittest.TestCase):
    def test_load_and_override_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(VALID_CONFIG, encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.dataset.name, "DEMO")
            self.assertEqual(config.dataset.retention["major"], 0.5)
            self.assertEqual(config.training.seeds, [7, 42])
            overridden = config.with_uniform_retention(0.75)
            self.assertEqual(overridden.dataset.retention, {"major": 0.75})
            self.assertEqual(config.dataset.retention, {"major": 0.5})

    def test_invalid_retention_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(VALID_CONFIG.replace("major: 0.5", "major: 1.5"), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
