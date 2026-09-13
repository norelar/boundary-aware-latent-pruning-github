from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from latent_pruning.data import scan_image_folder, stratified_split


class DataTests(unittest.TestCase):
    def test_recursive_scan_and_stratified_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, color in (("a", "red"), ("b", "blue")):
                nested = root / label / "cluster_0"
                nested.mkdir(parents=True)
                for index in range(6):
                    Image.new("RGB", (4, 4), color=color).save(nested / f"{index}.png")
            dataset = scan_image_folder(root, ["png"])
            self.assertEqual(dataset.classes, ("a", "b"))
            self.assertEqual(dataset.counts(), {"a": 6, "b": 6})
            train, validation = stratified_split(dataset.samples, 1 / 3, seed=42)
            self.assertEqual(len(train), 8)
            self.assertEqual(len(validation), 4)
            self.assertFalse({sample.path for sample in train} & {sample.path for sample in validation})
            self.assertEqual({sample.label for sample in validation}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
