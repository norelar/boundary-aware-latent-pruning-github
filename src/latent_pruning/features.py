"""VQ-VAE and raw-pixel feature extraction with deterministic caching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from latent_pruning.config import PruningConfig, VQVAEConfig
from latent_pruning.data import ImageSample
from latent_pruning.utils import stable_hash

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class FeatureSet:
    features: np.ndarray
    samples: list[ImageSample]
    representation: str

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"Feature matrix must be 2-D, got {self.features.shape}")
        if len(self.samples) != self.features.shape[0]:
            raise ValueError("Feature rows and sample paths are misaligned")
        if not np.isfinite(self.features).all():
            raise ValueError("Feature matrix contains NaN or infinite values")


def _load_rgb_array(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def extract_raw_pixel_features(
    samples: Sequence[ImageSample], image_size: int = 64
) -> FeatureSet:
    features = [
        _load_rgb_array(sample.path, image_size).reshape(-1)
        for sample in samples
    ]
    output = FeatureSet(np.asarray(features, dtype=np.float32), list(samples), "raw_pixels")
    output.validate()
    return output


def _feature_cache_key(samples: Sequence[ImageSample], config: VQVAEConfig) -> str:
    checkpoint = Path(config.checkpoint)
    checkpoint_metadata = None
    if checkpoint.exists():
        stat = checkpoint.stat()
        checkpoint_metadata = [str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns]
    payload = {
        "samples": [
            [str(sample.path), sample.path.stat().st_size, sample.path.stat().st_mtime_ns]
            for sample in samples
        ],
        "checkpoint": checkpoint_metadata,
        "image_size": config.image_size,
        "embedding_dim": config.embedding_dim,
        "num_embeddings": config.num_embeddings,
        "hidden_dims": config.hidden_dims,
        "representation": config.representation,
    }
    return stable_hash(payload, length=24)


def _read_cache(cache_path: Path, samples: Sequence[ImageSample], representation: str) -> FeatureSet | None:
    if not cache_path.is_file():
        return None
    with np.load(cache_path, allow_pickle=False) as archive:
        features = archive["features"]
        paths = archive["paths"].tolist()
    if paths != [str(sample.path) for sample in samples]:
        return None
    output = FeatureSet(features.astype(np.float32, copy=False), list(samples), representation)
    output.validate()
    return output


def extract_vqvae_features(
    samples: Sequence[ImageSample],
    config: VQVAEConfig,
    device: str,
    cache_dir: str | Path | None = None,
    model=None,
) -> FeatureSet:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError("Install experiment dependencies with: pip install -e '.[experiment]'") from exc

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"vqvae_{_feature_cache_key(samples, config)}.npz"
        cached = _read_cache(cache_path, samples, config.representation)
        if cached is not None:
            return cached

    class PathDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            array = _load_rgb_array(samples[index].path, config.image_size)
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    if model is None:
        from latent_pruning.models.vqvae import load_vqvae_checkpoint

        model = load_vqvae_checkpoint(config, device=device)
    model.eval()
    loader = DataLoader(
        PathDataset(),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            representation = model.representation(batch, kind=config.representation)
            rows.append(representation.detach().float().cpu().numpy())
    features = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    output = FeatureSet(features, list(samples), config.representation)
    output.validate()
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            features=output.features,
            paths=np.asarray([str(sample.path) for sample in samples]),
            metadata=np.asarray(json.dumps({"representation": config.representation})),
        )
    return output


def preprocess_features(
    features: np.ndarray,
    config: PruningConfig,
    seed: int,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2-D feature matrix, got {matrix.shape}")
    if config.scaler == "standard":
        matrix = StandardScaler().fit_transform(matrix)
    elif config.scaler == "minmax":
        matrix = MinMaxScaler().fit_transform(matrix)
    elif config.scaler != "none":
        raise ValueError(f"Unknown scaler: {config.scaler}")

    if config.pca_components is not None:
        maximum = min(int(config.pca_components), matrix.shape[0] - 1, matrix.shape[1])
        if maximum >= 2 and matrix.shape[1] > maximum:
            matrix = PCA(n_components=maximum, random_state=seed).fit_transform(matrix)
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessing produced NaN or infinite feature values")
    return matrix
