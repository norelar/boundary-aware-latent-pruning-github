"""Typed YAML configuration loading and validation."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is inconsistent."""


@dataclass
class DatasetConfig:
    name: str
    train_dir: Path
    test_dir: Path
    validation_dir: Path | None = None
    pruned_classes: list[str] = field(default_factory=list)
    retention: dict[str, float] = field(default_factory=dict)
    retention_sweep: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    extensions: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    )


@dataclass
class VQVAEConfig:
    checkpoint: Path
    image_size: int = 64
    in_channels: int = 3
    embedding_dim: int = 128
    num_embeddings: int = 512
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    beta: float = 0.25
    representation: str = "histogram"
    batch_size: int = 64
    num_workers: int = 2


@dataclass
class PruningConfig:
    n_clusters: int = 30
    n_init: int = 10
    nu: float = 0.05
    gamma: str | float = "scale"
    include_support_vectors: bool = True
    boundary_quantile: float = 0.20
    boundary_threshold: float | None = None
    n_margin_strata: int = 5
    scaler: str = "standard"
    pca_components: int | None = 128
    materialization: str = "copy"


@dataclass
class TrainingConfig:
    model: str = "resnet50"
    pretrained: bool = True
    image_size: int = 224
    hidden_dim: int = 256
    dropout: float = 0.30
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "step"
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.10
    validation_fraction: float = 0.15
    early_stopping_patience: int = 5
    loss: str = "cross_entropy"
    use_class_weights: bool = False
    mixed_precision: bool = True
    seeds: list[int] = field(default_factory=lambda: [7, 21, 42, 128, 224])


@dataclass
class ExperimentConfig:
    project_name: str
    output_dir: Path
    dataset: DatasetConfig
    vqvae: VQVAEConfig
    pruning: PruningConfig = field(default_factory=PruningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    source_path: Path | None = field(default=None, repr=False)

    def validate(self) -> None:
        errors: list[str] = []
        ds = self.dataset
        vq = self.vqvae
        pr = self.pruning
        tr = self.training

        if not ds.name.strip():
            errors.append("dataset.name must not be empty")
        if len(set(ds.pruned_classes)) != len(ds.pruned_classes):
            errors.append("dataset.pruned_classes contains duplicates")
        missing = sorted(set(ds.pruned_classes) - set(ds.retention))
        if missing:
            errors.append(f"retention is missing configured pruned classes: {missing}")
        extra = sorted(set(ds.retention) - set(ds.pruned_classes))
        if extra:
            errors.append(f"retention contains classes absent from pruned_classes: {extra}")
        for label, value in ds.retention.items():
            if not 0.0 < float(value) <= 1.0:
                errors.append(f"retention[{label!r}] must be in (0, 1], got {value}")
        for value in ds.retention_sweep:
            if not 0.0 < float(value) <= 1.0:
                errors.append(f"retention_sweep values must be in (0, 1], got {value}")

        normalized_extensions = []
        for extension in ds.extensions:
            extension = extension.lower()
            normalized_extensions.append(extension if extension.startswith(".") else f".{extension}")
        ds.extensions = sorted(set(normalized_extensions))

        if vq.representation not in {"encoding_indices", "histogram", "quantized", "pooled"}:
            errors.append(
                "vqvae.representation must be encoding_indices, histogram, quantized, or pooled"
            )
        if vq.image_size <= 0 or vq.embedding_dim <= 0 or vq.num_embeddings <= 1:
            errors.append("VQ-VAE dimensions must be positive and num_embeddings must exceed 1")
        if not vq.hidden_dims or any(int(x) <= 0 for x in vq.hidden_dims):
            errors.append("vqvae.hidden_dims must contain positive integers")

        if pr.n_clusters < 1 or pr.n_init < 1:
            errors.append("pruning.n_clusters and pruning.n_init must be at least 1")
        if not 0.0 < pr.nu < 1.0:
            errors.append("pruning.nu must be in (0, 1)")
        if not 0.0 <= pr.boundary_quantile <= 1.0:
            errors.append("pruning.boundary_quantile must be in [0, 1]")
        if pr.boundary_threshold is not None and pr.boundary_threshold < 0.0:
            errors.append("pruning.boundary_threshold must be non-negative or null")
        if pr.n_margin_strata < 1:
            errors.append("pruning.n_margin_strata must be at least 1")
        if pr.scaler not in {"standard", "minmax", "none"}:
            errors.append("pruning.scaler must be standard, minmax, or none")
        if pr.materialization not in {"copy", "symlink", "hardlink", "manifest"}:
            errors.append("pruning.materialization must be copy, symlink, hardlink, or manifest")

        if tr.epochs < 1 or tr.batch_size < 1:
            errors.append("training.epochs and training.batch_size must be at least 1")
        if not 0.0 <= tr.validation_fraction < 1.0:
            errors.append("training.validation_fraction must be in [0, 1)")
        if tr.optimizer not in {"adam", "adamw", "sgd"}:
            errors.append("training.optimizer must be adam, adamw, or sgd")
        if tr.scheduler not in {"step", "cosine", "plateau", "none"}:
            errors.append("training.scheduler must be step, cosine, plateau, or none")
        if tr.loss not in {"cross_entropy", "focal"}:
            errors.append("training.loss must be cross_entropy or focal")
        if not tr.seeds:
            errors.append("training.seeds must contain at least one integer")
        if len(set(tr.seeds)) != len(tr.seeds):
            errors.append("training.seeds contains duplicates")

        if errors:
            raise ConfigurationError("Invalid configuration:\n- " + "\n- ".join(errors))

    def with_uniform_retention(self, retention: float) -> "ExperimentConfig":
        if not 0.0 < retention <= 1.0:
            raise ConfigurationError(f"retention must be in (0, 1], got {retention}")
        clone = copy.deepcopy(self)
        clone.dataset.retention = {label: float(retention) for label in clone.dataset.pruned_classes}
        clone.validate()
        return clone

    def with_model(self, model: str) -> "ExperimentConfig":
        clone = copy.deepcopy(self)
        clone.training = replace(clone.training, model=model)
        clone.validate()
        return clone

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw.pop("source_path", None)

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(raw)


def _expand_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError("The YAML root must be a mapping")

    try:
        dataset_raw = dict(raw["dataset"])
        vqvae_raw = dict(raw["vqvae"])
    except KeyError as exc:
        raise ConfigurationError(f"Missing required configuration section: {exc.args[0]}") from exc

    dataset_raw["train_dir"] = _expand_path(dataset_raw["train_dir"])
    dataset_raw["test_dir"] = _expand_path(dataset_raw["test_dir"])
    dataset_raw["validation_dir"] = _expand_path(dataset_raw.get("validation_dir"))
    vqvae_raw["checkpoint"] = _expand_path(vqvae_raw["checkpoint"])

    config = ExperimentConfig(
        project_name=str(raw.get("project_name", "boundary_aware_latent_pruning")),
        output_dir=_expand_path(raw.get("output_dir", "outputs")) or Path("outputs"),
        dataset=DatasetConfig(**dataset_raw),
        vqvae=VQVAEConfig(**vqvae_raw),
        pruning=PruningConfig(**dict(raw.get("pruning", {}))),
        training=TrainingConfig(**dict(raw.get("training", {}))),
        source_path=source.resolve(),
    )
    config.validate()
    return config


def load_ablation_variants(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    variants = raw.get("variants") if isinstance(raw, dict) else None
    if not isinstance(variants, dict) or not variants:
        raise ConfigurationError(f"No variants found in {source}")
    return {str(name): dict(settings) for name, settings in variants.items()}
