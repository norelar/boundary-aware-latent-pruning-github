"""Dataset-level pruning orchestration and artifact creation."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import yaml

from latent_pruning.config import ExperimentConfig
from latent_pruning.data import ImageSample, materialize_samples, summarize_samples
from latent_pruning.features import extract_raw_pixel_features, extract_vqvae_features
from latent_pruning.selection import SelectionResult, select_features
from latent_pruning.utils import environment_metadata, slugify, stable_hash, write_json

FeatureMode = Literal["vqvae", "raw_pixels", "none"]


@dataclass
class PruningArtifact:
    selected_samples: list[ImageSample]
    selection_manifest: pd.DataFrame
    cluster_metrics: pd.DataFrame
    class_summary: pd.DataFrame
    output_dir: Path | None = None


def _validate_policy(samples: Sequence[ImageSample], config: ExperimentConfig) -> None:
    labels = {sample.label for sample in samples}
    configured = set(config.dataset.pruned_classes)
    missing = sorted(configured - labels)
    if missing:
        raise ValueError(
            f"Configured pruned classes are absent from {config.dataset.name}: {missing}; "
            f"available classes are {sorted(labels)}"
        )


def _selection_records(
    class_samples: Sequence[ImageSample], result: SelectionResult
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, sample in enumerate(class_samples):
        margin = result.abs_margins[index]
        records.append(
            {
                "path": str(sample.path),
                "class": sample.label,
                "class_index": sample.class_index,
                "selected": bool(result.selected_mask[index]),
                "cluster_id": int(result.cluster_labels[index]),
                "abs_margin": None if np.isnan(margin) else float(margin),
                "support_vector": bool(result.support_mask[index]),
                "boundary_priority": bool(result.boundary_mask[index]),
                "selection_reason": str(result.reasons[index]),
            }
        )
    return records


def prune_samples(
    samples: Sequence[ImageSample],
    config: ExperimentConfig,
    seed: int,
    selector: str = "proposed",
    feature_mode: FeatureMode = "vqvae",
    cache_dir: str | Path | None = None,
) -> PruningArtifact:
    _validate_policy(samples, config)
    if selector == "random":
        feature_mode = "none"
    by_class: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.label].append(sample)

    selected_samples: list[ImageSample] = []
    manifest_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    vqvae_model = None

    for label in sorted(by_class):
        class_samples = by_class[label]
        original_count = len(class_samples)
        if label not in config.dataset.retention:
            selected_samples.extend(class_samples)
            manifest_records.extend(
                {
                    "path": str(sample.path),
                    "class": sample.label,
                    "class_index": sample.class_index,
                    "selected": True,
                    "cluster_id": -1,
                    "abs_margin": None,
                    "support_vector": False,
                    "boundary_priority": False,
                    "selection_reason": "preserved_full_class",
                }
                for sample in class_samples
            )
            summary_records.append(
                {
                    "class": label,
                    "original_count": original_count,
                    "selected_count": original_count,
                    "effective_retention": 1.0,
                    "configured_retention": 1.0,
                    "status": "preserved_full_class",
                }
            )
            continue

        retention = float(config.dataset.retention[label])
        if selector == "random":
            features = np.zeros((original_count, 1), dtype=np.float32)
            result = select_features(
                features,
                retention,
                config.pruning,
                seed=seed,
                mode="random",
                preprocess=False,
            )
        else:
            if feature_mode == "vqvae":
                if vqvae_model is None:
                    from latent_pruning.models.vqvae import load_vqvae_checkpoint
                    from latent_pruning.utils import resolve_device

                    device = resolve_device("auto")
                    vqvae_model = load_vqvae_checkpoint(config.vqvae, device=device)
                feature_set = extract_vqvae_features(
                    class_samples,
                    config.vqvae,
                    device=next(vqvae_model.parameters()).device.type,
                    cache_dir=cache_dir,
                    model=vqvae_model,
                )
            elif feature_mode == "raw_pixels":
                feature_set = extract_raw_pixel_features(class_samples, image_size=config.vqvae.image_size)
            else:
                raise ValueError(f"Feature mode {feature_mode!r} is incompatible with selector {selector!r}")

            selector_mode = {
                "proposed": "proposed",
                "global_ocsvm": "global_ocsvm",
                "clustered_random": "clustered_random",
            }.get(selector)
            if selector_mode is None:
                raise ValueError(f"Unsupported selector: {selector}")
            result = select_features(
                feature_set.features,
                retention,
                config.pruning,
                seed=seed,
                mode=selector_mode,
            )

        selected = [class_samples[int(index)] for index in result.selected_indices]
        selected_samples.extend(selected)
        manifest_records.extend(_selection_records(class_samples, result))
        for metric in result.cluster_metrics:
            row = metric.to_dict()
            row.update(
                {
                    "dataset": config.dataset.name,
                    "class": label,
                    "seed": seed,
                    "configured_retention": retention,
                    "selector": selector,
                    "feature_mode": feature_mode,
                }
            )
            metric_records.append(row)
        summary_records.append(
            {
                "class": label,
                "original_count": original_count,
                "selected_count": len(selected),
                "effective_retention": len(selected) / original_count,
                "configured_retention": retention,
                "status": selector,
            }
        )

    selected_samples.sort(key=lambda sample: (sample.class_index, str(sample.path)))
    artifact = PruningArtifact(
        selected_samples=selected_samples,
        selection_manifest=pd.DataFrame(manifest_records).sort_values(["class_index", "path"]),
        cluster_metrics=pd.DataFrame(metric_records),
        class_summary=pd.DataFrame(summary_records).sort_values("class"),
    )
    expected = int(artifact.class_summary["selected_count"].sum())
    if len(artifact.selected_samples) != expected:
        raise RuntimeError("Class summary and selected sample list disagree")
    return artifact


def save_pruning_artifact(
    artifact: PruningArtifact,
    config: ExperimentConfig,
    seed: int,
    output_dir: str | Path | None = None,
    materialize: bool = False,
    overwrite: bool = False,
) -> PruningArtifact:
    retention_signature = stable_hash(config.dataset.retention, length=8)
    run_name = f"{slugify(config.dataset.name)}_seed-{seed}_ret-{retention_signature}"
    destination = Path(output_dir) if output_dir else config.output_dir / "pruning" / run_name
    destination.mkdir(parents=True, exist_ok=True)
    artifact.selection_manifest.to_csv(destination / "selection_manifest.csv", index=False)
    artifact.cluster_metrics.to_csv(destination / "cluster_metrics.csv", index=False)
    artifact.class_summary.to_csv(destination / "class_summary.csv", index=False)
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(
        destination / "run_metadata.json",
        {
            "dataset": config.dataset.name,
            "seed": seed,
            "selector": sorted(set(artifact.class_summary["status"]) - {"preserved_full_class"}),
            "retention": config.dataset.retention,
            "input_count": int(artifact.class_summary["original_count"].sum()),
            "selected_count": len(artifact.selected_samples),
            "selected_counts": summarize_samples(artifact.selected_samples),
            "environment": environment_metadata(),
        },
    )
    if materialize:
        materialize_samples(
            artifact.selected_samples,
            destination / "dataset",
            mode=config.pruning.materialization,
            overwrite=overwrite,
        )
    artifact.output_dir = destination
    return artifact


def run_pruning(
    samples: Sequence[ImageSample],
    config: ExperimentConfig,
    seed: int,
    selector: str = "proposed",
    feature_mode: FeatureMode = "vqvae",
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    materialize: bool = False,
    overwrite: bool = False,
) -> PruningArtifact:
    artifact = prune_samples(samples, config, seed, selector, feature_mode, cache_dir)
    return save_pruning_artifact(
        artifact,
        config,
        seed,
        output_dir=output_dir,
        materialize=materialize,
        overwrite=overwrite,
    )


def config_with_retention_map(
    config: ExperimentConfig, retention: dict[str, float]
) -> ExperimentConfig:
    clone = copy.deepcopy(config)
    clone.dataset.retention = {str(label): float(value) for label, value in retention.items()}
    clone.dataset.pruned_classes = list(clone.dataset.retention)
    clone.validate()
    return clone
