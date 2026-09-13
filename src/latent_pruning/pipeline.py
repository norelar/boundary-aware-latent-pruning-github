"""End-to-end baseline and pruning experiments across multiple seeds."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from latent_pruning.config import ExperimentConfig
from latent_pruning.data import (
    DatasetIndex,
    check_compatible_classes,
    scan_image_folder,
    stratified_split,
)
from latent_pruning.pruning import prune_samples, save_pruning_artifact
from latent_pruning.training import TrainingArtifact, train_classifier
from latent_pruning.utils import slugify, utc_run_stamp, write_json


def load_dataset_indexes(
    config: ExperimentConfig,
) -> tuple[DatasetIndex, DatasetIndex | None, DatasetIndex]:
    train = scan_image_folder(
        config.dataset.train_dir,
        config.dataset.extensions,
    )
    test = scan_image_folder(
        config.dataset.test_dir,
        config.dataset.extensions,
        expected_classes=train.classes,
    )
    check_compatible_classes(train, test)
    validation = None
    if config.dataset.validation_dir is not None:
        validation = scan_image_folder(
            config.dataset.validation_dir,
            config.dataset.extensions,
            expected_classes=train.classes,
        )
        check_compatible_classes(train, validation)
    return train, validation, test


def _aggregate(rows: list[dict[str, object]], output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "runs.csv", index=False)
    excluded = {"dataset", "model", "variant", "seed", "run_dir"}
    metric_columns = [
        column
        for column in raw.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(raw[column])
    ]
    group_columns = [column for column in ("dataset", "model", "variant") if column in raw]
    summary = raw.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "summary_mean_std.csv", index=False)
    return raw, summary


def run_baseline_suite(
    config: ExperimentConfig,
    model: str | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment = copy.deepcopy(config)
    if model:
        experiment.training.model = model
    experiment.validate()
    train_index, validation_index, test_index = load_dataset_indexes(experiment)
    run_root = Path(output_dir) if output_dir else (
        experiment.output_dir / "baseline" / f"{slugify(experiment.training.model)}_{utc_run_stamp()}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in list(seeds or experiment.training.seeds):
        if validation_index is None:
            fit, validation = stratified_split(
                train_index.samples,
                experiment.training.validation_fraction,
                seed,
            )
        else:
            fit, validation = list(train_index.samples), list(validation_index.samples)
        run_dir = run_root / f"seed_{seed}"
        artifact = train_classifier(
            fit,
            validation,
            test_index.samples,
            experiment,
            seed,
            run_dir,
            device=device,
        )
        rows.append(
            {
                "dataset": experiment.dataset.name,
                "model": experiment.training.model,
                "variant": "full_data",
                "seed": seed,
                "train_size": len(fit),
                "validation_size": len(validation),
                "test_size": len(test_index.samples),
                "run_dir": str(run_dir),
                **artifact.test_metrics,
            }
        )
    return _aggregate(rows, run_root)


def run_pruning_experiment(
    config: ExperimentConfig,
    model: str | None = None,
    retention: float | None = None,
    seeds: Sequence[int] | None = None,
    selector: str = "proposed",
    feature_mode: str = "vqvae",
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment = copy.deepcopy(config)
    if model:
        experiment.training.model = model
    if retention is not None:
        experiment = experiment.with_uniform_retention(retention)
    experiment.validate()
    effective_feature_mode = "none" if selector == "random" else feature_mode
    train_index, validation_index, test_index = load_dataset_indexes(experiment)
    signature = "-".join(
        f"{slugify(label)}-{value:g}" for label, value in sorted(experiment.dataset.retention.items())
    )
    run_root = Path(output_dir) if output_dir else (
        experiment.output_dir
        / "experiments"
        / f"{slugify(experiment.training.model)}_{slugify(signature)}_{utc_run_stamp()}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in list(seeds or experiment.training.seeds):
        if validation_index is None:
            fit, validation = stratified_split(
                train_index.samples,
                experiment.training.validation_fraction,
                seed,
            )
        else:
            fit, validation = list(train_index.samples), list(validation_index.samples)
        run_dir = run_root / f"seed_{seed}"
        selection = prune_samples(
            fit,
            experiment,
            seed,
            selector=selector,
            feature_mode=effective_feature_mode,
            cache_dir=experiment.output_dir / "feature_cache",
        )
        save_pruning_artifact(
            selection,
            experiment,
            seed,
            output_dir=run_dir / "selection",
            materialize=False,
        )
        training = train_classifier(
            selection.selected_samples,
            validation,
            test_index.samples,
            experiment,
            seed,
            run_dir / "training",
            device=device,
        )
        rows.append(
            {
                "dataset": experiment.dataset.name,
                "model": experiment.training.model,
                "variant": f"{selector}_{effective_feature_mode}",
                "seed": seed,
                "train_size_before": len(fit),
                "train_size_after": len(selection.selected_samples),
                "effective_total_retention": len(selection.selected_samples) / len(fit),
                "validation_size": len(validation),
                "test_size": len(test_index.samples),
                "run_dir": str(run_dir),
                **training.test_metrics,
            }
        )
    write_json(
        run_root / "experiment_definition.json",
        {
            "config": experiment.to_dict(),
            "selector": selector,
            "feature_mode": effective_feature_mode,
            "seeds": list(seeds or experiment.training.seeds),
        },
    )
    return _aggregate(rows, run_root)


def summarize_existing_runs(input_dir: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(input_dir)
    metric_files = sorted(source.rglob("test_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No test_metrics.json files found under {source}")
    import json

    rows: list[dict[str, object]] = []
    for metric_file in metric_files:
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
        metadata_file = metric_file.with_name("run_metadata.json")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
        rows.append(
            {
                "dataset": metadata.get("dataset", "unknown"),
                "model": metadata.get("model", "unknown"),
                "variant": metric_file.parent.parent.name,
                "seed": metadata.get("seed"),
                "run_dir": str(metric_file.parent),
                **metrics,
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return _aggregate(rows, output)
