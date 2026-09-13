"""Component and baseline ablation orchestration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from latent_pruning.config import ExperimentConfig, load_ablation_variants
from latent_pruning.data import ImageSample, stratified_split, summarize_samples, write_sample_manifest
from latent_pruning.features import extract_vqvae_features, preprocess_features
from latent_pruning.pipeline import _aggregate, load_dataset_indexes
from latent_pruning.pruning import prune_samples, save_pruning_artifact
from latent_pruning.training import train_classifier
from latent_pruning.utils import resolve_device, slugify, utc_run_stamp, write_json


def _classical_undersampling(
    samples: Sequence[ImageSample],
    config: ExperimentConfig,
    seed: int,
    method: str,
    cache_dir: Path,
) -> list[ImageSample]:
    try:
        from imblearn.under_sampling import NearMiss, TomekLinks
    except ImportError as exc:
        raise ImportError("Install ablation dependencies with: pip install -e '.[ablation]'") from exc
    device = resolve_device("auto")
    feature_set = extract_vqvae_features(
        samples,
        config.vqvae,
        device=device,
        cache_dir=cache_dir,
    )
    features = preprocess_features(feature_set.features, config.pruning, seed)
    labels = np.asarray([sample.class_index for sample in samples])
    sampler = NearMiss(version=1, n_neighbors=3) if method == "nearmiss" else TomekLinks()
    sampler.fit_resample(features, labels)
    indices = getattr(sampler, "sample_indices_", None)
    if indices is None:
        raise RuntimeError(f"{method} did not expose sample_indices_")
    return [samples[int(index)] for index in sorted(np.asarray(indices, dtype=int))]


def _save_classical_selection(
    original: Sequence[ImageSample],
    selected: Sequence[ImageSample],
    output: Path,
    method: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    selected_paths = {sample.path for sample in selected}
    pd.DataFrame(
        [
            {
                "path": str(sample.path),
                "class": sample.label,
                "class_index": sample.class_index,
                "selected": sample.path in selected_paths,
                "selection_reason": method if sample.path in selected_paths else "not_selected",
            }
            for sample in original
        ]
    ).to_csv(output / "selection_manifest.csv", index=False)
    write_sample_manifest(selected, output / "selected_samples.csv")
    write_json(
        output / "selection_summary.json",
        {
            "method": method,
            "original_count": len(original),
            "selected_count": len(selected),
            "class_counts": summarize_samples(selected),
        },
    )


def run_ablation_suite(
    config: ExperimentConfig,
    variants: Sequence[str],
    variants_file: str | Path,
    model: str | None = None,
    seeds: Sequence[int] | None = None,
    uniform_retention: float = 0.75,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = load_ablation_variants(variants_file)
    unknown = sorted(set(variants) - set(definitions))
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}; available={sorted(definitions)}")
    base = copy.deepcopy(config)
    if model:
        base.training.model = model
    base.validate()
    train_index, validation_index, test_index = load_dataset_indexes(base)
    root = Path(output_dir) if output_dir else (
        base.output_dir
        / "ablation"
        / f"{slugify(base.training.model)}_{utc_run_stamp()}"
    )
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in list(seeds or base.training.seeds):
        if validation_index is None:
            fit_samples, validation_samples = stratified_split(
                train_index.samples,
                base.training.validation_fraction,
                seed,
            )
        else:
            fit_samples = list(train_index.samples)
            validation_samples = list(validation_index.samples)

        for variant in variants:
            definition = definitions[variant]
            experiment = copy.deepcopy(base)
            experiment.training.loss = str(definition.get("loss", experiment.training.loss))
            experiment.training.use_class_weights = bool(
                definition.get("use_class_weights", experiment.training.use_class_weights)
            )
            selection_name = str(definition.get("selection", "full"))
            if selection_name not in {"full", "nearmiss", "tomeklinks"}:
                if definition.get("use_balanced_retention", False):
                    pass
                else:
                    experiment = experiment.with_uniform_retention(uniform_retention)
            run_dir = root / variant / f"seed_{seed}"
            cache_dir = base.output_dir / "feature_cache"

            if selection_name == "full":
                selected = list(fit_samples)
            elif selection_name in {"nearmiss", "tomeklinks"}:
                selected = _classical_undersampling(
                    fit_samples, experiment, seed, selection_name, cache_dir
                )
                _save_classical_selection(
                    fit_samples, selected, run_dir / "selection", selection_name
                )
            else:
                selector = {
                    "proposed": "proposed",
                    "random": "random",
                    "global_ocsvm": "global_ocsvm",
                    "clustered_random": "clustered_random",
                }[selection_name]
                feature_mode = str(definition.get("feature_mode", "vqvae"))
                selection = prune_samples(
                    fit_samples,
                    experiment,
                    seed,
                    selector=selector,
                    feature_mode=feature_mode,
                    cache_dir=cache_dir,
                )
                save_pruning_artifact(
                    selection,
                    experiment,
                    seed,
                    output_dir=run_dir / "selection",
                )
                selected = selection.selected_samples

            training = train_classifier(
                selected,
                validation_samples,
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
                    "variant": variant,
                    "seed": seed,
                    "train_size_before": len(fit_samples),
                    "train_size_after": len(selected),
                    "effective_total_retention": len(selected) / len(fit_samples),
                    "run_dir": str(run_dir),
                    **training.test_metrics,
                }
            )
    write_json(
        root / "ablation_definition.json",
        {
            "variants": {name: definitions[name] for name in variants},
            "uniform_retention": uniform_retention,
            "seeds": list(seeds or base.training.seeds),
            "config": base.to_dict(),
        },
    )
    return _aggregate(rows, root)
