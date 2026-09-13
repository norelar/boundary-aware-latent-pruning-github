"""Command-line interface for all repository workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

from latent_pruning.config import ConfigurationError, load_config


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Dataset YAML configuration")


def _seeds(values: Sequence[int] | None, defaults: Sequence[int]) -> list[int]:
    return list(values) if values else list(defaults)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent-pruning",
        description="Boundary-aware VQ-VAE latent-space dataset pruning",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-config", help="Validate YAML and optionally dataset folders")
    _common_config(validate)
    validate.add_argument("--check-data", action="store_true")
    validate.add_argument("--check-checkpoint", action="store_true")

    vqvae = subcommands.add_parser("train-vqvae", help="Train or resume a VQ-VAE")
    _common_config(vqvae)
    vqvae.add_argument("--output", type=Path)
    vqvae.add_argument("--epochs", type=int, default=300)
    vqvae.add_argument("--learning-rate", type=float, default=1e-4)
    vqvae.add_argument("--validation-fraction", type=float, default=0.10)
    vqvae.add_argument("--seed", type=int, default=42)
    vqvae.add_argument("--device", default="auto")
    vqvae.add_argument("--resume", type=Path)

    prune = subcommands.add_parser("prune", help="Run one pruning policy")
    _common_config(prune)
    prune.add_argument("--output", type=Path)
    prune.add_argument("--retention", type=float)
    prune.add_argument("--seed", type=int, default=42)
    prune.add_argument(
        "--selector",
        choices=["proposed", "global_ocsvm", "clustered_random", "random"],
        default="proposed",
    )
    prune.add_argument("--feature-mode", choices=["vqvae", "raw_pixels", "none"], default="vqvae")
    prune.add_argument("--materialize", action="store_true")
    prune.add_argument("--overwrite", action="store_true")

    sweep = subcommands.add_parser("prune-sweep", help="Run every configured retention level")
    _common_config(sweep)
    sweep.add_argument("--output", type=Path)
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--materialize", action="store_true")
    sweep.add_argument("--overwrite", action="store_true")

    baseline = subcommands.add_parser("train-classifier", help="Train on the full, unpruned training data")
    _common_config(baseline)
    baseline.add_argument("--output", type=Path)
    baseline.add_argument("--model")
    baseline.add_argument("--seeds", nargs="+", type=int)
    baseline.add_argument("--seed", type=int, help="Convenience alias for a single seed")
    baseline.add_argument("--device", default="auto")

    experiment = subcommands.add_parser(
        "run-experiment", help="Split, prune only the fit subset, train, and test once"
    )
    _common_config(experiment)
    experiment.add_argument("--output", type=Path)
    experiment.add_argument("--model")
    experiment.add_argument("--retention", type=float)
    experiment.add_argument("--seeds", nargs="+", type=int)
    experiment.add_argument("--seed", type=int)
    experiment.add_argument(
        "--selector",
        choices=["proposed", "global_ocsvm", "clustered_random", "random"],
        default="proposed",
    )
    experiment.add_argument("--feature-mode", choices=["vqvae", "raw_pixels", "none"], default="vqvae")
    experiment.add_argument("--device", default="auto")

    ablation = subcommands.add_parser("run-ablation", help="Run component and imbalance baselines")
    _common_config(ablation)
    ablation.add_argument("--output", type=Path)
    ablation.add_argument("--model")
    ablation.add_argument("--seeds", nargs="+", type=int)
    ablation.add_argument("--seed", type=int)
    ablation.add_argument("--uniform-retention", type=float, default=0.75)
    ablation.add_argument("--variants-file", type=Path, default=Path("configs/ablation_variants.yaml"))
    ablation.add_argument("--variants", nargs="+", required=True)
    ablation.add_argument("--device", default="auto")

    summarize = subcommands.add_parser("summarize-results", help="Aggregate completed structured runs")
    summarize.add_argument("--input", required=True, type=Path)
    summarize.add_argument("--output", required=True, type=Path)

    legacy = subcommands.add_parser("analyze-legacy-logs", help="Parse original notebook text logs")
    legacy.add_argument("--input", required=True, type=Path)
    legacy.add_argument("--output", required=True, type=Path)
    legacy.add_argument("--no-recursive", action="store_true")
    return parser


def _chosen_seeds(args, config) -> list[int]:
    if getattr(args, "seed", None) is not None:
        return [args.seed]
    return _seeds(getattr(args, "seeds", None), config.training.seeds)


def execute(args: argparse.Namespace) -> int:
    if args.command == "summarize-results":
        from latent_pruning.pipeline import summarize_existing_runs

        raw, summary = summarize_existing_runs(args.input, args.output)
        print(f"Aggregated {len(raw)} runs; summary saved to {args.output}")
        return 0
    if args.command == "analyze-legacy-logs":
        from latent_pruning.analysis.legacy_logs import analyze_legacy_directory

        clusters, performance = analyze_legacy_directory(
            args.input, args.output, recursive=not args.no_recursive
        )
        print(
            f"Parsed {len(clusters)} pruning-cluster rows and {len(performance)} performance runs; "
            f"outputs saved to {args.output}"
        )
        return 0

    config = load_config(args.config)
    if args.command == "validate-config":
        report: dict[str, object] = {"config": str(args.config), "status": "valid"}
        if args.check_checkpoint:
            if not config.vqvae.checkpoint.is_file():
                raise FileNotFoundError(f"VQ-VAE checkpoint not found: {config.vqvae.checkpoint}")
            report["checkpoint"] = str(config.vqvae.checkpoint)
        if args.check_data:
            from latent_pruning.pipeline import load_dataset_indexes

            train, validation, test = load_dataset_indexes(config)
            report["classes"] = list(train.classes)
            report["train_counts"] = train.counts()
            report["validation_counts"] = validation.counts() if validation else None
            report["test_counts"] = test.counts()
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
        return 0

    if args.command == "train-vqvae":
        from latent_pruning.data import scan_image_folder
        from latent_pruning.vqvae_training import train_vqvae

        train = scan_image_folder(config.dataset.train_dir, config.dataset.extensions)
        output = args.output or config.output_dir / "vqvae"
        checkpoint = train_vqvae(
            train.samples,
            config,
            output,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            device=args.device,
            resume=args.resume,
        )
        print(f"Best VQ-VAE checkpoint: {checkpoint}")
        return 0

    if args.command in {"prune", "prune-sweep"}:
        from latent_pruning.data import scan_image_folder
        from latent_pruning.pruning import run_pruning

        train = scan_image_folder(config.dataset.train_dir, config.dataset.extensions)
        if args.command == "prune":
            current = config.with_uniform_retention(args.retention) if args.retention else config
            artifact = run_pruning(
                train.samples,
                current,
                args.seed,
                selector=args.selector,
                feature_mode=args.feature_mode,
                cache_dir=current.output_dir / "feature_cache",
                output_dir=args.output,
                materialize=args.materialize,
                overwrite=args.overwrite,
            )
            print(f"Selected {len(artifact.selected_samples)} images; outputs: {artifact.output_dir}")
            return 0
        root = args.output or config.output_dir / "pruning_sweep"
        for retention in config.dataset.retention_sweep:
            current = config.with_uniform_retention(float(retention))
            artifact = run_pruning(
                train.samples,
                current,
                args.seed,
                cache_dir=current.output_dir / "feature_cache",
                output_dir=root / f"retention_{int(round(retention * 100)):03d}",
                materialize=args.materialize,
                overwrite=args.overwrite,
            )
            print(f"Retention {retention:.0%}: {len(artifact.selected_samples)} images")
        print(f"Sweep outputs: {root}")
        return 0

    if args.command == "train-classifier":
        from latent_pruning.pipeline import run_baseline_suite

        raw, _ = run_baseline_suite(
            config,
            model=args.model,
            seeds=_chosen_seeds(args, config),
            output_dir=args.output,
            device=args.device,
        )
        print(raw.to_string(index=False))
        return 0

    if args.command == "run-experiment":
        from latent_pruning.pipeline import run_pruning_experiment

        raw, _ = run_pruning_experiment(
            config,
            model=args.model,
            retention=args.retention,
            seeds=_chosen_seeds(args, config),
            selector=args.selector,
            feature_mode=args.feature_mode,
            output_dir=args.output,
            device=args.device,
        )
        print(raw.to_string(index=False))
        return 0

    if args.command == "run-ablation":
        from latent_pruning.ablation import run_ablation_suite

        raw, _ = run_ablation_suite(
            config,
            variants=args.variants,
            variants_file=args.variants_file,
            model=args.model,
            seeds=_chosen_seeds(args, config),
            uniform_retention=args.uniform_retention,
            output_dir=args.output,
            device=args.device,
        )
        print(raw.to_string(index=False))
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = execute(args)
    except (ConfigurationError, FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
