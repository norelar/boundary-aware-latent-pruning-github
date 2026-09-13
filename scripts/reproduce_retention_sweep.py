#!/usr/bin/env python3
"""Run configured datasets, retention levels, models, and seeds from Python."""

from __future__ import annotations

import argparse
from pathlib import Path

from latent_pruning.config import load_config
from latent_pruning.pipeline import run_pruning_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["resnet50"])
    parser.add_argument("--retentions", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    for config_path in args.configs:
        config = load_config(config_path)
        seeds = args.seeds or config.training.seeds
        for model in args.models:
            for retention in args.retentions:
                print(
                    f"Running dataset={config.dataset.name}, model={model}, "
                    f"retention={retention:.0%}, seeds={seeds}"
                )
                run_pruning_experiment(
                    config,
                    model=model,
                    retention=retention,
                    seeds=seeds,
                    device=args.device,
                )


if __name__ == "__main__":
    main()
