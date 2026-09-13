"""Parse and summarize the text logs produced by the original notebooks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def normalize_dataset_name(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {
        "HAM": "HAM10000",
        "HAM10000": "HAM10000",
        "COVID": "COVID19",
        "COVID19": "COVID19",
        "OASIS": "OASIS",
        "ADNI": "OASIS",
    }
    normalized = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return aliases.get(normalized, normalized)


def infer_dataset_from_name(name: str) -> str | None:
    upper = name.upper()
    for token in ("HAM10000", "HAM", "COVID19", "COVID", "OASIS", "ADNI"):
        if token in upper:
            return normalize_dataset_name(token)
    return None


def infer_seed(text: str, name: str = "") -> int | None:
    match = re.search(r"^\s*Seed\s*:\s*(\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return int(match.group(1))
    match = re.search(r"seed[_:\- ]*(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_pruning_log(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    seed = infer_seed(text, source.name)
    dataset = infer_dataset_from_name(source.name) or infer_dataset_from_name(text[:1000])
    current_class: str | None = None
    current_retention: int | None = None
    current: dict[str, Any] | None = None
    section: str | None = None
    records: list[dict[str, Any]] = []

    progress_class = re.compile(
        r",\s*([^=,\]]+)=\s*\d+\s*/\s*\d+\]\s*features_vectors shape is",
        flags=re.IGNORECASE,
    )
    explicit_class = re.compile(r"(?:class|classe)\s*[:=]\s*([^,|]+)", flags=re.IGNORECASE)
    retention_pattern = re.compile(r"pruning ratio is\s+(\d+)", flags=re.IGNORECASE)
    cluster_start = re.compile(r"^\s*Cluster\s+(\d+)\s+-", flags=re.IGNORECASE)
    copied_pattern = re.compile(r"(\d+)\s+images copi(?:e|é)es vers\s+(.+)$", flags=re.IGNORECASE)

    for line in lines:
        match = progress_class.search(line)
        if match:
            current_class = match.group(1).strip()
        match = explicit_class.search(line)
        if match and "classification" not in line.lower():
            candidate = match.group(1).strip()
            if candidate and len(candidate) < 100:
                current_class = candidate
        match = retention_pattern.search(line)
        if match:
            current_retention = int(match.group(1))
        match = cluster_start.search(line)
        if match:
            current = {
                "file": source.name,
                "path": str(source),
                "dataset": dataset,
                "seed": seed,
                "class": current_class,
                "retention_percent": current_retention,
                "cluster_id": int(match.group(1)),
            }
            section = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped == "--- Entire Cluster ---":
            section = "entire"
            continue
        if stripped == "--- Selected Samples ---":
            section = "selected"
            continue
        if stripped == "--- Non-Selected Samples ---":
            section = "non_selected"
            continue

        patterns: list[tuple[str, str, type]] = [
            ("cluster_size", r"Cluster size:\s*(\d+)", int),
            ("selected_size", r"Selected size:\s*(\d+)", int),
            ("support_count", r"Support vectors?:\s*(\d+)", int),
            ("cmc", r"Cluster Margin Compactness \(CMC\):\s*([-+\d.eE]+)", float),
            ("bfs", r"Boundary Focus Score \(BFS\):\s*([-+\d.eE]+)", float),
            ("cv", r"Stratification Score \(CV\):\s*([-+\d.eE]+)", float),
        ]
        for key, pattern, converter in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                current[key] = converter(match.group(1))
        match = re.search(r"Border points:\s*(\d+)\s*\(([\d.]+)%\)", line, flags=re.IGNORECASE)
        if match:
            current["border_count"] = int(match.group(1))
            current["border_percent"] = float(match.group(2))
            current["border_fraction"] = float(match.group(2)) / 100.0
        if section is not None:
            match = re.match(r"Mean \|distance\|:\s*([-+\d.eE]+)$", stripped, flags=re.IGNORECASE)
            if match:
                current[f"{section}_mean_abs_margin"] = float(match.group(1))
            match = re.match(r"Std\s+\|distance\|:\s*([-+\d.eE]+)$", stripped, flags=re.IGNORECASE)
            if match:
                current[f"{section}_std_abs_margin"] = float(match.group(1))
        match = copied_pattern.search(line)
        if match:
            current["copied"] = int(match.group(1))
            current["output_path"] = match.group(2).strip()
            path_match = re.search(r"[/\\](\d+)%[/\\]([^/\\]+)", current["output_path"])
            if path_match:
                current["retention_percent"] = current.get("retention_percent") or int(path_match.group(1))
                current["class"] = current.get("class") or path_match.group(2)
            records.append(current)
            current = None
            section = None

    frame = pd.DataFrame(records)
    if not frame.empty:
        if "selected_size" in frame and "border_count" in frame:
            denominator = frame["selected_size"].replace(0, np.nan)
            frame["completion_fraction"] = (1.0 - frame["border_count"] / denominator).clip(0, 1)
        if "selected_mean_abs_margin" in frame and "entire_mean_abs_margin" in frame:
            frame["mean_margin_gap"] = (
                frame["selected_mean_abs_margin"] - frame["entire_mean_abs_margin"]
            )
        if {"seed", "class", "retention_percent", "cluster_size"}.issubset(frame.columns):
            frame["cluster_rank"] = (
                frame.groupby(["seed", "class", "retention_percent"])["cluster_size"]
                .rank(method="first", ascending=False)
                .astype("Int64")
            )
    return frame


def parse_performance_filename(name: str) -> dict[str, Any]:
    stem = Path(name).stem
    dataset = infer_dataset_from_name(stem)
    retention: int | None = None
    matches = re.findall(r"(?:^|_)(\d{1,3})%?(?=_|$)", stem)
    for candidate in reversed(matches):
        value = int(candidate)
        if 0 <= value <= 100:
            retention = value
            break
    if retention is None and re.search(r"(?:100%|baseline|full[_ -]?data)", stem, flags=re.IGNORECASE):
        retention = 100
    model = stem
    cutoff = re.search(r"(?:_train_|_model|\d{8}_\d{6})", stem, flags=re.IGNORECASE)
    if cutoff:
        model = stem[: cutoff.start()]
    model = model.strip("_ -").lower() or "unknown"
    timestamp = re.search(r"(\d{8})[_T](\d{6})", stem)
    if retention is None and dataset is not None and timestamp is not None:
        # Original baseline filenames frequently ended directly in the dataset
        # name, without an explicit 100% token.
        retention = 100
    return {
        "model": model,
        "dataset": dataset,
        "retention_percent": retention,
        "run_tag": "_".join(timestamp.groups()) if timestamp else stem,
    }


def _normalize_metric(value: str, percent_mark: str | None = None) -> float:
    number = float(value)
    if percent_mark or number > 1.0:
        number /= 100.0
    return number


def parse_performance_log(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    metadata = parse_performance_filename(source.name)
    metrics: dict[str, float | None] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    # Notebook training lines: TEST | Loss ... | Acc ... | Precision ... | Recall ... | F1-score ...
    epoch_pattern = re.compile(
        r"(?:TEST|test)\s*(?:\||:)?.*?"
        r"(?:Acc|Accuracy)\s*[:=]\s*([\d.]+)(%?).*?"
        r"Precision\s*[:=]\s*([\d.]+)(%?).*?"
        r"Recall\s*[:=]\s*([\d.]+)(%?).*?"
        r"F1(?:-score)?\s*[:=]\s*([\d.]+)(%?)",
        flags=re.IGNORECASE,
    )
    matches = list(epoch_pattern.finditer(text))
    if matches:
        match = matches[-1]
        metrics = {
            "accuracy": _normalize_metric(match.group(1), match.group(2)),
            "precision": _normalize_metric(match.group(3), match.group(4)),
            "recall": _normalize_metric(match.group(5), match.group(6)),
            "f1": _normalize_metric(match.group(7), match.group(8)),
        }
    else:
        patterns = {
            "accuracy": [r"\baccuracy\b\s*[:=]\s*([\d.]+)(%?)", r"\bacc\b\s*[:=]\s*([\d.]+)(%?)"],
            "precision": [r"\bprecision\b\s*[:=]\s*([\d.]+)(%?)"],
            "recall": [r"\brecall\b\s*[:=]\s*([\d.]+)(%?)"],
            "f1": [r"\bf1(?:-score)?\b\s*[:=]\s*([\d.]+)(%?)", r"\bf-score\b\s*[:=]\s*([\d.]+)(%?)"],
        }
        for key, candidates in patterns.items():
            found = []
            for pattern in candidates:
                found.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
            if found:
                match = found[-1]
                metrics[key] = _normalize_metric(match.group(1), match.group(2))

        macro_matches = list(
            re.finditer(
                r"^\s*macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
        if macro_matches:
            match = macro_matches[-1]
            metrics["precision"] = metrics["precision"] or _normalize_metric(match.group(1))
            metrics["recall"] = metrics["recall"] or _normalize_metric(match.group(2))
            metrics["f1"] = metrics["f1"] or _normalize_metric(match.group(3))

    return {
        "file": source.name,
        "path": str(source),
        "seed": infer_seed(text, source.name),
        **metadata,
        **metrics,
    }


def _performance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in ("accuracy", "precision", "recall", "f1") if column in frame]
    group = ["model", "dataset", "retention_percent"]
    summary = frame.groupby(group, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def _cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in (
            "bfs",
            "cmc",
            "cv",
            "border_fraction",
            "completion_fraction",
            "mean_margin_gap",
            "selected_size",
        )
        if column in frame
    ]
    group = ["dataset", "class", "retention_percent"]
    summary = frame.groupby(group, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def _save_plots(frame: pd.DataFrame, output: Path) -> list[Path]:
    if frame.empty:
        return []
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    safe_group = frame.dropna(subset=["class", "retention_percent"])
    for class_name, group in safe_group.groupby("class"):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(class_name))
        if "bfs" in group and group["bfs"].notna().any():
            aggregate = group.groupby("retention_percent")["bfs"].agg(["mean", "std"]).reset_index()
            figure, axis = plt.subplots()
            axis.errorbar(aggregate["retention_percent"], aggregate["mean"], yerr=aggregate["std"], marker="o")
            axis.set(xlabel="Retention (%)", ylabel="Boundary Focus Score", title=f"BFS — {class_name}")
            axis.grid(alpha=0.3)
            path = plot_dir / f"bfs_by_retention_{safe_name}.png"
            figure.savefig(path, dpi=250, bbox_inches="tight")
            plt.close(figure)
            saved.append(path)
        if "border_fraction" in group and group["border_fraction"].notna().any():
            levels = sorted(group["retention_percent"].dropna().unique())
            values = [group.loc[group["retention_percent"] == level, "border_fraction"].dropna() for level in levels]
            figure, axis = plt.subplots()
            axis.boxplot(values, labels=[str(int(level)) for level in levels], showfliers=False)
            axis.set(xlabel="Retention (%)", ylabel="Boundary fraction", title=f"Boundary richness — {class_name}")
            path = plot_dir / f"boundary_fraction_{safe_name}.png"
            figure.savefig(path, dpi=250, bbox_inches="tight")
            plt.close(figure)
            saved.append(path)
    if {"cluster_size", "selected_size"}.issubset(frame.columns):
        valid = frame.dropna(subset=["cluster_size", "selected_size"])
        if not valid.empty:
            figure, axis = plt.subplots()
            axis.scatter(valid["cluster_size"], valid["selected_size"], s=14, alpha=0.6)
            axis.set(xlabel="Cluster size", ylabel="Selected size", title="Selection budget by cluster")
            axis.grid(alpha=0.3)
            path = plot_dir / "selected_vs_cluster_size.png"
            figure.savefig(path, dpi=250, bbox_inches="tight")
            plt.close(figure)
            saved.append(path)
    return saved


def analyze_legacy_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    recursive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(input_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Legacy log directory not found: {source}")
    files = source.rglob("*") if recursive else source.iterdir()
    files = sorted(
        path
        for path in files
        if path.is_file() and path.suffix.lower() in {".txt", ".log", ".out"}
    )
    pruning_frames: list[pd.DataFrame] = []
    performance_rows: list[dict[str, Any]] = []
    for path in files:
        text_head = path.read_text(encoding="utf-8", errors="ignore")[:20000]
        if "Boundary Focus Score" in text_head or "pruning ratio is" in text_head:
            parsed = parse_pruning_log(path)
            if not parsed.empty:
                pruning_frames.append(parsed)
        if any(token in text_head.lower() for token in ("f1", "precision", "accuracy", "acc:")):
            row = parse_performance_log(path)
            if any(row[key] is not None for key in ("accuracy", "precision", "recall", "f1")):
                performance_rows.append(row)

    clusters = pd.concat(pruning_frames, ignore_index=True) if pruning_frames else pd.DataFrame()
    performance = pd.DataFrame(performance_rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output / "legacy_pruning_clusters.csv", index=False)
    performance.to_csv(output / "legacy_performance_runs.csv", index=False)
    if not clusters.empty:
        _cluster_summary(clusters).to_csv(output / "legacy_pruning_summary.csv", index=False)
        _save_plots(clusters, output)
    if not performance.empty:
        _performance_summary(performance).to_csv(output / "legacy_performance_summary.csv", index=False)
    return clusters, performance
