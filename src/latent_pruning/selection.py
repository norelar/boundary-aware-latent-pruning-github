"""Exact-budget K-means and local OC-SVM boundary selection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM

from latent_pruning.config import PruningConfig
from latent_pruning.features import preprocess_features

SelectorMode = Literal["proposed", "global_ocsvm", "clustered_random", "random"]


@dataclass(frozen=True)
class ClusterMetrics:
    cluster_id: int
    cluster_size: int
    selected_size: int
    target_size: int
    support_count: int
    boundary_count: int
    selected_boundary_count: int
    completion_count: int
    mean_abs_margin: float | None
    std_abs_margin: float | None
    selected_mean_abs_margin: float | None
    selected_std_abs_margin: float | None
    non_selected_mean_abs_margin: float | None
    non_selected_std_abs_margin: float | None
    cmc: float | None
    bfs: float | None
    cv: float | None
    used_ocsvm: bool

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


@dataclass
class SelectionResult:
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    cluster_labels: np.ndarray
    abs_margins: np.ndarray
    support_mask: np.ndarray
    boundary_mask: np.ndarray
    reasons: np.ndarray
    cluster_metrics: list[ClusterMetrics]

    def validate(self, sample_count: int, target_count: int) -> None:
        arrays = (
            self.selected_mask,
            self.cluster_labels,
            self.abs_margins,
            self.support_mask,
            self.boundary_mask,
            self.reasons,
        )
        if any(len(array) != sample_count for array in arrays):
            raise RuntimeError("Selection output arrays do not align with the input samples")
        if len(self.selected_indices) != target_count:
            raise RuntimeError(
                f"Selection returned {len(self.selected_indices)} samples; expected {target_count}"
            )
        if len(np.unique(self.selected_indices)) != len(self.selected_indices):
            raise RuntimeError("Selection contains duplicate sample indices")
        if not np.array_equal(np.flatnonzero(self.selected_mask), self.selected_indices):
            raise RuntimeError("selected_mask and selected_indices disagree")


def target_count(sample_count: int, retention: float) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    return min(sample_count, max(1, int(math.ceil(sample_count * retention))))


def allocate_exact_counts(cluster_sizes: Sequence[int], total_target: int) -> np.ndarray:
    """Proportionally allocate an exact integer budget with capacity constraints."""
    sizes = np.asarray(cluster_sizes, dtype=int)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < 0):
        raise ValueError("cluster_sizes must be a non-empty sequence of non-negative integers")
    available = int(sizes.sum())
    if available < 1:
        raise ValueError("At least one sample must be available")
    total_target = min(available, max(1, int(total_target)))
    raw = sizes.astype(float) / available * total_target
    allocated = np.minimum(np.floor(raw).astype(int), sizes)
    non_empty = np.flatnonzero(sizes)
    lower = np.zeros_like(sizes)
    if total_target >= len(non_empty):
        lower[non_empty] = 1
        allocated = np.maximum(allocated, lower)

    residual = raw - np.floor(raw)
    while int(allocated.sum()) > total_target:
        candidates = np.flatnonzero(allocated > lower)
        if candidates.size == 0:
            raise RuntimeError("Cannot reduce the cluster allocation to the target")
        order = candidates[np.lexsort((candidates, residual[candidates]))]
        allocated[order[0]] -= 1
    while int(allocated.sum()) < total_target:
        candidates = np.flatnonzero(allocated < sizes)
        if candidates.size == 0:
            raise RuntimeError("Cannot increase the cluster allocation to the target")
        # Largest remainder first; cluster id is a deterministic tie breaker.
        order = candidates[np.lexsort((candidates, -residual[candidates]))]
        allocated[order[0]] += 1
        residual[order[0]] = -1.0
        if np.all(residual[candidates] < 0):
            residual = raw - allocated
    return allocated


def _equal_stratified_sample(
    candidates: np.ndarray,
    margins: np.ndarray,
    count: int,
    strata: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or candidates.size == 0:
        return np.asarray([], dtype=int)
    count = min(int(count), int(candidates.size))
    ordered = candidates[np.argsort(margins[candidates], kind="stable")]
    bins = [part for part in np.array_split(ordered, min(strata, len(ordered))) if len(part)]
    chosen: list[int] = []
    remaining_bins = [part.copy() for part in bins]
    # Round-robin produces an approximately equal contribution from near-to-far levels.
    while len(chosen) < count and any(len(part) for part in remaining_bins):
        for index, part in enumerate(remaining_bins):
            if len(chosen) >= count:
                break
            if len(part) == 0:
                continue
            position = int(rng.integers(0, len(part)))
            chosen.append(int(part[position]))
            remaining_bins[index] = np.delete(part, position)
    return np.asarray(chosen, dtype=int)


def _fit_boundary_model(
    features: np.ndarray,
    config: PruningConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    count = len(features)
    if count >= 3:
        try:
            model = OneClassSVM(kernel="rbf", gamma=config.gamma, nu=config.nu)
            model.fit(features)
            margins = np.abs(model.decision_function(features).reshape(-1))
            support_mask = np.zeros(count, dtype=bool)
            support_mask[np.asarray(model.support_, dtype=int)] = True
            return margins.astype(float), support_mask, True
        except (ValueError, FloatingPointError):
            pass

    # Stable fallback for tiny or degenerate clusters: the farthest points from
    # the centroid receive the smallest pseudo-margin and are treated as support.
    distance = np.linalg.norm(features - features.mean(axis=0, keepdims=True), axis=1)
    margins = distance.max(initial=0.0) - distance
    support_count = max(1, int(math.ceil(config.nu * count)))
    support_mask = np.zeros(count, dtype=bool)
    support_mask[np.argsort(margins, kind="stable")[:support_count]] = True
    return margins.astype(float), support_mask, False


def _margin_summary(
    cluster_id: int,
    margins: np.ndarray,
    selected: np.ndarray,
    support: np.ndarray,
    boundary: np.ndarray,
    target: int,
    used_ocsvm: bool,
) -> ClusterMetrics:
    not_selected = ~selected

    def mean(values: np.ndarray) -> float | None:
        return float(np.mean(values)) if values.size else None

    def std(values: np.ndarray) -> float | None:
        return float(np.std(values)) if values.size else None

    all_mean = mean(margins)
    selected_mean = mean(margins[selected])
    cmc = selected_mean / all_mean if all_mean not in (None, 0.0) and selected_mean is not None else None
    bfs = 1.0 - cmc if cmc is not None else None
    cv = (
        float(np.std(margins[selected]) / selected_mean)
        if selected.sum() > 1 and selected_mean not in (None, 0.0)
        else None
    )
    selected_boundary = int(np.sum(selected & boundary))
    return ClusterMetrics(
        cluster_id=cluster_id,
        cluster_size=len(margins),
        selected_size=int(selected.sum()),
        target_size=int(target),
        support_count=int(support.sum()),
        boundary_count=int(boundary.sum()),
        selected_boundary_count=selected_boundary,
        completion_count=int(selected.sum()) - selected_boundary,
        mean_abs_margin=all_mean,
        std_abs_margin=std(margins),
        selected_mean_abs_margin=selected_mean,
        selected_std_abs_margin=std(margins[selected]),
        non_selected_mean_abs_margin=mean(margins[not_selected]),
        non_selected_std_abs_margin=std(margins[not_selected]),
        cmc=cmc,
        bfs=bfs,
        cv=cv,
        used_ocsvm=used_ocsvm,
    )


def _empty_margin_summary(cluster_id: int, cluster_size: int, selected_size: int) -> ClusterMetrics:
    return ClusterMetrics(
        cluster_id=cluster_id,
        cluster_size=cluster_size,
        selected_size=selected_size,
        target_size=selected_size,
        support_count=0,
        boundary_count=0,
        selected_boundary_count=0,
        completion_count=selected_size,
        mean_abs_margin=None,
        std_abs_margin=None,
        selected_mean_abs_margin=None,
        selected_std_abs_margin=None,
        non_selected_mean_abs_margin=None,
        non_selected_std_abs_margin=None,
        cmc=None,
        bfs=None,
        cv=None,
        used_ocsvm=False,
    )


def _select_boundary_cluster(
    features: np.ndarray,
    target: int,
    config: PruningConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    count = len(features)
    margins, support, used_ocsvm = _fit_boundary_model(features, config)
    if config.boundary_threshold is not None:
        adjacent = margins <= config.boundary_threshold
    else:
        threshold = float(np.quantile(margins, config.boundary_quantile))
        adjacent = margins <= threshold
    boundary = adjacent | support if config.include_support_vectors else adjacent

    selected = np.zeros(count, dtype=bool)
    reasons = np.full(count, "not_selected", dtype=object)
    priority = np.flatnonzero(boundary)
    priority = priority[np.argsort(margins[priority], kind="stable")]
    if len(priority) >= target:
        chosen = priority[:target]
        selected[chosen] = True
        reasons[chosen] = "boundary"
        return selected, margins, support, boundary, used_ocsvm

    selected[priority] = True
    reasons[priority] = "boundary"
    remaining = np.flatnonzero(~selected)
    rng = np.random.default_rng(seed)
    completion = _equal_stratified_sample(
        remaining,
        margins,
        target - int(selected.sum()),
        config.n_margin_strata,
        rng,
    )
    selected[completion] = True
    reasons[completion] = "stratified_completion"
    if int(selected.sum()) != target:
        raise RuntimeError("Stratified completion failed to satisfy the exact cluster budget")
    return selected, margins, support, boundary, used_ocsvm


def select_features(
    features: np.ndarray,
    retention: float,
    config: PruningConfig,
    seed: int,
    mode: SelectorMode = "proposed",
    preprocess: bool = True,
) -> SelectionResult:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"Expected a non-empty 2-D feature matrix, got {matrix.shape}")
    count = matrix.shape[0]
    target = target_count(count, retention)
    if preprocess and mode != "random":
        matrix = preprocess_features(matrix, config, seed)

    selected_mask = np.zeros(count, dtype=bool)
    cluster_labels = np.full(count, -1, dtype=int)
    margins = np.full(count, np.nan, dtype=float)
    support_mask = np.zeros(count, dtype=bool)
    boundary_mask = np.zeros(count, dtype=bool)
    reasons = np.full(count, "not_selected", dtype=object)
    metrics: list[ClusterMetrics] = []
    rng = np.random.default_rng(seed)

    if mode == "random":
        chosen = np.sort(rng.choice(count, size=target, replace=False))
        selected_mask[chosen] = True
        cluster_labels[:] = 0
        reasons[chosen] = "random"
        metrics.append(_empty_margin_summary(0, count, target))
    else:
        cluster_count = 1 if mode == "global_ocsvm" else min(config.n_clusters, count)
        if cluster_count == 1:
            labels = np.zeros(count, dtype=int)
        else:
            labels = KMeans(
                n_clusters=cluster_count,
                n_init=config.n_init,
                random_state=seed,
            ).fit_predict(matrix)
        cluster_labels[:] = labels
        cluster_ids = np.sort(np.unique(labels))
        sizes = [int(np.sum(labels == cluster_id)) for cluster_id in cluster_ids]
        targets = allocate_exact_counts(sizes, target)

        for position, (cluster_id, cluster_target) in enumerate(zip(cluster_ids, targets, strict=True)):
            global_indices = np.flatnonzero(labels == cluster_id)
            local_count = len(global_indices)
            if cluster_target == 0:
                metrics.append(_empty_margin_summary(int(cluster_id), local_count, 0))
                continue
            if mode == "clustered_random":
                local_selected = np.zeros(local_count, dtype=bool)
                chosen = rng.choice(local_count, size=int(cluster_target), replace=False)
                local_selected[chosen] = True
                selected_mask[global_indices[local_selected]] = True
                reasons[global_indices[local_selected]] = "clustered_random"
                metrics.append(
                    _empty_margin_summary(int(cluster_id), local_count, int(cluster_target))
                )
                continue

            local_selected, local_margin, local_support, local_boundary, used_ocsvm = _select_boundary_cluster(
                matrix[global_indices],
                int(cluster_target),
                config,
                seed + 1009 * (position + 1),
            )
            selected_mask[global_indices] = local_selected
            margins[global_indices] = local_margin
            support_mask[global_indices] = local_support
            boundary_mask[global_indices] = local_boundary
            local_reasons = np.full(local_count, "not_selected", dtype=object)
            local_reasons[local_selected & local_boundary] = "boundary"
            local_reasons[local_selected & ~local_boundary] = "stratified_completion"
            reasons[global_indices] = local_reasons
            metrics.append(
                _margin_summary(
                    int(cluster_id),
                    local_margin,
                    local_selected,
                    local_support,
                    local_boundary,
                    int(cluster_target),
                    used_ocsvm,
                )
            )

    selected_indices = np.flatnonzero(selected_mask)
    result = SelectionResult(
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        cluster_labels=cluster_labels,
        abs_margins=margins,
        support_mask=support_mask,
        boundary_mask=boundary_mask,
        reasons=reasons,
        cluster_metrics=metrics,
    )
    result.validate(count, target)
    return result
