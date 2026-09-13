# Output reference

## Pruning outputs

| File | Contents |
|---|---|
| `resolved_config.yaml` | Exact configuration used after CLI overrides |
| `selection_manifest.csv` | One row per candidate image, selected flag, cluster, margin, and reason |
| `class_summary.csv` | Original/selected class counts and effective retention |
| `cluster_metrics.csv` | OC-SVM support, boundary, CMC, BFS, CV, and margin statistics |
| `run_metadata.json` | Dataset, seed, totals, package versions, and environment |
| `dataset/` | Optional materialized `ImageFolder` tree |

Selection reasons are `boundary`, `stratified_completion`,
`preserved_full_class`, `random`, `clustered_random`, or `not_selected`.

Cluster Margin Compactness and Boundary Focus Score are recorded as

```text
CMC = mean(|margin| selected) / mean(|margin| cluster)
BFS = 1 - CMC
```

The coefficient of variation (`cv`) measures dispersion among selected absolute
margins. These are diagnostic scores, not classification metrics.

## Training outputs

| File | Contents |
|---|---|
| `best_model.pt` | Best validation macro-F1 checkpoint and metadata |
| `history.csv` | Per-epoch train/validation loss and metrics |
| `validation_metrics.json` | Metrics used for model selection audit |
| `test_metrics.json` | Single final held-out evaluation |
| `per_class_metrics.csv` | Per-class report |
| `confusion_matrix.csv/.png` | Exact values and publication-quality rendering |
| `validation_predictions.csv` | Validation probabilities |
| `test_predictions.csv` | Held-out probabilities |
| `*_manifest.csv` | Exact paths assigned to each split |

Metrics include accuracy, balanced accuracy, macro and weighted
precision/recall/F1, MCC, macro OvR ROC-AUC, NLL, and 15-bin ECE.

## Multi-seed outputs

`runs.csv` contains seed-level values. `summary_mean_std.csv` groups by dataset,
model, and variant and adds `_mean`, `_std`, and `_count` suffixes.

## Legacy log analysis

`analyze-legacy-logs` creates:

- `legacy_pruning_clusters.csv`
- `legacy_pruning_summary.csv`
- `legacy_performance_runs.csv`
- `legacy_performance_summary.csv`
- `plots/*.png`

The parser accepts the English/French messages and inconsistent filename
patterns present in the supplied notebooks.
