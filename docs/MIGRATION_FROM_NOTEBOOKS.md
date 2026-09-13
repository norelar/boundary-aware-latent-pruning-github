# Migration from the supplied notebooks

| Supplied source | Canonical repository code |
|---|---|
| COVID-19 notebook | `configs/covid19.yaml`, `pruning.py`, `training.py` |
| HAM10000 notebook | `configs/ham10000.yaml`, shared package modules |
| OASIS notebook | `configs/oasis.yaml`, shared package modules |
| Log Files Analysis notebook | `analysis/legacy_logs.py` |
| Ablation Python script | `ablation.py`, `configs/ablation_variants.yaml` |

The legacy notebooks are not bundled in the distributed repository. They
contained duplicated definitions, Colab-specific paths, and partially executed
exploratory cells. The modules listed above are their tested replacements.

## Corrections made during modularization

1. COVID-19, OASIS, and HAM configurations are isolated. In the concatenated
   ablation file, later variables overwrote earlier dataset settings.
2. Retention is explicitly represented as a fraction kept. Original variables
   named `prune_ratio` were passed as `portion=n/100`, meaning retention.
3. Exact target counts replace independent per-cluster flooring and incomplete
   margin-stratum filling.
4. Support vectors and boundary-adjacent samples are both configurable priority
   sources, consistent with the method description.
5. Non-pruned classes are included automatically in every materialized set.
6. VQ-VAE feature extraction is batched and cached; paths and rows are validated
   before selection.
7. The classifier is selected on validation macro-F1. The notebooks repeatedly
   evaluated the test loader during training and sometimes saved the best test
   checkpoint, which creates optimistic test estimates.
8. Class labels are always ordered, rather than stored in Python sets, so
   confusion-matrix axes remain correct.
9. Logs are structured as CSV/JSON. A compatible parser remains available for
   older text logs.
10. Every seed writes its exact split and selection manifests.

These corrections preserve the experimental components while making the
reported protocol auditable and suitable for a public research repository.
