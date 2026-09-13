# Reproducibility protocol

## 1. Freeze the data definition

Keep the test directory unchanged across every experiment. If no explicit
validation directory is supplied, the code creates a 15% stratified validation
split from the original training directory for each seed. Pruning is then
performed only on the remaining fit subset. This prevents VQ-VAE feature
selection, K-means, and OC-SVM decisions from seeing validation or test images.

For strict isolation, the VQ-VAE checkpoint itself must be trained only on the
fit subset (or on an external corpus). A checkpoint trained on the complete
original training folder has indirectly seen the later classifier-validation
images even when their feature rows are excluded from K-means and OC-SVM.

Record any patient/subject grouping before using this repository. The generic
folder scanner cannot infer patient identifiers from image filenames. For
datasets containing multiple images per patient, create subject-disjoint
train/validation/test folders before running the pipeline.

## 2. Configure the VQ-VAE checkpoint exactly

Checkpoint tensor shapes depend on all of the following fields:

- `embedding_dim`
- `num_embeddings`
- `hidden_dims`
- `in_channels`

The supplied notebook settings are represented in the three YAML files.
Loading remains strict by default, so an architecture/checkpoint mismatch fails
immediately rather than silently skipping tensors.

## 3. Choose and report the latent representation

Four representations are implemented:

- `encoding_indices`: flattened discrete code indices, matching the initial
  notebook implementation;
- `histogram`: normalized codebook-usage frequency, matching the later
  corrected notebook cells and recommended as the default;
- `quantized`: flattened quantized latent tensors;
- `pooled`: spatial mean of the quantized latent tensor.

The representation is part of the feature-cache fingerprint and is written to
the resolved configuration for every run.

## 4. Retention and class policy

Retention is defined as the fraction kept. The exact class target is

```text
target = min(N_class, max(1, ceil(retention × N_class)))
```

The target is proportionally allocated across K-means clusters with an exact
integer correction. All classes absent from `dataset.retention` are preserved
in full.

For component ablations, the code applies a uniform 75% retention to each
configured majority class by default. `balanced_proposed` instead uses the
dataset-specific class map in the main configuration.

## 5. Seeds and deterministic behavior

Default seeds are:

```text
7, 21, 42, 128, 224
```

Python, NumPy, PyTorch, CUDA, K-means, data-loader generators, and selection
RNGs are seeded. CUDA kernels that do not have deterministic implementations
may still vary across devices or PyTorch/CUDA releases; the environment
versions are stored in each run.

## 6. Model selection and final evaluation

The best checkpoint is chosen using validation macro-F1. The test loader is not
called during training. After early stopping, the best validation checkpoint is
loaded and the held-out test set is evaluated exactly once.

Primary reporting files are:

- `runs.csv`: one row per seed;
- `summary_mean_std.csv`: mean, standard deviation, and count across seeds;
- `test_metrics.json`: seed-level test metrics;
- `per_class_metrics.csv`: class-specific precision, recall, F1, and support;
- `test_predictions.csv`: auditable probabilities and labels.
