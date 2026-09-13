# Reproducibility protocol



## 1. Configure the VQ-VAE checkpoint exactly

Checkpoint tensor shapes depend on all of the following fields:

- `embedding_dim`
- `num_embeddings`
- `hidden_dims`
- `in_channels`


## 2. Choose and report the latent representation

Four representations are implemented:

- `encoding_indices`: flattened discrete code indices, matching the initial
  notebook implementation;
- `histogram`: normalized codebook-usage frequency, matching the later
  corrected notebook cells and recommended as the default;
- `quantized`: flattened quantized latent tensors;
- `pooled`: spatial mean of the quantized latent tensor.


## 3. Retention and class policy

Retention is defined as the fraction kept. The exact class target is

```text
target = min(N_class, max(1, ceil(retention × N_class)))
```

The target is proportionally allocated across K-means clusters with an exact
integer correction. All classes absent from `dataset.retention` are preserved
in full.


## 4. Seeds and deterministic behavior

Default seeds are:

```text
7, 21, 42, 128, 224
```

Python, NumPy, PyTorch, CUDA, K-means, data-loader generators, and selection
RNGs are seeded. CUDA kernels that do not have deterministic implementations
may still vary across devices or PyTorch/CUDA releases; the environment
versions are stored in each run.


