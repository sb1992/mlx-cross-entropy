# mlx-cce

Memory-efficient [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

Standard cross-entropy materializes a `[batch, vocab]` logits matrix that dominates memory for large vocabularies (128K+ tokens). CCE fuses the linear projection and loss computation into compiled Metal kernels using simdgroup MMA, **never allocating the full logit matrix**.

Based on ["Cut Your Losses in Large-Vocabulary Language Models"](https://arxiv.org/abs/2411.09009) (Wijmans et al., 2024). See also Apple's [reference implementation](https://github.com/apple/ml-cross-entropy) (PyTorch/Triton).

## Install

```bash
pip install mlx-cce
```

This compiles the C++ and Metal kernels natively on your machine. Requires macOS with Apple Silicon and MLX >= 0.22.0.

From source:

```bash
git clone https://github.com/ShreyBhatia/mlx-cce.git
cd mlx-cce && pip install -e .
```

## Usage

```python
from mlx_cce import linear_cross_entropy

# Instead of:
#   logits = hidden_states @ classifier.T   # [B, V] — huge
#   loss = nn.losses.cross_entropy(logits, targets)

# Use:
loss = linear_cross_entropy(hidden_states, classifier_weight, targets)
```

Handles batched sequences too — `[B, T, D]` inputs are flattened automatically:

```python
loss = linear_cross_entropy(
    hidden_states,       # [B, T, D]
    classifier_weight,   # [V, D]
    targets,             # [B, T]
    ignore_index=-100,
)
```

### With mlx-lm

mlx-lm's trainer accepts a custom `loss` function — no fork needed:

```python
import mlx.core as mx
from mlx_cce import linear_cross_entropy
from mlx_lm.tuner.trainer import train

def cce_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    hidden = model.model(inputs)

    if model.args.tie_word_embeddings:
        c = model.model.embed_tokens.weight
    else:
        c = model.lm_head.weight
        # Dequantize if model is quantized
        if hasattr(model.lm_head, "scales"):
            c = mx.dequantize(
                c, model.lm_head.scales, model.lm_head.biases,
                model.lm_head.group_size, model.lm_head.bits,
            )

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    targets = mx.where(mask, targets, mx.full(targets.shape, -100))
    ntoks = mask.sum()

    loss = linear_cross_entropy(hidden, c, targets, reduction="sum") / ntoks
    return loss, ntoks

# train(model, optimizer, dataset, loss=cce_loss)
```

## Performance

B=1024, V=128256, D=2048, float16, M-series Apple Silicon:

| Pass | Standard CE | CCE | Ratio |
|------|------------|-----|-------|
| Forward | 36.3 ms | 39.7 ms | 1.09x |
| Backward | 20.2 ms | 16.6 ms | 0.82x |
| **Total** | **56.5 ms** | **57.1 ms** | **~1.0x** |

Peak memory is significantly lower since the `[1024, 128256]` logit matrix (~500 MB in float16) is never allocated.

## How it works

1. **Tiled MMA forward** — tiles `e @ c.T` into 32x32 blocks using `simdgroup_multiply_accumulate`. Each tile computes local max and sum-exp for numerically stable logsumexp without storing the full `[B, V]` result.

2. **Gradient filtering** — the backward pass reuses per-tile max values from the forward to skip tiles where the softmax contribution is below `2^{-12}`. For trained models this filters 75-95% of tiles.

3. **Atomic accumulation** — gradients for `dE` are accumulated via `atomic_fetch_add_explicit`, eliminating intermediate buffers.

## API

```python
linear_cross_entropy(e, c, targets, bias=None, reduction="mean", ignore_index=-100)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `e` | `mx.array` | required | Hidden states `[..., D]` |
| `c` | `mx.array` | required | Classifier weights `[V, D]` |
| `targets` | `mx.array` | required | Target indices `[...]` |
| `bias` | `mx.array` | `None` | Classifier bias `[V]` |
| `reduction` | `str` | `"mean"` | `"mean"`, `"sum"`, or `"none"` |
| `ignore_index` | `int` | `-100` | Target index to mask |

Lower-level access: `cce_loss(e, c, targets)` takes flat `[B, D]` inputs directly.

## Development

```bash
pip install -e ".[dev]"
pytest
python benchmarks/bench_synthetic.py
```

## Reference implementation

The `reference/` directory contains a pure Python + Metal JIT implementation of the same algorithm using `mx.fast.metal_kernel`. It's useful for understanding the approach and debugging but is not installed by pip — the compiled C++ kernels are ~1x faster and used by default.

## Citation

```bibtex
@article{wijmans2024cut,
  title={Cut Your Losses in Large-Vocabulary Language Models},
  author={Wijmans, Erik and Kollar, Thomas and Bisk, Yonatan},
  journal={arXiv preprint arXiv:2411.09009},
  year={2024}
}
```

## License

MIT
