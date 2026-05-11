# mlx-cross-entropy

Memory-efficient cross-entropy loss for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. An MLX-native implementation of [Cut Cross-Entropy (CCE)](https://arxiv.org/abs/2411.09009) with compiled Metal kernels.

## The problem

As vocabularies grow (128K+ tokens in Llama 3, 256K in Gemma 2), the cross-entropy loss dominates training memory. Standard cross-entropy materializes a full `[batch × seq_len, vocab]` logits matrix:

| Model | Batch | Seq Len | Logit Matrix | CCE Overhead | Reduction |
|-------|-------|---------|--------------|--------------|-----------|
| Llama 3.2 1B | 4 | 512 | **1.1 GB** | 33 MB | **32x** |
| Llama 3.2 1B | 4 | 1024 | **2.1 GB** | 66 MB | **32x** |
| Llama 3.1 8B | 4 | 1024 | **2.1 GB** | 66 MB | **32x** |
| Gemma 2 2B | 4 | 512 | **2.1 GB** | 66 MB | **32x** |

CCE computes the loss **without ever allocating this matrix**, reducing the memory footprint of the loss computation by ~32x. This can be the difference between fitting a training run in memory or not.

## How it works

CCE fuses the linear projection (`e @ c.T`) and loss computation into a single Metal kernel:

1. **Tiled MMA** — the forward pass tiles the matmul into 32x32 blocks using Apple Silicon's `simdgroup_multiply_accumulate` hardware. Each tile computes local max and sum-exp for a numerically stable logsumexp, without ever storing the full result.

2. **Gradient filtering** — the backward pass reuses per-tile max values from the forward to skip tiles where `softmax < 2^{-12}`. For trained models this filters **75-95% of vocabulary tiles**, making the backward pass faster than standard CE.

3. **Atomic accumulation** — gradients are accumulated directly via `atomic_fetch_add_explicit`, eliminating intermediate `[batch, num_tiles, D]` buffers.

Based on ["Cut Your Losses in Large-Vocabulary Language Models"](https://arxiv.org/abs/2411.09009) (Wijmans et al., ICLR 2025). See Apple's [reference implementation](https://github.com/apple/ml-cross-entropy) for PyTorch/Triton.

## Install

```bash
pip install mlx-cross-entropy
```

Compiles C++ and Metal kernels natively. Requires macOS with Apple Silicon and MLX >= 0.22.0.

From source:

```bash
git clone https://github.com/sb1992/mlx-cross-entropy.git
cd mlx-cross-entropy && pip install -e .
```

## Usage

```python
from mlx_cce import linear_cross_entropy

# Standard (materializes huge logit matrix):
#   logits = hidden_states @ classifier.T   # [B, V] — dominates memory
#   loss = nn.losses.cross_entropy(logits, targets)

# CCE (never allocates logits):
loss = linear_cross_entropy(hidden_states, classifier_weight, targets)
```

Handles batched sequences — `[B, T, D]` inputs are flattened automatically:

```python
loss = linear_cross_entropy(
    hidden_states,       # [B, T, D]
    classifier_weight,   # [V, D]
    targets,             # [B, T]
    ignore_index=-100,
)
```

### With mlx-lm

mlx-lm's trainer accepts a custom `loss` callable — no fork needed:

```python
import mlx.core as mx
from mlx_cce import linear_cross_entropy

def cce_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Get hidden states before the output head
    hidden = model.model(inputs)

    # Get classifier weights (handle both tied and separate lm_head)
    if model.args.tie_word_embeddings:
        embed = model.model.embed_tokens
    else:
        embed = model.lm_head

    # Apply sequence mask
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    targets = mx.where(mask, targets, mx.full(targets.shape, -100))
    ntoks = mask.sum()

    # CCE reads quantized weights directly — no mx.dequantize() needed
    if hasattr(embed, "scales"):
        loss = linear_cross_entropy(
            hidden, embed.weight, targets, reduction="sum",
            scales=embed.scales, biases=embed.biases,
            group_size=embed.group_size, bits=embed.bits,
        ) / ntoks
    else:
        loss = linear_cross_entropy(
            hidden, embed.weight, targets, reduction="sum",
        ) / ntoks
    return loss, ntoks

# from mlx_lm.tuner.trainer import train
# train(model, optimizer, dataset, loss=cce_loss)
```

### Computing related quantities

`linear_cross_entropy` with `reduction="none"` returns per-token negative log-likelihoods, useful for:

```python
nll = linear_cross_entropy(e, c, targets, reduction="none")  # [B] or [B, T]

# Perplexity
ppl = mx.exp(mx.mean(nll))

# DPO-style preference loss
dpo_loss = -mx.log(mx.sigmoid(nll_rejected.sum() - nll_preferred.sum()))
```

## Performance

Isolated kernel benchmarks (B=1024, V=128256, D=2048, M-series):

| Pass | Standard CE | CCE | |
|------|------------|-----|---|
| Forward | 36.3 ms | 39.7 ms | 1.09x |
| Backward | 20.2 ms | 16.6 ms | **0.82x** |
| **Total** | **56.5 ms** | **57.1 ms** | ~1.0x |

CCE backward is faster due to gradient filtering (skipping 75-95% of tiles). Forward is slightly slower due to tiled reduction overhead. Total throughput is comparable, with dramatically less memory.

**Quantized models:** CCE reads packed 4-bit weights directly — no `mx.dequantize()` needed. Pass `scales`, `biases`, `group_size`, and `bits` to `linear_cross_entropy` and the Metal kernel dequantizes on-the-fly per tile, saving ~500 MB of memory that a full dequantization would allocate.

## API

```python
linear_cross_entropy(e, c, targets, bias=None, reduction="mean", ignore_index=-100,
                     scales=None, biases=None, group_size=64, bits=4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `e` | `mx.array` | required | Hidden states `[..., D]` |
| `c` | `mx.array` | required | Classifier weights `[V, D]` or packed `[V, D/8]` (uint32) |
| `targets` | `mx.array` | required | Target indices `[...]` |
| `bias` | `mx.array` | `None` | Classifier bias `[V]` |
| `reduction` | `str` | `"mean"` | `"mean"`, `"sum"`, or `"none"` |
| `ignore_index` | `int` | `-100` | Target index to mask |
| `scales` | `mx.array` | `None` | Quantization scales `[V, D/group_size]` |
| `biases` | `mx.array` | `None` | Quantization biases `[V, D/group_size]` |
| `group_size` | `int` | `64` | Quantization group size (32, 64, or 128) |
| `bits` | `int` | `4` | Quantization bit width (only 4 supported) |

Lower-level: `cce_loss(e, c, targets)` takes flat `[B, D]` inputs directly.

## Development

```bash
git clone https://github.com/sb1992/mlx-cross-entropy.git
cd mlx-cross-entropy
pip install -e ".[dev]"
pytest
```

## Reference implementation

The `reference/` directory contains a pure Python implementation using `mx.fast.metal_kernel` (Metal JIT). Useful for understanding the algorithm and debugging, but not installed — the compiled C++ kernels are used by default.

## Citation

```bibtex
@inproceedings{wijmans2025cut,
  author    = {Erik Wijmans and Brody Huval and Alexander Hertzberg and
               Vladlen Koltun and Philipp Kr\"ahenb\"uhl},
  title     = {Cut Your Losses in Large-Vocabulary Language Models},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
}
```

## License

MIT
