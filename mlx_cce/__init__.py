"""mlx-cce: Memory-efficient Cut Cross-Entropy for Apple Silicon.

Implements the approach from "Cut Your Losses in Large-Vocabulary Language
Models" (Wijmans et al., 2024) using compiled Metal kernels with simdgroup
MMA on Apple Silicon.

    from mlx_cce import linear_cross_entropy
    loss = linear_cross_entropy(hidden_states, classifier_weight, targets)
"""

__version__ = "0.1.0"

import mlx.core as mx
from ._ext import cut_cross_entropy, cce_forward_raw, cce_backward_raw

__all__ = ["linear_cross_entropy", "cce_loss", "cut_cross_entropy"]

_SENTINEL_BIAS = mx.array([])


@mx.custom_function
def _cce_fused(e, c, targets, bias):
    has_bias = bias.size > 0
    results = cce_forward_raw(e, c, targets, bias=bias if has_bias else None)
    tile_max_flat, tile_sum_exp_flat, neg_target_logit, lse = results

    B = e.shape[0]
    V = c.shape[0]
    num_v_tiles = (V + 31) // 32

    tile_max_2d = tile_max_flat.reshape(B, num_v_tiles)
    return neg_target_logit + lse, lse, tile_max_2d


@_cce_fused.vjp
def _cce_fused_vjp(primals, cotangents, outputs):
    e, c, targets, bias = primals
    d_nll, d_lse, d_tile_max = cotangents
    nll, lse, tile_max = outputs
    has_bias = bias.size > 0
    dE = cce_backward_raw(
        e, c, lse, targets, d_nll,
        tile_max.reshape(-1),
        bias=bias if has_bias else None,
    )
    return (
        dE,
        mx.zeros_like(c),
        mx.zeros(targets.shape, dtype=e.dtype),
        mx.zeros_like(bias),
    )


def cce_loss(e, c, targets, bias=None, reduction="mean", ignore_index=-100):
    B, D = e.shape
    V = c.shape[0]
    bias_arg = bias if bias is not None else _SENTINEL_BIAS

    if ignore_index is not None:
        valid = mx.not_equal(targets, mx.array(ignore_index))
        safe_targets = mx.where(valid, targets, mx.zeros_like(targets)).astype(mx.uint32)
    else:
        valid = mx.ones((B,), dtype=mx.bool_)
        safe_targets = targets.astype(mx.uint32)

    nll, lse, tile_max = _cce_fused(e, c, safe_targets, bias_arg)
    valid_f = valid.astype(mx.float32)

    if reduction == "none":
        return nll * valid_f
    elif reduction == "sum":
        return mx.sum(nll * valid_f)
    elif reduction == "mean":
        count = mx.maximum(mx.sum(valid), mx.array(1))
        return mx.sum(nll * valid_f) / count.astype(mx.float32)
    raise ValueError(f"Unknown reduction: {reduction}")


def linear_cross_entropy(
    e, c, targets, bias=None, reduction="mean", ignore_index=-100
):
    """Memory-efficient cross-entropy that never materializes [B, V] logits.

    Args:
        e: Hidden states [B, D] or [B, T, D] (flattened automatically)
        c: Classifier weights [V, D]
        targets: Target indices [B] or [B, T]
        bias: Optional classifier bias [V]
        reduction: "mean", "sum", or "none"
        ignore_index: Target value to mask (-100 default)

    Returns:
        Loss scalar (or per-sample if reduction="none")
    """
    orig_shape = e.shape[:-1]
    D = e.shape[-1]
    e_flat = e.reshape(-1, D)
    targets_flat = targets.reshape(-1)

    loss = cce_loss(
        e_flat, c, targets_flat,
        bias=bias, reduction=reduction, ignore_index=ignore_index,
    )

    if reduction == "none":
        return loss.reshape(orig_shape)
    return loss
