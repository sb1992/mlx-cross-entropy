"""Core CCE operations — custom VJP wrappers for autograd integration.

Internal module. Public API is in __init__.py.
"""

import mlx.core as mx

from ._metal_backward import cce_backward_dE, cce_backward_full
from ._fused_forward import fused_cce_forward
from ._fused_backward import (
    fused_cce_backward_v3,
    fused_cce_backward_v3_full,
)
from ._reference import linear_cross_entropy_reference

_SENTINEL_BIAS = mx.zeros((0,))


# --- Fused MMA forward + v3 backward (dE only, best for LoRA/frozen head) ---

@mx.custom_function
def _cce_fused_v3(e, c, targets, bias):
    has_bias = bias.size > 0
    nll, lse, tile_max = fused_cce_forward(
        e, c, targets, bias=bias if has_bias else None, return_tile_max=True
    )
    return nll, lse, tile_max


@_cce_fused_v3.vjp
def _cce_fused_v3_vjp(primals, cotangents, outputs):
    e, c, targets, bias = primals
    d_nll, d_lse, d_tile_max = cotangents
    nll, lse, tile_max = outputs
    has_bias = bias.size > 0
    dE = fused_cce_backward_v3(
        e, c, lse, targets, d_nll, tile_max,
        bias=bias if has_bias else None,
    )
    return (
        dE,
        mx.zeros_like(c),
        mx.zeros(targets.shape, dtype=e.dtype),
        mx.zeros_like(bias),
    )


# --- Fused MMA forward + v3_full backward (dE + dC + dBias, full training) ---

@mx.custom_function
def _cce_fused_v3_full(e, c, targets, bias):
    has_bias = bias.size > 0
    nll, lse, tile_max = fused_cce_forward(
        e, c, targets, bias=bias if has_bias else None, return_tile_max=True
    )
    return nll, lse, tile_max


@_cce_fused_v3_full.vjp
def _cce_fused_v3_full_vjp(primals, cotangents, outputs):
    e, c, targets, bias = primals
    d_nll, d_lse, d_tile_max = cotangents
    nll, lse, tile_max = outputs
    has_bias = bias.size > 0
    dE, dC, dBias = fused_cce_backward_v3_full(
        e, c, lse, targets, d_nll, tile_max,
        bias=bias if has_bias else None,
    )
    return (
        dE,
        dC,
        mx.zeros(targets.shape, dtype=e.dtype),
        dBias if dBias is not None else mx.zeros_like(bias),
    )


# --- Chunked forward + matmul backward (fallback for non-aligned shapes) ---

@mx.custom_function
def _cce_chunked(e, c, targets, bias):
    has_bias = bias.size > 0
    B = e.shape[0]
    V = c.shape[0]

    target_logit = mx.sum(e.astype(mx.float32) * c[targets].astype(mx.float32), axis=-1)
    if has_bias:
        target_logit = target_logit + bias[targets].astype(mx.float32)

    chunk_size = min(4096, V)
    running_max = mx.full((B,), -1e38, dtype=mx.float32)
    running_sum = mx.zeros((B,), dtype=mx.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        logits_chunk = (e @ c[v_start:v_end].T).astype(mx.float32)
        if has_bias:
            logits_chunk = logits_chunk + bias[v_start:v_end].astype(mx.float32)[None, :]
        chunk_max = mx.max(logits_chunk, axis=-1)
        new_max = mx.maximum(running_max, chunk_max)
        running_sum = (
            running_sum * mx.exp(running_max - new_max)
            + mx.sum(mx.exp(logits_chunk - new_max[:, None]), axis=-1)
        )
        running_max = new_max
        mx.eval(running_max, running_sum)

    lse = running_max + mx.log(running_sum)
    nll = lse - target_logit
    return nll, lse


@_cce_chunked.vjp
def _cce_chunked_vjp(primals, cotangents, outputs):
    e, c, targets, bias = primals
    d_nll, d_lse = cotangents
    nll, lse = outputs
    has_bias = bias.size > 0
    dE = cce_backward_dE(
        e, c, lse, targets, d_nll,
        bias=bias if has_bias else None,
    )
    return (
        dE,
        mx.zeros_like(c),
        mx.zeros(targets.shape, dtype=e.dtype),
        mx.zeros_like(bias),
    )


@mx.custom_function
def _cce_chunked_full(e, c, targets, bias):
    has_bias = bias.size > 0
    B = e.shape[0]
    V = c.shape[0]

    target_logit = mx.sum(e.astype(mx.float32) * c[targets].astype(mx.float32), axis=-1)
    if has_bias:
        target_logit = target_logit + bias[targets].astype(mx.float32)

    chunk_size = min(4096, V)
    running_max = mx.full((B,), -1e38, dtype=mx.float32)
    running_sum = mx.zeros((B,), dtype=mx.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        logits_chunk = (e @ c[v_start:v_end].T).astype(mx.float32)
        if has_bias:
            logits_chunk = logits_chunk + bias[v_start:v_end].astype(mx.float32)[None, :]
        chunk_max = mx.max(logits_chunk, axis=-1)
        new_max = mx.maximum(running_max, chunk_max)
        running_sum = (
            running_sum * mx.exp(running_max - new_max)
            + mx.sum(mx.exp(logits_chunk - new_max[:, None]), axis=-1)
        )
        running_max = new_max
        mx.eval(running_max, running_sum)

    lse = running_max + mx.log(running_sum)
    nll = lse - target_logit
    return nll, lse


@_cce_chunked_full.vjp
def _cce_chunked_full_vjp(primals, cotangents, outputs):
    e, c, targets, bias = primals
    d_nll, d_lse = cotangents
    nll, lse = outputs
    has_bias = bias.size > 0
    dE, dC, dBias = cce_backward_full(
        e, c, lse, targets, d_nll,
        bias=bias if has_bias else None,
    )
    return (
        dE,
        dC,
        mx.zeros(targets.shape, dtype=e.dtype),
        dBias if dBias is not None else mx.zeros_like(bias),
    )


# --- Reduction helper ---

def _apply_reduction(per_sample, targets, orig_shape, reduction, ignore_index, dtype):
    valid = targets != ignore_index
    per_sample = per_sample * valid

    if reduction == "none":
        return per_sample.reshape(orig_shape)
    elif reduction == "sum":
        return per_sample.sum()
    elif reduction == "mean":
        count = mx.maximum(valid.sum(), 1)
        return per_sample.sum() / count
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


# --- Public implementation ---

def linear_cross_entropy_impl(
    e, c, targets,
    bias=None,
    reduction="mean",
    ignore_index=-100,
    shift=0,
    return_lse=False,
    compute_all_grads=False,
):
    if shift > 0:
        e = e[..., :-shift, :]
        targets = targets[..., shift:]

    orig_shape = targets.shape
    e = e.reshape(-1, e.shape[-1])
    targets = targets.reshape(-1)

    B, D = e.shape
    V = c.shape[0]
    bias_arg = bias if bias is not None else _SENTINEL_BIAS

    can_fuse = (B % 32 == 0) and (V % 32 == 0) and (D % 8 == 0)

    if can_fuse:
        if compute_all_grads:
            nll, lse, _tile_max = _cce_fused_v3_full(e, c, targets, bias_arg)
        else:
            nll, lse, _tile_max = _cce_fused_v3(e, c, targets, bias_arg)
    else:
        if compute_all_grads:
            nll, lse = _cce_chunked_full(e, c, targets, bias_arg)
        else:
            nll, lse = _cce_chunked(e, c, targets, bias_arg)

    loss = _apply_reduction(
        nll.astype(e.dtype), targets, orig_shape, reduction, ignore_index, e.dtype
    )

    if return_lse:
        return loss, lse
    return loss
