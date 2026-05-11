"""Gradient correctness: CCE backward vs standard cross-entropy backward."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_cce import linear_cross_entropy


def _inputs(B, D, V, seed=42):
    mx.random.seed(seed)
    e = mx.random.normal((B, D))
    c = mx.random.normal((V, D))
    targets = mx.random.randint(0, V, (B,))
    mx.eval(e, c, targets)
    return e, c, targets


def _cosine(a, b):
    a, b = np.array(a).ravel(), np.array(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


SHAPES = [
    (32, 64, 1024),
    (64, 128, 4096),
    (128, 256, 32000),
    (32, 2048, 128256),
]


class TestGradientDE:
    @pytest.mark.parametrize("B,D,V", SHAPES)
    def test_dE_cosine(self, B, D, V):
        e, c, targets = _inputs(B, D, V)

        ref_fn = lambda e_: nn.losses.cross_entropy(
            e_ @ c.T, targets, reduction="mean"
        )
        cce_fn = lambda e_: linear_cross_entropy(e_, c, targets)

        _, ref_g = mx.value_and_grad(ref_fn)(e)
        _, cce_g = mx.value_and_grad(cce_fn)(e)
        mx.eval(ref_g, cce_g)
        assert _cosine(ref_g, cce_g) > 0.9999

    def test_dE_ignore_index(self):
        e, c, targets = _inputs(32, 64, 1024)
        targets_masked = mx.where(
            mx.arange(32) < 16, targets, mx.full((32,), -100)
        )
        mx.eval(targets_masked)

        cce_fn = lambda e_: linear_cross_entropy(
            e_, c, targets_masked, ignore_index=-100
        )
        _, g = mx.value_and_grad(cce_fn)(e)
        mx.eval(g)
        masked_grad = np.array(g)[16:]
        assert np.allclose(masked_grad, 0, atol=1e-6)

    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    def test_dE_reduction(self, reduction):
        e, c, targets = _inputs(32, 64, 1024)

        ref_fn = lambda e_: nn.losses.cross_entropy(
            e_ @ c.T, targets, reduction=reduction
        )
        cce_fn = lambda e_: linear_cross_entropy(
            e_, c, targets, reduction=reduction
        )
        if reduction == "none":
            ref_fn_s = lambda e_: ref_fn(e_).sum()
            cce_fn_s = lambda e_: cce_fn(e_).sum()
        else:
            ref_fn_s, cce_fn_s = ref_fn, cce_fn

        _, ref_g = mx.value_and_grad(ref_fn_s)(e)
        _, cce_g = mx.value_and_grad(cce_fn_s)(e)
        mx.eval(ref_g, cce_g)
        assert _cosine(ref_g, cce_g) > 0.9999

    def test_dE_with_bias(self):
        e, c, targets = _inputs(32, 64, 1024)
        bias = mx.random.normal((1024,))
        mx.eval(bias)

        ref_fn = lambda e_: nn.losses.cross_entropy(
            e_ @ c.T + bias, targets, reduction="mean"
        )
        cce_fn = lambda e_: linear_cross_entropy(e_, c, targets, bias=bias)

        _, ref_g = mx.value_and_grad(ref_fn)(e)
        _, cce_g = mx.value_and_grad(cce_fn)(e)
        mx.eval(ref_g, cce_g)
        assert _cosine(ref_g, cce_g) > 0.999

    def test_dE_batched_sequence(self):
        B, T, D, V = 4, 128, 256, 4096
        mx.random.seed(42)
        e = mx.random.normal((B, T, D))
        c = mx.random.normal((V, D))
        targets = mx.random.randint(0, V, (B, T))
        mx.eval(e, c, targets)

        ref_fn = lambda e_: nn.losses.cross_entropy(
            e_.reshape(B * T, D) @ c.T, targets.reshape(-1), reduction="mean"
        )
        cce_fn = lambda e_: linear_cross_entropy(e_, c, targets)

        _, ref_g = mx.value_and_grad(ref_fn)(e)
        _, cce_g = mx.value_and_grad(cce_fn)(e)
        mx.eval(ref_g, cce_g)
        assert _cosine(ref_g, cce_g) > 0.9999
