"""Forward correctness: CCE vs nn.losses.cross_entropy."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_cce import linear_cross_entropy, cce_loss


def _inputs(B, D, V, seed=42):
    mx.random.seed(seed)
    e = mx.random.normal((B, D))
    c = mx.random.normal((V, D))
    targets = mx.random.randint(0, V, (B,))
    mx.eval(e, c, targets)
    return e, c, targets


def _ref_loss(e, c, targets, reduction="mean", ignore_index=-100):
    logits = e @ c.T
    return nn.losses.cross_entropy(logits, targets, reduction=reduction)


SHAPES = [
    (32, 64, 1024),
    (64, 128, 4096),
    (128, 256, 32000),
    (32, 2048, 128256),
]


class TestForward:
    @pytest.mark.parametrize("B,D,V", SHAPES)
    def test_loss_matches_reference(self, B, D, V):
        e, c, targets = _inputs(B, D, V)
        ref = _ref_loss(e, c, targets)
        got = linear_cross_entropy(e, c, targets)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-3)

    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    def test_reduction_modes(self, reduction):
        e, c, targets = _inputs(32, 64, 1024)
        ref = _ref_loss(e, c, targets, reduction=reduction)
        got = linear_cross_entropy(e, c, targets, reduction=reduction)
        mx.eval(ref, got)
        np.testing.assert_allclose(np.array(got), np.array(ref), rtol=1e-2, atol=1e-3)

    def test_ignore_index(self):
        e, c, targets = _inputs(32, 64, 1024)
        targets_masked = mx.where(
            mx.arange(32) < 16, targets, mx.full((32,), -100)
        )
        mx.eval(targets_masked)
        logits = e @ c.T
        ce_per = nn.losses.cross_entropy(logits, targets, reduction="none")
        mx.eval(ce_per)
        ref = mx.sum(ce_per[:16]).item() / 16.0

        got = linear_cross_entropy(e, c, targets_masked, ignore_index=-100)
        mx.eval(got)
        np.testing.assert_allclose(got.item(), ref, rtol=1e-2)

    def test_ignore_index_none(self):
        e, c, targets = _inputs(32, 64, 1024)
        ref = _ref_loss(e, c, targets)
        got = cce_loss(e, c, targets, ignore_index=None)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-2)

    def test_bias(self):
        e, c, targets = _inputs(32, 64, 1024)
        bias = mx.random.normal((1024,))
        mx.eval(bias)
        logits = e @ c.T + bias
        ref = nn.losses.cross_entropy(logits, targets, reduction="mean")
        got = linear_cross_entropy(e, c, targets, bias=bias)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-3)

    def test_batched_sequence(self):
        B, T, D, V = 4, 128, 256, 4096
        mx.random.seed(42)
        e = mx.random.normal((B, T, D))
        c = mx.random.normal((V, D))
        targets = mx.random.randint(0, V, (B, T))
        mx.eval(e, c, targets)

        logits = e.reshape(B * T, D) @ c.T
        ref = nn.losses.cross_entropy(
            logits, targets.reshape(-1), reduction="mean"
        )
        got = linear_cross_entropy(e, c, targets)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-3)

    def test_cce_loss_matches_cut_cross_entropy(self):
        e, c, targets = _inputs(32, 64, 1024)
        from mlx_cce import cut_cross_entropy
        got_raw = cut_cross_entropy(e, c, targets.astype(mx.uint32))
        got_py = cce_loss(e, c, targets)
        mx.eval(got_raw, got_py)
        np.testing.assert_allclose(got_raw.item(), got_py.item(), rtol=1e-5)

    def test_llm_scale(self):
        e, c, targets = _inputs(32, 2048, 128256)
        ref = _ref_loss(e, c, targets)
        got = linear_cross_entropy(e, c, targets)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-3)

    @pytest.mark.parametrize("B,D,V,gs", [
        (32, 64, 1024, 32),
        (32, 128, 4096, 64),
        (64, 256, 32000, 128),
    ])
    def test_quantized_matches_dequantized(self, B, D, V, gs):
        mx.random.seed(42)
        e = mx.random.normal((B, D)).astype(mx.float16)
        c_full = mx.random.normal((V, D)).astype(mx.float16)
        targets = mx.random.randint(0, V, (B,))
        c_weight, c_scales, c_biases = mx.quantize(c_full, group_size=gs, bits=4)
        c_deq = mx.dequantize(c_weight, c_scales, c_biases, gs, 4)
        mx.eval(e, targets, c_weight, c_scales, c_biases, c_deq)

        ref = linear_cross_entropy(e, c_deq, targets)
        got = linear_cross_entropy(e, c_weight, targets,
                                   scales=c_scales, biases=c_biases,
                                   group_size=gs, bits=4)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-4)

    def test_quantized_gradient(self):
        B, D, V, gs = 32, 128, 1024, 64
        mx.random.seed(42)
        e = mx.random.normal((B, D)).astype(mx.float16)
        c_full = mx.random.normal((V, D)).astype(mx.float16)
        targets = mx.random.randint(0, V, (B,))
        c_weight, c_scales, c_biases = mx.quantize(c_full, group_size=gs, bits=4)
        c_deq = mx.dequantize(c_weight, c_scales, c_biases, gs, 4)
        mx.eval(e, targets, c_weight, c_scales, c_biases, c_deq)

        grad_ref = mx.grad(lambda x: linear_cross_entropy(x, c_deq, targets))(e)
        grad_q = mx.grad(lambda x: linear_cross_entropy(
            x, c_weight, targets, scales=c_scales, biases=c_biases,
            group_size=gs, bits=4))(e)
        mx.eval(grad_ref, grad_q)

        rel = (mx.max(mx.abs(grad_ref - grad_q)) / mx.max(mx.abs(grad_ref))).item()
        assert rel < 0.01, f"Gradient relative error {rel:.6f} > 0.01"

    def test_quantized_ignore_index(self):
        B, D, V, gs = 32, 128, 1024, 64
        mx.random.seed(42)
        e = mx.random.normal((B, D)).astype(mx.float16)
        c_full = mx.random.normal((V, D)).astype(mx.float16)
        targets = mx.random.randint(0, V, (B,))
        targets_masked = mx.where(mx.arange(B) < 16, targets, mx.full((B,), -100))
        c_weight, c_scales, c_biases = mx.quantize(c_full, group_size=gs, bits=4)
        c_deq = mx.dequantize(c_weight, c_scales, c_biases, gs, 4)
        mx.eval(e, targets_masked, c_weight, c_scales, c_biases, c_deq)

        ref = linear_cross_entropy(e, c_deq, targets_masked, ignore_index=-100)
        got = linear_cross_entropy(e, c_weight, targets_masked, ignore_index=-100,
                                   scales=c_scales, biases=c_biases, group_size=gs, bits=4)
        mx.eval(ref, got)
        np.testing.assert_allclose(got.item(), ref.item(), rtol=1e-4)
