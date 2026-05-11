"""mlx-cce: Memory-efficient Cut Cross-Entropy for Apple Silicon.

Implements the fused cross-entropy approach from "Cut Your Losses in
Large-Vocabulary Language Models" (Wijmans et al., 2024) using Metal
simdgroup MMA on Apple Silicon via MLX.

Basic usage:
    from mlx_cce import linear_cross_entropy

    loss = linear_cross_entropy(hidden_states, classifier_weight, targets)
"""

__version__ = "0.1.0"

import mlx.core as mx
import mlx.nn as nn

from ._ops import linear_cross_entropy_impl
from ._reference import linear_cross_entropy_reference


def linear_cross_entropy(
    e,
    c,
    targets,
    bias=None,
    reduction="mean",
    ignore_index=-100,
    shift=0,
    return_lse=False,
    impl="auto",
    compute_all_grads=False,
):
    """Memory-efficient cross-entropy that never materializes [B, V] logits.

    Replaces the standard pattern of ``logits = e @ c.T`` followed by
    ``cross_entropy(logits, targets)`` with a fused kernel that computes
    the loss directly from embeddings and classifier weights.

    Args:
        e: Hidden states, shape ``(..., D)``
        c: Classifier weight matrix, shape ``(V, D)``
        targets: Target indices, shape ``(...)``
        bias: Optional classifier bias, shape ``(V,)``
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``
        ignore_index: Target value to ignore (default: -100)
        shift: Shift targets by this many positions (for causal LM)
        return_lse: If True, also return log-sum-exp values
        impl: ``"auto"`` (default), ``"fused"``, ``"chunked"``, or ``"reference"``
        compute_all_grads: If True, compute gradients for ``c`` and ``bias``
            too (needed for full training). If False (default), only compute
            gradient for ``e`` (sufficient for LoRA / frozen-head fine-tuning).

    Returns:
        Loss scalar (or per-sample if reduction="none").
        If return_lse=True, returns (loss, lse) tuple.
    """
    if impl == "reference":
        return linear_cross_entropy_reference(
            e, c, targets,
            bias=bias,
            reduction=reduction,
            ignore_index=ignore_index,
            shift=shift,
            return_lse=return_lse,
        )

    if impl == "auto":
        e_flat = e.reshape(-1, e.shape[-1])
        B, D = e_flat.shape
        V = c.shape[0]
        if B % 32 == 0 and V % 32 == 0 and D % 8 == 0:
            impl = "fused"
        else:
            impl = "chunked"

    return linear_cross_entropy_impl(
        e, c, targets,
        bias=bias,
        reduction=reduction,
        ignore_index=ignore_index,
        shift=shift,
        return_lse=return_lse,
        compute_all_grads=compute_all_grads,
    )


class LinearCrossEntropy(nn.Module):
    """Drop-in module for memory-efficient cross-entropy loss.

    Usage::

        criterion = LinearCrossEntropy(reduction="mean")
        loss = criterion(hidden_states, classifier_weight, targets)
    """

    def __init__(
        self,
        reduction="mean",
        ignore_index=-100,
        shift=0,
        compute_all_grads=False,
    ):
        super().__init__()
        self._reduction = reduction
        self._ignore_index = ignore_index
        self._shift = shift
        self._compute_all_grads = compute_all_grads

    def __call__(self, e, c, targets, bias=None):
        return linear_cross_entropy(
            e, c, targets,
            bias=bias,
            reduction=self._reduction,
            ignore_index=self._ignore_index,
            shift=self._shift,
            compute_all_grads=self._compute_all_grads,
        )


__all__ = [
    "linear_cross_entropy",
    "LinearCrossEntropy",
    "linear_cross_entropy_reference",
]
