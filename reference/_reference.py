import mlx.core as mx
import mlx.nn as nn


def linear_cross_entropy_reference(
    e,
    c,
    targets,
    bias=None,
    reduction="mean",
    ignore_index=-100,
    shift=0,
    return_lse=False,
):
    if shift > 0:
        e = e[..., :-shift, :]
        targets = targets[..., shift:]

    orig_shape = targets.shape
    e = e.reshape(-1, e.shape[-1])
    targets = targets.reshape(-1)
    B = targets.shape[0]

    logits = e @ c.T
    if bias is not None:
        logits = logits + bias[None, :]

    per_sample = nn.losses.cross_entropy(logits, targets, reduction="none")

    if return_lse:
        log_sum_exp = mx.logsumexp(logits, axis=-1)

    valid = targets != ignore_index
    per_sample = per_sample * valid

    if reduction == "none":
        loss = per_sample.reshape(orig_shape)
    elif reduction == "sum":
        loss = per_sample.sum()
    elif reduction == "mean":
        count = mx.maximum(valid.sum(), 1)
        loss = per_sample.sum() / count
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    if return_lse:
        return loss, log_sum_exp
    return loss
