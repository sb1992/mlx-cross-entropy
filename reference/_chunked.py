import mlx.core as mx


def linear_cross_entropy_chunked(
    e,
    c,
    targets,
    bias=None,
    reduction="mean",
    ignore_index=-100,
    shift=0,
    return_lse=False,
    chunk_size=4096,
):
    if shift > 0:
        e = e[..., :-shift, :]
        targets = targets[..., shift:]

    orig_shape = targets.shape
    e = e.reshape(-1, e.shape[-1])
    targets = targets.reshape(-1)
    B = targets.shape[0]
    V = c.shape[0]

    # Gather correct-token logits: O(B*D) memory, not O(B*V)
    target_logits = mx.sum(e * c[targets], axis=-1)
    if bias is not None:
        target_logits = target_logits + bias[targets]

    # Online logsumexp across vocab chunks
    running_max = mx.full((B,), -1e38, dtype=mx.float32)
    running_sum = mx.zeros((B,), dtype=mx.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        c_chunk = c[v_start:v_end]
        logits_chunk = e @ c_chunk.T  # [B, chunk]
        if bias is not None:
            logits_chunk = logits_chunk + bias[v_start:v_end][None, :]

        logits_chunk = logits_chunk.astype(mx.float32)
        chunk_max = mx.max(logits_chunk, axis=-1)  # [B]

        new_max = mx.maximum(running_max, chunk_max)
        running_sum = (
            running_sum * mx.exp(running_max - new_max)
            + mx.sum(mx.exp(logits_chunk - new_max[:, None]), axis=-1)
        )
        running_max = new_max

        mx.eval(running_max, running_sum)

    lse = running_max + mx.log(running_sum)
    per_sample = (lse - target_logits.astype(mx.float32)).astype(e.dtype)

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
        return loss, lse
    return loss
