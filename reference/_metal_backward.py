"""Backward pass for CCE — computes dE (and optionally dC, dBias).

Strategies:
  - "matmul": single e@c.T matmul, autograd-friendly (used by VJP during training)
  - "metal": fused Metal kernel — computes dot products, softmax, and gradient
    accumulation in a single pass per V-tile. Never materializes the [B, V]
    logits/softmax matrix. Uses partial outputs reduced across V-tiles.
  - "filtered": chunked with per-chunk softmax filtering — skips gradient computation
    for vocab chunks where all softmax values are below precision threshold.
  - "chunked": loop with mx.eval per chunk, memory-optimal (standalone use only —
    mx.eval has no effect inside autograd VJP context)
"""

import math
import mlx.core as mx

FILTER_LOG_EPS = math.log(2**-12)

# ---------------------------------------------------------------------------
# Fused Metal backward kernel
# Grid: 2D — (B * BLOCK_SIZE, V_TILES, 1)
# Each threadgroup handles one (token, V_tile) pair.
# For each vocab entry in the tile:
#   1. Compute dot(e[b], c[v]) using SIMD reduction over D
#   2. Compute weight = d_nll[b] * exp(dot - lse[b]) - (v==target)*d_nll[b]
#   3. Accumulate dE[b, d] += weight * c[v, d] per lane
# After the V loop, reduce across SIMD groups and write partial output.
# Python reduces partial_dE[B, V_TILES, D] → dE[B, D] by summing axis=1.
# ---------------------------------------------------------------------------

CCE_BACKWARD_DE_SOURCE = """
    uint b = threadgroup_position_in_grid.x;
    uint v_tile_id = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint num_simd_groups = simdgroups_per_threadgroup;
    uint block_size = threads_per_threadgroup.x;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint B = e_shape[0];

    uint num_v_tiles = (V + TILE_V - 1) / TILE_V;
    uint v_tile_start = v_tile_id * TILE_V;
    uint v_tile_end = min(v_tile_start + TILE_V, V);
    uint tile_size = v_tile_end - v_tile_start;

    float d_nll_b = d_nll_val[b];
    float lse_b = lse[b];
    uint target_b = targets[b];

    threadgroup float shared_e[4096];
    for (uint d = tid; d < D; d += block_size) {
        shared_e[d] = (float)e[b * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Split V-tile among SIMD groups for maximum parallelism
    uint entries_per_group = (tile_size + num_simd_groups - 1) / num_simd_groups;
    uint sg_start = v_tile_start + simd_group * entries_per_group;
    uint sg_end = min(sg_start + entries_per_group, v_tile_end);

    uint d_per_lane = (D + 31) / 32;
    float dE_local[128];
    for (uint i = 0; i < d_per_lane; i++) dE_local[i] = 0.0f;

    for (uint vi = sg_start; vi < sg_end; vi++) {
        float partial = 0.0f;
        for (uint d = simd_lane; d < D; d += 32) {
            partial += shared_e[d] * (float)c[vi * D + d];
        }
        float dot_val = simd_sum(partial);

        float weight = d_nll_b * exp(dot_val - lse_b);
        if (vi == target_b) weight -= d_nll_b;

        uint idx = 0;
        for (uint d = simd_lane; d < D; d += 32) {
            dE_local[idx] += weight * (float)c[vi * D + d];
            idx++;
        }
    }

    // Reduce across SIMD groups (reuse shared_e)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint idx = 0;
        for (uint d = simd_lane; d < D; d += 32) { shared_e[d] = dE_local[idx++]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 1; g < num_simd_groups; g++) {
        if (simd_group == g) {
            uint idx = 0;
            for (uint d = simd_lane; d < D; d += 32) { shared_e[d] += dE_local[idx++]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = b * num_v_tiles + v_tile_id;
    for (uint d = tid; d < D; d += block_size) {
        partial_dE[out_row * D + d] = shared_e[d];
    }
"""

CCE_BACKWARD_DE_BIAS_SOURCE = """
    uint b = threadgroup_position_in_grid.x;
    uint v_tile_id = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint num_simd_groups = simdgroups_per_threadgroup;
    uint block_size = threads_per_threadgroup.x;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint B = e_shape[0];

    uint num_v_tiles = (V + TILE_V - 1) / TILE_V;
    uint v_tile_start = v_tile_id * TILE_V;
    uint v_tile_end = min(v_tile_start + TILE_V, V);
    uint tile_size = v_tile_end - v_tile_start;

    float d_nll_b = d_nll_val[b];
    float lse_b = lse[b];
    uint target_b = targets[b];

    threadgroup float shared_e[4096];
    for (uint d = tid; d < D; d += block_size) {
        shared_e[d] = (float)e[b * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint entries_per_group = (tile_size + num_simd_groups - 1) / num_simd_groups;
    uint sg_start = v_tile_start + simd_group * entries_per_group;
    uint sg_end = min(sg_start + entries_per_group, v_tile_end);

    uint d_per_lane = (D + 31) / 32;
    float dE_local[128];
    for (uint i = 0; i < d_per_lane; i++) dE_local[i] = 0.0f;

    for (uint vi = sg_start; vi < sg_end; vi++) {
        float partial = 0.0f;
        for (uint d = simd_lane; d < D; d += 32) {
            partial += shared_e[d] * (float)c[vi * D + d];
        }
        float dot_val = simd_sum(partial);
        dot_val += (float)bias[vi];

        float weight = d_nll_b * exp(dot_val - lse_b);
        if (vi == target_b) weight -= d_nll_b;

        uint idx = 0;
        for (uint d = simd_lane; d < D; d += 32) {
            dE_local[idx] += weight * (float)c[vi * D + d];
            idx++;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint idx = 0;
        for (uint d = simd_lane; d < D; d += 32) { shared_e[d] = dE_local[idx++]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 1; g < num_simd_groups; g++) {
        if (simd_group == g) {
            uint idx = 0;
            for (uint d = simd_lane; d < D; d += 32) { shared_e[d] += dE_local[idx++]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = b * num_v_tiles + v_tile_id;
    for (uint d = tid; d < D; d += block_size) {
        partial_dE[out_row * D + d] = shared_e[d];
    }
"""

BWD_TILE_V = 4096
BWD_BLOCK_SIZE = 256
_bwd_kernel_cache = {}


def _get_backward_kernel(has_bias):
    key = ("cce_backward", has_bias)
    if key not in _bwd_kernel_cache:
        if has_bias:
            _bwd_kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_backward_dE_bias",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val"],
                output_names=["partial_dE"],
                source=CCE_BACKWARD_DE_BIAS_SOURCE,
                header=f"constant uint TILE_V = {BWD_TILE_V};",
                ensure_row_contiguous=True,
            )
        else:
            _bwd_kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_backward_dE",
                input_names=["e", "c", "lse", "targets", "d_nll_val"],
                output_names=["partial_dE"],
                source=CCE_BACKWARD_DE_SOURCE,
                header=f"constant uint TILE_V = {BWD_TILE_V};",
                ensure_row_contiguous=True,
            )
    return _bwd_kernel_cache[key]


def cce_backward_dE_metal(e, c, lse, targets, d_nll, bias=None):
    """Fused Metal backward: never materializes [B, V] logits/softmax."""
    B, D = e.shape
    V = c.shape[0]
    num_v_tiles = (V + BWD_TILE_V - 1) // BWD_TILE_V

    kernel = _get_backward_kernel(has_bias=bias is not None)
    if bias is not None:
        inputs = [e, c, bias, lse, targets, d_nll]
    else:
        inputs = [e, c, lse, targets, d_nll]

    (partial_dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(B * BWD_BLOCK_SIZE, num_v_tiles, 1),
        threadgroup=(BWD_BLOCK_SIZE, 1, 1),
        output_shapes=[(B * num_v_tiles * D,)],
        output_dtypes=[mx.float32],
    )

    dE = partial_dE.reshape(B, num_v_tiles, D).sum(axis=1)
    return dE.astype(e.dtype)


def cce_backward_dE(e, c, lse, targets, d_nll, bias=None):
    e_f32 = e.astype(mx.float32)
    c_f32 = c.astype(mx.float32)

    logits = e_f32 @ c_f32.T
    if bias is not None:
        logits = logits + bias.astype(mx.float32)[None, :]

    softmax = mx.exp(logits - lse[:, None])
    dE = (d_nll[:, None] * softmax) @ c_f32 - d_nll[:, None] * c_f32[targets]
    return dE.astype(e.dtype)


def cce_backward_full(e, c, lse, targets, d_nll, bias=None):
    V = c.shape[0]
    e_f32 = e.astype(mx.float32)
    c_f32 = c.astype(mx.float32)

    logits = e_f32 @ c_f32.T
    if bias is not None:
        logits = logits + bias.astype(mx.float32)[None, :]

    softmax = mx.exp(logits - lse[:, None])
    one_hot = (mx.arange(V)[None, :] == targets[:, None]).astype(mx.float32)
    dlogits = d_nll[:, None] * (softmax - one_hot)

    dE = (dlogits @ c_f32).astype(e.dtype)
    dC = (dlogits.T @ e_f32).astype(c.dtype)
    dBias = dlogits.sum(axis=0).astype(bias.dtype) if bias is not None else None
    return dE, dC, dBias


def cce_backward_dE_filtered(e, c, lse, targets, d_nll, bias=None, chunk_size=4096):
    """Gradient-filtered backward: skip chunks where softmax < 2^-12 for ALL tokens."""
    B, D = e.shape
    V = c.shape[0]
    e_f32 = e.astype(mx.float32)

    # Target contribution: dE -= d_nll * c[targets] (always exact)
    dE = -d_nll[:, None] * c[targets].astype(mx.float32)

    # Softmax contribution: dE += d_nll * softmax @ c, chunked with filtering
    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        c_chunk = c[v_start:v_end].astype(mx.float32)

        logits_chunk = e_f32 @ c_chunk.T
        if bias is not None:
            logits_chunk = logits_chunk + bias[v_start:v_end].astype(mx.float32)[None, :]

        # Check: can we skip this chunk's gradient?
        # max(softmax) = exp(max(logits) - lse). If < eps, all softmax ≈ 0.
        chunk_max_log_sm = mx.max(mx.max(logits_chunk, axis=-1) - lse).item()
        if chunk_max_log_sm < FILTER_LOG_EPS:
            continue

        softmax_chunk = mx.exp(logits_chunk - lse[:, None])
        dE = dE + (d_nll[:, None] * softmax_chunk) @ c_chunk

    return dE.astype(e.dtype)


def cce_backward_full_filtered(e, c, lse, targets, d_nll, bias=None, chunk_size=4096):
    """Gradient-filtered backward with dC and dBias.

    Avoids materializing [B, V] by computing target contributions per-chunk.
    """
    B, D = e.shape
    V = c.shape[0]
    e_f32 = e.astype(mx.float32)

    dE = -d_nll[:, None] * c[targets].astype(mx.float32)
    dC_chunks = []
    dBias_chunks = [] if bias is not None else None

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        chunk_len = v_end - v_start
        c_chunk = c[v_start:v_end].astype(mx.float32)

        local_idx = targets - v_start
        in_chunk = (local_idx >= 0) & (local_idx < chunk_len)
        local_idx_safe = mx.clip(local_idx, 0, chunk_len - 1)
        one_hot_chunk = (
            (mx.arange(chunk_len)[None, :] == local_idx_safe[:, None])
            & in_chunk[:, None]
        ).astype(mx.float32)

        logits_chunk = e_f32 @ c_chunk.T
        if bias is not None:
            logits_chunk = logits_chunk + bias[v_start:v_end].astype(mx.float32)[None, :]

        chunk_max_log_sm = mx.max(mx.max(logits_chunk, axis=-1) - lse).item()

        if chunk_max_log_sm < FILTER_LOG_EPS:
            has_targets = mx.any(in_chunk).item()
            if not has_targets:
                dC_chunks.append(mx.zeros((chunk_len, D), dtype=mx.float32))
                if dBias_chunks is not None:
                    dBias_chunks.append(mx.zeros((chunk_len,), dtype=mx.float32))
                continue
            target_dlogits = -d_nll[:, None] * one_hot_chunk
            dC_chunks.append(target_dlogits.T @ e_f32)
            if dBias_chunks is not None:
                dBias_chunks.append(target_dlogits.sum(axis=0))
        else:
            softmax_chunk = mx.exp(logits_chunk - lse[:, None])
            dE = dE + (d_nll[:, None] * softmax_chunk) @ c_chunk
            dlogits_chunk = d_nll[:, None] * (softmax_chunk - one_hot_chunk)
            dC_chunks.append(dlogits_chunk.T @ e_f32)
            if dBias_chunks is not None:
                dBias_chunks.append(dlogits_chunk.sum(axis=0))

    dE = dE.astype(e.dtype)
    dC = mx.concatenate(dC_chunks, axis=0).astype(c.dtype)
    dBias = mx.concatenate(dBias_chunks, axis=0).astype(bias.dtype) if dBias_chunks is not None else None
    return dE, dC, dBias


def cce_backward_dE_chunked(e, c, lse, targets, d_nll, bias=None, chunk_size=4096):
    """Memory-optimal chunked backward. Only effective outside autograd context."""
    B, D = e.shape
    V = c.shape[0]

    dE = mx.zeros((B, D), dtype=mx.float32)
    lse_col = lse[:, None]
    e_f32 = e.astype(mx.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        c_chunk = c[v_start:v_end].astype(mx.float32)

        logits_chunk = e_f32 @ c_chunk.T
        if bias is not None:
            logits_chunk = logits_chunk + bias[v_start:v_end].astype(mx.float32)[None, :]
        softmax_chunk = mx.exp(logits_chunk - lse_col)
        del logits_chunk

        target_in_chunk = (targets[:, None] == mx.arange(v_start, v_end)[None, :])
        dlogits_chunk = d_nll[:, None] * (softmax_chunk - target_in_chunk.astype(mx.float32))
        del softmax_chunk, target_in_chunk

        dE = dE + dlogits_chunk @ c_chunk
        del dlogits_chunk, c_chunk
        mx.eval(dE)

    return dE.astype(e.dtype)


def cce_backward_full_chunked(e, c, lse, targets, d_nll, bias=None, chunk_size=4096):
    """Memory-optimal chunked backward with dC. Only effective outside autograd context."""
    B, D = e.shape
    V = c.shape[0]

    dE = mx.zeros((B, D), dtype=mx.float32)
    dC_chunks = []
    dBias_chunks = [] if bias is not None else None
    lse_col = lse[:, None]
    e_f32 = e.astype(mx.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        c_chunk = c[v_start:v_end].astype(mx.float32)

        logits_chunk = e_f32 @ c_chunk.T
        if bias is not None:
            logits_chunk = logits_chunk + bias[v_start:v_end].astype(mx.float32)[None, :]

        softmax_chunk = mx.exp(logits_chunk - lse_col)
        del logits_chunk
        target_in_chunk = (targets[:, None] == mx.arange(v_start, v_end)[None, :])
        dlogits_chunk = d_nll[:, None] * (softmax_chunk - target_in_chunk.astype(mx.float32))
        del softmax_chunk, target_in_chunk

        dE = dE + dlogits_chunk @ c_chunk
        del c_chunk
        dC_chunks.append(dlogits_chunk.T @ e_f32)

        if dBias_chunks is not None:
            dBias_chunks.append(dlogits_chunk.sum(axis=0))

        del dlogits_chunk
        mx.eval(dE, dC_chunks[-1])

    dC = mx.concatenate(dC_chunks, axis=0)

    dE = dE.astype(e.dtype)
    dC = dC.astype(c.dtype)

    if dBias_chunks is not None:
        dBias = mx.concatenate(dBias_chunks, axis=0).astype(bias.dtype)
    else:
        dBias = None

    return dE, dC, dBias
