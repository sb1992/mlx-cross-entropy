"""Custom Metal kernel for CCE forward pass with SIMD-parallel dot products and V-tiling."""

import mlx.core as mx

# ---------------------------------------------------------------------------
# SIMD + V-tiled forward kernel
# Grid: 2D — (B * BLOCK_SIZE, V_TILES, 1)
# Each threadgroup handles one (token, V_tile) pair.
# Within the threadgroup, SIMD groups split the V_tile further.
# SIMD lanes cooperate on each dot product over the D dimension.
# Outputs partial logsumexp: partial_max[B * V_TILES], partial_sum[B * V_TILES]
# ---------------------------------------------------------------------------

CCE_PARTIAL_SOURCE = """
    uint b = threadgroup_position_in_grid.x;
    uint v_tile_id = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint num_simd_groups = simdgroups_per_threadgroup;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint block_size = threads_per_threadgroup.x;
    uint B = e_shape[0];

    uint num_v_tiles = (V + TILE_V - 1) / TILE_V;
    uint v_tile_start = v_tile_id * TILE_V;
    uint v_tile_end = min(v_tile_start + TILE_V, V);
    uint tile_size = v_tile_end - v_tile_start;

    // Load e[b] into shared memory
    threadgroup float shared_e[4096];
    for (uint d = tid; d < D; d += block_size) {
        shared_e[d] = (float)e[b * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group gets a sub-range within this V tile
    uint entries_per_group = (tile_size + num_simd_groups - 1) / num_simd_groups;
    uint sg_start = v_tile_start + simd_group * entries_per_group;
    uint sg_end = min(sg_start + entries_per_group, v_tile_end);

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (uint vi = sg_start; vi < sg_end; vi++) {
        float partial = 0.0f;
        for (uint d = simd_lane; d < D; d += 32) {
            partial += shared_e[d] * (float)c[vi * D + d];
        }
        float dot_val = simd_sum(partial);

        if (simd_lane == 0) {
            if (dot_val > local_max) {
                local_sum = local_sum * exp(local_max - dot_val) + 1.0f;
                local_max = dot_val;
            } else {
                local_sum += exp(dot_val - local_max);
            }
        }
    }

    // Reduce across SIMD groups within this threadgroup
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];

    if (simd_lane == 0) {
        tg_max[simd_group] = local_max;
        tg_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_max = -INFINITY;
        float final_sum = 0.0f;
        for (uint g = 0; g < num_simd_groups; g++) {
            float gmax = tg_max[g];
            float gsum = tg_sum[g];
            if (gmax > final_max) {
                final_sum = final_sum * exp(final_max - gmax) + gsum;
                final_max = gmax;
            } else {
                final_sum += gsum * exp(gmax - final_max);
            }
        }
        uint out_idx = b * num_v_tiles + v_tile_id;
        partial_max[out_idx] = final_max;
        partial_sum[out_idx] = final_sum;
    }
"""

CCE_PARTIAL_BIAS_SOURCE = """
    uint b = threadgroup_position_in_grid.x;
    uint v_tile_id = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint num_simd_groups = simdgroups_per_threadgroup;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint block_size = threads_per_threadgroup.x;
    uint B = e_shape[0];

    uint num_v_tiles = (V + TILE_V - 1) / TILE_V;
    uint v_tile_start = v_tile_id * TILE_V;
    uint v_tile_end = min(v_tile_start + TILE_V, V);
    uint tile_size = v_tile_end - v_tile_start;

    threadgroup float shared_e[4096];
    for (uint d = tid; d < D; d += block_size) {
        shared_e[d] = (float)e[b * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint entries_per_group = (tile_size + num_simd_groups - 1) / num_simd_groups;
    uint sg_start = v_tile_start + simd_group * entries_per_group;
    uint sg_end = min(sg_start + entries_per_group, v_tile_end);

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (uint vi = sg_start; vi < sg_end; vi++) {
        float partial = 0.0f;
        for (uint d = simd_lane; d < D; d += 32) {
            partial += shared_e[d] * (float)c[vi * D + d];
        }
        float dot_val = simd_sum(partial);
        if (simd_lane == 0) dot_val += (float)bias[vi];

        if (simd_lane == 0) {
            if (dot_val > local_max) {
                local_sum = local_sum * exp(local_max - dot_val) + 1.0f;
                local_max = dot_val;
            } else {
                local_sum += exp(dot_val - local_max);
            }
        }
    }

    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    if (simd_lane == 0) {
        tg_max[simd_group] = local_max;
        tg_sum[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_max = -INFINITY;
        float final_sum = 0.0f;
        for (uint g = 0; g < num_simd_groups; g++) {
            float gmax = tg_max[g];
            float gsum = tg_sum[g];
            if (gmax > final_max) {
                final_sum = final_sum * exp(final_max - gmax) + gsum;
                final_max = gmax;
            } else {
                final_sum += gsum * exp(gmax - final_max);
            }
        }
        uint out_idx = b * num_v_tiles + v_tile_id;
        partial_max[out_idx] = final_max;
        partial_sum[out_idx] = final_sum;
    }
"""

_kernel_cache = {}

TILE_V = 4096
BLOCK_SIZE = 256


def _get_partial_kernel(has_bias):
    key = ("cce_partial", has_bias)
    if key not in _kernel_cache:
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_partial_bias",
                input_names=["e", "c", "bias"],
                output_names=["partial_max", "partial_sum"],
                source=CCE_PARTIAL_BIAS_SOURCE,
                header=f"constant uint TILE_V = {TILE_V};",
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_partial",
                input_names=["e", "c"],
                output_names=["partial_max", "partial_sum"],
                source=CCE_PARTIAL_SOURCE,
                header=f"constant uint TILE_V = {TILE_V};",
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def cce_forward_metal(e, c, targets, bias=None):
    B = e.shape[0]
    V = c.shape[0]
    num_v_tiles = (V + TILE_V - 1) // TILE_V

    kernel = _get_partial_kernel(has_bias=bias is not None)
    inputs = [e, c, bias] if bias is not None else [e, c]

    p_max, p_sum = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(B * BLOCK_SIZE, num_v_tiles, 1),
        threadgroup=(BLOCK_SIZE, 1, 1),
        output_shapes=[(B * num_v_tiles,), (B * num_v_tiles,)],
        output_dtypes=[mx.float32, mx.float32],
    )

    # Reduce partial logsumexp across V-tiles
    p_max = p_max.reshape(B, num_v_tiles)
    p_sum = p_sum.reshape(B, num_v_tiles)

    global_max = mx.max(p_max, axis=1)  # [B]
    shifted = p_sum * mx.exp(p_max - global_max[:, None])
    total_sum = mx.sum(shifted, axis=1)  # [B]
    lse = global_max + mx.log(total_sum)

    # Target logit — simple gather + dot, O(B*D)
    target_logit = mx.sum(
        e.astype(mx.float32) * c[targets].astype(mx.float32), axis=-1
    )
    if bias is not None:
        target_logit = target_logit + bias[targets].astype(mx.float32)

    nll = lse - target_logit
    return nll, lse
