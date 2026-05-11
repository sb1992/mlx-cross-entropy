"""Fused CCE forward: simdgroup MMA dots + per-tile logsumexp.

Single kernel dispatch computes tile-level max/sum_exp for the full
e @ c.T logsumexp, eliminating mx.eval() synchronization overhead.

Grid: (B, V/32, 1), threadgroup (32, 1, 1)
Each threadgroup: 4B×4V MMA → 32×32 dots → shared mem → row-wise max/sum_exp
Python reduces tile_max[B, T], tile_sum_exp[B, T] → lse[B]
"""

import mlx.core as mx


def _build_fused_source(has_bias, compute_logit_avg=False, fuse_target=False):
    bias_add = " + (float)bias[vs + j]" if has_bias else ""
    target_bias_add = " + (float)bias[my_target]" if has_bias else ""
    logit_avg_block = ""
    if compute_logit_avg:
        logit_avg_block = """
    // Column sums for vocab sorting (raw logits, no bias)
    {
        float col_sum = 0.0f;
        for (uint i = 0; i < 32; i++) {
            col_sum += shared[i * 32 + tid];
        }
        atomic_fetch_add_explicit(
            (device atomic<float>*)(logit_avg + vs + tid),
            col_sum, memory_order_relaxed);
    }
"""
    target_extract_block = ""
    if fuse_target:
        target_extract_block = f"""
    // Extract target logit from MMA result (same computation as logsumexp)
    {{
        uint my_target = targets[bs + tid];
        if (my_target >= vs && my_target < vs + 32) {{
            float tgt_logit = shared[tid * 32 + (my_target - vs)]{target_bias_add};
            neg_target_logit[bs + tid] = -tgt_logit;
        }}
    }}
"""
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint bs = tg_x * 32;
    uint vs = tg_y * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint num_v_tiles = (V + 31) / 32;

    threadgroup float shared[1024];

    simdgroup_float8x8 d00,d01,d02,d03,d10,d11,d12,d13;
    simdgroup_float8x8 d20,d21,d22,d23,d30,d31,d32,d33;
    d00.thread_elements()[0]=0;d00.thread_elements()[1]=0;
    d01.thread_elements()[0]=0;d01.thread_elements()[1]=0;
    d02.thread_elements()[0]=0;d02.thread_elements()[1]=0;
    d03.thread_elements()[0]=0;d03.thread_elements()[1]=0;
    d10.thread_elements()[0]=0;d10.thread_elements()[1]=0;
    d11.thread_elements()[0]=0;d11.thread_elements()[1]=0;
    d12.thread_elements()[0]=0;d12.thread_elements()[1]=0;
    d13.thread_elements()[0]=0;d13.thread_elements()[1]=0;
    d20.thread_elements()[0]=0;d20.thread_elements()[1]=0;
    d21.thread_elements()[0]=0;d21.thread_elements()[1]=0;
    d22.thread_elements()[0]=0;d22.thread_elements()[1]=0;
    d23.thread_elements()[0]=0;d23.thread_elements()[1]=0;
    d30.thread_elements()[0]=0;d30.thread_elements()[1]=0;
    d31.thread_elements()[0]=0;d31.thread_elements()[1]=0;
    d32.thread_elements()[0]=0;d32.thread_elements()[1]=0;
    d33.thread_elements()[0]=0;d33.thread_elements()[1]=0;

    for (uint d = 0; d < D; d += 8) {{
        simdgroup_float8x8 a0,a1,a2,a3,b0,b1,b2,b3;
        simdgroup_load(a0, e + bs * D + d, (ulong)D);
        simdgroup_load(a1, e + (bs+8) * D + d, (ulong)D);
        simdgroup_load(a2, e + (bs+16) * D + d, (ulong)D);
        simdgroup_load(a3, e + (bs+24) * D + d, (ulong)D);
        simdgroup_load(b0, c + vs * D + d, (ulong)D, ulong2(0,0), true);
        simdgroup_load(b1, c + (vs+8) * D + d, (ulong)D, ulong2(0,0), true);
        simdgroup_load(b2, c + (vs+16) * D + d, (ulong)D, ulong2(0,0), true);
        simdgroup_load(b3, c + (vs+24) * D + d, (ulong)D, ulong2(0,0), true);

        simdgroup_multiply_accumulate(d00,a0,b0,d00);
        simdgroup_multiply_accumulate(d01,a0,b1,d01);
        simdgroup_multiply_accumulate(d02,a0,b2,d02);
        simdgroup_multiply_accumulate(d03,a0,b3,d03);
        simdgroup_multiply_accumulate(d10,a1,b0,d10);
        simdgroup_multiply_accumulate(d11,a1,b1,d11);
        simdgroup_multiply_accumulate(d12,a1,b2,d12);
        simdgroup_multiply_accumulate(d13,a1,b3,d13);
        simdgroup_multiply_accumulate(d20,a2,b0,d20);
        simdgroup_multiply_accumulate(d21,a2,b1,d21);
        simdgroup_multiply_accumulate(d22,a2,b2,d22);
        simdgroup_multiply_accumulate(d23,a2,b3,d23);
        simdgroup_multiply_accumulate(d30,a3,b0,d30);
        simdgroup_multiply_accumulate(d31,a3,b1,d31);
        simdgroup_multiply_accumulate(d32,a3,b2,d32);
        simdgroup_multiply_accumulate(d33,a3,b3,d33);
    }}

    simdgroup_store(d00, shared + 0, (ulong)32);
    simdgroup_store(d01, shared + 8, (ulong)32);
    simdgroup_store(d02, shared + 16, (ulong)32);
    simdgroup_store(d03, shared + 24, (ulong)32);
    simdgroup_store(d10, shared + 256, (ulong)32);
    simdgroup_store(d11, shared + 264, (ulong)32);
    simdgroup_store(d12, shared + 272, (ulong)32);
    simdgroup_store(d13, shared + 280, (ulong)32);
    simdgroup_store(d20, shared + 512, (ulong)32);
    simdgroup_store(d21, shared + 520, (ulong)32);
    simdgroup_store(d22, shared + 528, (ulong)32);
    simdgroup_store(d23, shared + 536, (ulong)32);
    simdgroup_store(d30, shared + 768, (ulong)32);
    simdgroup_store(d31, shared + 776, (ulong)32);
    simdgroup_store(d32, shared + 784, (ulong)32);
    simdgroup_store(d33, shared + 792, (ulong)32);

    threadgroup_barrier(mem_flags::mem_threadgroup);
{logit_avg_block}{target_extract_block}
    float row_max = -1e38f;
    for (uint j = 0; j < 32; j++) {{
        float val = shared[tid * 32 + j]{bias_add};
        row_max = fmax(row_max, val);
    }}
    float row_sum = 0.0f;
    for (uint j = 0; j < 32; j++) {{
        float val = shared[tid * 32 + j]{bias_add};
        row_sum += exp(val - row_max);
    }}

    uint out_idx = (bs + tid) * num_v_tiles + tg_y;
    tile_max[out_idx] = row_max;
    tile_sum_exp[out_idx] = row_sum;
"""


_kernel_cache = {}


def _get_fused_kernel(has_bias, compute_logit_avg=False, fuse_target=False):
    key = ("cce_fused_fwd", has_bias, compute_logit_avg, fuse_target)
    if key not in _kernel_cache:
        input_names = ["e", "c"]
        if has_bias:
            input_names.append("bias")
        if fuse_target:
            input_names.append("targets")
        output_names = ["tile_max", "tile_sum_exp"]
        if compute_logit_avg:
            output_names.append("logit_avg")
        if fuse_target:
            output_names.append("neg_target_logit")
        parts = ["cce_fused_fwd"]
        if has_bias:
            parts.append("b")
        if compute_logit_avg:
            parts.append("la")
        if fuse_target:
            parts.append("ft")
        name = "_".join(parts)
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=_build_fused_source(
                has_bias=has_bias,
                compute_logit_avg=compute_logit_avg,
                fuse_target=fuse_target,
            ),
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def fused_cce_forward(e, c, targets, bias=None, return_tile_max=False,
                      return_logit_avg=False, fuse_target=True):
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    num_b_tiles = B // 32
    num_v_tiles = V // 32

    kernel = _get_fused_kernel(
        has_bias=bias is not None,
        compute_logit_avg=return_logit_avg,
        fuse_target=fuse_target,
    )
    inputs = [e, c]
    if bias is not None:
        inputs.append(bias)
    if fuse_target:
        inputs.append(targets)

    output_shapes = [(B * num_v_tiles,), (B * num_v_tiles,)]
    output_dtypes = [mx.float32, mx.float32]
    needs_init = return_logit_avg
    if return_logit_avg:
        output_shapes.append((V,))
        output_dtypes.append(mx.float32)
    if fuse_target:
        output_shapes.append((B,))
        output_dtypes.append(mx.float32)

    kwargs = {}
    if needs_init:
        kwargs["init_value"] = 0.0

    results = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 32, num_v_tiles, 1),
        threadgroup=(32, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        **kwargs,
    )

    tile_max_flat = results[0]
    tile_sum_exp_flat = results[1]
    out_idx = 2
    logit_avg = None
    if return_logit_avg:
        logit_avg = results[out_idx]
        out_idx += 1
    neg_target_logit = None
    if fuse_target:
        neg_target_logit = results[out_idx]

    tile_max_2d = tile_max_flat.reshape(B, num_v_tiles)
    tile_sum_exp_2d = tile_sum_exp_flat.reshape(B, num_v_tiles)

    global_max = mx.max(tile_max_2d, axis=-1)
    rescaled = tile_sum_exp_2d * mx.exp(tile_max_2d - global_max[:, None])
    lse = global_max + mx.log(mx.sum(rescaled, axis=-1))

    if fuse_target:
        nll = lse + neg_target_logit
    else:
        target_logit = mx.sum(e.astype(mx.float32) * c[targets].astype(mx.float32), axis=-1)
        if bias is not None:
            target_logit = target_logit + bias[targets].astype(mx.float32)
        nll = lse - target_logit

    if return_tile_max and return_logit_avg:
        return nll, lse, tile_max_2d, logit_avg / B
    if return_tile_max:
        return nll, lse, tile_max_2d
    if return_logit_avg:
        return nll, lse, logit_avg / B
    return nll, lse
