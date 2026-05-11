"""Fused CCE backward: MMA dot recomputation + gradient accumulation.

Three implementations:
1. Scalar: SIMD group 0 computes MMA dots, all 8 groups do scalar gradient accum
2. MMA: All-MMA approach — dots via MMA, then gradient W×c via MMA hardware too
3. Filtered: Uses tile_max from forward to skip V-tiles where softmax < 2^-12.
   Single kernel launch, all filtering decisions on-GPU (no Python loop/.item() sync).

All recompute e @ c.T tile-by-tile (never materializes [B, V]).
Grid: (B/32, num_v_tiles, 1), threadgroup (256, 1, 1) = 8 SIMD groups
Outputs partial_dE[B, num_v_tiles, D], Python sums axis=1.
"""

import math
import mlx.core as mx

FILTER_LOG_EPS = math.log(2**-12)


BWD_V_TILE = 4096


def _build_fused_backward_source(has_bias):
    bias_add = " + (float)bias[vs + j]" if has_bias else ""
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);
    uint num_v_tiles = (V + BWD_V_TILE - 1) / BWD_V_TILE;

    threadgroup float shared[1024];

    // Each SIMD group handles 4 tokens
    uint t_start = simd_group * 4;
    uint d_per_lane = (D + 31) / 32;

    // Private accumulators: 4 tokens × D/32 per lane
    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {{
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }}

    // Pre-load per-token scalars for this group's 4 tokens
    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {{
        my_d_nll[i] = d_nll_val[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = targets[bs + t_start + i];
    }}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{

        // SIMD group 0 does MMA
        if (simd_group == 0) {{
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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All 8 SIMD groups do gradient accumulation
        // Each handles 4 tokens: t_start .. t_start+3
        #define ACCUM_TOKEN(TI, DE_ARR) {{ \
            uint t_##TI = t_start + TI; \
            float w_##TI[32]; \
            for (uint j = 0; j < 32; j++) {{ \
                float dot_val = shared[t_##TI * 32 + j]{bias_add}; \
                float wj = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \
                if ((vs + j) == my_target[TI]) wj -= my_d_nll[TI]; \
                w_##TI[j] = wj; \
            }} \
            uint idx_##TI = 0; \
            for (uint d = simd_lane; d < D; d += 32) {{ \
                float acc = 0.0f; \
                for (uint j = 0; j < 32; j++) {{ \
                    acc += w_##TI[j] * (float)c[(vs + j) * D + d]; \
                }} \
                DE_ARR[idx_##TI] += acc; \
                idx_##TI++; \
            }} \
        }}
        ACCUM_TOKEN(0, dE0)
        ACCUM_TOKEN(1, dE1)
        ACCUM_TOKEN(2, dE2)
        ACCUM_TOKEN(3, dE3)
        #undef ACCUM_TOKEN

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write results: each SIMD group writes its 4 tokens
    #define WRITE_TOKEN(TI, DE_ARR) {{ \
        uint out_row_##TI = (bs + t_start + TI) * num_v_tiles + tg_y; \
        uint widx_##TI = 0; \
        for (uint d = simd_lane; d < D; d += 32) {{ \
            partial_dE[out_row_##TI * D + d] = DE_ARR[widx_##TI++]; \
        }} \
    }}
    WRITE_TOKEN(0, dE0)
    WRITE_TOKEN(1, dE1)
    WRITE_TOKEN(2, dE2)
    WRITE_TOKEN(3, dE3)
    #undef WRITE_TOKEN
"""


def _build_fused_backward_mma_source(has_bias, cols_per_group):
    bias_add = " + (float)bias[vs + col]" if has_bias else ""

    # Generate named gradient accumulators (avoid array indexing → register spills)
    decl_lines = []
    init_lines = []
    for rt in range(4):
        for ct in range(cols_per_group):
            name = f"g{rt}_{ct}"
            decl_lines.append(f"    simdgroup_float8x8 {name};")
            init_lines.append(f"    {name}.thread_elements()[0]=0.0f; {name}.thread_elements()[1]=0.0f;")
    decl_block = "\n".join(decl_lines)
    init_block = "\n".join(init_lines)

    # Generate unrolled MMA gradient loop body
    mac_lines = []
    for ct in range(cols_per_group):
        mac_lines.append(f"            {{ simdgroup_float8x8 cb; simdgroup_load(cb, c + (vs + k * 8) * D + d_offset + {ct * 8}, (ulong)D);")
        for rt in range(4):
            mac_lines.append(f"              simdgroup_multiply_accumulate(g{rt}_{ct}, w{rt}, cb, g{rt}_{ct});")
        mac_lines.append(f"            }}")
    mac_block = "\n".join(mac_lines)

    # Generate output stores
    store_lines = []
    for rt in range(4):
        for ct in range(cols_per_group):
            store_lines.append(f"    simdgroup_store(g{rt}_{ct}, partial_dE + (bs + {rt * 8}) * num_v_tiles * D + tg_y * D + d_offset + {ct * 8}, out_stride);")
    store_block = "\n".join(store_lines)

    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);
    uint num_v_tiles = (V + BWD_V_TILE - 1) / BWD_V_TILE;

    threadgroup float shared[1024];

    uint d_offset = simd_group * {cols_per_group * 8};

{decl_block}
{init_block}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{

        if (simd_group == 0) {{
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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute W = softmax weights from dots → overwrite shared
        for (uint i = tid; i < 1024; i += 256) {{
            uint row = i / 32;
            uint col = i % 32;
            float dn = d_nll_val[bs + row];
            float l = lse[bs + row];
            float dot = shared[i]{bias_add};
            float w = dn * exp(dot - l);
            if ((vs + col) == targets[bs + row]) w -= dn;
            shared[i] = w;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA gradient: W[shared,32×32] × c[device,32×D/8] per group
        for (uint k = 0; k < 4; k++) {{
            simdgroup_float8x8 w0, w1, w2, w3;
            simdgroup_load(w0, shared + 0 * 256 + k * 8, (ulong)32);
            simdgroup_load(w1, shared + 1 * 256 + k * 8, (ulong)32);
            simdgroup_load(w2, shared + 2 * 256 + k * 8, (ulong)32);
            simdgroup_load(w3, shared + 3 * 256 + k * 8, (ulong)32);

{mac_block}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    ulong out_stride = (ulong)(num_v_tiles * D);
{store_block}
"""


_kernel_cache = {}


def _get_fused_backward_kernel(has_bias):
    key = ("cce_fused_bwd", has_bias)
    if key not in _kernel_cache:
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_fused_bwd_bias",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val"],
                output_names=["partial_dE"],
                source=_build_fused_backward_source(has_bias=True),
                header=f"constant uint BWD_V_TILE = {BWD_V_TILE};",
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_fused_bwd",
                input_names=["e", "c", "lse", "targets", "d_nll_val"],
                output_names=["partial_dE"],
                source=_build_fused_backward_source(has_bias=False),
                header=f"constant uint BWD_V_TILE = {BWD_V_TILE};",
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def _get_fused_backward_mma_kernel(has_bias, cols_per_group):
    key = ("cce_fused_bwd_mma", has_bias, cols_per_group)
    if key not in _kernel_cache:
        header = f"constant uint BWD_V_TILE = {BWD_V_TILE};"
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name=f"cce_fused_bwd_mma_bias_{cols_per_group}",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val"],
                output_names=["partial_dE"],
                source=_build_fused_backward_mma_source(has_bias=True, cols_per_group=cols_per_group),
                header=header,
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name=f"cce_fused_bwd_mma_{cols_per_group}",
                input_names=["e", "c", "lse", "targets", "d_nll_val"],
                output_names=["partial_dE"],
                source=_build_fused_backward_mma_source(has_bias=False, cols_per_group=cols_per_group),
                header=header,
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def fused_cce_backward_dE_mma(e, c, lse, targets, d_nll, bias=None):
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 64 == 0, f"D must be divisible by 64 for MMA backward, got {D}"

    cols_per_group = D // 64
    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_fused_backward_mma_kernel(has_bias=bias is not None, cols_per_group=cols_per_group)
    if bias is not None:
        inputs = [e, c, bias, lse, targets, d_nll]
    else:
        inputs = [e, c, lse, targets, d_nll]

    (partial_dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * num_v_tiles * D,)],
        output_dtypes=[mx.float32],
    )

    dE = partial_dE.reshape(B, num_v_tiles, D).sum(axis=1)
    return dE.astype(e.dtype)


def chunked_cce_backward_dE(e, c, lse, targets, d_nll, bias=None, chunk_size=4096, use_eval=True):
    B, D = e.shape
    V = c.shape[0]
    e_f32 = e.astype(mx.float32)
    chunk_size = min(chunk_size, V)

    dE = -d_nll[:, None] * c[targets].astype(mx.float32)
    if use_eval:
        mx.eval(dE)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        c_chunk = c[v_start:v_end]
        logits = (e_f32 @ c_chunk.T).astype(mx.float32)
        if bias is not None:
            logits = logits + bias[v_start:v_end].astype(mx.float32)[None, :]
        sm = mx.exp(logits - lse[:, None])
        dE = dE + (d_nll[:, None] * sm) @ c_chunk.astype(mx.float32)
        if use_eval:
            mx.eval(dE)

    return dE.astype(e.dtype)


def fused_cce_backward_dE(e, c, lse, targets, d_nll, bias=None):
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_fused_backward_kernel(has_bias=bias is not None)
    if bias is not None:
        inputs = [e, c, bias, lse, targets, d_nll]
    else:
        inputs = [e, c, lse, targets, d_nll]

    (partial_dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * num_v_tiles * D,)],
        output_dtypes=[mx.float32],
    )

    dE = partial_dE.reshape(B, num_v_tiles, D).sum(axis=1)
    return dE.astype(e.dtype)


# ---------------------------------------------------------------------------
# Filtered backward: uses tile_max from forward to skip V-tiles on-GPU
# ---------------------------------------------------------------------------

def _build_filtered_backward_source(has_bias):
    bias_add = " + (float)bias[vs + j]" if has_bias else ""
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint num_fwd_tiles = (V + 31) / 32;

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);
    uint num_v_tiles = (V + BWD_V_TILE - 1) / BWD_V_TILE;

    threadgroup float shared[1024];

    uint t_start = simd_group * 4;
    uint d_per_lane = (D + 31) / 32;

    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {{
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }}

    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {{
        my_d_nll[i] = d_nll_val[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = targets[bs + t_start + i];
    }}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{
        // --- GPU-side gradient filtering using tile_max from forward ---
        if (tid == 0) {{
            uint fwd_tile = vs / 32;
            float flag = 0.0f;
            for (uint i = 0; i < 32; i++) {{
                if (tile_max[(bs + i) * num_fwd_tiles + fwd_tile] - lse[bs + i] >= FILTER_THRESHOLD) {{
                    flag = 1.0f;
                    break;
                }}
            }}
            shared[0] = flag;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[0] < 0.5f) continue;

        // MMA dots (SIMD group 0 only)
        if (simd_group == 0) {{
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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All 8 SIMD groups do gradient accumulation (softmax part only, no target sub)
        #define ACCUM_TOKEN(TI, DE_ARR) {{ \
            uint t_##TI = t_start + TI; \
            float w_##TI[32]; \
            for (uint j = 0; j < 32; j++) {{ \
                float dot_val = shared[t_##TI * 32 + j]{bias_add}; \
                w_##TI[j] = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \
            }} \
            uint idx_##TI = 0; \
            for (uint d = simd_lane; d < D; d += 32) {{ \
                float acc = 0.0f; \
                for (uint j = 0; j < 32; j++) {{ \
                    acc += w_##TI[j] * (float)c[(vs + j) * D + d]; \
                }} \
                DE_ARR[idx_##TI] += acc; \
                idx_##TI++; \
            }} \
        }}
        ACCUM_TOKEN(0, dE0)
        ACCUM_TOKEN(1, dE1)
        ACCUM_TOKEN(2, dE2)
        ACCUM_TOKEN(3, dE3)
        #undef ACCUM_TOKEN

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #define WRITE_TOKEN(TI, DE_ARR) {{ \
        uint out_row_##TI = (bs + t_start + TI) * num_v_tiles + tg_y; \
        uint widx_##TI = 0; \
        for (uint d = simd_lane; d < D; d += 32) {{ \
            partial_dE[out_row_##TI * D + d] = DE_ARR[widx_##TI++]; \
        }} \
    }}
    WRITE_TOKEN(0, dE0)
    WRITE_TOKEN(1, dE1)
    WRITE_TOKEN(2, dE2)
    WRITE_TOKEN(3, dE3)
    #undef WRITE_TOKEN
"""


def _get_filtered_backward_kernel(has_bias):
    key = ("cce_filtered_bwd", has_bias)
    if key not in _kernel_cache:
        header = (
            f"constant uint BWD_V_TILE = {BWD_V_TILE};\n"
            f"constant float FILTER_THRESHOLD = {FILTER_LOG_EPS:.10f}f;"
        )
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_filtered_bwd_bias",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["partial_dE"],
                source=_build_filtered_backward_source(has_bias=True),
                header=header,
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_filtered_bwd",
                input_names=["e", "c", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["partial_dE"],
                source=_build_filtered_backward_source(has_bias=False),
                header=header,
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def fused_cce_backward_filtered(e, c, lse, targets, d_nll, tile_max, bias=None):
    """Filtered Metal backward: skips V-tiles where softmax < 2^-12 on-GPU.

    Separates target contribution (always exact) from softmax gradient (filtered).
    Kernel computes: dE_softmax = sum_over_significant_tiles(d_nll * softmax @ c)
    Python adds: dE_target = -d_nll * c[targets]
    """
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_filtered_backward_kernel(has_bias=bias is not None)
    if bias is not None:
        inputs = [e, c, bias, lse, targets, d_nll, tile_max]
    else:
        inputs = [e, c, lse, targets, d_nll, tile_max]

    (partial_dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * num_v_tiles * D,)],
        output_dtypes=[mx.float32],
    )

    dE_softmax = partial_dE.reshape(B, num_v_tiles, D).sum(axis=1)
    dE_target = -d_nll[:, None] * c[targets].astype(mx.float32)
    dE = dE_softmax + dE_target
    return dE.astype(e.dtype)


# ---------------------------------------------------------------------------
# v2 backward: paper-aligned rewrite
# - One threadgroup per B-tile, iterates over ALL V (no partial arrays)
# - Recomputes logits via MMA for filtering (no saved tile_max)
# - Target subtraction inside kernel (no Python dE_target)
# - Output: dE[B, D] directly
# ---------------------------------------------------------------------------

def _build_backward_v2_source(has_bias):
    bias_add = " + (float)bias[vs + j]" if has_bias else ""
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint num_fwd_tiles = (V + 31) / 32;

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);
    uint num_v_tiles = (V + BWD_V_TILE - 1) / BWD_V_TILE;

    threadgroup float shared[1024];

    uint t_start = simd_group * 4;
    uint d_per_lane = (D + 31) / 32;

    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {{
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }}

    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {{
        my_d_nll[i] = d_nll_val[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = targets[bs + t_start + i];
    }}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{
        // --- Pre-MMA filter: tile_max + target check ---
        if (tid == 0) {{
            uint fwd_tile = vs / 32;
            float flag = 0.0f;
            for (uint i = 0; i < 32; i++) {{
                uint t = targets[bs + i];
                if (t >= vs && t < vs + 32) {{ flag = 1.0f; break; }}
                if (tile_max[(bs + i) * num_fwd_tiles + fwd_tile] - lse[bs + i] >= FILTER_THRESHOLD) {{
                    flag = 1.0f;
                    break;
                }}
            }}
            shared[0] = flag;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[0] < 0.5f) continue;

        // SIMD group 0: MMA to compute 32x32 logit tile
        if (simd_group == 0) {{
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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Gradient: w = d_nll * (softmax - one_hot(target)) ---
        #define ACCUM_TOKEN(TI, DE_ARR) {{ \
            uint t_##TI = t_start + TI; \
            float w_##TI[32]; \
            for (uint j = 0; j < 32; j++) {{ \
                float dot_val = shared[t_##TI * 32 + j]{bias_add}; \
                float wj = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \
                if ((vs + j) == my_target[TI]) wj -= my_d_nll[TI]; \
                w_##TI[j] = wj; \
            }} \
            uint idx_##TI = 0; \
            for (uint d = simd_lane; d < D; d += 32) {{ \
                float acc = 0.0f; \
                for (uint j = 0; j < 32; j++) {{ \
                    acc += w_##TI[j] * (float)c[(vs + j) * D + d]; \
                }} \
                DE_ARR[idx_##TI] += acc; \
                idx_##TI++; \
            }} \
        }}
        ACCUM_TOKEN(0, dE0)
        ACCUM_TOKEN(1, dE1)
        ACCUM_TOKEN(2, dE2)
        ACCUM_TOKEN(3, dE3)
        #undef ACCUM_TOKEN

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #define WRITE_TOKEN(TI, DE_ARR) {{ \
        uint out_row_##TI = (bs + t_start + TI) * num_v_tiles + tg_y; \
        uint widx_##TI = 0; \
        for (uint d = simd_lane; d < D; d += 32) {{ \
            partial_dE[out_row_##TI * D + d] = DE_ARR[widx_##TI++]; \
        }} \
    }}
    WRITE_TOKEN(0, dE0)
    WRITE_TOKEN(1, dE1)
    WRITE_TOKEN(2, dE2)
    WRITE_TOKEN(3, dE3)
    #undef WRITE_TOKEN
"""


def _get_backward_v2_kernel(has_bias):
    key = ("cce_bwd_v2", has_bias)
    if key not in _kernel_cache:
        header = (
            f"constant uint BWD_V_TILE = {BWD_V_TILE};\n"
            f"constant float FILTER_THRESHOLD = {FILTER_LOG_EPS:.10f}f;"
        )
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v2b",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["partial_dE"],
                source=_build_backward_v2_source(has_bias=True),
                header=header,
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v2nb",
                input_names=["e", "c", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["partial_dE"],
                source=_build_backward_v2_source(has_bias=False),
                header=header,
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def fused_cce_backward_v2(e, c, lse, targets, d_nll, tile_max, bias=None):
    """Fast filtered backward with target subtraction in kernel.

    Uses tile_max from forward for pre-MMA filtering (skips MMA for filtered tiles).
    Target contribution computed inside kernel (no Python-side dE_target).
    Target-containing tiles are never filtered.
    """
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_backward_v2_kernel(has_bias=bias is not None)
    if bias is not None:
        inputs = [e, c, bias, lse, targets, d_nll, tile_max]
    else:
        inputs = [e, c, lse, targets, d_nll, tile_max]

    (partial_dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * num_v_tiles * D,)],
        output_dtypes=[mx.float32],
    )

    dE = partial_dE.reshape(B, num_v_tiles, D).sum(axis=1)
    return dE.astype(e.dtype)


# ---------------------------------------------------------------------------
# v3 backward: on-GPU atomic gradient accumulation (GAP 1)
# Same as v2 (target subtraction, pre-MMA filtering) but writes dE[B, D]
# directly via atomic_fetch_add_explicit — eliminates partial_dE arrays.
# ---------------------------------------------------------------------------

def _build_backward_v3_source(has_bias):
    bias_add = " + (float)bias[vs + j]" if has_bias else ""
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint num_fwd_tiles = (V + 31) / 32;

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);

    threadgroup float shared[1024];

    uint t_start = simd_group * 4;
    uint d_per_lane = (D + 31) / 32;

    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {{
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }}

    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {{
        my_d_nll[i] = d_nll_val[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = targets[bs + t_start + i];
    }}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{
        if (tid == 0) {{
            uint fwd_tile = vs / 32;
            float flag = 0.0f;
            for (uint i = 0; i < 32; i++) {{
                uint t = targets[bs + i];
                if (t >= vs && t < vs + 32) {{ flag = 1.0f; break; }}
                if (tile_max[(bs + i) * num_fwd_tiles + fwd_tile] - lse[bs + i] >= FILTER_THRESHOLD) {{
                    flag = 1.0f;
                    break;
                }}
            }}
            shared[0] = flag;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[0] < 0.5f) continue;

        if (simd_group == 0) {{
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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #define ACCUM_TOKEN(TI, DE_ARR) {{ \
            uint t_##TI = t_start + TI; \
            float w_##TI[32]; \
            for (uint j = 0; j < 32; j++) {{ \
                float dot_val = shared[t_##TI * 32 + j]{bias_add}; \
                float wj = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \
                if ((vs + j) == my_target[TI]) wj -= my_d_nll[TI]; \
                w_##TI[j] = wj; \
            }} \
            uint idx_##TI = 0; \
            for (uint d = simd_lane; d < D; d += 32) {{ \
                float acc = 0.0f; \
                for (uint j = 0; j < 32; j++) {{ \
                    acc += w_##TI[j] * (float)c[(vs + j) * D + d]; \
                }} \
                DE_ARR[idx_##TI] += acc; \
                idx_##TI++; \
            }} \
        }}
        ACCUM_TOKEN(0, dE0)
        ACCUM_TOKEN(1, dE1)
        ACCUM_TOKEN(2, dE2)
        ACCUM_TOKEN(3, dE3)
        #undef ACCUM_TOKEN

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Atomic add accumulated gradients to output dE[B, D]
    #define WRITE_TOKEN(TI, DE_ARR) {{ \
        uint widx_##TI = 0; \
        for (uint d = simd_lane; d < D; d += 32) {{ \
            atomic_fetch_add_explicit( \
                (device atomic<float>*)(dE + (bs + t_start + TI) * D + d), \
                DE_ARR[widx_##TI++], memory_order_relaxed); \
        }} \
    }}
    WRITE_TOKEN(0, dE0)
    WRITE_TOKEN(1, dE1)
    WRITE_TOKEN(2, dE2)
    WRITE_TOKEN(3, dE3)
    #undef WRITE_TOKEN
"""


def _get_backward_v3_kernel(has_bias):
    key = ("cce_bwd_v3", has_bias)
    if key not in _kernel_cache:
        header = (
            f"constant uint BWD_V_TILE = {BWD_V_TILE};\n"
            f"constant float FILTER_THRESHOLD = {FILTER_LOG_EPS:.10f}f;"
        )
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v3b",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["dE"],
                source=_build_backward_v3_source(has_bias=True),
                header=header,
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v3nb",
                input_names=["e", "c", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["dE"],
                source=_build_backward_v3_source(has_bias=False),
                header=header,
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def fused_cce_backward_v3(e, c, lse, targets, d_nll, tile_max, bias=None):
    """v3: atomic accumulation + pre-MMA filter + target subtraction."""
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_backward_v3_kernel(has_bias=bias is not None)
    if bias is not None:
        inputs = [e, c, bias, lse, targets, d_nll, tile_max]
    else:
        inputs = [e, c, lse, targets, d_nll, tile_max]

    (dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * D,)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )

    return dE.reshape(B, D).astype(e.dtype)


# ---------------------------------------------------------------------------
# v4 backward: vocab-sorted + post-MMA filtering + atomic output (GAP 3)
# Takes c_sorted (pre-gathered by Python) and sorted_targets.
# No tile_max — filters after MMA using recomputed logits.
# ---------------------------------------------------------------------------

def _build_backward_v4_source(has_bias):
    bias_add = " + (float)bias_sorted[vs + j]" if has_bias else ""
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_sorted_shape[0];

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);

    threadgroup float shared[1025];

    uint t_start = simd_group * 4;
    uint d_per_lane = (D + 31) / 32;

    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {{
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }}

    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {{
        my_d_nll[i] = d_nll_val[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = sorted_targets[bs + t_start + i];
    }}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{

        // SIMD group 0: MMA to compute 32x32 logit tile
        if (simd_group == 0) {{
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
                simdgroup_load(b0, c_sorted + vs * D + d, (ulong)D, ulong2(0,0), true);
                simdgroup_load(b1, c_sorted + (vs+8) * D + d, (ulong)D, ulong2(0,0), true);
                simdgroup_load(b2, c_sorted + (vs+16) * D + d, (ulong)D, ulong2(0,0), true);
                simdgroup_load(b3, c_sorted + (vs+24) * D + d, (ulong)D, ulong2(0,0), true);

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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Post-MMA filter: check logits and targets
        if (tid == 0) {{
            float flag = 0.0f;
            for (uint i = 0; i < 32 && flag < 0.5f; i++) {{
                uint t = sorted_targets[bs + i];
                if (t >= vs && t < vs + 32) {{ flag = 1.0f; break; }}
                for (uint j = 0; j < 32; j++) {{
                    if (shared[i * 32 + j] - lse[bs + i] >= FILTER_THRESHOLD) {{
                        flag = 1.0f; break;
                    }}
                }}
            }}
            shared[1024] = flag;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[1024] < 0.5f) continue;

        #define ACCUM_TOKEN(TI, DE_ARR) {{ \
            uint t_##TI = t_start + TI; \
            float w_##TI[32]; \
            for (uint j = 0; j < 32; j++) {{ \
                float dot_val = shared[t_##TI * 32 + j]{bias_add}; \
                float wj = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \
                if ((vs + j) == my_target[TI]) wj -= my_d_nll[TI]; \
                w_##TI[j] = wj; \
            }} \
            uint idx_##TI = 0; \
            for (uint d = simd_lane; d < D; d += 32) {{ \
                float acc = 0.0f; \
                for (uint j = 0; j < 32; j++) {{ \
                    acc += w_##TI[j] * (float)c_sorted[(vs + j) * D + d]; \
                }} \
                DE_ARR[idx_##TI] += acc; \
                idx_##TI++; \
            }} \
        }}
        ACCUM_TOKEN(0, dE0)
        ACCUM_TOKEN(1, dE1)
        ACCUM_TOKEN(2, dE2)
        ACCUM_TOKEN(3, dE3)
        #undef ACCUM_TOKEN

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #define WRITE_TOKEN(TI, DE_ARR) {{ \
        uint widx_##TI = 0; \
        for (uint d = simd_lane; d < D; d += 32) {{ \
            atomic_fetch_add_explicit( \
                (device atomic<float>*)(dE + (bs + t_start + TI) * D + d), \
                DE_ARR[widx_##TI++], memory_order_relaxed); \
        }} \
    }}
    WRITE_TOKEN(0, dE0)
    WRITE_TOKEN(1, dE1)
    WRITE_TOKEN(2, dE2)
    WRITE_TOKEN(3, dE3)
    #undef WRITE_TOKEN
"""


def _get_backward_v4_kernel(has_bias):
    key = ("cce_bwd_v4", has_bias)
    if key not in _kernel_cache:
        header = (
            f"constant uint BWD_V_TILE = {BWD_V_TILE};\n"
            f"constant float FILTER_THRESHOLD = {FILTER_LOG_EPS:.10f}f;"
        )
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v4b",
                input_names=["e", "c_sorted", "bias_sorted", "lse", "sorted_targets", "d_nll_val"],
                output_names=["dE"],
                source=_build_backward_v4_source(has_bias=True),
                header=header,
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v4nb",
                input_names=["e", "c_sorted", "lse", "sorted_targets", "d_nll_val"],
                output_names=["dE"],
                source=_build_backward_v4_source(has_bias=False),
                header=header,
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def fused_cce_backward_v4(e, c_sorted, lse, sorted_targets, d_nll, bias_sorted=None):
    """v4: vocab-sorted backward with post-MMA filtering + atomic output."""
    B, D = e.shape
    V = c_sorted.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_backward_v4_kernel(has_bias=bias_sorted is not None)
    if bias_sorted is not None:
        inputs = [e, c_sorted, bias_sorted, lse, sorted_targets, d_nll]
    else:
        inputs = [e, c_sorted, lse, sorted_targets, d_nll]

    (dE,) = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * D,)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )

    return dE.reshape(B, D).astype(e.dtype)


# ---------------------------------------------------------------------------
# v3_full backward: dE + dC + dBias with pre-MMA filtering (GAP 6)
# Same filtering/MMA as v3, adds atomic dC and dBias accumulation.
# ---------------------------------------------------------------------------

def _build_backward_v3_full_source(has_bias):
    bias_add = " + (float)bias[vs + j]" if has_bias else ""
    dbias_block = ""
    if has_bias:
        dbias_block = (
            "            if (simd_lane == 0) { \\\n"
            "                for (uint j = 0; j < 32; j++) { \\\n"
            "                    atomic_fetch_add_explicit( \\\n"
            "                        (device atomic<float>*)(dBias + vs + j), \\\n"
            "                        w_##TI[j], memory_order_relaxed); \\\n"
            "                } \\\n"
            "            } \\\n"
        )
    return f"""\
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;
    uint bs = tg_x * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];
    uint num_fwd_tiles = (V + 31) / 32;

    uint v_tile_start = tg_y * BWD_V_TILE;
    uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);

    threadgroup float shared[1024];

    uint t_start = simd_group * 4;
    uint d_per_lane = (D + 31) / 32;

    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {{
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }}

    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {{
        my_d_nll[i] = d_nll_val[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = targets[bs + t_start + i];
    }}

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {{
        if (tid == 0) {{
            uint fwd_tile = vs / 32;
            float flag = 0.0f;
            for (uint i = 0; i < 32; i++) {{
                uint t = targets[bs + i];
                if (t >= vs && t < vs + 32) {{ flag = 1.0f; break; }}
                if (tile_max[(bs + i) * num_fwd_tiles + fwd_tile] - lse[bs + i] >= FILTER_THRESHOLD) {{
                    flag = 1.0f;
                    break;
                }}
            }}
            shared[0] = flag;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[0] < 0.5f) continue;

        if (simd_group == 0) {{
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
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #define ACCUM_FULL(TI, DE_ARR) {{ \\
            uint t_##TI = t_start + TI; \\
            float w_##TI[32]; \\
            for (uint j = 0; j < 32; j++) {{ \\
                float dot_val = shared[t_##TI * 32 + j]{bias_add}; \\
                float wj = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \\
                if ((vs + j) == my_target[TI]) wj -= my_d_nll[TI]; \\
                w_##TI[j] = wj; \\
            }} \\
            uint idx_##TI = 0; \\
            for (uint d = simd_lane; d < D; d += 32) {{ \\
                float acc = 0.0f; \\
                for (uint j = 0; j < 32; j++) {{ \\
                    acc += w_##TI[j] * (float)c[(vs + j) * D + d]; \\
                }} \\
                DE_ARR[idx_##TI] += acc; \\
                idx_##TI++; \\
            }} \\
            for (uint d = simd_lane; d < D; d += 32) {{ \\
                float e_val = (float)e[(bs + t_##TI) * D + d]; \\
                for (uint j = 0; j < 32; j++) {{ \\
                    atomic_fetch_add_explicit( \\
                        (device atomic<float>*)(dC + (vs + j) * D + d), \\
                        w_##TI[j] * e_val, memory_order_relaxed); \\
                }} \\
            }} \\
{dbias_block}        }}
        ACCUM_FULL(0, dE0)
        ACCUM_FULL(1, dE1)
        ACCUM_FULL(2, dE2)
        ACCUM_FULL(3, dE3)
        #undef ACCUM_FULL

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #define WRITE_TOKEN(TI, DE_ARR) {{ \\
        uint widx_##TI = 0; \\
        for (uint d = simd_lane; d < D; d += 32) {{ \\
            atomic_fetch_add_explicit( \\
                (device atomic<float>*)(dE + (bs + t_start + TI) * D + d), \\
                DE_ARR[widx_##TI++], memory_order_relaxed); \\
        }} \\
    }}
    WRITE_TOKEN(0, dE0)
    WRITE_TOKEN(1, dE1)
    WRITE_TOKEN(2, dE2)
    WRITE_TOKEN(3, dE3)
    #undef WRITE_TOKEN
"""


def _get_backward_v3_full_kernel(has_bias):
    key = ("cce_bwd_v3_full", has_bias)
    if key not in _kernel_cache:
        header = (
            f"constant uint BWD_V_TILE = {BWD_V_TILE};\n"
            f"constant float FILTER_THRESHOLD = {FILTER_LOG_EPS:.10f}f;"
        )
        if has_bias:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v3fb",
                input_names=["e", "c", "bias", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["dE", "dC", "dBias"],
                source=_build_backward_v3_full_source(has_bias=True),
                header=header,
                ensure_row_contiguous=True,
            )
        else:
            _kernel_cache[key] = mx.fast.metal_kernel(
                name="cce_bwd_v3fnb",
                input_names=["e", "c", "lse", "targets", "d_nll_val", "tile_max"],
                output_names=["dE", "dC"],
                source=_build_backward_v3_full_source(has_bias=False),
                header=header,
                ensure_row_contiguous=True,
            )
    return _kernel_cache[key]


def fused_cce_backward_v3_full(e, c, lse, targets, d_nll, tile_max, bias=None):
    """v3_full: atomic dE + dC + dBias, pre-MMA filter, target subtraction."""
    B, D = e.shape
    V = c.shape[0]
    assert B % 32 == 0, f"B must be divisible by 32, got {B}"
    assert V % 32 == 0, f"V must be divisible by 32, got {V}"
    assert D % 8 == 0, f"D must be divisible by 8, got {D}"

    has_bias = bias is not None
    num_b_tiles = B // 32
    num_v_tiles = (V + BWD_V_TILE - 1) // BWD_V_TILE

    kernel = _get_backward_v3_full_kernel(has_bias=has_bias)
    if has_bias:
        inputs = [e, c, bias, lse, targets, d_nll, tile_max]
    else:
        inputs = [e, c, lse, targets, d_nll, tile_max]

    output_shapes = [(B * D,), (V * D,)]
    output_dtypes = [mx.float32, mx.float32]
    if has_bias:
        output_shapes.append((V,))
        output_dtypes.append(mx.float32)

    results = kernel(
        inputs=inputs,
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 256, num_v_tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        init_value=0.0,
    )

    dE = results[0].reshape(B, D).astype(e.dtype)
    dC = results[1].reshape(V, D).astype(c.dtype)
    dBias = results[2].astype(bias.dtype) if has_bias else None
    return dE, dC, dBias
