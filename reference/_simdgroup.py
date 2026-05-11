"""High-performance simdgroup_matrix matmul kernel for e @ c.T.

Uses Apple Silicon's simdgroup_float8x8 matrix multiply hardware
(simdgroup_multiply_accumulate) with 4B×4V register blocking:
each SIMD group computes a 32×32 output tile using 16 accumulators,
with hardware transpose via simdgroup_load(..., true).

Beats MLX's built-in matmul at the primary CCE shape (B=256, D=768, V=32K)
and stays competitive at larger sizes.
"""

import mlx.core as mx

# 4B×4V kernel: 32×32 output per SIMD group, 16 accumulators
# Loads: 4 a-blocks (e) + 4 b-blocks (c, transposed via hardware flag)
# MMA ops: 16 per D-chunk → 2.0 MMA/load ratio
CCE_MATMUL_4B4V_SOURCE = """
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint bs = tg_x * 32;
    uint vs = tg_y * 32;
    uint D = e_shape[1];
    uint V = c_shape[0];

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

    for (uint d = 0; d < D; d += 8) {
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
    }

    simdgroup_store(d00, out + bs*V + vs, (ulong)V);
    simdgroup_store(d01, out + bs*V + vs+8, (ulong)V);
    simdgroup_store(d02, out + bs*V + vs+16, (ulong)V);
    simdgroup_store(d03, out + bs*V + vs+24, (ulong)V);
    simdgroup_store(d10, out + (bs+8)*V + vs, (ulong)V);
    simdgroup_store(d11, out + (bs+8)*V + vs+8, (ulong)V);
    simdgroup_store(d12, out + (bs+8)*V + vs+16, (ulong)V);
    simdgroup_store(d13, out + (bs+8)*V + vs+24, (ulong)V);
    simdgroup_store(d20, out + (bs+16)*V + vs, (ulong)V);
    simdgroup_store(d21, out + (bs+16)*V + vs+8, (ulong)V);
    simdgroup_store(d22, out + (bs+16)*V + vs+16, (ulong)V);
    simdgroup_store(d23, out + (bs+16)*V + vs+24, (ulong)V);
    simdgroup_store(d30, out + (bs+24)*V + vs, (ulong)V);
    simdgroup_store(d31, out + (bs+24)*V + vs+8, (ulong)V);
    simdgroup_store(d32, out + (bs+24)*V + vs+16, (ulong)V);
    simdgroup_store(d33, out + (bs+24)*V + vs+24, (ulong)V);
"""

_kernel_cache = {}


def _get_matmul_kernel():
    key = "cce_matmul_4b4v"
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name="cce_matmul_4b4v",
            input_names=["e", "c"],
            output_names=["out"],
            source=CCE_MATMUL_4B4V_SOURCE,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def simdgroup_matmul_eT_c(e, c):
    """Compute e @ c.T using simdgroup_matrix hardware.

    Args:
        e: [B, D] float32, B must be divisible by 32
        c: [V, D] float32, V must be divisible by 32, D must be divisible by 8

    Returns:
        [B, V] float32 = e @ c.T
    """
    B, D = e.shape
    V = c.shape[0]

    kernel = _get_matmul_kernel()
    num_b_tiles = B // 32
    num_v_tiles = (V + 31) // 32

    (out,) = kernel(
        inputs=[e, c],
        template=[("T", e.dtype)],
        grid=(num_b_tiles * 32, num_v_tiles, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B * V,)],
        output_dtypes=[mx.float32],
    )
    return out.reshape(B, V)
