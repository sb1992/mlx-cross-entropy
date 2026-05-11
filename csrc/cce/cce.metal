// Copyright © 2024 Apple Inc.
// Cut Cross-Entropy Metal kernels — fused forward + filtered backward
// Reference: "Cut Your Losses in Large-Vocabulary Language Models"
//            Wijmans et al., arXiv 2411.09009

#include <metal_atomic>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// ==========================================================================
// FORWARD: tiled MMA logits + logsumexp stats, never materialize [B, V]
// Grid: (B/32, V/32, 1)   Threadgroup: (32, 1, 1) = 1 SIMD group
// Single SIMD group iterates 4 row blocks of 8 rows × 32 cols each.
// ==========================================================================

template <typename T, bool HAS_BIAS, bool FUSE_TARGET>
[[kernel]] void cce_forward(
    const device T* e           [[buffer(0)]],
    const device T* c           [[buffer(1)]],
    const device T* bias        [[buffer(2)]],
    const device uint32_t* targets [[buffer(3)]],
    device float* tile_max      [[buffer(4)]],
    device float* tile_sum_exp  [[buffer(5)]],
    device float* neg_target_logit [[buffer(6)]],
    constant const uint3& shape [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {

    const uint D = shape[1];
    const uint V = shape[2];
    const uint num_v_tiles = (V + 31) / 32;
    const uint bs = tgid.x * 32;
    const uint vs = tgid.y * 32;

    threadgroup float shared[1024];

    simdgroup_float8x8 d00,d01,d02,d03,d10,d11,d12,d13;
    simdgroup_float8x8 d20,d21,d22,d23,d30,d31,d32,d33;
    d00.thread_elements()[0]=0; d00.thread_elements()[1]=0;
    d01.thread_elements()[0]=0; d01.thread_elements()[1]=0;
    d02.thread_elements()[0]=0; d02.thread_elements()[1]=0;
    d03.thread_elements()[0]=0; d03.thread_elements()[1]=0;
    d10.thread_elements()[0]=0; d10.thread_elements()[1]=0;
    d11.thread_elements()[0]=0; d11.thread_elements()[1]=0;
    d12.thread_elements()[0]=0; d12.thread_elements()[1]=0;
    d13.thread_elements()[0]=0; d13.thread_elements()[1]=0;
    d20.thread_elements()[0]=0; d20.thread_elements()[1]=0;
    d21.thread_elements()[0]=0; d21.thread_elements()[1]=0;
    d22.thread_elements()[0]=0; d22.thread_elements()[1]=0;
    d23.thread_elements()[0]=0; d23.thread_elements()[1]=0;
    d30.thread_elements()[0]=0; d30.thread_elements()[1]=0;
    d31.thread_elements()[0]=0; d31.thread_elements()[1]=0;
    d32.thread_elements()[0]=0; d32.thread_elements()[1]=0;
    d33.thread_elements()[0]=0; d33.thread_elements()[1]=0;

    for (uint k = 0; k < D; k += 8) {
        simdgroup_matrix<T, 8, 8> a0, a1, a2, a3, b0, b1, b2, b3;
        simdgroup_load(a0, e + bs * D + k, (ulong)D);
        simdgroup_load(a1, e + (bs + 8) * D + k, (ulong)D);
        simdgroup_load(a2, e + (bs + 16) * D + k, (ulong)D);
        simdgroup_load(a3, e + (bs + 24) * D + k, (ulong)D);
        simdgroup_load(b0, c + vs * D + k, (ulong)D, ulong2(0, 0), true);
        simdgroup_load(b1, c + (vs + 8) * D + k, (ulong)D, ulong2(0, 0), true);
        simdgroup_load(b2, c + (vs + 16) * D + k, (ulong)D, ulong2(0, 0), true);
        simdgroup_load(b3, c + (vs + 24) * D + k, (ulong)D, ulong2(0, 0), true);

        simdgroup_multiply_accumulate(d00, a0, b0, d00);
        simdgroup_multiply_accumulate(d01, a0, b1, d01);
        simdgroup_multiply_accumulate(d02, a0, b2, d02);
        simdgroup_multiply_accumulate(d03, a0, b3, d03);
        simdgroup_multiply_accumulate(d10, a1, b0, d10);
        simdgroup_multiply_accumulate(d11, a1, b1, d11);
        simdgroup_multiply_accumulate(d12, a1, b2, d12);
        simdgroup_multiply_accumulate(d13, a1, b3, d13);
        simdgroup_multiply_accumulate(d20, a2, b0, d20);
        simdgroup_multiply_accumulate(d21, a2, b1, d21);
        simdgroup_multiply_accumulate(d22, a2, b2, d22);
        simdgroup_multiply_accumulate(d23, a2, b3, d23);
        simdgroup_multiply_accumulate(d30, a3, b0, d30);
        simdgroup_multiply_accumulate(d31, a3, b1, d31);
        simdgroup_multiply_accumulate(d32, a3, b2, d32);
        simdgroup_multiply_accumulate(d33, a3, b3, d33);
    }

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

    // First 32 threads handle the 32 rows for reduction
    if (tid < 32) {
        if (FUSE_TARGET) {
            uint my_target = targets[bs + tid];
            if (my_target >= vs && my_target < vs + 32) {
                float tgt_logit = shared[tid * 32 + (my_target - vs)];
                if (HAS_BIAS) tgt_logit += static_cast<float>(bias[my_target]);
                neg_target_logit[bs + tid] = -tgt_logit;
            }
        }

        float row_max = -1e38f;
        for (uint j = 0; j < 32; j++) {
            float val = shared[tid * 32 + j];
            if (HAS_BIAS) val += static_cast<float>(bias[vs + j]);
            row_max = fmax(row_max, val);
        }

        float row_sum = 0.0f;
        for (uint j = 0; j < 32; j++) {
            float val = shared[tid * 32 + j];
            if (HAS_BIAS) val += static_cast<float>(bias[vs + j]);
            row_sum += exp(val - row_max);
        }

        uint out_idx = (bs + tid) * num_v_tiles + tgid.y;
        tile_max[out_idx] = row_max;
        tile_sum_exp[out_idx] = row_sum;
    }
}

// ==========================================================================
// BACKWARD: gradient filtering + atomic accumulation
// Grid: (B/32, V/4096, 1)   Threadgroup: (256, 1, 1) = 8 SIMD groups
// SIMD group 0 computes 32×32 MMA (16 accumulators), all 8 accumulate grads.
// ==========================================================================

constant constexpr float FILTER_LOG_EPS = -8.3177661667f;
constant constexpr uint BWD_V_TILE = 4096;

template <typename T, bool HAS_BIAS, bool COMPUTE_DC>
[[kernel]] void cce_backward(
    const device T* e              [[buffer(0)]],
    const device T* c              [[buffer(1)]],
    const device T* bias           [[buffer(2)]],
    const device float* lse        [[buffer(3)]],
    const device uint32_t* targets [[buffer(4)]],
    const device float* d_nll      [[buffer(5)]],
    const device float* tile_max_buf [[buffer(6)]],
    device float* dE               [[buffer(7)]],
    device float* dC               [[buffer(8)]],
    device float* dBias            [[buffer(9)]],
    constant const uint3& shape    [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {

    const uint D = shape[1];
    const uint V = shape[2];
    const uint num_fwd_tiles = (V + 31) / 32;

    const uint bs = tgid.x * 32;
    const uint v_tile_start = tgid.y * BWD_V_TILE;
    const uint v_tile_end = min(v_tile_start + BWD_V_TILE, V);
    const uint t_start = simd_group * 4;

    threadgroup float shared[1024];

    const uint d_per_lane = (D + 31) / 32;
    float dE0[128], dE1[128], dE2[128], dE3[128];
    for (uint i = 0; i < d_per_lane; i++) {
        dE0[i] = 0.0f; dE1[i] = 0.0f;
        dE2[i] = 0.0f; dE3[i] = 0.0f;
    }

    float my_d_nll[4], my_lse[4];
    uint my_target[4];
    for (uint i = 0; i < 4; i++) {
        my_d_nll[i] = d_nll[bs + t_start + i];
        my_lse[i] = lse[bs + t_start + i];
        my_target[i] = targets[bs + t_start + i];
    }

    for (uint vs = v_tile_start; vs < v_tile_end; vs += 32) {
        if (tid == 0) {
            uint fwd_tile = vs / 32;
            float flag = 0.0f;
            for (uint i = 0; i < 32; i++) {
                uint t = targets[bs + i];
                if (t >= vs && t < vs + 32) { flag = 1.0f; break; }
                if (tile_max_buf[(bs + i) * num_fwd_tiles + fwd_tile] - lse[bs + i] >= FILTER_LOG_EPS) {
                    flag = 1.0f; break;
                }
            }
            shared[0] = flag;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared[0] < 0.5f) continue;

        if (simd_group == 0) {
            simdgroup_float8x8 d00,d01,d02,d03,d10,d11,d12,d13;
            simdgroup_float8x8 d20,d21,d22,d23,d30,d31,d32,d33;
            d00.thread_elements()[0]=0; d00.thread_elements()[1]=0;
            d01.thread_elements()[0]=0; d01.thread_elements()[1]=0;
            d02.thread_elements()[0]=0; d02.thread_elements()[1]=0;
            d03.thread_elements()[0]=0; d03.thread_elements()[1]=0;
            d10.thread_elements()[0]=0; d10.thread_elements()[1]=0;
            d11.thread_elements()[0]=0; d11.thread_elements()[1]=0;
            d12.thread_elements()[0]=0; d12.thread_elements()[1]=0;
            d13.thread_elements()[0]=0; d13.thread_elements()[1]=0;
            d20.thread_elements()[0]=0; d20.thread_elements()[1]=0;
            d21.thread_elements()[0]=0; d21.thread_elements()[1]=0;
            d22.thread_elements()[0]=0; d22.thread_elements()[1]=0;
            d23.thread_elements()[0]=0; d23.thread_elements()[1]=0;
            d30.thread_elements()[0]=0; d30.thread_elements()[1]=0;
            d31.thread_elements()[0]=0; d31.thread_elements()[1]=0;
            d32.thread_elements()[0]=0; d32.thread_elements()[1]=0;
            d33.thread_elements()[0]=0; d33.thread_elements()[1]=0;

            for (uint k = 0; k < D; k += 8) {
                simdgroup_matrix<T, 8, 8> a0, a1, a2, a3, b0, b1, b2, b3;
                simdgroup_load(a0, e + bs * D + k, (ulong)D);
                simdgroup_load(a1, e + (bs + 8) * D + k, (ulong)D);
                simdgroup_load(a2, e + (bs + 16) * D + k, (ulong)D);
                simdgroup_load(a3, e + (bs + 24) * D + k, (ulong)D);
                simdgroup_load(b0, c + vs * D + k, (ulong)D, ulong2(0, 0), true);
                simdgroup_load(b1, c + (vs + 8) * D + k, (ulong)D, ulong2(0, 0), true);
                simdgroup_load(b2, c + (vs + 16) * D + k, (ulong)D, ulong2(0, 0), true);
                simdgroup_load(b3, c + (vs + 24) * D + k, (ulong)D, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(d00, a0, b0, d00);
                simdgroup_multiply_accumulate(d01, a0, b1, d01);
                simdgroup_multiply_accumulate(d02, a0, b2, d02);
                simdgroup_multiply_accumulate(d03, a0, b3, d03);
                simdgroup_multiply_accumulate(d10, a1, b0, d10);
                simdgroup_multiply_accumulate(d11, a1, b1, d11);
                simdgroup_multiply_accumulate(d12, a1, b2, d12);
                simdgroup_multiply_accumulate(d13, a1, b3, d13);
                simdgroup_multiply_accumulate(d20, a2, b0, d20);
                simdgroup_multiply_accumulate(d21, a2, b1, d21);
                simdgroup_multiply_accumulate(d22, a2, b2, d22);
                simdgroup_multiply_accumulate(d23, a2, b3, d23);
                simdgroup_multiply_accumulate(d30, a3, b0, d30);
                simdgroup_multiply_accumulate(d31, a3, b1, d31);
                simdgroup_multiply_accumulate(d32, a3, b2, d32);
                simdgroup_multiply_accumulate(d33, a3, b3, d33);
            }

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
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #define PROCESS_TOKEN(TI) { \
            float w[32]; \
            for (uint j = 0; j < 32; j++) { \
                float dot_val = shared[(t_start + TI) * 32 + j]; \
                if (HAS_BIAS) dot_val += static_cast<float>(bias[vs + j]); \
                float wj = my_d_nll[TI] * exp(dot_val - my_lse[TI]); \
                if ((vs + j) == my_target[TI]) wj -= my_d_nll[TI]; \
                w[j] = wj; \
            } \
            uint idx = 0; \
            for (uint d = simd_lane; d < D; d += 32) { \
                float acc = 0.0f; \
                for (uint j = 0; j < 32; j++) { \
                    acc += w[j] * static_cast<float>(c[(vs + j) * D + d]); \
                } \
                dE##TI[idx] += acc; \
                idx++; \
            } \
            if (COMPUTE_DC) { \
                for (uint d = simd_lane; d < D; d += 32) { \
                    float e_val = static_cast<float>(e[(bs + t_start + TI) * D + d]); \
                    for (uint j = 0; j < 32; j++) { \
                        atomic_fetch_add_explicit( \
                            (device atomic<float>*)&dC[(vs + j) * D + d], w[j] * e_val, memory_order_relaxed); \
                    } \
                } \
                if (HAS_BIAS && simd_lane == 0) { \
                    for (uint j = 0; j < 32; j++) { \
                        atomic_fetch_add_explicit((device atomic<float>*)&dBias[vs + j], w[j], memory_order_relaxed); \
                    } \
                } \
            } \
        }

        PROCESS_TOKEN(0)
        PROCESS_TOKEN(1)
        PROCESS_TOKEN(2)
        PROCESS_TOKEN(3)
        #undef PROCESS_TOKEN

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #define WRITE_DE(TI) { \
        uint idx = 0; \
        for (uint d = simd_lane; d < D; d += 32) { \
            atomic_fetch_add_explicit((device atomic<float>*)&dE[(bs + t_start + TI) * D + d], dE##TI[idx], memory_order_relaxed); \
            idx++; \
        } \
    }
    WRITE_DE(0) WRITE_DE(1) WRITE_DE(2) WRITE_DE(3)
    #undef WRITE_DE
}

// ==========================================================================
// Reduction: tile_max + tile_sum_exp → lse  (one thread per batch element)
// ==========================================================================

[[kernel]] void reduce_tile_to_lse(
    const device float* tile_max     [[buffer(0)]],
    const device float* tile_sum_exp [[buffer(1)]],
    device float* lse                [[buffer(2)]],
    constant const uint2& params     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {

    uint B = params[0];
    uint num_tiles = params[1];
    if (tid >= B) return;

    const device float* tm = tile_max + tid * num_tiles;
    const device float* ts = tile_sum_exp + tid * num_tiles;

    float gmax = -INFINITY;
    for (uint v = 0; v < num_tiles; v++)
        gmax = max(gmax, tm[v]);

    float gsum = 0.0f;
    for (uint v = 0; v < num_tiles; v++)
        gsum += ts[v] * exp(tm[v] - gmax);

    lse[tid] = gmax + log(gsum);
}

// ==========================================================================
// FILL: GPU-side zero initialization
// ==========================================================================

[[kernel]] void fill_zero_f32(
    device float* buf [[buffer(0)]],
    constant const uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < count) buf[tid] = 0.0f;
}

// ==========================================================================
// Template instantiations
// ==========================================================================

// Forward: <T, HAS_BIAS, FUSE_TARGET>
#define instantiate_cce_fwd(tname, type, bias, target) \
  template [[host_name("cce_fwd_" #tname "_b" #bias "_t" #target)]] \
  [[kernel]] void cce_forward<type, bias, target>( \
      const device type*, const device type*, const device type*, \
      const device uint32_t*, device float*, device float*, \
      device float*, constant const uint3&, uint3, uint);

instantiate_cce_fwd(float32, float, false, false)
instantiate_cce_fwd(float32, float, true,  false)
instantiate_cce_fwd(float32, float, false, true)
instantiate_cce_fwd(float32, float, true,  true)
instantiate_cce_fwd(float16, half, false, false)
instantiate_cce_fwd(float16, half, true,  false)
instantiate_cce_fwd(float16, half, false, true)
instantiate_cce_fwd(float16, half, true,  true)
instantiate_cce_fwd(bfloat16, bfloat16_t, false, false)
instantiate_cce_fwd(bfloat16, bfloat16_t, true,  false)
instantiate_cce_fwd(bfloat16, bfloat16_t, false, true)
instantiate_cce_fwd(bfloat16, bfloat16_t, true,  true)
#undef instantiate_cce_fwd

// Backward: <T, HAS_BIAS, COMPUTE_DC>
#define instantiate_cce_bwd(tname, type, bias, dc) \
  template [[host_name("cce_bwd_" #tname "_b" #bias "_dc" #dc)]] \
  [[kernel]] void cce_backward<type, bias, dc>( \
      const device type*, const device type*, const device type*, \
      const device float*, const device uint32_t*, const device float*, \
      const device float*, device float*, device float*, \
      device float*, constant const uint3&, uint3, uint, uint, uint);

instantiate_cce_bwd(float32, float, false, false)
instantiate_cce_bwd(float32, float, true,  false)
instantiate_cce_bwd(float32, float, false, true)
instantiate_cce_bwd(float32, float, true,  true)
instantiate_cce_bwd(float16, half, false, false)
instantiate_cce_bwd(float16, half, true,  false)
instantiate_cce_bwd(float16, half, false, true)
instantiate_cce_bwd(float16, half, true,  true)
instantiate_cce_bwd(bfloat16, bfloat16_t, false, false)
instantiate_cce_bwd(bfloat16, bfloat16_t, true,  false)
instantiate_cce_bwd(bfloat16, bfloat16_t, false, true)
instantiate_cce_bwd(bfloat16, bfloat16_t, true,  true)
#undef instantiate_cce_bwd
