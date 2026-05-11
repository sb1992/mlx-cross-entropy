#include <cstring>
#include <dlfcn.h>
#include <filesystem>

#include "mlx/utils.h"

#include "cce/cce.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace cce {

static std::string current_binary_dir() {
  static std::string dir = []() -> std::string {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

// =========================================================================
// Public API
// =========================================================================

mx::array cut_cross_entropy(
    const mx::array& e,
    const mx::array& c,
    const mx::array& targets,
    const mx::array& bias,
    const std::string& reduction,
    int ignore_index,
    bool compute_all_grads,
    mx::StreamOrDevice s) {

  if (e.ndim() != 2)
    throw std::invalid_argument(
        "[cut_cross_entropy] e must be 2D [B, D], got " +
        std::to_string(e.ndim()) + "D");
  if (c.ndim() != 2)
    throw std::invalid_argument(
        "[cut_cross_entropy] c must be 2D [V, D], got " +
        std::to_string(c.ndim()) + "D");
  if (targets.ndim() != 1)
    throw std::invalid_argument(
        "[cut_cross_entropy] targets must be 1D [B], got " +
        std::to_string(targets.ndim()) + "D");

  int B = e.shape(0);
  int D = e.shape(1);
  int V = c.shape(0);

  if (c.shape(1) != D)
    throw std::invalid_argument(
        "[cut_cross_entropy] c.shape[1] must equal e.shape[1] (D)");
  if (targets.shape(0) != B)
    throw std::invalid_argument(
        "[cut_cross_entropy] targets.shape[0] must equal e.shape[0] (B)");
  if (B % 32 != 0 || V % 32 != 0 || D % 8 != 0)
    throw std::invalid_argument(
        "[cut_cross_entropy] requires B%32==0, V%32==0, D%8==0. Got B=" +
        std::to_string(B) + ", V=" + std::to_string(V) + ", D=" +
        std::to_string(D));

  bool has_bias = bias.size() > 0;
  if (has_bias && (bias.ndim() != 1 || bias.shape(0) != V))
    throw std::invalid_argument(
        "[cut_cross_entropy] bias must be 1D [V]");

  auto stream = mx::to_stream(s);
  int num_v_tiles = (V + 31) / 32;

  // Primitive outputs: tile_max[B*V/32], tile_sum_exp[B*V/32], neg_target_logit[B], lse[B]
  // lse is computed on-GPU via reduce_tile_to_lse kernel inside eval_gpu
  std::vector<mx::Shape> out_shapes = {
      {B * num_v_tiles},   // tile_max
      {B * num_v_tiles},   // tile_sum_exp
      {B},                 // neg_target_logit
      {B}};                // lse (reduced from tile stats on GPU)
  std::vector<mx::Dtype> out_dtypes = {
      mx::float32, mx::float32, mx::float32, mx::float32};

  std::vector<mx::array> inputs = {e, c, targets};
  if (has_bias) inputs.push_back(bias);

  auto prim = std::make_shared<CutCrossEntropy>(
      stream, has_bias, compute_all_grads);

  auto results = mx::array::make_arrays(
      out_shapes, out_dtypes, prim, inputs);

  auto& lse = results[3];
  auto& neg_target_logit = results[2];
  auto nll = lse + neg_target_logit;

  if (reduction == "mean") {
    auto valid = mx::not_equal(targets, mx::array(ignore_index));
    nll = nll * mx::astype(valid, mx::float32);
    auto count = mx::maximum(mx::sum(valid), mx::array(1));
    return mx::sum(nll) / mx::astype(count, mx::float32);
  } else if (reduction == "sum") {
    auto valid = mx::not_equal(targets, mx::array(ignore_index));
    nll = nll * mx::astype(valid, mx::float32);
    return mx::sum(nll);
  } else if (reduction == "none") {
    auto valid = mx::not_equal(targets, mx::array(ignore_index));
    return nll * mx::astype(valid, mx::float32);
  }
  throw std::invalid_argument(
      "[cut_cross_entropy] Unknown reduction: " + reduction);
}

// =========================================================================
// Raw forward: returns (tile_max_flat, tile_sum_exp_flat, neg_target_logit, lse)
// For use with mx.custom_function in Python
// =========================================================================

std::vector<mx::array> cce_forward_raw(
    const mx::array& e,
    const mx::array& c,
    const mx::array& targets,
    const mx::array& bias,
    mx::StreamOrDevice s) {

  int B = e.shape(0), D = e.shape(1), V = c.shape(0);
  bool has_bias = bias.size() > 0;
  auto stream = mx::to_stream(s);
  int num_v_tiles = (V + 31) / 32;

  std::vector<mx::Shape> out_shapes = {
      {B * num_v_tiles}, {B * num_v_tiles}, {B}, {B}};
  std::vector<mx::Dtype> out_dtypes = {
      mx::float32, mx::float32, mx::float32, mx::float32};

  std::vector<mx::array> inputs = {e, c, targets};
  if (has_bias) inputs.push_back(bias);

  auto prim = std::make_shared<CutCrossEntropy>(stream, has_bias, false);
  return mx::array::make_arrays(out_shapes, out_dtypes, prim, inputs);
}

// =========================================================================
// Raw backward: takes forward outputs + d_nll, returns dE
// For use with mx.custom_function in Python
// =========================================================================

mx::array cce_backward_raw(
    const mx::array& e,
    const mx::array& c,
    const mx::array& lse,
    const mx::array& targets,
    const mx::array& d_nll,
    const mx::array& tile_max,
    const mx::array& bias,
    mx::StreamOrDevice s) {

  int B = e.shape(0), D = e.shape(1), V = c.shape(0);
  bool has_bias = bias.size() > 0;
  auto stream = mx::to_stream(s);

  std::vector<mx::array> bwd_inputs = {e, c};
  bwd_inputs.push_back(has_bias ? bias : mx::array({}));
  bwd_inputs.push_back(lse);
  bwd_inputs.push_back(targets);
  bwd_inputs.push_back(d_nll);
  bwd_inputs.push_back(tile_max);

  std::vector<mx::Shape> bwd_shapes = {{B, D}};
  std::vector<mx::Dtype> bwd_dtypes = {mx::float32};

  auto bwd_prim = std::make_shared<CutCrossEntropyBwd>(stream, has_bias, false);
  auto vjps = mx::array::make_arrays(bwd_shapes, bwd_dtypes, bwd_prim, bwd_inputs);
  return mx::astype(vjps[0], e.dtype());
}

// =========================================================================
// CPU eval — not implemented, GPU-only operation
// =========================================================================

void CutCrossEntropy::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[CutCrossEntropy] CPU evaluation not implemented. "
      "Use mx.gpu stream.");
}

void CutCrossEntropyBwd::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[CutCrossEntropyBwd] CPU evaluation not implemented. "
      "Use mx.gpu stream.");
}

// =========================================================================
// GPU dispatch (forward)
// =========================================================================

#ifdef _METAL_

void CutCrossEntropy::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& d = mx::metal::device(stream().device);

  auto& e = inputs[0];
  auto& c = inputs[1];
  auto& targets = inputs[2];

  uint32_t B = e.shape(0);
  uint32_t D = e.shape(1);
  uint32_t V = c.shape(0);
  uint32_t num_v_tiles = (V + 31) / 32;

  std::string type_name;
  switch (e.dtype()) {
    case mx::float32: type_name = "float32"; break;
    case mx::float16: type_name = "float16"; break;
    case mx::bfloat16: type_name = "bfloat16"; break;
    default: throw std::runtime_error("CCE: unsupported dtype");
  }

  std::string kernel_name = "cce_fwd_" + type_name +
      "_b" + (has_bias_ ? "true" : "false") + "_ttrue";

  auto lib = d.get_library("mlx_cce_ext", current_binary_dir());
  auto kernel = d.get_kernel(kernel_name, lib);
  auto& enc = mx::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(kernel);

  auto& tile_max = outputs[0];
  auto& tile_sum_exp = outputs[1];
  auto& neg_tgt = outputs[2];

  auto& lse = outputs[3];

  tile_max.set_data(mx::allocator::malloc(tile_max.nbytes()));
  tile_sum_exp.set_data(mx::allocator::malloc(tile_sum_exp.nbytes()));
  neg_tgt.set_data(mx::allocator::malloc(neg_tgt.nbytes()));
  lse.set_data(mx::allocator::malloc(lse.nbytes()));

  // Forward MMA kernel
  enc.set_input_array(e, 0);
  enc.set_input_array(c, 1);
  enc.set_input_array(has_bias_ ? inputs[3] : e, 2);
  enc.set_input_array(targets, 3);
  enc.set_output_array(tile_max, 4);
  enc.set_output_array(tile_sum_exp, 5);
  enc.set_output_array(neg_tgt, 6);

  uint32_t shape[3] = {B, D, V};
  enc.set_bytes(shape, sizeof(shape), 7);

  uint32_t num_b_tiles = B / 32;
  MTL::Size grid(num_b_tiles, num_v_tiles, 1);
  MTL::Size tg(32, 1, 1);
  enc.dispatch_threadgroups(grid, tg);

  // Reduce tile stats → lse on GPU (avoids intermediate graph ops)
  auto reduce_kernel = d.get_kernel("reduce_tile_to_lse", lib);
  enc.set_compute_pipeline_state(reduce_kernel);
  enc.set_input_array(tile_max, 0);
  enc.set_input_array(tile_sum_exp, 1);
  enc.set_output_array(lse, 2);
  uint32_t reduce_params[2] = {B, num_v_tiles};
  enc.set_bytes(reduce_params, sizeof(reduce_params), 3);
  constexpr uint32_t reduce_tg = 256;
  enc.dispatch_threadgroups(
      MTL::Size((B + reduce_tg - 1) / reduce_tg, 1, 1),
      MTL::Size(reduce_tg, 1, 1));
}

// =========================================================================
// GPU dispatch (backward)
// =========================================================================

void CutCrossEntropyBwd::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& d = mx::metal::device(stream().device);

  auto& e = inputs[0];
  auto& c = inputs[1];
  auto& lse = inputs[3];
  auto& targets = inputs[4];
  auto& d_nll = inputs[5];
  auto& tile_max_buf = inputs[6];

  uint32_t B = e.shape(0);
  uint32_t D = e.shape(1);
  uint32_t V = c.shape(0);
  uint32_t num_v_tiles = (V + 4095) / 4096;

  std::string type_name;
  switch (e.dtype()) {
    case mx::float32: type_name = "float32"; break;
    case mx::float16: type_name = "float16"; break;
    case mx::bfloat16: type_name = "bfloat16"; break;
    default: throw std::runtime_error("CCE backward: unsupported dtype");
  }

  std::string kernel_name = "cce_bwd_" + type_name +
      "_b" + (has_bias_ ? "true" : "false") +
      "_dc" + (compute_all_grads_ ? "true" : "false");

  auto lib = d.get_library("mlx_cce_ext", current_binary_dir());
  auto& enc = mx::metal::get_command_encoder(stream());

  // Zero-fill outputs via GPU kernel (must use GPU to avoid memset/GPU races)
  auto zero_alloc = [&](mx::array& buf) {
    buf.set_data(mx::allocator::malloc(buf.nbytes()));
    auto fill_kernel = d.get_kernel("fill_zero_f32", lib);
    enc.set_compute_pipeline_state(fill_kernel);
    enc.set_output_array(buf, 0);
    uint32_t count = buf.size();
    enc.set_bytes(&count, sizeof(uint32_t), 1);
    constexpr uint32_t fill_tg = 256;
    enc.dispatch_threadgroups(
        MTL::Size((count + fill_tg - 1) / fill_tg, 1, 1),
        MTL::Size(fill_tg, 1, 1));
  };

  auto& dE = outputs[0];
  zero_alloc(dE);

  if (compute_all_grads_) {
    auto& dC = outputs[1];
    zero_alloc(dC);

    if (has_bias_) {
      auto& dBias = outputs[2];
      zero_alloc(dBias);
    }
  }

  // Dispatch backward kernel
  auto kernel = d.get_kernel(kernel_name, lib);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(e, 0);
  enc.set_input_array(c, 1);
  enc.set_input_array(has_bias_ ? inputs[2] : e, 2);
  enc.set_input_array(lse, 3);
  enc.set_input_array(targets, 4);
  enc.set_input_array(d_nll, 5);
  enc.set_input_array(tile_max_buf, 6);
  enc.set_output_array(dE, 7);

  if (compute_all_grads_) {
    enc.set_output_array(outputs[1], 8);
    if (has_bias_) {
      enc.set_output_array(outputs[2], 9);
    } else {
      enc.set_output_array(dE, 8);
      enc.set_output_array(dE, 9);
    }
  } else {
    enc.set_output_array(dE, 8);
    enc.set_output_array(dE, 9);
  }

  uint32_t shape[3] = {B, D, V};
  enc.set_bytes(shape, sizeof(shape), 10);

  uint32_t num_b_tiles = B / 32;
  MTL::Size grid(num_b_tiles, num_v_tiles, 1);
  MTL::Size tg(256, 1, 1);
  enc.dispatch_threadgroups(grid, tg);
}

#else

void CutCrossEntropy::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("CutCrossEntropy: Metal not built.");
}

void CutCrossEntropyBwd::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("CutCrossEntropyBwd: Metal not built.");
}

#endif

// =========================================================================
// VJP — create backward primitive
// =========================================================================

std::vector<mx::array> CutCrossEntropy::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {

  auto& e = primals[0];
  auto& c = primals[1];
  auto& targets = primals[2];
  // cotangents[3] = d_loss/d_lse; cotangents[2] = d_loss/d_neg_target_logit
  // nll = lse + neg_target_logit, so d_nll flows equally to both
  auto& d_nll = cotangents[3];

  int B = e.shape(0), D = e.shape(1), V = c.shape(0);

  // outputs: [0]=tile_max_flat, [1]=tile_sum_exp_flat, [2]=neg_target_logit, [3]=lse
  auto& lse = outputs[3];
  auto& tile_max = outputs[0];

  std::vector<mx::array> bwd_inputs = {e, c};
  bwd_inputs.push_back(has_bias_ ? primals[3] : mx::array({}));
  bwd_inputs.push_back(lse);
  bwd_inputs.push_back(targets);
  bwd_inputs.push_back(d_nll);
  bwd_inputs.push_back(tile_max);

  std::vector<mx::Shape> bwd_shapes = {{B, D}};
  std::vector<mx::Dtype> bwd_dtypes = {mx::float32};

  if (compute_all_grads_) {
    bwd_shapes.push_back({V, D});
    bwd_dtypes.push_back(mx::float32);
    if (has_bias_) {
      bwd_shapes.push_back({V});
      bwd_dtypes.push_back(mx::float32);
    }
  }

  auto bwd_prim = std::make_shared<CutCrossEntropyBwd>(
      stream(), has_bias_, compute_all_grads_);

  auto vjps = mx::array::make_arrays(
      bwd_shapes, bwd_dtypes, bwd_prim, bwd_inputs);

  std::vector<mx::array> result;
  for (auto& arg : argnums) {
    if (arg == 0) {
      result.push_back(mx::astype(vjps[0], e.dtype()));
    } else if (arg == 1 && compute_all_grads_) {
      result.push_back(mx::astype(vjps[1], c.dtype()));
    } else if (arg == 3 && compute_all_grads_ && has_bias_) {
      result.push_back(mx::astype(vjps.back(), primals[3].dtype()));
    } else {
      result.push_back(mx::zeros_like(primals[arg]));
    }
  }
  return result;
}

}  // namespace cce
