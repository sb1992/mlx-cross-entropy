#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace cce {

mx::array cut_cross_entropy(
    const mx::array& e,
    const mx::array& c,
    const mx::array& targets,
    const mx::array& bias,
    const std::string& reduction,
    int ignore_index,
    bool compute_all_grads,
    mx::StreamOrDevice s = {});

std::vector<mx::array> cce_forward_raw(
    const mx::array& e,
    const mx::array& c,
    const mx::array& targets,
    const mx::array& bias,
    mx::StreamOrDevice s = {});

mx::array cce_backward_raw(
    const mx::array& e,
    const mx::array& c,
    const mx::array& lse,
    const mx::array& targets,
    const mx::array& d_nll,
    const mx::array& tile_max,
    const mx::array& bias,
    mx::StreamOrDevice s = {});

std::vector<mx::array> cce_forward_raw_quantized(
    const mx::array& e,
    const mx::array& c_weight,
    const mx::array& c_scales,
    const mx::array& c_biases,
    const mx::array& targets,
    int group_size,
    int bits,
    const mx::array& bias,
    mx::StreamOrDevice s = {});

mx::array cce_backward_raw_quantized(
    const mx::array& e,
    const mx::array& c_weight,
    const mx::array& c_scales,
    const mx::array& c_biases,
    const mx::array& lse,
    const mx::array& targets,
    const mx::array& d_nll,
    const mx::array& tile_max,
    int group_size,
    int bits,
    const mx::array& bias,
    mx::StreamOrDevice s = {});

class CutCrossEntropy : public mx::Primitive {
 public:
  explicit CutCrossEntropy(
      mx::Stream stream,
      bool has_bias,
      bool compute_all_grads)
      : mx::Primitive(stream),
        has_bias_(has_bias),
        compute_all_grads_(compute_all_grads) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  const char* name() const override { return "CutCrossEntropy"; }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto& o = static_cast<const CutCrossEntropy&>(other);
    return has_bias_ == o.has_bias_ &&
           compute_all_grads_ == o.compute_all_grads_;
  }

 private:
  bool has_bias_;
  bool compute_all_grads_;
};

class CutCrossEntropyBwd : public mx::Primitive {
 public:
  explicit CutCrossEntropyBwd(
      mx::Stream stream,
      bool has_bias,
      bool compute_all_grads)
      : mx::Primitive(stream),
        has_bias_(has_bias),
        compute_all_grads_(compute_all_grads) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "CutCrossEntropyBwd"; }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto& o = static_cast<const CutCrossEntropyBwd&>(other);
    return has_bias_ == o.has_bias_ &&
           compute_all_grads_ == o.compute_all_grads_;
  }

 private:
  bool has_bias_;
  bool compute_all_grads_;
};

class CutCrossEntropyQuantized : public mx::Primitive {
 public:
  explicit CutCrossEntropyQuantized(
      mx::Stream stream,
      bool has_bias,
      int group_size)
      : mx::Primitive(stream),
        has_bias_(has_bias),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "CutCrossEntropyQuantized"; }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto& o = static_cast<const CutCrossEntropyQuantized&>(other);
    return has_bias_ == o.has_bias_ && group_size_ == o.group_size_;
  }

 private:
  bool has_bias_;
  int group_size_;
};

class CutCrossEntropyQuantizedBwd : public mx::Primitive {
 public:
  explicit CutCrossEntropyQuantizedBwd(
      mx::Stream stream,
      bool has_bias,
      int group_size)
      : mx::Primitive(stream),
        has_bias_(has_bias),
        group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "CutCrossEntropyQuantizedBwd"; }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto& o = static_cast<const CutCrossEntropyQuantizedBwd&>(other);
    return has_bias_ == o.has_bias_ && group_size_ == o.group_size_;
  }

 private:
  bool has_bias_;
  int group_size_;
};

}  // namespace cce
