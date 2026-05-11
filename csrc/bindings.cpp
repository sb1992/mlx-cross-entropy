#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "cce/cce.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Cut Cross-Entropy: memory-efficient cross-entropy with native Metal kernels";

  m.def(
      "cut_cross_entropy",
      [](const mx::array& e,
         const mx::array& c,
         const mx::array& targets,
         const std::optional<mx::array>& bias,
         const std::string& reduction,
         int ignore_index,
         bool compute_all_grads,
         mx::StreamOrDevice s) {
        return cce::cut_cross_entropy(
            e, c, targets,
            bias.value_or(mx::array({})),
            reduction, ignore_index, compute_all_grads, s);
      },
      "e"_a,
      "c"_a,
      "targets"_a,
      "bias"_a = nb::none(),
      "reduction"_a = "mean",
      "ignore_index"_a = -100,
      "compute_all_grads"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
Memory-efficient cross-entropy that never materializes [B, V] logits.

Computes loss = logsumexp(e @ c.T) - e[targets] . c[targets]
using tiled MMA + gradient filtering on Apple Silicon.

Args:
    e: Hidden states [B, D]
    c: Classifier weights [V, D]
    targets: Target indices [B]
    bias: Optional classifier bias [V]
    reduction: "mean", "sum", or "none"
    ignore_index: Target value to mask (-100 default)
    compute_all_grads: If True, compute dC and dBias too

Returns:
    Loss scalar (or [B] if reduction="none")
)");

  m.def(
      "cce_forward_raw",
      [](const mx::array& e,
         const mx::array& c,
         const mx::array& targets,
         const std::optional<mx::array>& bias,
         mx::StreamOrDevice s) {
        return cce::cce_forward_raw(
            e, c, targets, bias.value_or(mx::array({})), s);
      },
      "e"_a, "c"_a, "targets"_a, "bias"_a = nb::none(),
      nb::kw_only(), "stream"_a = nb::none());

  m.def(
      "cce_backward_raw",
      [](const mx::array& e,
         const mx::array& c,
         const mx::array& lse,
         const mx::array& targets,
         const mx::array& d_nll,
         const mx::array& tile_max,
         const std::optional<mx::array>& bias,
         mx::StreamOrDevice s) {
        return cce::cce_backward_raw(
            e, c, lse, targets, d_nll, tile_max,
            bias.value_or(mx::array({})), s);
      },
      "e"_a, "c"_a, "lse"_a, "targets"_a, "d_nll"_a, "tile_max"_a,
      "bias"_a = nb::none(), nb::kw_only(), "stream"_a = nb::none());

  m.def(
      "cce_forward_raw_quantized",
      [](const mx::array& e,
         const mx::array& c_weight,
         const mx::array& c_scales,
         const mx::array& c_biases,
         const mx::array& targets,
         int group_size,
         int bits,
         const std::optional<mx::array>& bias,
         mx::StreamOrDevice s) {
        return cce::cce_forward_raw_quantized(
            e, c_weight, c_scales, c_biases, targets,
            group_size, bits, bias.value_or(mx::array({})), s);
      },
      "e"_a, "c_weight"_a, "c_scales"_a, "c_biases"_a, "targets"_a,
      "group_size"_a, "bits"_a,
      "bias"_a = nb::none(), nb::kw_only(), "stream"_a = nb::none());

  m.def(
      "cce_backward_raw_quantized",
      [](const mx::array& e,
         const mx::array& c_weight,
         const mx::array& c_scales,
         const mx::array& c_biases,
         const mx::array& lse,
         const mx::array& targets,
         const mx::array& d_nll,
         const mx::array& tile_max,
         int group_size,
         int bits,
         const std::optional<mx::array>& bias,
         mx::StreamOrDevice s) {
        return cce::cce_backward_raw_quantized(
            e, c_weight, c_scales, c_biases, lse, targets, d_nll, tile_max,
            group_size, bits, bias.value_or(mx::array({})), s);
      },
      "e"_a, "c_weight"_a, "c_scales"_a, "c_biases"_a,
      "lse"_a, "targets"_a, "d_nll"_a, "tile_max"_a,
      "group_size"_a, "bits"_a,
      "bias"_a = nb::none(), nb::kw_only(), "stream"_a = nb::none());
}
