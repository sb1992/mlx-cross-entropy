"""Benchmark: speed, memory, and accuracy across configurations."""

import time
import mlx.core as mx
import numpy as np
import gc


def _time_fn(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        result = fn()
        mx.eval(result) if not isinstance(result, tuple) else mx.eval(*result)

    times = []
    for _ in range(repeat):
        mx.synchronize()
        t0 = time.perf_counter()
        result = fn()
        mx.eval(result) if not isinstance(result, tuple) else mx.eval(*result)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def _measure_peak(fn):
    gc.collect()
    mx.eval(mx.zeros(1))
    mx.synchronize()
    mx.reset_peak_memory()
    result = fn()
    mx.eval(result) if not isinstance(result, tuple) else mx.eval(*result)
    mx.synchronize()
    return mx.get_peak_memory() / 1024**2


def run_benchmark():
    from mlx_cce import linear_cross_entropy

    configs = [
        (128,  768,  32_000),
        (256,  768,  32_000),
        (512,  768,  32_000),
        (1024, 768,  32_000),
        (1024, 2048, 128_000),
        (4096, 768,  32_000),
    ]

    print()
    print("=" * 90)
    print("  FORWARD PASS: reference vs fused (simdgroup MMA)")
    print("=" * 90)
    print(f"{'Config':>24} | {'Time (ms)':^22} | {'Peak Memory (MB)':^35}")
    print(f"{'B x D x V':>24} | {'ref':>8} {'fused':>8} {'ratio':>6} | {'ref':>8} {'fused':>8} {'saved':>9}")
    print("-" * 90)

    for B, D, V in configs:
        mx.random.seed(0)
        e = mx.random.normal((B, D))
        c = mx.random.normal((V, D))
        targets = mx.random.randint(0, V, (B,))
        mx.eval(e, c, targets)

        ref_ms = _time_fn(lambda: linear_cross_entropy(e, c, targets, impl="reference"))
        fused_ms = _time_fn(lambda: linear_cross_entropy(e, c, targets))

        ref_mem = _measure_peak(lambda: linear_cross_entropy(e, c, targets, impl="reference"))
        fused_mem = _measure_peak(lambda: linear_cross_entropy(e, c, targets))

        ratio = fused_ms / ref_ms
        saved = (1 - fused_mem / ref_mem) * 100

        label = f"{B}x{D}x{V}"
        print(
            f"{label:>24} | "
            f"{ref_ms:>8.2f} {fused_ms:>8.2f} {ratio:>5.2f}x | "
            f"{ref_mem:>8.1f} {fused_mem:>8.1f} {saved:>8.1f}%"
        )

    print("=" * 90)
    print()

    # Fwd+Bwd
    bwd_configs = [
        (256,  768,  32_000),
        (1024, 768,  32_000),
        (4096, 768,  32_000),
    ]

    print("=" * 90)
    print("  FWD+BWD (all grads): reference vs fused")
    print("=" * 90)
    print(f"{'Config':>22} | {'Time (ms)':^22} | {'Peak Memory (MB)':^35}")
    print(f"{'B x D x V':>22} | {'ref':>8} {'fused':>8} {'ratio':>6} | {'ref':>8} {'fused':>8} {'saved':>9}")
    print("-" * 90)

    for B, D, V in bwd_configs:
        mx.random.seed(0)
        e = mx.random.normal((B, D))
        c = mx.random.normal((V, D))
        targets = mx.random.randint(0, V, (B,))
        mx.eval(e, c, targets)

        def ref_grad():
            return mx.value_and_grad(
                lambda e_: linear_cross_entropy(e_, c, targets, impl="reference")
            )(e)

        def fused_grad():
            return mx.value_and_grad(
                lambda e_: linear_cross_entropy(e_, c, targets, compute_all_grads=True)
            )(e)

        ref_total = _time_fn(ref_grad)
        fused_total = _time_fn(fused_grad)

        ref_mem = _measure_peak(ref_grad)
        fused_mem = _measure_peak(fused_grad)

        ratio = fused_total / ref_total
        saved = (1 - fused_mem / ref_mem) * 100

        label = f"{B}x{D}x{V}"
        print(
            f"{label:>22} | "
            f"{ref_total:>8.2f} {fused_total:>8.2f} {ratio:>5.2f}x | "
            f"{ref_mem:>8.1f} {fused_mem:>8.1f} {saved:>8.1f}%"
        )

    print("=" * 90)
    print()

    # Accuracy
    print("=" * 70)
    print("  ACCURACY: Max Absolute Error vs Reference (fp32)")
    print("=" * 70)
    print(f"{'Config':>22} | {'Loss':>10} {'dE':>10} {'dC':>10} {'dBias':>10}")
    print("-" * 70)

    accuracy_configs = [
        (64,   128,  1024),
        (256,  768,  32000),
        (1024, 2048, 128_000),
    ]

    for B, D, V in accuracy_configs:
        mx.random.seed(0)
        e = mx.random.normal((B, D))
        c = mx.random.normal((V, D))
        targets = mx.random.randint(0, V, (B,))
        bias = mx.random.normal((V,))
        mx.eval(e, c, targets, bias)

        ref_loss = linear_cross_entropy(e, c, targets, bias=bias, impl="reference")
        fused_loss = linear_cross_entropy(e, c, targets, bias=bias)
        mx.eval(ref_loss, fused_loss)
        loss_err = abs(fused_loss.item() - ref_loss.item())

        ref_dE = mx.value_and_grad(
            lambda e_: linear_cross_entropy(e_, c, targets, bias=bias, impl="reference")
        )(e)[1]
        fused_dE = mx.value_and_grad(
            lambda e_: linear_cross_entropy(e_, c, targets, bias=bias)
        )(e)[1]
        mx.eval(ref_dE, fused_dE)
        dE_err = np.max(np.abs(np.array(ref_dE) - np.array(fused_dE)))

        ref_dC = mx.value_and_grad(
            lambda c_: linear_cross_entropy(e, c_, targets, bias=bias, impl="reference")
        )(c)[1]
        fused_dC = mx.value_and_grad(
            lambda c_: linear_cross_entropy(
                e, c_, targets, bias=bias, compute_all_grads=True
            )
        )(c)[1]
        mx.eval(ref_dC, fused_dC)
        dC_err = np.max(np.abs(np.array(ref_dC) - np.array(fused_dC)))

        ref_dB = mx.grad(
            lambda b: linear_cross_entropy(e, c, targets, bias=b, impl="reference")
        )(bias)
        fused_dB = mx.grad(
            lambda b: linear_cross_entropy(
                e, c, targets, bias=b, compute_all_grads=True
            )
        )(bias)
        mx.eval(ref_dB, fused_dB)
        dB_err = np.max(np.abs(np.array(ref_dB) - np.array(fused_dB)))

        label = f"{B}x{D}x{V}"
        print(
            f"{label:>22} | "
            f"{loss_err:>10.2e} {dE_err:>10.2e} {dC_err:>10.2e} {dB_err:>10.2e}"
        )

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
