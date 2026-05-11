"""Memory comparison: reference vs chunked vs metal at increasing vocab sizes."""

import mlx.core as mx

from mlx_cce import linear_cross_entropy


def measure_peak(fn):
    mx.eval(mx.zeros(1))
    mx.synchronize()
    mx.reset_peak_memory()
    result = fn()
    mx.eval(result) if not isinstance(result, tuple) else mx.eval(*result)
    mx.synchronize()
    return mx.get_peak_memory()


def run():
    configs = [
        (1024, 768,  32_000),
        (1024, 2048, 128_000),
        (4096, 768,  32_000),
        (4096, 2048, 128_000),
    ]

    print(f"{'Config':>22} | {'Reference':>12} {'Chunked':>12} {'Metal':>12} | {'Chunked Save':>12} {'Metal Save':>12}")
    print("-" * 95)

    for B, D, V in configs:
        mx.random.seed(0)
        e = mx.random.normal((B, D))
        c = mx.random.normal((V, D))
        targets = mx.random.randint(0, V, (B,))
        mx.eval(e, c, targets)

        ref = measure_peak(lambda: linear_cross_entropy(e, c, targets, impl="reference")) / 1024**2
        chk = measure_peak(lambda: linear_cross_entropy(e, c, targets, impl="chunked")) / 1024**2
        met = measure_peak(lambda: linear_cross_entropy(e, c, targets, impl="metal")) / 1024**2

        chk_save = (1 - chk / ref) * 100
        met_save = (1 - met / ref) * 100

        label = f"{B}x{D}x{V}"
        print(
            f"{label:>22} | "
            f"{ref:>10.1f}MB {chk:>10.1f}MB {met:>10.1f}MB | "
            f"{chk_save:>10.1f}%  {met_save:>10.1f}%"
        )


if __name__ == "__main__":
    run()
