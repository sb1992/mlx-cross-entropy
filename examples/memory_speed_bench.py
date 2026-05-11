"""Memory & speed benchmark: standard CE vs CCE (C++ extension).

Measures peak memory and wall time for forward+backward at increasing
sequence lengths, showing where CCE's O(B*D) vs CE's O(B*V) advantage kicks in.
"""

import sys
sys.path.insert(0, "/Users/shraey/.superset/worktrees/mlx-cce/claude-test")

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_cce_native import cut_cross_entropy


def pad_to_32(x, axis=0):
    pad_n = (32 - x.shape[axis] % 32) % 32
    if pad_n == 0:
        return x, 0
    shape = list(x.shape)
    shape[axis] = pad_n
    return mx.concatenate([x, mx.zeros(shape, dtype=x.dtype)], axis=axis), pad_n


def measure(fn, warmup=2, repeats=5):
    """Returns (peak_memory_MB, avg_time_ms) or None if OOM."""
    try:
        for _ in range(warmup):
            r = fn()
            if isinstance(r, (list, tuple)):
                mx.eval(*r)
            else:
                mx.eval(r)

        times = []
        peak = 0
        for _ in range(repeats):
            mx.synchronize()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            r = fn()
            if isinstance(r, (list, tuple)):
                mx.eval(*r)
            else:
                mx.eval(r)
            mx.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            peak = max(peak, mx.get_peak_memory())

        return peak / 1024**2, sum(times) / len(times)
    except Exception:
        return None, None


def ce_forward_backward(h, w, targets):
    def loss_fn(h_in):
        logits = h_in @ w.T
        return nn.losses.cross_entropy(logits, targets).mean()
    return mx.value_and_grad(loss_fn)(h)


def cce_forward_backward(h, w_padded, targets, V_pad, orig_n):
    def loss_fn(h_in):
        h_p, n_pad = pad_to_32(h_in)
        t_p, _ = pad_to_32(targets)
        loss = cut_cross_entropy(
            h_p, w_padded, t_p,
            reduction="none", ignore_index=V_pad + 1)
        if n_pad > 0:
            loss = loss[:orig_n]
        return loss.mean()
    return mx.value_and_grad(loss_fn)(h)


def main():
    D = 2048
    V = 128256  # Llama 3 vocab

    configs = [
        (4, 128),     # N=512
        (4, 512),     # N=2048
        (4, 1024),    # N=4096
        (8, 1024),    # N=8192
        (8, 2048),    # N=16384
        (16, 2048),   # N=32768
    ]

    mx.random.seed(42)
    w = mx.random.normal((V, D)).astype(mx.float32) * 0.02
    V_pad = ((V + 31) // 32) * 32
    w_padded = mx.concatenate([w, mx.zeros((V_pad - V, D), dtype=mx.float32)])
    mx.eval(w, w_padded)

    logit_per_row_mb = V * 4 / 1024**2

    print(f"D={D}, V={V:,}")
    print(f"Per-token logit row: {logit_per_row_mb:.1f} MB  (this is what CCE avoids materializing)")
    print()
    print(f"{'B×S':>8s} {'N':>6s} {'logits':>8s} │ {'CE mem':>9s} {'CCE mem':>9s} {'Save%':>6s} │ {'CE time':>9s} {'CCE time':>9s} {'Ratio':>6s}")
    print("─" * 90)

    for B, S in configs:
        N = B * S
        logit_mb = N * logit_per_row_mb

        h = mx.random.normal((N, D)).astype(mx.float32) * 0.1
        t = mx.random.randint(0, high=V, shape=(N,)).astype(mx.uint32)
        mx.eval(h, t)

        ce_mem, ce_time = measure(lambda: ce_forward_backward(h, w, t))
        cce_mem, cce_time = measure(lambda: cce_forward_backward(h, w_padded, t, V_pad, N))

        label = f"{B}×{S}"

        if ce_mem is not None and cce_mem is not None:
            savings = (1 - cce_mem / ce_mem) * 100
            ratio = f"{ce_time / cce_time:.2f}x"
            print(
                f"{label:>8s} {N:>6d} {logit_mb:>6.0f}MB │ "
                f"{ce_mem:>7.0f}MB {cce_mem:>7.0f}MB {savings:>5.1f}% │ "
                f"{ce_time:>7.1f}ms {cce_time:>7.1f}ms {ratio:>6s}"
            )
        elif ce_mem is None and cce_mem is not None:
            print(
                f"{label:>8s} {N:>6d} {logit_mb:>6.0f}MB │ "
                f"    OOM   {cce_mem:>7.0f}MB   ----  │ "
                f"    OOM   {cce_time:>7.1f}ms   ----"
            )
        elif ce_mem is not None:
            print(
                f"{label:>8s} {N:>6d} {logit_mb:>6.0f}MB │ "
                f"{ce_mem:>7.0f}MB     OOM     ----  │ "
                f"{ce_time:>7.1f}ms     OOM     ----"
            )

    print()
    print("Key insight: CCE never allocates the [N, V] logit tensor.")
    print(f"  CE memory scales as O(N × V) = O(N × {V:,})")
    print(f"  CCE memory scales as O(N × D) = O(N × {D:,})  ({V/D:.0f}× smaller per token)")


if __name__ == "__main__":
    main()
