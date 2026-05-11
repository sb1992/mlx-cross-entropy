"""Training validation: compare reference CE vs CCE on a tiny transformer.

Trains two identical TinyLM models (V=1024, D=256, H=4, L=2, ~2.6M params)
on random next-token-prediction data. Both should produce matching loss curves,
proving CCE gradients are correct for real training.
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_cce import linear_cross_entropy


class MLP(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.attn = nn.MultiHeadAttention(D, H)
        self.n2 = nn.LayerNorm(D)
        self.mlp = MLP(D)

    def __call__(self, x, mask):
        q = self.n1(x)
        x = x + self.attn(q, q, q, mask=mask)
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, V, D, H, L):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = [Block(D, H) for _ in range(L)]
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def __call__(self, tokens):
        B, S = tokens.shape
        x = self.embed(tokens)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(S).astype(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln(x)

    def loss_reference(self, tokens):
        x = self(tokens[:, :-1])
        logits = self.head(x)
        targets = tokens[:, 1:]
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        return mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))

    def loss_cce(self, tokens):
        x = self(tokens[:, :-1])
        targets = tokens[:, 1:]
        return linear_cross_entropy(
            x, self.head.weight, targets,
            compute_all_grads=True,
            reduction="mean",
        )


def train(model, loss_fn, data, optimizer, steps):
    losses = []

    def step(model, batch):
        loss = loss_fn(model, batch)
        return loss

    loss_and_grad = nn.value_and_grad(model, step)

    for i in range(steps):
        batch = data[i % len(data)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"  step {i+1:3d}: loss = {losses[-1]:.4f}")

    return losses


def main():
    V, D, H, L = 1024, 256, 4, 2
    batch_size, seq_len = 4, 64
    num_steps = 100
    lr = 1e-3
    num_batches = 20

    print(f"TinyLM: V={V}, D={D}, H={H}, L={L}")
    print(f"Training: batch={batch_size}, seq={seq_len}, steps={num_steps}, lr={lr}")
    print()

    mx.random.seed(42)
    data = [mx.random.randint(0, V, (batch_size, seq_len + 1)) for _ in range(num_batches)]
    mx.eval(*data)

    mx.random.seed(0)
    model_ref = TinyLM(V, D, H, L)
    mx.eval(model_ref.parameters())
    ref_params = [p.copy() for p in model_ref.parameters().values()] if hasattr(model_ref.parameters(), 'values') else None

    mx.random.seed(0)
    model_cce = TinyLM(V, D, H, L)
    mx.eval(model_cce.parameters())

    from mlx.utils import tree_flatten
    ref_flat = [x.sum().item() for _, x in tree_flatten(model_ref.parameters())]
    cce_flat = [x.sum().item() for _, x in tree_flatten(model_cce.parameters())]
    param_match = all(abs(a - b) < 1e-10 for a, b in zip(ref_flat, cce_flat))
    print(f"Initial params match: {param_match}")
    print()

    opt_ref = optim.Adam(learning_rate=lr)
    opt_cce = optim.Adam(learning_rate=lr)

    print("=== Reference CE ===")
    t0 = time.perf_counter()
    losses_ref = train(model_ref, TinyLM.loss_reference, data, opt_ref, num_steps)
    t_ref = time.perf_counter() - t0

    print()
    print("=== CCE (fused v3) ===")
    t0 = time.perf_counter()
    losses_cce = train(model_cce, TinyLM.loss_cce, data, opt_cce, num_steps)
    t_cce = time.perf_counter() - t0

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    diffs = [abs(a - b) for a, b in zip(losses_ref, losses_cce)]
    max_diff = max(diffs)
    avg_diff = sum(diffs) / len(diffs)
    final_ref = losses_ref[-1]
    final_cce = losses_cce[-1]

    print(f"Reference: final loss = {final_ref:.4f}  ({t_ref:.1f}s)")
    print(f"CCE:       final loss = {final_cce:.4f}  ({t_cce:.1f}s)")
    print(f"Max loss diff:  {max_diff:.2e}")
    print(f"Avg loss diff:  {avg_diff:.2e}")
    print()

    converged = final_ref < losses_ref[0] and final_cce < losses_cce[0]
    matched = max_diff < 0.1
    print(f"Both converged:   {'YES' if converged else 'NO'}")
    print(f"Losses matched:   {'YES' if matched else 'NO'} (max diff < 0.1)")

    if converged and matched:
        print("\nVALIDATION PASSED: CCE produces correct gradients for training.")
    else:
        print("\nVALIDATION FAILED")
        if not converged:
            print("  - One or both models did not converge")
        if not matched:
            print(f"  - Loss curves diverged (max diff = {max_diff:.2e})")

    print()
    print("Loss curve (every 10 steps):")
    print(f"{'Step':>6}  {'Reference':>10}  {'CCE':>10}  {'Diff':>10}")
    for i in range(0, num_steps, 10):
        print(f"{i+1:>6}  {losses_ref[i]:>10.4f}  {losses_cce[i]:>10.4f}  {diffs[i]:>10.2e}")


if __name__ == "__main__":
    main()
