"""Quick LoRA training test: standard CE vs CCE on Llama 3.2 1B."""

import sys
sys.path.insert(0, "/Users/shraey/.superset/worktrees/mlx-cce/claude-test")

import time
import mlx
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_cce import linear_cross_entropy

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def dequantize_if_needed(layer):
    if hasattr(layer, "scales"):
        return mx.dequantize(
            layer.weight, layer.scales, layer.biases,
            layer.group_size, layer.bits)
    return layer.weight


def pad_to(x, n, axis=0):
    pad_n = (n - x.shape[axis] % n) % n
    if pad_n == 0:
        return x, 0
    shape = list(x.shape)
    shape[axis] = pad_n
    return mx.concatenate([x, mx.zeros(shape, dtype=x.dtype)], axis=axis), pad_n


def standard_loss(model, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets).mean()


def cce_loss(model, inputs, targets):
    hidden = model.model(inputs)
    w = dequantize_if_needed(model.model.embed_tokens)

    B, S, D = hidden.shape
    h = hidden.reshape(-1, D).astype(mx.float32)
    t = targets.reshape(-1).astype(mx.uint32)

    # Pad vocab to %32
    V = w.shape[0]
    v_pad = (32 - V % 32) % 32
    if v_pad > 0:
        w = mx.concatenate([w, mx.zeros((v_pad, D), dtype=w.dtype)])

    # Pad batch to %32
    h, n_pad = pad_to(h, 32)
    t, _ = pad_to(t, 32)

    loss = linear_cross_entropy(
        h, w.astype(mx.float32), t,
        reduction="none",
        ignore_index=w.shape[0] + 1)

    # Remove padding, mean over real tokens
    if n_pad > 0:
        loss = loss[:B * S]
    return loss.mean()


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_ID)

    # Apply LoRA to last 4 layers
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.freeze()
    linear_to_lora_layers(model, 4, {"rank": 8, "scale": 20.0, "dropout": 0.0})

    trainable = sum(p.size for _, p in model.trainable_parameters().items()
                    if hasattr(p, 'size')) if hasattr(model, 'trainable_parameters') else "?"

    # Count trainable params
    n_train = 0
    for k, v in model.parameters().items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, mx.array) and "lora" in str(k) + str(kk):
                    n_train += vv.size
    print(f"Model loaded, LoRA applied")

    # Create synthetic batch
    V = 128256
    seq_len = 64
    batch = mx.random.randint(0, high=V, shape=(2, seq_len + 1)).astype(mx.int32)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Test that both losses compute and are close
    loss_ce = standard_loss(model, inputs, targets)
    loss_cce = cce_loss(model, inputs, targets)
    mx.eval(loss_ce, loss_cce)
    print(f"\nForward comparison:")
    print(f"  Standard CE: {loss_ce.item():.6f}")
    print(f"  CCE:         {loss_cce.item():.6f}")
    print(f"  Rel diff:    {abs(loss_ce.item() - loss_cce.item()) / abs(loss_ce.item()):.6f}")

    # Test gradient computation
    print(f"\nGradient test (5 steps each):")

    optimizer_ce = mlx.optimizers.Adam(learning_rate=1e-4)
    optimizer_cce = mlx.optimizers.Adam(learning_rate=1e-4)

    # Train with standard CE
    loss_grad_ce = nn.value_and_grad(model, lambda m, x, t: standard_loss(m, x, t))
    ce_losses = []
    t0 = time.perf_counter()
    for step in range(5):
        loss, grads = loss_grad_ce(model, inputs, targets)
        optimizer_ce.update(model, grads)
        mx.eval(model.parameters(), optimizer_ce.state, loss)
        ce_losses.append(loss.item())
    ce_time = time.perf_counter() - t0

    print(f"  CE losses:  {['%.4f' % l for l in ce_losses]}")
    print(f"  CE time:    {ce_time:.2f}s ({ce_time/5*1000:.0f} ms/step)")

    # Reload model for fair CCE comparison
    model2, _ = load(MODEL_ID)
    model2.freeze()
    linear_to_lora_layers(model2, 4, {"rank": 8, "scale": 20.0, "dropout": 0.0})
    optimizer_cce = mlx.optimizers.Adam(learning_rate=1e-4)

    loss_grad_cce = nn.value_and_grad(model2, lambda m, x, t: cce_loss(m, x, t))
    cce_losses = []
    t0 = time.perf_counter()
    for step in range(5):
        loss, grads = loss_grad_cce(model2, inputs, targets)
        optimizer_cce.update(model2, grads)
        mx.eval(model2.parameters(), optimizer_cce.state, loss)
        cce_losses.append(loss.item())
    cce_time = time.perf_counter() - t0

    print(f"  CCE losses: {['%.4f' % l for l in cce_losses]}")
    print(f"  CCE time:   {cce_time:.2f}s ({cce_time/5*1000:.0f} ms/step)")

    # Compare loss curves
    print(f"\n  Loss curves match: ", end="")
    max_diff = max(abs(a - b) for a, b in zip(ce_losses, cce_losses))
    print(f"{'YES' if max_diff < 0.1 else 'NO'} (max diff: {max_diff:.4f})")
    print(f"  Speed ratio: {ce_time/cce_time:.2f}x (CCE vs CE)")


if __name__ == "__main__":
    main()
