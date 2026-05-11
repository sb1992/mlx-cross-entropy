"""LoRA training comparison with real text (peaked softmax) vs random."""

import sys
sys.path.insert(0, "/Users/shraey/.superset/worktrees/mlx-cce/claude-test")

import time
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
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


def ce_loss(model, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets).mean()


def cce_loss(model, inputs, targets):
    hidden = model.model(inputs)
    w = dequantize_if_needed(model.model.embed_tokens)
    B, S, D = hidden.shape
    h = hidden.reshape(-1, D).astype(mx.float32)
    t = targets.reshape(-1).astype(mx.uint32)
    V = w.shape[0]
    v_pad = (32 - V % 32) % 32
    if v_pad > 0:
        w = mx.concatenate([w, mx.zeros((v_pad, D), dtype=w.dtype)])
    h, n_pad = pad_to(h, 32)
    t, _ = pad_to(t, 32)
    loss = linear_cross_entropy(
        h, w.astype(mx.float32), t,
        reduction="none", ignore_index=w.shape[0] + 1)
    if n_pad > 0:
        loss = loss[:B * S]
    return loss.mean()


def run_comparison(model, init_state, inputs, targets, label, steps=10):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    # Forward
    model.load_weights(init_state, strict=False)
    mx.eval(model.parameters())
    l_ce = ce_loss(model, inputs, targets)
    l_cce = cce_loss(model, inputs, targets)
    mx.eval(l_ce, l_cce)
    print(f"Forward: CE={l_ce.item():.6f}  CCE={l_cce.item():.6f}  diff={abs(l_ce.item()-l_cce.item()):.6f}")

    # Gradient comparison
    grad_ce_fn = nn.value_and_grad(model, ce_loss)
    grad_cce_fn = nn.value_and_grad(model, cce_loss)
    _, grads_ce = grad_ce_fn(model, inputs, targets)
    _, grads_cce = grad_cce_fn(model, inputs, targets)
    ce_flat = mlx.utils.tree_flatten(grads_ce)
    cce_flat = mlx.utils.tree_flatten(grads_cce)
    mx.eval([v for _, v in ce_flat], [v for _, v in cce_flat])

    close = 0
    total = 0
    for (name, ga), (_, gb) in zip(ce_flat, cce_flat):
        na = mx.sqrt(mx.sum(ga.astype(mx.float32)**2)).item()
        nb = mx.sqrt(mx.sum(gb.astype(mx.float32)**2)).item()
        if na < 1e-10 and nb < 1e-10:
            close += 1
            total += 1
            continue
        total += 1
        if na > 1e-10 and nb > 1e-10:
            cos = mx.sum(ga.astype(mx.float32)*gb.astype(mx.float32)).item()/(na*nb)
            rel = abs(na - nb) / na
            if rel < 0.1 and cos > 0.9:
                close += 1
    print(f"Gradients: {close}/{total} match (<10% norm + >0.9 cos)")

    # Training
    model.load_weights(init_state, strict=False)
    mx.eval(model.parameters())
    opt_ce = mlx.optimizers.Adam(learning_rate=1e-4)
    gf_ce = nn.value_and_grad(model, ce_loss)
    ce_losses = []
    for _ in range(steps):
        loss, grads = gf_ce(model, inputs, targets)
        opt_ce.update(model, grads)
        mx.eval(model.parameters(), opt_ce.state, loss)
        ce_losses.append(loss.item())

    model.load_weights(init_state, strict=False)
    mx.eval(model.parameters())
    opt_cce = mlx.optimizers.Adam(learning_rate=1e-4)
    gf_cce = nn.value_and_grad(model, cce_loss)
    cce_losses = []
    for _ in range(steps):
        loss, grads = gf_cce(model, inputs, targets)
        opt_cce.update(model, grads)
        mx.eval(model.parameters(), opt_cce.state, loss)
        cce_losses.append(loss.item())

    print(f"CE  losses: {['%.4f' % l for l in ce_losses]}")
    print(f"CCE losses: {['%.4f' % l for l in cce_losses]}")
    max_diff = max(abs(a-b) for a,b in zip(ce_losses, cce_losses))
    print(f"Max loss diff: {max_diff:.4f}")


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_ID)
    model.freeze()
    linear_to_lora_layers(model, 4, {"rank": 8, "scale": 20.0, "dropout": 0.0})
    mx.eval(model.parameters())
    init_state = list(mlx.utils.tree_flatten(model.trainable_parameters()))

    V = 128256

    # Random inputs
    batch_rand = mx.random.randint(0, high=V, shape=(2, 65)).astype(mx.int32)
    run_comparison(model, init_state, batch_rand[:, :-1], batch_rand[:, 1:],
                   "Random inputs (flat softmax)")

    # Real text
    texts = [
        "The quick brown fox jumps over the lazy dog and runs across the field into the sunset.",
        "Machine learning models can be fine-tuned using parameter-efficient methods like LoRA."
    ]
    encoded = [tokenizer.encode(t) for t in texts]
    max_len = min(max(len(e) for e in encoded), 65)
    for i in range(len(encoded)):
        if len(encoded[i]) < max_len:
            encoded[i] = encoded[i] + [tokenizer.eos_token_id] * (max_len - len(encoded[i]))
        else:
            encoded[i] = encoded[i][:max_len]
    tok = mx.array(encoded).astype(mx.int32)
    run_comparison(model, init_state, tok[:, :-1], tok[:, 1:],
                   "Real text (peaked softmax)")


if __name__ == "__main__":
    main()
