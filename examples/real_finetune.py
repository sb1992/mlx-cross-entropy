"""Real LoRA fine-tuning: CE vs CCE on actual text data."""

import sys
sys.path.insert(0, "/Users/shraey/.superset/worktrees/mlx-cce/claude-test")

import time
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load, generate
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_cce import linear_cross_entropy

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

TRAIN_DATA = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Python is a programming language created by Guido van Rossum in 1991.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "The Great Wall of China is over 13,000 miles long and was built over many centuries.",
    "Albert Einstein published his theory of general relativity in 1915.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Shakespeare wrote approximately 37 plays during his lifetime.",
    "The human body contains approximately 206 bones in the adult skeleton.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    "Jupiter is the largest planet in our solar system with a mass of 1.9 times ten to the 27 kilograms.",
    "The French Revolution began in 1789 and fundamentally changed European politics.",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic level.",
    "The Amazon rainforest produces about 20 percent of the world's oxygen supply.",
]

GEN_PROMPTS = [
    "The capital of France",
    "Python is a programming",
    "The speed of light",
    "Machine learning is",
]


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


def tokenize_batch(tokenizer, texts, max_len=64):
    encoded = [tokenizer.encode(t) for t in texts]
    ml = min(max(len(e) for e in encoded), max_len)
    pad_id = tokenizer.eos_token_id or 0
    padded = []
    for e in encoded:
        if len(e) < ml:
            padded.append(e + [pad_id] * (ml - len(e)))
        else:
            padded.append(e[:ml])
    return mx.array(padded).astype(mx.int32)


def train(model, tokenizer, init_state, loss_fn, name, steps=50, lr=1e-4):
    model.load_weights(init_state, strict=False)
    mx.eval(model.parameters())
    opt = mlx.optimizers.Adam(learning_rate=lr)
    grad_fn = nn.value_and_grad(model, loss_fn)

    batch_size = 4
    losses = []
    t0 = time.perf_counter()

    for step in range(steps):
        start = (step * batch_size) % len(TRAIN_DATA)
        texts = [TRAIN_DATA[(start + i) % len(TRAIN_DATA)] for i in range(batch_size)]
        tokens = tokenize_batch(tokenizer, texts)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        loss, grads = grad_fn(model, inputs, targets)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        losses.append(loss.item())

        if step % 10 == 0 or step == steps - 1:
            print(f"  [{name}] step {step:3d}: loss={loss.item():.4f}")

    elapsed = time.perf_counter() - t0
    print(f"  [{name}] {steps} steps in {elapsed:.1f}s ({elapsed/steps*1000:.0f} ms/step)")
    return losses


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_ID)
    model.freeze()
    linear_to_lora_layers(model, 4, {"rank": 8, "scale": 20.0, "dropout": 0.0})
    mx.eval(model.parameters())
    init_state = list(mlx.utils.tree_flatten(model.trainable_parameters()))

    n_train = sum(v.size for _, v in init_state)
    print(f"Trainable params: {n_train:,}")

    # Pre-training generation
    print("\n--- Pre-training generation ---")
    model.load_weights(init_state, strict=False)
    for p in GEN_PROMPTS[:2]:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        print(f"  '{p}' -> {out[:80]}")

    # Train with CE
    print("\n--- CE Training (50 steps) ---")
    ce_losses = train(model, tokenizer, init_state, ce_loss, "CE", steps=80)

    # Save CE-trained state and generate
    ce_state = list(mlx.utils.tree_flatten(model.trainable_parameters()))
    print("\n--- CE generation (after training) ---")
    for p in GEN_PROMPTS:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        print(f"  '{p}' -> {out[:100]}")

    # Train with CCE
    print("\n--- CCE Training (50 steps) ---")
    cce_losses = train(model, tokenizer, init_state, cce_loss, "CCE", steps=80)

    cce_state = list(mlx.utils.tree_flatten(model.trainable_parameters()))
    print("\n--- CCE generation (after training) ---")
    for p in GEN_PROMPTS:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        print(f"  '{p}' -> {out[:100]}")

    # Compare loss curves
    print("\n--- Loss curve comparison ---")
    print(f"{'Step':>4s}  {'CE':>8s}  {'CCE':>8s}  {'Diff':>8s}")
    for i in range(0, len(ce_losses), 5):
        d = abs(ce_losses[i] - cce_losses[i])
        print(f"{i:4d}  {ce_losses[i]:8.4f}  {cce_losses[i]:8.4f}  {d:8.4f}")

    max_diff = max(abs(a-b) for a,b in zip(ce_losses, cce_losses))
    avg_diff = sum(abs(a-b) for a,b in zip(ce_losses, cce_losses)) / len(ce_losses)
    print(f"\nMax loss diff: {max_diff:.4f}")
    print(f"Avg loss diff: {avg_diff:.4f}")

    # Compare final LoRA weights
    print("\n--- LoRA weight comparison (after training) ---")
    total_cos = 0
    count = 0
    for (name, wce), (_, wcce) in zip(ce_state, cce_state):
        wce_f = wce.astype(mx.float32)
        wcce_f = wcce.astype(mx.float32)
        nce = mx.sqrt(mx.sum(wce_f**2)).item()
        ncce = mx.sqrt(mx.sum(wcce_f**2)).item()
        if nce > 1e-10 and ncce > 1e-10:
            cos = mx.sum(wce_f * wcce_f).item() / (nce * ncce)
            total_cos += cos
            count += 1
            if "layers.15" in name and "lora_b" in name:
                print(f"  {name}: cos={cos:.4f} norm_ratio={ncce/nce:.4f}")
    if count > 0:
        print(f"  Avg cosine (non-zero params): {total_cos/count:.4f}")


if __name__ == "__main__":
    main()
