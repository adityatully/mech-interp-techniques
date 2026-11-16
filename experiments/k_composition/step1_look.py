"""Step 1, the slow version: look at ONE short example instead of averages.

8 ordinary words, then the same 8 words again. For every position we print:
  - where head 0.7 (previous-token head) looks most
  - where head 1.10 (induction head) looks most
  - what the model predicts next, and with what probability
and we save the two attention patterns as heatmaps.

Run on coe9 (interp env), from the repo root:
    HF_HUB_OFFLINE=1 python experiments/k_composition/step1_look.py
Then open results/k_composition/step1_look.png in Cursor.
"""

import matplotlib

matplotlib.use("Agg")  # no screen on coe9: draw straight to a file
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from model import TOKENIZER_NAME, AttnOnlyTransformer

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AttnOnlyTransformer.from_pretrained(device=device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

words = [" apple", " river", " seven", " doctor", " blue", " window", " music", " stone"]
ids = [tokenizer.encode(w)[0] for w in words]
assert all(len(tokenizer.encode(w)) == 1 for w in words), "each word must be one token"

tokens = torch.tensor([[tokenizer.bos_token_id] + ids + ids], device=device)  # [1, 17]
labels = [tokenizer.decode([t]) for t in tokens[0].tolist()]
labels[0] = "BOS"

logits, cache = model(tokens, return_cache=True)
probs = logits[0].float().softmax(-1)  # [17, d_vocab]
pat_prev = cache["pattern", 0][0, 7]  # [q_pos, k_pos] = [17, 17]  head 0.7
pat_ind = cache["pattern", 1][0, 10]  # [17, 17]                  head 1.10

print(f"{'pos':>3s} {'token':9s} | {'0.7 looks at':16s} | {'1.10 looks at':16s} | model predicts next")
for p in range(len(labels)):
    k_prev = pat_prev[p].argmax().item()
    k_ind = pat_ind[p].argmax().item()
    top = probs[p].argmax().item()
    print(f"{p:3d} {labels[p]:9s} | {k_prev:2d} {labels[k_prev]:8s} {pat_prev[p, k_prev]:.2f} "
          f"| {k_ind:2d} {labels[k_ind]:8s} {pat_ind[p, k_ind]:.2f} "
          f"| {tokenizer.decode([top])!r} ({probs[p, top]:.2f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, pat, title in [(axes[0], pat_prev, "head 0.7 (previous-token)"), (axes[1], pat_ind, "head 1.10 (induction)")]:
    ax.imshow(pat.cpu(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(17), labels, rotation=90)
    ax.set_yticks(range(17), labels)
    ax.set_xlabel("key: the token being looked AT")
    ax.set_ylabel("query: the token doing the looking")
    ax.set_title(title)
fig.tight_layout()
out = "results/k_composition/step1_look.png"
fig.savefig(out, dpi=110)
print(f"\nsaved {out}")
