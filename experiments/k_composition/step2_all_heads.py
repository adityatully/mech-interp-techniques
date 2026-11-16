"""Step 2a: is head 0.7's pipe curve special? Compare every layer-0 head against 1.10's key.

For each layer-0 head h (0..11):
    M_h = W_O[0,h] @ W_K[1,10]        [64, 768] @ [768, 64] -> [64, 64]
    s   = singular values of M_h      [64]

Two different things can make a head "special", so we print both:
    size  : ||M_h||_F = sqrt(sum of s_i^2)   how much in total 1.10's key hears from head h
    shape : pipes to 90%                      how many pipes hold 90% of that (concentrated or spread out)
A head with a bigger W_O gets bigger s_i everywhere, so size alone can mislead. Look at both.

PREDICT before running:
  1. Will 0.7 have the largest size? By a lot (2x? 5x?) or a little?
  2. Will 0.7 need fewer, the same, or more pipes to reach 90% than the other heads?
  3. Where will 0.9 and 0.4 (the weak prev-token heads) land?

Run on coe9 (interp env), from the repo root:
    HF_HUB_OFFLINE=1 python experiments/k_composition/step2_all_heads.py | tee results/k_composition/step2_all_heads.txt
Then open results/k_composition/step2_all_heads.png
"""

import matplotlib

matplotlib.use("Agg")  # no screen on coe9: draw straight to a file
import matplotlib.pyplot as plt
import torch

from model import load_state_dict

sd = load_state_dict()
W_O = sd["blocks.0.attn.W_O"]  # [head, d_head, d_model] = [12, 64, 768]   layer 0 writes
W_K = sd["blocks.1.attn.W_K"]  # [head, d_model, d_head] = [12, 768, 64]   layer 1 key reads

READER = 10  # head 1.10

print(f"{'head':>5s} {'s_1':>7s} {'s_10':>7s} {'s_20':>7s} {'size':>7s} {'pipes to 90%':>13s}")
all_s = []
for h in range(12):
    M = W_O[h] @ W_K[READER]  # [64, 64]
    s = torch.linalg.svdvals(M)  # [64], largest first
    all_s.append(s)

    size = (s**2).sum().sqrt()  # = ||M||_F
    cumul = (s**2).cumsum(dim=0) / (s**2).sum()  # [64], same "cumul" column as step2_svd.py
    pipes_90 = (cumul < 0.9).sum().item() + 1  # first i where cumul reaches 0.9

    print(f"  0.{h:<2d} {s[0]:7.2f} {s[9]:7.2f} {s[19]:7.2f} {size:7.2f} {pipes_90:13d}")

# Plot: one curve per head, s_i against i. Head 0.7 in red, everything else grey.
fig, ax = plt.subplots(figsize=(8, 5))
for h, s in enumerate(all_s):
    if h == 7:
        ax.plot(range(1, 65), s, color="red", linewidth=2.5, label="0.7 (prev-token head)")
    else:
        ax.plot(range(1, 65), s, color="grey", linewidth=1, alpha=0.6)
        ax.annotate(f"0.{h}", (1, s[0].item()), fontsize=7, color="grey")
ax.set_xlabel("pipe i (sorted, biggest first)")
ax.set_ylabel("s_i")
ax.set_title(f"singular values of W_O[0,h] @ W_K[1,{READER}], one curve per layer-0 head")
ax.legend()
fig.tight_layout()
out = "results/k_composition/step2_all_heads.png"
fig.savefig(out, dpi=110)
print(f"\nsaved {out}")
