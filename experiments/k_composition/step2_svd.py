"""Step 2, first look: the 64 singular values of M = W_O[0,7] @ W_K[1,10].

Weights only. No forward pass, no data.

    z of head 0.7         [64]        what 0.7 computed at one position
    z @ W_O[0,7]          [768]       what 0.7 writes into the residual stream
    ... @ W_K[1,10]       [64]        how that changes 1.10's key
    M = W_O[0,7] @ W_K[1,10]   [64, 64]   goes straight from 0.7's z to 1.10's key

SVD splits M into 64 separate pipes. Pipe i carries the part of z along v_i
into the key along u_i, multiplied by s_i (how much M stretches that direction).

One question: are a few s_i large and the rest small, or are they all about the same?

PREDICT before running: how many pipes do you think are "large"? 1? 5? 30? all 64?

Run on coe9 (interp env), from the repo root:
    HF_HUB_OFFLINE=1 python experiments/k_composition/step2_svd.py | tee results/k_composition/step2_svd.txt
"""

import torch

from model import load_state_dict

sd = load_state_dict()  # dict: name -> tensor (fp32)

W_O = sd["blocks.0.attn.W_O"]  # [head, d_head, d_model] = [12, 64, 768]   layer 0 writes
W_K = sd["blocks.1.attn.W_K"]  # [head, d_model, d_head] = [12, 768, 64]   layer 1 key reads

M = W_O[7] @ W_K[10]  # [64, 768] @ [768, 64] -> [64, 64]
print("M shape:", tuple(M.shape))

s = torch.linalg.svdvals(M)  # [64], already sorted largest -> smallest

# Three ways to read the same 64 numbers:
#   s        raw stretch factor of each pipe
#   s / s[0] size relative to the biggest pipe (1.00 for the first)
#   share    s_i^2 / sum(s^2): fraction of M's total "size" in pipe i.
#            (sum of s_i^2 = sum of all squared entries of M, so the shares add up to 1)
share = s**2 / (s**2).sum()
cumulative = share.cumsum(dim=0)  # running total of share

print(f"\n{'i':>3s} {'s_i':>8s} {'s_i/s_1':>8s} {'share':>7s} {'cumul':>7s}")
for i in range(len(s)):
    print(f"{i + 1:3d} {s[i]:8.3f} {s[i] / s[0]:8.3f} {share[i]:7.3f} {cumulative[i]:7.3f}")
