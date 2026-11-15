"""Piece 2 check: one attention layer (layer 0), checked three ways.

Question : is our Attention class correct, and does layer 0 contain a previous-token head?
Metrics  :
  A. pattern rows sum to 1, and the upper triangle (future keys) is exactly 0
  B. our z matches PyTorch's own attention (F.scaled_dot_product_attention), max |diff| < 1e-4
  C. previous-token score per head = mean over i of pattern[i, i-1], on random tokens
Expected :
  A, B pass (pure math, (1)).
  C: head 0.7 scores far above the other 11 heads (known answer from ARENA, (2)).
     A random head over a causal window of ~25 keys would score ~1/25 ≈ 0.04 by chance.

PREDICT before running:
  1. Shapes of q, scores, pattern, z, head_out, attn_out for the "cat" prompt (13 tokens).
  2. Roughly what previous-token score will head 0.7 get? 0.3? 0.9? 0.99?
  3. Run again with --no-pos (W_pos removed from Q and K). What happens to head 0.7's score, and why?
     Hint: without position, what does a key at position j know about itself?

Run on coe9 (interp env), from the repo root:
    python experiments/k_composition/s2_attention.py | tee results/k_composition/s2_attention.txt
    python experiments/k_composition/s2_attention.py --no-pos | tee results/k_composition/s2_attention_nopos.txt
"""

import sys

import einops
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import TOKENIZER_NAME, Attention, Embed, PosEmbed, load_state_dict

torch.set_grad_enabled(False)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
NO_POS = "--no-pos" in sys.argv

sd = load_state_dict()
embed = Embed(sd["embed.W_E"]).to(device)
pos_embed = PosEmbed(sd["pos_embed.W_pos"]).to(device)
attn0 = Attention(sd, layer=0).to(device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print(f"device={device}  no_pos={NO_POS}")


def run_layer0(tokens):
    resid = embed(tokens)  # [batch, pos, d_model]  layer 0's input is just the embedding
    pos_vectors = pos_embed(tokens)  # [batch, pos, d_model]
    if NO_POS:
        pos_vectors = torch.zeros_like(pos_vectors)
    return attn0(resid, pos_vectors)


# ---- shapes on a real prompt -----------------------------------------------------------
prompt = "The cat sat on the mat. The cat sat on the"
tokens = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt)], device=device)  # [1, 13]
attn_out, c = run_layer0(tokens)
print("\nshapes:")
for name in ["q", "k", "v", "scores", "pattern", "z", "head_out"]:
    print(f"  {name:9s} {tuple(c[name].shape)}")
print(f"  {'attn_out':9s} {tuple(attn_out.shape)}")

# ---- A: pattern is a valid causal attention pattern ------------------------------------
pattern = c["pattern"]  # [batch, head, q_pos, k_pos]
row_sums = pattern.sum(-1)  # [batch, head, q_pos]
upper = torch.triu(pattern, diagonal=1)  # entries where k_pos > q_pos
print(f"\nA. max |row_sum - 1| = {(row_sums - 1).abs().max().item():.2e}   "
      f"max future weight = {upper.abs().max().item():.2e}")

# ---- B: compare with PyTorch's built-in attention ----------------------------------------
# F.scaled_dot_product_attention wants [batch, head, pos, d_head]; ours are [batch, pos, head, d_head].
q, k, v = (einops.rearrange(c[n], "b pos head d -> b head pos d") for n in ["q", "k", "v"])
z_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [batch, head, pos, d_head]
z_ours = einops.rearrange(c["z"], "b pos head d -> b head pos d")
print(f"B. max |z_ours - z_torch| = {(z_ours - z_torch).abs().max().item():.2e}")

# ---- C: previous-token score on random tokens ------------------------------------------
# Random tokens: no meaning, no grammar. A head that still attends to i-1 is doing it by POSITION.
batch, seq_len = 10, 50
rand = torch.randint(0, sd["embed.W_E"].shape[0], (batch, seq_len), device=device)
bos = torch.full((batch, 1), tokenizer.bos_token_id, device=device)
rand_tokens = torch.cat([bos, rand], dim=1)  # [batch, 51]
_, c = run_layer0(rand_tokens)
pattern = c["pattern"]  # [batch, head, q_pos, k_pos]

# diagonal(offset=-1) picks pattern[..., i, i-1] for i = 1..seq_len  -> [batch, head, seq_len]
prev_attn = pattern.diagonal(offset=-1, dim1=-2, dim2=-1)
prev_score = prev_attn.mean(dim=(0, -1))  # [head]
print("\nC. previous-token score per layer-0 head (mean of pattern[i, i-1]):")
for h in prev_score.argsort(descending=True).tolist():
    print(f"   0.{h:<3d} {prev_score[h].item():.3f}  {'#' * int(40 * prev_score[h].item())}")
