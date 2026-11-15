"""Piece 0: download ARENA's 2-layer attention-only model and list every weight tensor.

Question : what tensors does the checkpoint hold, and what does each axis mean?
Metric   : none yet; we print name, shape and dtype for each tensor.
Expected : you predict the shapes BEFORE running (fill PREDICTIONS below).

Model facts (from ARENA 1.2's HookedTransformerConfig):
    d_model=768, 
    d_head=64, 
    n_heads=12, 
    n_layers=2, 
    n_ctx=2048,  context winow size is 2048 tokens
    d_vocab=50278
    attn_only=True          -> no MLPs
    normalization_type=None -> no LayerNorm anywhere
    positional_embedding_type="shortformer"
                            -> W_pos is added to the Q and K inputs of every layer,
                               NOT to the residual stream (so V and the residual carry no position)
    tokenizer = EleutherAI/gpt-neox-20b
Known answer (ARENA): 0.7 = previous-token head; 1.4 and 1.10 = induction heads.

Convention reminder: the checkpoint is in TransformerLens layout (tokens are ROWS): each token is a row vector in the residual strea, 
so like if we consdier only one token is [1x768] so each head would be like ex W_q = [768, 64] so q = [1x64]
    resid [pos, d_model];  W_Q[h] is [d_model, d_head], so q = resid @ W_Q[h]
Your notes use columns and [out, in] weights, i.e. the transpose.

Run on coe9 (interp env):
    python experiments/k_composition/s0_load_weights.py
"""

import sys

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "arena-education/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"  # 184 MB, lands in $HF_HOME (/data/coe9/hf_cache)

# ---- PREDICTIONS: write the shape you expect for each, before running -------------
# embed.W_E                 ?
# pos_embed.W_pos           ?
# blocks.0.attn.W_Q         ?
# blocks.0.attn.b_Q         ?
# blocks.0.attn.W_O         ?
# blocks.0.attn.b_O         ?
# unembed.W_U               ?
# unembed.b_U               ?
# Anything in the file that is not a learned weight?  ?
# ------------------------------------------------------------------------------------

# Optional: pass a local .pth path as the first argument (e.g. if coe9 can't reach the HF Hub).
weights_path = sys.argv[1] if len(sys.argv) > 1 else hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)  # dict[str, Tensor]

print(f"{len(state_dict)} tensors in {weights_path}\n")
print(f"{'name':32s} {'shape':24s} {'dtype':10s} {'numel':>12s}")
total = 0
for name, tensor in state_dict.items():
    print(f"{name:32s} {str(list(tensor.shape)):24s} {str(tensor.dtype):10s} {tensor.numel():>12,}")
    total += tensor.numel()
print(f"\ntotal elements: {total:,}  (~{total * 4 / 1e6:.0f} MB in fp32)")
