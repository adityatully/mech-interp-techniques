"""Piece 1 check: the "zero-layer model" = embed -> unembed, no attention.

Question : do Embed / PosEmbed / Unembed give the right shapes, and what does the
           direct path W_E @ W_U predict on its own?
Metric   : shape asserts; top-5 next-token predictions per position; one equality check.
Expected :
  - logits[b, p] depend ONLY on the token at p (no attention, no position in the residual),
    so the same token at two positions gives bit-identical logits.
  - The direct path learns bigram statistics: "given this token, what usually comes next".
    (2) strong empirical evidence, Elhage et al. 2021 on attention-only models.

PREDICT before running:
  1. Shape of tokens, resid, pos_vectors, logits for the prompt below?
  2. Top prediction after "The"? After " cat" (1st time)? After " cat" (2nd time)?
  3. Can this model predict " sat" after the second " cat" by using the first sentence? Why / why not?
  4. (from piece 0) What value does IGNORE hold in the fp16 file?

Run on coe9 (interp env), from the repo root:
    python experiments/k_composition/s1_zero_layer.py
"""

import torch
from transformers import AutoTokenizer

from model import TOKENIZER_NAME, Embed, PosEmbed, Unembed, load_state_dict

torch.set_grad_enabled(False)

sd = load_state_dict()
print("IGNORE in file:", sd["blocks.0.attn.IGNORE"].item())

embed = Embed(sd["embed.W_E"])
pos_embed = PosEmbed(sd["pos_embed.W_pos"])
unembed = Unembed(sd["unembed.W_U"], sd["unembed.b_U"])

# ---- tokenize (boilerplate) ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
prompt = "The cat sat on the mat. The cat sat on the"
# The NeoX tokenizer does not add BOS on its own. TransformerLens prepends it, and the model
# was trained that way, so we prepend it by hand. Position 0 is then BOS, not "The".
ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
tokens = torch.tensor([ids])  # [batch, pos] = [1, seq_len]
str_tokens = [tokenizer.decode([i]) for i in ids]
print(f"\nBOS id = {tokenizer.bos_token_id} ({tokenizer.bos_token!r})")
print("tokens:", [f"{p}:{s!r}" for p, s in enumerate(str_tokens)])

# ---- forward through the direct path ------------------------------------------------
resid = embed(tokens)  # [batch, pos, d_model]
pos_vectors = pos_embed(tokens)  # [batch, pos, d_model]  (built, but NOT added to resid)
logits = unembed(resid)  # [batch, pos, d_vocab]

batch, seq_len = tokens.shape
assert resid.shape == (batch, seq_len, 768), resid.shape
assert pos_vectors.shape == (batch, seq_len, 768), pos_vectors.shape
assert logits.shape == (batch, seq_len, 50278), logits.shape
print(f"\ntokens {tuple(tokens.shape)}  resid {tuple(resid.shape)}  "
      f"pos_vectors {tuple(pos_vectors.shape)}  logits {tuple(logits.shape)}")

# ---- what does the direct path predict? ---------------------------------------------
print("\nTop-5 next-token predictions of the zero-layer model:")
for p in range(seq_len):
    top = logits[0, p].topk(5).indices.tolist()
    print(f"  {p:2d} {str_tokens[p]!r:10s} -> {[tokenizer.decode([t]) for t in top]}")

# ---- equality check: same token, different position ---------------------------------
first_cat, second_cat = [p for p, s in enumerate(str_tokens) if s == " cat"]
same = torch.equal(logits[0, first_cat], logits[0, second_cat])
print(f"\nlogits at ' cat' pos {first_cat} == pos {second_cat}: {same}")
