"""Piece 3 check: the full 2-layer model on one prompt with a repeat in it.

Question : with attention on, does the model use the first sentence to predict the second?
Metric   : for each position p, the probability the model gives to the TRUE next token (p+1),
           for the full model and for the zero-layer model (embed -> unembed only).
Expected :
  - First sentence: both models are unsure (this is a generic sentence).
  - Second sentence, after " cat" / " sat" / " on": the full model should be far more confident
    than the first time, because it can look back at what followed these tokens before.
    The zero-layer model gives the SAME number both times (it saw that in piece 1).

PREDICT before running:
  1. P(" sat" | ... " cat") at the 1st " cat" vs the 2nd " cat", full model. Rough numbers?
  2. Which head(s) do you think are responsible for the jump? How would you check?

Run on coe9 (interp env), from the repo root:
    python experiments/k_composition/s3_full_model.py | tee results/k_composition/s3_full_model.txt
"""

import torch
from transformers import AutoTokenizer

from model import TOKENIZER_NAME, AttnOnlyTransformer

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AttnOnlyTransformer.from_pretrained(device=device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

prompt = "The cat sat on the mat. The cat sat on the"
ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
tokens = torch.tensor([ids], device=device)  # [1, pos]
str_tokens = [tokenizer.decode([i]) for i in ids]

logits_full = model(tokens)  # [1, pos, d_vocab]
logits_zero = model.unembed(model.embed(tokens))  # [1, pos, d_vocab]  direct path only

# Probability of the true next token. Logits at p predict token p+1, so the last position
# has no target and the first token (BOS) is never predicted. Off-by-one is the classic bug here.
targets = tokens[0, 1:]  # [pos-1]


def p_true_next(logits):  # logits [1, pos, d_vocab] -> [pos-1]
    probs = logits[0, :-1].float().softmax(dim=-1)  # [pos-1, d_vocab]  fp32 for metrics
    return probs.gather(-1, targets[:, None])[:, 0]


p_full, p_zero = p_true_next(logits_full), p_true_next(logits_zero)

print(f"{'pos':>3s} {'current':10s} {'true next':10s} {'P zero-layer':>13s} {'P full':>8s}   full model top-3")
for p in range(len(ids) - 1):
    top3 = [tokenizer.decode([t]) for t in logits_full[0, p].topk(3).indices.tolist()]
    print(f"{p:3d} {str_tokens[p]!r:10s} {str_tokens[p + 1]!r:10s} {p_zero[p].item():13.4f} {p_full[p].item():8.4f}   {top3}")

loss_full = -p_full.log()
print(f"\nmean loss (nats), full model: 1st sentence {loss_full[:7].mean():.2f} | 2nd sentence {loss_full[7:].mean():.2f}")
