"""Step 1: find the previous-token and induction heads from behaviour.

Question : which of the 24 heads attend to i-1 (previous token), and which attend to the
           token AFTER the previous copy of the current token (induction)?
Metrics  :
  prev score      = mean pattern[i, i-1]    over queries 1..100
  induction score = mean pattern[i, i-49]   over queries 51..100 (second copy only)
  loss            = mean next-token loss on the first copy vs the second copy
Control  : the same run where the "second copy" is NEW random tokens (no repeat).
           Induction scores should fall to chance and the second-half loss should stay high.
           If they don't, the "induction score" is measuring something else.
Expected : (known answer, ARENA)  0.7 = prev-token head;  1.4 and 1.10 = induction heads.
           Loss: first copy high (random tokens are unpredictable), second copy much lower.
           Chance level for a score ~ uniform attention over the causal window.

Layout of one sequence (BOS shifts everything by 1, but not the gap between copies):
    pos:    0     1  2 ... 50    51 52 ... 100
    token: BOS    t1 t2 ... t50  t1 t2 ... t50
    query at 51 (t1, second time) -> induction head should attend to 2 (t2, the token after t1's first copy)
                                     51 - 49 = 2

PREDICT before running:
  1. Mean loss on the first copy and on the second copy, roughly? (ln(50278) ≈ 10.8 nats is "uniform guessing")
  2. Induction score of the best head: 0.3? 0.6? 0.9?
  3. In the control, what happens to 0.7's prev-token score? (Hint: does it need a repeat?)

Run on coe9 (interp env), from the repo root:
    python experiments/k_composition/step1_find_heads.py | tee results/k_composition/step1_find_heads.txt
"""

import json

import torch
from transformers import AutoTokenizer

from model import TOKENIZER_NAME, AttnOnlyTransformer

torch.set_grad_enabled(False)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH, SEQ_LEN = 20, 50  # 20 sequences, each 50 random tokens

model = AttnOnlyTransformer.from_pretrained(device=device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
d_vocab = model.embed.W_E.shape[0]


def make_tokens(repeat: bool) -> torch.Tensor:
    """[BOS, t1..t50, t1..t50] if repeat, else [BOS, t1..t50, u1..u50] with fresh u's."""
    first = torch.randint(0, d_vocab, (BATCH, SEQ_LEN), device=device)  # [batch, 50]
    second = first if repeat else torch.randint(0, d_vocab, (BATCH, SEQ_LEN), device=device)
    bos = torch.full((BATCH, 1), tokenizer.bos_token_id, device=device)  # [batch, 1]
    return torch.cat([bos, first, second], dim=1)  # [batch, 101]


def per_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Loss for predicting token p+1 from position p.  -> [batch, 100]

    Index j of the result is the loss on token j+1:
        j = 0..49   -> tokens 1..50    (first copy)
        j = 50..99  -> tokens 51..100  (second copy)
    """
    log_probs = logits[:, :-1].float().log_softmax(dim=-1)  # [batch, 100, d_vocab]  fp32
    targets = tokens[:, 1:]  # [batch, 100]
    return -log_probs.gather(-1, targets[..., None])[..., 0]  # [batch, 100]


def head_scores(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """-> prev [layer, head], induction [layer, head]"""
    prev, induction = [], []
    for layer in range(2):
        pattern = cache["pattern", layer]  # [batch, head, q_pos=101, k_pos=101]

        # diagonal(offset=-d) collects pattern[i, i-d] for every query i >= d.
        # offset -1:  queries 1..100              -> 100 entries
        prev_diag = pattern.diagonal(offset=-1, dim1=-2, dim2=-1)  # [batch, head, 100]
        prev.append(prev_diag.mean(dim=(0, -1)))  # [head]

        # offset -(SEQ_LEN-1) = -49:  queries 49..100 -> 52 entries; entry e is query 49+e.
        # We want queries 51..100 (second copy) -> entries 2..51.
        ind_diag = pattern.diagonal(offset=-(SEQ_LEN - 1), dim1=-2, dim2=-1)  # [batch, head, 52]
        ind_diag = ind_diag[..., 2:]  # [batch, head, 50]  queries 51..100
        induction.append(ind_diag.mean(dim=(0, -1)))  # [head]
    return torch.stack(prev), torch.stack(induction)  # [layer, head] each


results = {}
for name, repeat in [("repeated", True), ("control (no repeat)", False)]:
    tokens = make_tokens(repeat)  # [batch, 101]
    logits, cache = model(tokens, return_cache=True)
    loss = per_token_loss(logits, tokens)  # [batch, 100]
    prev, induction = head_scores(cache)  # [2, 12] each

    print(f"\n===== {name} =====")
    print(f"loss  first copy {loss[:, :SEQ_LEN].mean():.2f}   second copy {loss[:, SEQ_LEN:].mean():.2f}  (nats)")
    for label, scores in [("prev-token", prev), ("induction", induction)]:
        flat = scores.flatten()  # [24], index = layer*12 + head
        order = flat.argsort(descending=True)[:5].tolist()
        top = "  ".join(f"{i // 12}.{i % 12}={flat[i]:.3f}" for i in order)
        print(f"top-5 {label:10s} {top}")
    print("induction score, all heads (rows = layer):")
    for layer in range(2):
        print(f"  L{layer}: " + " ".join(f"{s:.2f}" for s in induction[layer].tolist()))

    results[name] = dict(
        loss_first=loss[:, :SEQ_LEN].mean().item(),
        loss_second=loss[:, SEQ_LEN:].mean().item(),
        prev=prev.tolist(),
        induction=induction.tolist(),
    )

# Chance level: uniform attention over the causal window, averaged over the same queries.
q = torch.arange(SEQ_LEN + 1, 2 * SEQ_LEN + 1).float()  # queries 51..100
print(f"\nchance (uniform causal attention) for the induction score: {(1 / (q + 1)).mean():.3f}")

out = "results/k_composition/step1_scores.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"saved {out}")
