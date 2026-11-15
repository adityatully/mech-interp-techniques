"""ARENA's 2-layer attention-only model, written from scratch.

Built piece by piece:
    piece 1 (this file so far): weight loader, Embed, PosEmbed, Unembed
    piece 2: Attention
    piece 3: the full model + activation cache + interventions

Shape names used in every comment:
    batch, pos            one prompt per batch row, one token per position
    d_model = 768         residual stream width
    d_vocab = 50278       vocab size (the tokenizer has slightly fewer real tokens; the rest are padding)
    n_ctx   = 2048        max sequence length

Convention: TransformerLens layout, tokens are ROWS.
    resid [batch, pos, d_model];  W_U [d_model, d_vocab];  logits = resid @ W_U
"""

import sys

import einops
import torch
from huggingface_hub import hf_hub_download
from torch import nn

REPO_ID = "arena-education/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"


def load_state_dict(path: str | None = None) -> dict[str, torch.Tensor]:
    """Load the checkpoint and cast every tensor to fp32.

    Why fp32: the file is fp16 (~3 significant digits). That is enough to run the model,
    but too coarse for SVDs, principal angles and small loss differences later on.
    """
    if path is None:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {name: t.float() for name, t in sd.items()}


class Embed(nn.Module):
    """Token id -> its row of W_E.

    W_E [d_vocab, d_model]: row t is the vector token t writes into the residual stream.
    Indexing a row is the same as one_hot(t) @ W_E (in your column notes: W_E x_onehot,
    which picks out a column). We index instead of multiplying because the one-hot
    vector would be 50,278 numbers to select one row.
    """

    def __init__(self, W_E: torch.Tensor):
        super().__init__()
        self.W_E = nn.Parameter(W_E, requires_grad=False)  # [d_vocab, d_model]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [batch, pos] (int64)  ->  [batch, pos, d_model] ->[batch_size, sequence_length, d_model]
        return self.W_E[tokens] # this expression directly indexes by the token ids to get the embedding for each token in the batch


class PosEmbed(nn.Module):
    """Position -> its row of W_pos.

    Shortformer model: this output is NOT added to the residual stream.
    It is added only to the query and key inputs of each attention layer (piece 2).
    So the residual stream itself carries no positional information.
    """

    def __init__(self, W_pos: torch.Tensor):
        super().__init__()
        self.W_pos = nn.Parameter(W_pos, requires_grad=False)  # [n_ctx, d_model]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [batch, pos]  ->  [batch, pos, d_model]
        # Only the sequence length matters; every batch row gets the same position vectors.
        batch, seq_len = tokens.shape
        pos_vectors = self.W_pos[:seq_len]  # [pos, d_model]
        return einops.repeat(pos_vectors, "pos d_model -> batch pos d_model", batch=batch)


class Unembed(nn.Module):
    """Residual stream -> logits over the vocab.

    W_U [d_model, d_vocab]: column t is the direction that, read from the residual stream,
    raises token t's logit. logit_t = resid . W_U[:, t] + b_U[t].
    """

    def __init__(self, W_U: torch.Tensor, b_U: torch.Tensor):
        super().__init__()
        self.W_U = nn.Parameter(W_U, requires_grad=False)  # [d_model, d_vocab]
        self.b_U = nn.Parameter(b_U, requires_grad=False)  # [d_vocab]

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        # resid [batch, pos, d_model]  ->  logits [batch, pos, d_vocab]
        return einops.einsum(resid, self.W_U, "batch pos d_model, d_model d_vocab -> batch pos d_vocab") + self.b_U


if __name__ == "__main__":
    sd = load_state_dict(sys.argv[1] if len(sys.argv) > 1 else None)
    print({k: tuple(v.shape) for k, v in sd.items() if not k.startswith("blocks")})
