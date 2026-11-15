"""ARENA's 2-layer attention-only model, written from scratch.

Built piece by piece:
    piece 1: weight loader, Embed, PosEmbed, Unembed
    piece 2: Attention
    piece 3 (this file so far): the full model (cache + interventions: you design them next)

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


class Attention(nn.Module):
    """One attention layer: 12 heads in parallel, each reading and writing the residual stream.
    we are keeping head_index as a dimension itself, but yeah can mostly ignore them for clarity. 

    Extra shape names: head = 12, d_head = 64, q_pos / k_pos = query / key position.

    Weights for layer l (TransformerLens layout, tokens are rows):
        W_Q, W_K, W_V  [head, d_model, d_head]   read:  q = x @ W_Q[h]     [pos, 64]
        W_O            [head, d_head, d_model]   write: out = z @ W_O[h]   [pos, 768]
        b_Q, b_K, b_V  [head, d_head]
        b_O            [d_model]                 one bias for the whole layer, not per head
    In your column notes these are W_Q[h].T = [64, 768] (out, in), etc.

    Shortformer: Q and K read (resid + pos); V reads resid alone.
    So WHERE a head looks can depend on position, but WHAT it moves (v, then W_O) cannot.
    """

    def __init__(self, sd: dict[str, torch.Tensor], layer: int):
        super().__init__()
        p = f"blocks.{layer}.attn."
        for name in ["W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"]:
            setattr(self, name, nn.Parameter(sd[p + name], requires_grad=False)) 
        self.d_head = self.W_Q.shape[-1]  # 64

    def forward(self, resid: torch.Tensor, pos_vectors: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # resid, pos_vectors [batch_size, pos or sequencelength, d_model]
        qk_input = resid + pos_vectors  # [batch, pos, d_model]  shortformer: position only here
        v_input = resid  # [batch, pos, d_model]

        # ---- read: project into each head's 64-dim subspace --------------------------
        q = einops.einsum(qk_input, self.W_Q, "batch pos d_model, head d_model d_head -> batch pos head d_head") + self.b_Q
        k = einops.einsum(qk_input, self.W_K, "batch pos d_model, head d_model d_head -> batch pos head d_head") + self.b_K
        v = einops.einsum(v_input, self.W_V, "batch pos d_model, head d_model d_head -> batch pos head d_head") + self.b_V

        # ---- where to look: QK circuit ------------------------------------------------
        # scores[b, h, i, j] = q_i . k_j / sqrt(d_head)
        # Why the scale: if q and k entries have variance ~1, q.k is a sum of 64 terms with
        # variance ~64. Dividing by sqrt(64) brings it back to ~1, so softmax doesn't saturate.
        scores = einops.einsum(q, k, "batch q_pos head d_head, batch k_pos head d_head -> batch head q_pos k_pos")
        scores = scores / self.d_head**0.5

        # Causal mask: query i may only look at keys j <= i. Entries above the diagonal -> -inf,
        # so softmax gives them exactly 0. (Every row keeps its diagonal, so no row is all -inf.)
        seq_len = scores.shape[-1]
        future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        pattern = scores.softmax(dim=-1)  # [batch, head, q_pos, k_pos], each row sums to 1

        # ---- what to move: OV circuit -------------------------------------------------
        # z_i = sum_j pattern[i, j] * v_j   (a weighted average of value vectors)
        z = einops.einsum(pattern, v, "batch head q_pos k_pos, batch k_pos head d_head -> batch q_pos head d_head")
        # Each head writes back into the 768-dim stream through its own W_O.
        head_out = einops.einsum(z, self.W_O, "batch pos head d_head, head d_head d_model -> batch pos head d_model")
        attn_out = head_out.sum(dim=2) + self.b_O  # [batch, pos, d_model]  heads simply add up

        cache = dict(q=q, k=k, v=v, scores=scores, pattern=pattern, z=z, head_out=head_out)
        return attn_out, cache


class AttnOnlyTransformer(nn.Module):
    """The full model: embed -> attn layer 0 -> attn layer 1 -> unembed. Nothing else.

    The residual stream is a running sum. Each layer READS the current sum and ADDS its output:
        resid_0 = W_E[tokens]                              what the tokens are
        resid_1 = resid_0 + attn0(resid_0, pos)            + what layer 0 moved
        resid_2 = resid_1 + attn1(resid_1, pos)            + what layer 1 moved
        logits  = resid_2 @ W_U + b_U
    So layer 1 sees everything layer 0 wrote. That is the only way composition can happen.
    No LayerNorm between layers (normalization_type=None), so the sum is literally a sum.
    """

    def __init__(self, sd: dict[str, torch.Tensor]):
        super().__init__()
        self.embed = Embed(sd["embed.W_E"])
        self.pos_embed = PosEmbed(sd["pos_embed.W_pos"])
        self.layers = nn.ModuleList([Attention(sd, layer=0), Attention(sd, layer=1)])
        self.unembed = Unembed(sd["unembed.W_U"], sd["unembed.b_U"])

    @classmethod
    def from_pretrained(cls, path: str | None = None, device: str = "cpu") -> "AttnOnlyTransformer":
        return cls(load_state_dict(path)).to(device)

    def forward(self, tokens: torch.Tensor, return_cache: bool = False):
        # tokens [batch, pos]  ->  logits [batch, pos, d_vocab]            (default)
        #                      ->  (logits, cache)                          (if return_cache=True)
        #
        # cache is a flat dict keyed by (name, layer), like TransformerLens:
        #   cache["pattern", 1]     [batch, head, q_pos, k_pos]   layer 1's attention pattern
        #   cache["resid_pre", 1]   [batch, pos, d_model]         what layer 1 reads (= resid_0 + attn0 out)
        #   cache["attn_out", 0]    [batch, pos, d_model]         what layer 0 adds to the stream
        #   cache["resid_final"]    [batch, pos, d_model]         what the unembed reads (no layer number)
        cache = {}
        resid = self.embed(tokens)  # [batch, pos, d_model]
        pos_vectors = self.pos_embed(tokens)  # [batch, pos, d_model]  fed to every layer's Q and K
        cache["pos_vectors"] = pos_vectors
        for layer, attn in enumerate(self.layers):  # layer = 0, 1
            cache["resid_pre", layer] = resid
            attn_out, layer_cache = attn(resid, pos_vectors)  # layer_cache: q, k, v, scores, pattern, z, head_out
            for name, tensor in layer_cache.items():
                cache[name, layer] = tensor  # e.g. "pattern" -> cache["pattern", 0]
            cache["attn_out", layer] = attn_out
            resid = resid + attn_out
        cache["resid_final"] = resid
        logits = self.unembed(resid)  # [batch, pos, d_vocab]
        if return_cache:
            return logits, cache
        return logits


if __name__ == "__main__":
    sd = load_state_dict(sys.argv[1] if len(sys.argv) > 1 else None)
    print({k: tuple(v.shape) for k, v in sd.items() if not k.startswith("blocks")})
