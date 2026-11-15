# Experiment: Seeing K-composition inside induction heads

*Started 2026-11-15. Mech interp learning project.*

## 1. The idea in one line

Find the exact subspace where a previous-token head **writes** and an induction head's **key** **reads**. Then check, by measuring and by breaking it, that this overlap is what makes induction work.

## 2. My understanding (how I was thinking)

**Setup (GPT-2 Small shapes)**
- Residual stream: 768 dims, with no privileged basis. Single coordinates mean nothing, but directions do.
- Each head reads and writes through small subspaces: W_Q, W_K, W_V are each 64×768, and W_O is 768×64.

**The three types of composition**
- **Q-composition:** W_Q reads a subspace that an earlier head wrote to. The destination decides *what to ask* from context it already collected.
- **K-composition:** W_K reads a subspace that an earlier head wrote to. The source changes *how it answers* (what it "advertises").
- **V-composition:** W_V reads a subspace that an earlier head wrote to. This changes *what moves*, not where the head looks.
- Composition can be partial. A head does not have to read only from what an earlier head wrote.

**Induction**
- Sequence: `… A B … A` → predict `B`.
- K-composition lets B advertise "a token A came before me".
- Example from the paper's figure (*Mr and Mrs Dursley … Mr D|urs|ley*):
  - The current token is the second `D`. It attends back to the first `urs` and predicts `urs`.
  - A previous-token head wrote "the token before me was D" into the residual stream at the first `urs`.
  - The induction head's QK circuit at `D` looks for the key that says "prev = D", and it matches.
  - The hard part is getting the attention pattern right. Copying is easy.

## 3. Questions I wanted to answer

1. Can I see concretely that induction heads exist, and that **K-composition** is the part they really need?
2. Can I find **which subspace** the earlier head wrote to and **which subspace** the later head's W_K reads from?
3. Can I find the **overlap**? The single dimensions have no meaning, but the overlap carries the information "the token before me was D".
4. Can I change that message with **minimal edits** and watch how induction and next-token prediction change?
5. Can I **destroy** K-composition and look at the output?
6. Ideas I considered: SAEs, steering vectors, weight-space programming.

## 4. Corrections from discussion

- **Overlap is graded, not a count.** "The key reads 24 of the 64 directions" is not accurate. Two 64-dim subspaces meet at 64 *principal angles*, each from 0° to 90°. A dimension count only makes sense after I choose a threshold.
- **The sliding-window idea does not work.** The overlap has no ordered dimensions, so "sliding" the message is a basis-dependent permutation with no meaning. Changes that *do* have meaning:
  - **Remove** the message (project it out)
  - **Scale** the message
  - **Replace** "prev = D" with "prev = X"
- **One head reads two different subspaces.** The induction head's **W_K** reads the composed message ("prev = D"). Its **W_V** reads the token identity of `urs` itself (the direct path). Copying does not need composition.
- **The subspaces are learned, not random.** "Random" only means they are not lined up with the coordinate axes.
- **SAEs and steering vectors are not needed here.** The weights (W_O¹, W_K²) already define the subspace. An SAE on the previous-token head's output is a possible later extension.
- **The idea is not new.** Elhage et al. (2021) measured composition in a 2-layer attention-only model. Olsson et al. (2022) studied induction heads in larger models, and ARENA 1.2 has exercises on it. This is useful, because I can check my results against known answers.

## 5. Experiment steps

**Model:** start with `attn-only-2l` (TransformerLens). It has 2 layers and no MLPs, so the circuit is clean. Then repeat on GPT-2 Small. Known heads there (confirm with my own scores): **4.11** previous-token head, **5.5** and **6.9** induction heads.

**Rules:**
- Do one step per session.
- Write the code from a blank file on coe9.
- Use ARENA solutions only as an answer key.

### Step 1: Find the heads (behavior)
- Input: 50 random tokens, repeated once (100 tokens in total).
- For each head, compute:
  - **Previous-token score:** mean attention from i to i − 1
  - **Induction score:** mean attention from i to i − 49 (second half)
- Also record the **loss on the second half**. This is the metric for every later step.
- Expected result: one strong previous-token head in layer 0, and one strong induction head in layer 1.

### Step 2: Measure the overlap (weights only)

| Object | Shape | Meaning |
|---|---|---|
| W_O¹ | 768 × 64 | Its column space is where head 1 writes |
| W_K² | 64 × 768 | Its row space is where head 2's key reads |
| **W_K² W_O¹** | **64 × 64** | **The overlap** |
| W_K² W_OV¹ | 64 × 768 | The full K-composition path |

- Compute the SVD of W_K² W_O¹. The number of large singular values gives the effective overlap dimension.
- Compute principal angles: Q₁ = orth(W_O¹) and Q₂ = orth(W_K²ᵀ), each 768×64. The singular values of Q₂ᵀQ₁ are the cosines of the angles.
- **Random baseline:** two random 64-dim subspaces in 768 dims already have top cosines of about 0.5. Always compare against this.
- Repeat with W_Q² and W_V². Expected: K overlap is strong, Q overlap is weak, and V overlap is close to random.
- Compute composition scores for all earlier heads: ‖W_K² W_OV¹‖_F / (‖W_K²‖_F ‖W_OV¹‖_F), and the same for Q and V.

### Step 3: Token-level check
- Compute W_Eᵀ W_Q²ᵀ W_K² W_OV¹ W_E for about 2,000 common tokens (a 2000 × 2000 matrix).
- Expected result: a large diagonal, because query token D matches a key that says "prev = D".

### Step 4: Causal test — is K-composition required?
Remove head 1's output from **one** of head 2's inputs at a time:
- Key input only → expected: induction breaks
- Query input only → expected: little change
- Value input only → expected: little change

Measure the induction score and the second-half loss each time. This is path patching.

### Step 5: Minimal subspace ablation
- From the key input only, project out the top-k singular directions of W_K² W_O¹, mapped into the 768-dim stream through W_O¹.
- Sweep k from 1 to 64 and plot the induction score against k.
- Control: project out random k-dim subspaces instead.
- This is the concrete answer to "how many dimensions carry the message".

### Step 6: Rewrite the message
- At the source position (the first `urs`), replace head 1's output with the output it gives when the previous token is X.
- Expected result: the induction head now attends there when the current token is X, not D.
- **Weight-space version:** replace W_K² with W_K²(I − P), where P projects onto the overlap directions.

## 6. Caveats
- **LayerNorm:** it sits between head 1 and W_K². Its scale is one number per token, so it changes the size of the message but not its direction. Use `fold_ln=True`.
- **GPT-2 has several writers.** More than one head (and the MLPs) can write previous-token information, so the message subspace can be the combined span of several heads.

## 7. References
- Elhage et al., 2021. *A Mathematical Framework for Transformer Circuits.* https://transformer-circuits.pub/2021/framework/index.html
- Olsson et al., 2022. *In-context Learning and Induction Heads.* https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- ARENA Chapter 1.2: Intro to Mech Interp (induction heads, composition scores).

## 8. Results log
| Step | Date | Result | Notes |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |

## 9. Build log (model from scratch)
Raw outputs: `results/k_composition/` (save with `python <script> | tee results/k_composition/<script>.txt`).

| Piece | Date | Result | What I learned (draft, edit me) |
|---|---|---|---|
| 0 load weights | 2026-11-15 | 24 tensors, all fp16. No LN, no MLP. `mask` and `IGNORE` are buffers, not learned weights. W_E and W_U are not tied. | ~92% of parameters are W_E + W_U; the attention circuit is ~4.7M params. |
| 1 zero-layer | 2026-11-15 | Shapes OK. Same token at pos 2 and 9 gives identical logits. Top predictions are mostly word-piece completions: ` cat`→`abolism`, ` sat`→`irical`, ` on`→` behalf`, `.`→`jpg`/`png`, BOS→`Q`/`Introduction`/`/*`. | The direct path `W_E W_U` (+ `b_U`) is a rank-≤768 approximation of a bigram table: P(next \| current token only). Without attention it can't use context, so it can't predict ` sat` after the 2nd ` cat`. |
| 2 attention | 2026-11-15 | Rows sum to 1 (1e-7), no future weight, z matches `F.scaled_dot_product_attention` (7e-7). Prev-token score on random tokens: **0.7 = 0.995**, 0.9 = 0.27, 0.4 = 0.22, rest near chance (≈0.07 = mean of 1/(i+1)). With `--no-pos`: 0.7 drops to 0.092, every head ≈ chance. | Head 0.7 finds i−1 purely through W_pos in Q and K. Without position, a key only knows its token, and on random tokens the token at i−1 is not special. 0.9 and 0.4 may also write some prev-token info, so 0.7 may not be the only K-composition source. |
