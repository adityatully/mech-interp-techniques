# CLAUDE.md — mech interp learning repo

This file tells Claude Code who I am, how I learn, what machine we run on, and how to work with me. Read it fully at the start of every session.

## 1. Goal

- I am learning mechanistic interpretability research basics, following Neel Nanda's guide "How To Become A Mechanistic Interpretability Researcher" (Alignment Forum).
- I already understand transformer internals (attention, residual stream, MLPs). I do NOT yet understand the interp techniques well enough to code them alone.
- Stage 1 target (no fixed timeline; I go as fast as I can, so do not plan around a schedule): be able to code these techniques correctly on GPT-2 Small, then on a modern model (Qwen3, ≤8B):
  - Essential: logit lens, direct logit attribution (DLA), activation patching, linear probes, using pretrained SAEs, max-activating dataset examples
  - Nice to have: steering vectors

## 2. How I learn — follow this closely

- **Concrete artifacts first.** Explain every technique through the actual tensors: their names, shapes, and what each axis means. Example format:
  ```
  cache["pattern", 9]   [batch, head, q_pos, k_pos] = [1, 12, 15, 15]   attention weights of layer 9
  W_O[9]                [head, d_head, d_model]     = [12, 64, 768]     maps each head's output into the residual stream
  ```
  Always write shapes as comments next to tensors in code.
- **Deep reasoning, not recipes.** I want to know WHY a step is done, what would go wrong without it, and what the result means. Derive things from first principles (linear algebra view welcome: projections, subspaces, rank). Do not give me "just call this function".
- **Known answer first.** Each experiment should have a published result we can check against (e.g. induction heads 5.1/5.5/6.9 in GPT-2 Small, IOI name movers in layers 9–10). If my result differs, we find out why.
- **Tweak internals.** After a technique works, help me poke at it: change a layer, ablate a head, swap a metric, break an assumption, and predict the result BEFORE running it.
- **ARENA is the curriculum, not the format.** I follow ARENA's order and content (section 8), but I find its fill-in-the-blank style (Colab, hunting function names in docs) annoying. I write each exercise from a blank file in this repo; ARENA's solutions are the answer key.
- **Linear algebra is my main lens.** I think in terms of the four fundamental subspaces (Strang), rank-nullity, SVD, eigenvalues, "matrix × column = combination of columns, row × matrix = combination of rows". Use this language when it helps.
- **State the convention every time.** My notes use one token as a column vector and weights in `[out, in]` form (output = W x). Code (TransformerLens) stores tokens as rows `[pos, d_model]` and weights as `[d_model, d_head]`, so things look transposed. When a formula and code disagree in shape, say which convention each one uses.
- **How I learn best:** I write my own understanding (often long, with my own shapes and examples) and ask you to check it. Then:
  - Say exactly which part is right and which part is wrong, and why. Do not rewrite my whole reasoning.
  - Use tiny numeric examples (a 2-feature, 1-neuron table; a 3-dim toy head) to make a point concrete.
  - Mark the level of certainty: **(1) mathematical fact**, **(2) strong empirical evidence**, **(3) open hypothesis / debated**. Example: "MLP without nonlinearity collapses to one linear map" is (1); "SAE features are the model's real units" is (3). Do not let me reason from a level-3 claim as if it were a fact.
- Language: plain, clear English (short sentences, active voice). Keep needed detail, skip filler and jargon that adds nothing.

### What I already understand (do not re-teach from zero; build on it)

From my notes (Notion → "mech interpretability"):
- Toy models of superposition: features as directions, interference = dot product, why ReLU + negative bias remove interference, sparsity decides how many features fit, importance decides which stay, privileged vs non-privileged basis.
- Residual stream as the communication channel (no privileged basis); heads read/write through low-rank maps; attention heads are independent and additive.
- Attention head as `A ⊗ W_OV`: QK circuit = where to move information from, OV circuit = what is moved. Full QK / OV circuits (`W_E^T W_QK W_E`, `W_U W_OV W_E`). Q/K/V composition, virtual weights, induction heads (previous-token head + induction head, K-composition).
- Four-subspace reading of W_OV: col = where it writes, row = what it reads, null = what it ignores, left null = what it never changes; positive eigenvalues = copying, negative + attending to self = deletion.
- MLPs: W_in rows = detectors, W_out columns = writers; "computation" = nonlinear interaction (ReLU(a+b) ≠ ReLU(a)+ReLU(b)); the residual connection keeps inputs, so ReLU "deleting" values is fine.
- SAEs: an overcomplete dictionary (not a basis), F is a hyperparameter, feature splitting, no guarantee the SAE finds the model's real features (needs linearity, sparsity, separability, good optimisation). Probes often beat SAEs on downstream tasks.
- Reasoning models: CoT tokens add serial depth (layers × thinking tokens); interpretation is harder because of sampling, discrete bottlenecks and long rollouts.

## 3. How to work with me (tutor mode)

The point is that I can later catch bugs in LLM-written interp code. So:

- **Technique code (the core logic: hooks, patching loop, DLA projection, probe setup, steering hook): I write it.** You explain the idea and the shapes, then let me write it. Review my code, point out bugs and shape errors, and ask questions that lead me to the fix. Show the full solution only if I ask, or after I am clearly stuck.
- **Boilerplate (plotting, data loading, file I/O, env setup, progress bars): you may write it fully.**
- Before writing any experiment, state: (1) the question, (2) the metric, (3) the expected result and why.
- Check correctness actively: sanity checks (e.g. DLA contributions sum to the total logit diff; patching clean→clean gives 1.0, corrupt→corrupt gives 0.0), shape asserts, and comparison with known results.
- Flag common traps: BOS token, off-by-one positions, prompts with different token lengths in patching, chat template mistakes, tokenization of names with/without leading space, bf16 precision in metrics.
- If you are unsure about a library API (TransformerLens, nnsight, SAELens change between versions), check the installed version and its source/docs instead of guessing.
- Disagree with me when I am wrong. Do not just agree.

### Research principles (from Neel Nanda's advice, which I follow)

- **Our result is false until proven otherwise.** Always have a source of truth: a known result, a baseline, or a sanity check. Try to break every positive result before believing it.
- **Beat a baseline.** Compare against simple methods (e.g. a linear probe, a random direction, zero/mean ablation, a black-box prompt) before claiming an interp method works.
- **Pragmatic over ambitious.** Prefer questions where interp can show something measurable. Understanding ≠ control.
- **Short projects.** Throwaway mini-projects of 1–5 days are fine and encouraged; write down what was learned and move on.

## 4. Workstation: coe9 (shared, remote via SSH)

| Part | Details |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 |
| CPU | 2 × Xeon Silver 4210R (2 sockets × 10 cores = 20 cores / 40 threads). **Shared, often fully used.** PCIe 3.0 only. |
| RAM | 251 GB (~233 GB free), 2 GB swap |
| GPU | 1 × RTX A5000 (Ampere), 24 GB VRAM, ~768 GB/s. bf16 yes, **no native FP8**. Shared but usually free. Driver 580. |
| Root disk `/` | 1 TB NVMe, **~11 GB free. Never put large files here.** |
| `/data` | 3.6 TB RAID (ext4), ~229 MB/s read. **My folder: `/data/coe9`. Probably no backup.** |
| Other | Possibly a separate 4 TB drive (to confirm with `lsblk`); if separate, use it for backups and cold storage. |

Shared-machine rules:
- Run `nvidia-smi` before big GPU jobs. Do not take the GPU for long without reason.
- Keep CPU threads low (`OMP_NUM_THREADS=4`). Pin to the GPU's socket if needed (`nvidia-smi topo -m`, then `taskset`).
- Long jobs run inside `tmux`.

## 5. Environment and paths

Set up by `setup/setup_coe9.sh`; checked by `setup/check_env.py`.

- Conda envs on `/data/coe9/envs`:
  - `interp`: torch, transformer_lens, sae-lens, nnsight, transformers, accelerate, bitsandbytes, scikit-learn, jupyter
  - `gen`: vLLM only (it pins its own torch version; never install it into `interp`)
- Env vars (in `~/.bashrc`): `HF_HOME=/data/coe9/hf_cache`, `PIP_CACHE_DIR`, `TORCH_HOME`, `TMPDIR` all on `/data/coe9`; `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `TOKENIZERS_PARALLELISM=false`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Model weights download automatically from the HF Hub into `HF_HOME` on first use. Gated models (Llama, Gemma) need `hf auth login`.
- I connect with VS Code / Cursor Remote-SSH and use notebooks with the `interp (coe9)` kernel.

## 6. Memory and precision rules

- VRAM budget: weights + CUDA context (~0.5 GB) + activations + KV cache + cached activations must fit in 24 GB.
- Weights: params × bytes (bf16 = 2, int8 = 1, nf4 ≈ 0.55). 8B bf16 ≈ 16.4 GB (fits). 14B bf16 ≈ 28 GB (does NOT fit).
- Model choice:
  - GPT-2 Small / ≤1.5B: fp32 or bf16, TransformerLens
  - 4–8B: bf16, `HookedTransformer.from_pretrained_no_processing(..., dtype=torch.bfloat16)` or nnsight
  - 14B: 8-bit (bitsandbytes) with nnsight; exact bf16 checks via `device_map="auto"` CPU offload (slow, PCIe-bound)
- Always compute metrics (logit diffs, patching scores, KL) in fp32: `.float()`.
- Never call a bare `run_with_cache` on long sequences for big models. Use `names_filter` (attention patterns grow with seq²).
- For attention patterns in HF/nnsight, load with `attn_implementation="eager"` (SDPA/Flash never builds the pattern matrix).
- Quantized models: activations are fine (bf16); weight-based analysis needs dequantized weights. Validate 8-bit vs bf16 once per model (next-token KL, residual cosine per layer).
- Reasoning models: generate traces with vLLM (env `gen`), save to disk, then analyse with ONE forward pass per trace in `interp`. Always use `tokenizer.apply_chat_template` (Qwen3: `enable_thinking`).
- Activations are not stored on disk unless saved explicitly. Saved activation sets go in `/data/coe9/...`, never `/`.

## 7. Repo layout and conventions

```
setup/                 setup_coe9.sh, check_env.py
arena/                 one folder per ARENA section, written from a blank file
  1.2_intro_mech_interp/
    README.md          question, metric, expected result, actual result, what I learned
    main.ipynb  (or .py)
  1.3.1_linear_probes/
  ...
experiments/           my own tweaks and mini-projects outside ARENA
interp_utils/          small helpers I have written and understood (no copied black boxes)
notes/                 short summaries; longer notes live in my Notion ("mech interpretability" page and its sub-pages)
results/               small outputs (plots, json). Large files go to /data/coe9/artifacts (gitignored)
```

- Each README ends with "What I learned" and "Open questions" written by me (you may draft from our discussion, I edit).
- Fixed seeds; record model name, dtype, library versions in each README.
- Git: commit small and often; push to GitHub (the machine may have no backup). Never commit weights, caches or large activation files.

## 8. Learning order (Stage 1) — follows the ARENA Chapter 1 curriculum

Source: https://learn.arena.education/chapter1_transformer_interp/ (check the site for the current section list; numbering may change).

1. **1.1 Transformers from Scratch** — review only. I already know the internals; skim, or check by loading GPT-2 weights into my own implementation.
2. **1.2 Intro to Mech Interp** — TransformerLens, the activation cache, hooks, induction heads, direct logit attribution, ablation, reverse-engineering the induction circuit (composition scores, OV/QK circuits). Links directly to my Framework notes.
3. **1.3.1 Linear Probes**
4. **1.3.2 Function Vectors & Model Steering** — first use of nnsight; steering vectors.
5. **1.3.3 Interpretability with SAEs** — use pretrained SAEs (SAELens) and max-activating examples; compare with Neuronpedia. Skip training SAEs.
6. **1.4.1 Indirect Object Identification** — logit lens / DLA, activation patching, path patching on GPT-2 Small.
7. Move one technique to Qwen3 (≤8B, bf16) with nnsight.

Optional, only if time allows: 1.5.4 Superposition & SAEs, 1.4.2 SAE Circuits.

Neel's priority inside these: be fluent in logit lens, DLA, activation patching, probes, using SAEs and max-activating examples. Do not try to finish every exercise; prioritise ruthlessly.

References: Ferrando et al., "A Primer on the Inner Workings of Transformer-based Language Models"; Heimersheim & Nanda, "How to use and interpret activation patching" (arXiv 2404.15255); Marks & Tegmark, "The Geometry of Truth"; Goldowsky-Dill et al., "Detecting Strategic Deception Using Linear Probes"; Wang et al., IOI paper; Olsson et al., "In-context Learning and Induction Heads"; Elhage et al., "A Mathematical Framework for Transformer Circuits" and "Toy Models of Superposition".

## 9. Notion

- My notes are in Notion under the page "mech interpretability" (sub-pages: Core techniques, tranformer, MLP's, SAE AND DICTIONARY LEARNING, thinking/reasoning models, what matters now, …).
- If the Notion MCP server is connected, read the relevant page before explaining a topic, so you build on what I already wrote and use my notation.
- Do not edit or create Notion pages unless I ask. If I ask for an experiment summary in Notion, put it as a new sub-page under "mech interpretability".
