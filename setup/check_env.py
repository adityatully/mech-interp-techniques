"""Check the `interp` environment on coe9.

Run on coe9 after setup_coe9.sh, with the interp env active:
    conda activate /data/coe9/envs/interp && python setup/check_env.py

Each check prints OK / WARN / FAIL. Exit code is 1 if any check FAILs.
"""

import importlib
import importlib.metadata as md
import os
import shutil
import sys

BASE = "/data/coe9"
failed = False


def report(status: str, name: str, detail: str = "") -> None:
    global failed
    if status == "FAIL":
        failed = True
    print(f"[{status:4}] {name:28} {detail}")


# --- 1. Python and the active env -------------------------------------------
report("OK", "python", sys.version.split()[0])
in_env = sys.prefix.startswith(f"{BASE}/envs/interp")
report("OK" if in_env else "FAIL", "active env", sys.prefix)

# --- 2. Env vars: caches must live on /data, not on / -----------------------
for var in ["HF_HOME", "PIP_CACHE_DIR", "TORCH_HOME", "TMPDIR"]:
    val = os.environ.get(var)
    ok = val is not None and val.startswith(BASE)
    report("OK" if ok else "FAIL", var, str(val))
for var, want in [("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                  ("TOKENIZERS_PARALLELISM", "false")]:
    val = os.environ.get(var)
    report("OK" if val == want else "WARN", var, f"{val} (want {want})")

# --- 3. Disk space -----------------------------------------------------------
for path in ["/", BASE]:
    if os.path.exists(path):
        free_gb = shutil.disk_usage(path).free / 1e9
        report("OK" if free_gb > 5 else "WARN", f"free space {path}", f"{free_gb:.0f} GB")
    else:
        report("FAIL", f"free space {path}", "path does not exist")

# --- 4. Library versions (record these in each README) ----------------------
PACKAGES = ["torch", "transformer_lens", "sae-lens", "nnsight", "transformers",
            "accelerate", "bitsandbytes", "scikit-learn", "einops", "jaxtyping"]
for pkg in PACKAGES:
    try:
        report("OK", pkg, md.version(pkg))
    except md.PackageNotFoundError:
        report("FAIL", pkg, "not installed")

# --- 5. GPU ------------------------------------------------------------------
try:
    import torch
except ImportError:
    report("FAIL", "torch import", "cannot import torch; stopping")
    sys.exit(1)

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    report("OK", "gpu", f"{props.name}, CUDA {torch.version.cuda}")
    report("OK" if free / 1e9 > 4 else "WARN", "gpu memory",
           f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB (shared machine)")
    # bf16 matmul must work: we load 4-8B models in bf16.
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    report("OK", "bf16 matmul", f"dtype {(x @ x).dtype}")
else:
    report("FAIL", "gpu", "torch.cuda.is_available() is False")

# --- 6. End-to-end: GPT-2 Small through TransformerLens ---------------------
# Known answer: top next token after this prompt is " Paris".
try:
    tl = importlib.import_module("transformer_lens")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tl.HookedTransformer.from_pretrained("gpt2", device=device)
    prompt = "The Eiffel Tower is located in the city of"
    tokens = model.to_tokens(prompt)                      # [batch, pos] = [1, n_pos], BOS at pos 0
    logits, cache = model.run_with_cache(tokens)          # logits [batch, pos, d_vocab] = [1, n_pos, 50257]
    resid = cache["resid_post", 11]                       # [batch, pos, d_model] = [1, n_pos, 768], last layer
    assert logits.shape == (1, tokens.shape[1], model.cfg.d_vocab), logits.shape
    assert resid.shape == (1, tokens.shape[1], model.cfg.d_model), resid.shape
    top = model.to_string(logits[0, -1].argmax())
    report("OK" if top == " Paris" else "WARN", "gpt2 forward pass",
           f"top next token = {top!r} (expect ' Paris'), on {device}")
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    cached = os.path.isdir(hub) and any("gpt2" in d for d in os.listdir(hub))
    report("OK" if cached else "WARN", "weights cached in HF_HOME", hub)
except Exception as e:  # noqa: BLE001 - report any failure, do not crash the check
    report("FAIL", "gpt2 forward pass", f"{type(e).__name__}: {e}")

print()
print("FAILED: fix the FAIL lines above." if failed else "All checks passed.")
sys.exit(1 if failed else 0)
