#!/usr/bin/env bash
# Set up the mech-interp environment on coe9.
# Everything heavy (conda, envs, caches, tmp) goes on /data/coe9, never on / (~11 GB free).
# Safe to re-run: each step skips itself if already done.
#
# Usage (on coe9, inside tmux):   bash setup/setup_coe9.sh
set -euo pipefail

BASE=/data/coe9
CONDA_DIR=$BASE/miniforge3
ENVS_DIR=$BASE/envs
PY_VER=3.11

# ---------------------------------------------------------------------------
# 1. Directories on /data
# ---------------------------------------------------------------------------
mkdir -p "$BASE"/{envs,hf_cache,pip_cache,torch_cache,tmp,artifacts}

# ---------------------------------------------------------------------------
# 2. Env vars in ~/.bashrc (one marked block, added once)
# ---------------------------------------------------------------------------
MARK="# >>> coe9 interp env >>>"
if ! grep -qF "$MARK" ~/.bashrc; then
  cat >> ~/.bashrc <<EOF

$MARK
export HF_HOME=$BASE/hf_cache
export PIP_CACHE_DIR=$BASE/pip_cache
export TORCH_HOME=$BASE/torch_cache
export TMPDIR=$BASE/tmp
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# <<< coe9 interp env <<<
EOF
  echo "[setup] added env block to ~/.bashrc"
fi
# Use the same vars for the rest of this script.
export HF_HOME=$BASE/hf_cache PIP_CACHE_DIR=$BASE/pip_cache TORCH_HOME=$BASE/torch_cache \
       TMPDIR=$BASE/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---------------------------------------------------------------------------
# 3. Conda (Miniforge) on /data, unless conda is already available
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
else
  if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "[setup] installing Miniforge to $CONDA_DIR"
    curl -fsSL -o "$TMPDIR/miniforge.sh" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$TMPDIR/miniforge.sh" -b -p "$CONDA_DIR"
    rm "$TMPDIR/miniforge.sh"
    "$CONDA_DIR/bin/conda" init bash
  fi
  CONDA_BASE=$CONDA_DIR
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Store package cache on /data too.
conda config --add pkgs_dirs "$BASE/conda_pkgs" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. `interp` env: analysis stack
# ---------------------------------------------------------------------------
if [ ! -d "$ENVS_DIR/interp" ]; then
  conda create -y -p "$ENVS_DIR/interp" python=$PY_VER
fi
conda activate "$ENVS_DIR/interp"
pip install --upgrade pip
# Default PyPI torch wheel ships CUDA 12.x; driver 580 supports it.
pip install torch
# TransformerLens 4.x: HookedTransformer is gone; use TransformerBridge (see CLAUDE.md section 6).
pip install "transformer_lens>=4.4,<5" sae-lens nnsight transformers accelerate bitsandbytes \
            scikit-learn jupyter ipykernel einops jaxtyping plotly matplotlib pandas tqdm
python -m ipykernel install --user --name interp --display-name "interp (coe9)"
conda deactivate

# ---------------------------------------------------------------------------
# 5. `gen` env: vLLM only (it pins its own torch; never mix with interp)
# ---------------------------------------------------------------------------
if [ ! -d "$ENVS_DIR/gen" ]; then
  conda create -y -p "$ENVS_DIR/gen" python=$PY_VER
fi
conda activate "$ENVS_DIR/gen"
pip install --upgrade pip
pip install vllm
conda deactivate

echo
echo "[setup] done. Open a new shell (or: source ~/.bashrc), then run:"
echo "    conda activate $ENVS_DIR/interp && python setup/check_env.py"
