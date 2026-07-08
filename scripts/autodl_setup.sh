#!/usr/bin/env bash
# One-shot environment setup for AutoDL (Linux + CUDA + conda pre-installed).
# Usage:  bash scripts/autodl_setup.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-slimkt}"
PY_VER="${PY_VER:-3.10}"

echo "[slim-kt] Keeping data/outputs inside the repo folder (on the AutoDL data disk)."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs}"
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"

# --- AutoDL network: use domestic mirrors so HuggingFace / pip are fast ---
echo "[slim-kt] Configuring HuggingFace + pip mirrors (AutoDL China network)."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
mkdir -p "$HF_HOME"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# Persist the env vars for future shells
{
  echo "export DATA_ROOT=$DATA_ROOT"
  echo "export OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "export HF_ENDPOINT=$HF_ENDPOINT"
  echo "export HF_HOME=$HF_HOME"
} >> ~/.bashrc

# --- conda env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -y -n "$ENV_NAME" "python=${PY_VER}"
fi
conda activate "$ENV_NAME"

# --- torch: install the CUDA build matching the AutoDL image FIRST ---
# AutoDL images usually ship a matching torch already. If `import torch` works
# and torch.cuda.is_available() is True, skip this block.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[slim-kt] Installing PyTorch (CUDA 12.1 wheels). Adjust cu121 to your driver if needed."
  pip install -i "$PIP_INDEX" torch --index-url https://download.pytorch.org/whl/cu121 || \
  pip install -i "$PIP_INDEX" torch
fi

# --- project deps ---
pip install -i "$PIP_INDEX" -r requirements.txt
pip install -i "$PIP_INDEX" -e .

echo "[slim-kt] Sanity check:"
python - <<'PY'
import torch
print("  torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "[slim-kt] Done. Run:  conda activate ${ENV_NAME}"
