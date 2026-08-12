#!/usr/bin/env bash
# setup_prod.sh — Deterministic RunPod dependency bootstrap for vLLM production serving
#
# TARGET ENVIRONMENT
#   Template : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   GPU      : A100 40G / H100 80G recommended (vLLM tensor-parallel serving)
#   CUDA     : 12.4  →  wheel index https://download.pytorch.org/whl/cu124
#
# VERIFIED COMPATIBLE MATRIX
#   torch==2.5.1+cu124 | vllm==0.6.6.post1
#   fastapi<0.113 | starlette<0.38 | streamlit<1.36 | xgrammar<0.1.15
#
# USAGE
#   bash setup_prod.sh
#   python -m vllm.entrypoints.openai.api_server \
#       --model outputs/production_vllm_workspace \
#       --dtype bfloat16 --port 8000 --host 0.0.0.0 \
#       --enable-auto-tool-choice --tool-call-parser hermes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "  RPM-Agent — RunPod Production (vLLM) Bootstrap"
echo "================================================================"

# ── Step 1: Upgrade pip ──────────────────────────────────────────────
echo ""
echo "[1/4] Upgrading pip..."
python3 -m pip install --upgrade pip

# ── Step 2: Pin PyTorch ──────────────────────────────────────────────
# vLLM 0.6.x is validated against torch 2.4–2.5; cu124 matches the image.
echo ""
echo "[2/4] Installing pinned PyTorch 2.5.1+cu124..."
pip install \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --upgrade

# ── Step 3: Install app UI + vLLM serving stack ──────────────────────
# ninja must be present before vLLM/xgrammar JIT-compile CUDA kernels.
# xgrammar is capped below 0.1.15 (from_huggingface binding removed).
echo ""
echo "[3/4] Installing application and vLLM serving dependencies..."
pip install -r "${ROOT}/requirements.txt" -r "${ROOT}/requirements-prod.txt"

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
python3 - <<'EOF'
from importlib.metadata import version

import fastapi
import ninja  # noqa: F401  — required for PyTorch JIT CUDA extensions
import starlette
import streamlit
import torch
import uvicorn
import vllm
import xgrammar  # noqa: F401

print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  vllm           : {vllm.__version__}")
print(f"  fastapi        : {fastapi.__version__}")
print(f"  starlette      : {starlette.__version__}")
print(f"  uvicorn        : {uvicorn.__version__}")
print(f"  streamlit      : {streamlit.__version__}")
print(f"  xgrammar       : {version('xgrammar')}")
print(f"  ninja          : {version('ninja')}")
EOF

echo ""
echo "================================================================"
echo "  Environment ready."
echo ""
echo "  Launch server:"
echo "    python -m vllm.entrypoints.openai.api_server \\"
echo "        --model outputs/production_vllm_workspace \\"
echo "        --dtype bfloat16 --port 8000 --host 0.0.0.0 \\"
echo "        --enable-auto-tool-choice --tool-call-parser hermes"
echo "================================================================"
