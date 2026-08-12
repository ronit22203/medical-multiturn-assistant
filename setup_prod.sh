#!/usr/bin/env bash
# setup_prod.sh — Deterministic RunPod dependency bootstrap for vLLM production serving
#
# TARGET ENVIRONMENT
#   Template : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   GPU      : A100 40G / H100 80G recommended (vLLM tensor-parallel serving)
#   CUDA     : 12.4  →  wheel index https://download.pytorch.org/whl/cu124
#
# VERIFIED COMPATIBLE MATRIX
#   torch==2.5.1+cu124 | vllm==0.6.6.post1 | fastapi==0.115.5 | uvicorn==0.32.1
#
# USAGE
#   bash setup_prod.sh
#   python -m vllm.entrypoints.openai.api_server \
#       --model outputs/production_vllm_workspace \
#       --dtype bfloat16 --port 8000
set -euo pipefail

echo "================================================================"
echo "  RPM-Agent — RunPod Production (vLLM) Bootstrap"
echo "================================================================"

# ── Step 1: Upgrade pip ──────────────────────────────────────────────
echo ""
echo "[1/3] Upgrading pip..."
python3 -m pip install --upgrade pip

# ── Step 2: Pin PyTorch ──────────────────────────────────────────────
# vLLM 0.6.x is validated against torch 2.4–2.5; cu124 matches the image.
echo ""
echo "[2/3] Installing pinned PyTorch 2.5.1+cu124..."
pip install \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --upgrade

# ── Step 3: Install vLLM serving stack ──────────────────────────────
# vLLM bundles its own transformers; do not install separately.
# fastapi/uvicorn pinned to versions vLLM 0.6.x has been tested against.
echo ""
echo "[3/3] Installing vLLM serving stack..."
pip install \
    "vllm==0.6.6.post1" \
    "fastapi==0.115.5" \
    "uvicorn==0.32.1" \
    "pydantic==2.10.3" \
    "huggingface_hub==0.26.5"

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python3 - <<'EOF'
import torch, vllm, fastapi, uvicorn
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  vllm           : {vllm.__version__}")
print(f"  fastapi        : {fastapi.__version__}")
print(f"  uvicorn        : {uvicorn.__version__}")
EOF

echo ""
echo "================================================================"
echo "  Environment ready."
echo ""
echo "  Launch server:"
echo "    python -m vllm.entrypoints.openai.api_server \\"
echo "        --model outputs/production_vllm_workspace \\"
echo "        --dtype bfloat16 --port 8000"
echo "================================================================"

