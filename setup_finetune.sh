#!/usr/bin/env bash
# setup_finetune.sh — Deterministic RunPod dependency bootstrap for Unsloth SFT
#
# TARGET ENVIRONMENT
#   Template : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   GPU      : RTX 4090 (24 GB) or A100 (40/80 GB)
#   CUDA     : 12.4  →  wheel index https://download.pytorch.org/whl/cu124
#
# VERIFIED WORKING COMBINATION (2026-08-11, Unsloth 2026.8.12)
#   torch==2.5.1+cu124 | torchao==0.18.0 | trl==0.24.0
#   transformers==5.5.0 | peft==0.20.0 | accelerate==1.14.0 | bitsandbytes==0.50.0
#
# USAGE
#   bash setup_finetune.sh          # run once after pod spin-up
#   python scripts/02_finetune.py   # then launch training
set -euo pipefail

echo "================================================================"
echo "  RPM-Agent — RunPod Finetune Bootstrap"
echo "================================================================"

# ── Step 1: Purge packages that came with the base image ─────────────
# The template ships torch==2.4.1 and a stale torchao that uses
# torch.int1 (not available until torch>=2.5). Purge first.
echo ""
echo "[1/4] Purging stale base-image packages..."
pip uninstall -y \
    torch torchvision torchaudio torchao \
    unsloth unsloth_zoo \
    trl transformers datasets \
    peft accelerate bitsandbytes \
    2>/dev/null || true   # tolerate packages that aren't installed

# ── Step 2: Pin PyTorch foundation ───────────────────────────────────
# Must come before unsloth so torchao resolves against torch 2.5.1.
# cu124 wheel matches CUDA 12.4.1 on the RunPod devel image.
echo ""
echo "[2/4] Installing pinned PyTorch 2.5.1+cu124..."
pip install \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --upgrade

# ── Step 3: Install training stack (let unsloth resolve its own deps) ─
# DO NOT use --no-deps here — unsloth_zoo requires specific trl/transformers
# versions; letting it resolve ensures a consistent, tested combination.
# torchao is installed first, pinned to the version unsloth_zoo 2026.8.x
# requires and that is compatible with torch 2.5.1.
echo ""
echo "[3/4] Installing training stack..."
pip install "torchao==0.18.0"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ── Step 4: Verify environment ────────────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
python3 - <<'EOF'
import torch, unsloth, trl, transformers, peft, accelerate, bitsandbytes
print(f"  torch          : {torch.__version__}")
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")
print(f"  unsloth        : {unsloth.__version__}")
print(f"  trl            : {trl.__version__}")
print(f"  transformers   : {transformers.__version__}")
print(f"  peft           : {peft.__version__}")
print(f"  accelerate     : {accelerate.__version__}")
print(f"  bitsandbytes   : {bitsandbytes.__version__}")
EOF

echo ""
echo "================================================================"
echo "  Environment ready. Run: python scripts/02_finetune.py"
echo "================================================================"

