#!/bin/bash
set -e

echo "--- INITIATING RUNPOD DEPENDENCY LOCKDOWN ---"

# 1. Purge the baseline environment to prevent hidden conflicts
echo "[1/4] Purging default environment variables..."
pip uninstall -y unsloth trl torchao transformers datasets torch peft accelerate bitsandbytes

# 2. Lock the PyTorch foundation (Pinning to 2.5.1 to avoid torch.int1)
echo "[2/4] Installing locked PyTorch foundation..."
pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# 3. Pin the specific problem packages that caused the death spiral
echo "[3/4] Pinning strict library matrix..."
pip install "torchao>=0.13.0,<0.18.0"
pip install "trl==0.9.6" 
pip install "transformers==4.43.3"
pip install "datasets==2.19.0"
pip install "peft==0.11.1" "accelerate==0.33.0" "bitsandbytes==0.43.2"

# 4. Install Unsloth with --no-deps so it cannot override our pinned matrix
echo "[4/4] Installing Unsloth..."
pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "xformers==0.0.28.post3"

echo "--- ENVIRONMENT LOCKED AND READY ---"
