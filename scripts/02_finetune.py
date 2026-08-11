"""
scripts/02_finetune.py — Unsloth SFT Forge
=========================================
Trains a LoRA adapter on Qwen2.5-7B-Instruct using the RPM synthetic corpus,
then exports two production artifacts:
  - outputs/production_vllm_workspace/  : merged 16-bit tensors for vLLM cloud serving
  - outputs/local_gguf_workspace/       : q4_k_m GGUF binary for Ollama edge serving

REQUIREMENTS
  Runs on a CUDA GPU instance (RunPod A100/H100).  Do NOT run on Apple Silicon.

INSTALL (CUDA instance only — excluded from the local uv environment)
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes
"""
# unsloth MUST be imported first — it patches torch internals at import time.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH: int = 2048
DTYPE = None          # Auto-selects bfloat16 on Ampere+ (A100/H100); fp16 otherwise
LOAD_IN_4BIT: bool = True  # Keeps 7B within 24 GB VRAM during training

MODEL_NAME: str = "unsloth/Qwen2.5-7B-Instruct"  # Unsloth-patched weights (2× throughput)
DATASET_PATH: str = "data/synthetic/sft_dataset.jsonl"

OUTPUT_CHECKPOINTS: str = "outputs/checkpoints"
OUTPUT_VLLM: str = "outputs/production_vllm_workspace"
OUTPUT_GGUF: str = "outputs/local_gguf_workspace"

# ---------------------------------------------------------------------------
print("=" * 60)
print("  UNSLOTH SFT FORGE — RPM-Agent LoRA Training Pipeline")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Load Base Model + Tokenizer
# ---------------------------------------------------------------------------
print("\n[1/5] Loading base model and tokenizer...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply Qwen's ChatML template and register <|im_start|> / <|im_end|> as stop tokens.
# Critical: if omitted, the exported GGUF will hallucinate formatting indefinitely.
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)
print(f"  Model loaded: {MODEL_NAME}")
print(f"  bfloat16 supported: {FastLanguageModel.is_bfloat16_supported()}")

# ---------------------------------------------------------------------------
# 3. LoRA Adapter Configuration
# ---------------------------------------------------------------------------
print("\n[2/5] Attaching LoRA adapter...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,      # Unsloth-optimised: no dropout required for stable SFT
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% VRAM reduction
)
print("  LoRA adapter attached (r=16, alpha=32, all projection modules)")

# ---------------------------------------------------------------------------
# 4. Dataset Loading
# ---------------------------------------------------------------------------
print("\n[3/5] Loading synthetic SFT dataset...")

dataset = load_dataset(
    "json",
    data_files={"train": DATASET_PATH},
    split="train",
)
print(f"  Loaded {len(dataset)} training examples from {DATASET_PATH}")

# The "text" field is pre-rendered ChatML by 01_generate_data.py.
# Do NOT call apply_chat_template again — that would double-wrap the formatting.

# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
print("\n[4/5] Initialising SFTTrainer...")

# Batch size 2, grad accum 4 → effective batch = 8.
# Dataset: 300 rows → ~38 steps / epoch.  max_steps=60 ≈ 1.5 epochs.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",   # Pre-rendered ChatML string; no formatter needed
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,               # Must be False: packing corrupts multi-turn attention masks
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_CHECKPOINTS,
    ),
)

print("  Starting training run...")
trainer_stats = trainer.train()
print(f"\n  Training complete — {trainer_stats.metrics['train_runtime']:.1f}s total")
print(f"  Final train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")

# ---------------------------------------------------------------------------
# 6. Artifact Export
# ---------------------------------------------------------------------------
print("\n[5/5] Exporting production artifacts...")

# --- Artifact A: vLLM 16-bit ---
print(f"\n  [A] Merging LoRA into base weights (16-bit) → {OUTPUT_VLLM}")
print("      (This may take 3–5 minutes)")
model.save_pretrained_merged(OUTPUT_VLLM, tokenizer, save_method="merged_16bit")
print(f"  [A] vLLM artifact saved to {OUTPUT_VLLM}/")

# --- Artifact B: Ollama GGUF q4_k_m ---
print(f"\n  [B] Quantising to q4_k_m GGUF → {OUTPUT_GGUF}")
print("      (Unsloth clones llama.cpp on first run — allow 5–10 minutes)")
model.save_pretrained_gguf(OUTPUT_GGUF, tokenizer, quantization_method="q4_k_m")
print(f"  [B] GGUF artifact saved to {OUTPUT_GGUF}/")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Artifact A (vLLM):  {OUTPUT_VLLM}/")
print(f"  Artifact B (Ollama): {OUTPUT_GGUF}/*.gguf")
print("\n  SCP the GGUF back to your Mac:")
print(f"    scp -r runpod:~/medical-multiturn-assistant/{OUTPUT_GGUF} ./outputs/")
print("\n  Then build the Ollama model (see README — Training section).")
