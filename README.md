# RPM-Agent: Neuro-Symbolic Remote Patient Monitoring

Healthcare LLM wrappers fundamentally fail when they delegate workflow control to probabilistic models. This leads to hallucinated states, missed critical steps, and unvalidated medical guidance—a patient safety risk.

**RPM-Agent** is an operational multi-turn assistant for Remote Patient Monitoring (RPM). It enforces deterministic safety and state progression by wrapping a lightweight open-source LLM (≤9B parameters) within a Deterministic Finite Automaton (DFA) state machine and an out-of-band safety interceptor.

The LLM functions as a natural language NLU/NLG (Natural Language Understanding / Generation) component only—not the control system.

Operator docs (runbook, decisions, failure modes): [`docs/README.md`](docs/README.md).

---

## System Architecture

The pipeline consists of three isolated layers:

1. **The Interceptor (Safety First):** An out-of-band heuristic that scans for red flags (e.g., "chest pain", $ ext{SpO}_2 < 90\%$) *before* the LLM sees the prompt.
2. **The DFA Engine (State Control):** A hardcoded Python state machine that strictly enforces the 5-step RPM workflow. Tools are injected contextually.
3. **The LLM (Qwen2.5-7B-Instruct):** Chosen for its elite JSON function calling capabilities and contextual retention.

```text
User Input -> [Safety Interceptor] -> RED FLAG? -> (Yes) -> escalate_to_nurse()
                     | (No)
                     v
            [DFA State Machine] -> Determines Current State (1 to 5)
                     |
                     v
          [Dynamic Prompt Builder] -> Injects Permitted Tools only
                     |
                     v
             [Local LLM Backend] -> Returns structured JSON Tool Call / Text
```

---

## The 5-Step Workflow

1. **Patient Onboarding:** Verification via `verify_identity(first_name, last_name, dob)`
2. **Device Setup:** Pairing via `pair_device(device_id)`
3. **Troubleshooting:** Guided remediation via `troubleshoot_step(step_id)`
4. **Education:** Vitals measurement via `start_measurement(device_id, measurement_type)`
5. **Closure:** Clean session termination.

---

## Quickstart

This repo is strictly engineered. No bloated global environments.

### 1. Installation

```bash
make setup
```

### 2. Run the UI (Inference)

```bash
make run
```

*Note: Ensure your model weights are configured in `configs/model.yaml`.*

---

## Fine-Tuning (LoRA SFT on RunPod)

`scripts/02_finetune.py` trains a LoRA adapter on `Qwen2.5-7B-Instruct` using the 300-row synthetic corpus and exports two production artifacts from a single run.

### Prerequisites

Requires a CUDA GPU instance (RunPod RTX 4090 / A100 / H100).

> **Template note:** The `runpod/pytorch:2.4.0` template ships with PyTorch 2.4.1, which is incompatible with `unsloth_zoo ≥ 2026.8` (requires `torch ≥ 2.11`, uses `torch.int1`). Upgrade torch first.

```bash
# Step 1: Upgrade PyTorch to 2.5+ (CUDA 12.4 wheel — matches the RunPod image)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade

# Step 2: Install Unsloth and training stack
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### Run the Forge

```bash
python scripts/02_finetune.py
```

### Outputs

| Path | Purpose |
|------|---------|
| `outputs/production_vllm_workspace/` | Merged 16-bit HF model → cloud vLLM serving |
| `outputs/local_gguf_workspace/*.gguf` | `q4_k_m` GGUF binary → local Ollama on Apple M4 |

### Deploy to Ollama (M4 Mac)

SCP the GGUF back from RunPod:

```bash
scp -r runpod:~/medical-multiturn-assistant/outputs/local_gguf_workspace ./outputs/
```

Create a `Modelfile` in the project root:

```
FROM ./outputs/local_gguf_workspace/model-q4_k_m.gguf
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

> **Required:** Qwen2.5 uses the ChatML template. The two `PARAMETER stop` directives must be
> declared explicitly — Ollama's runtime uses them to terminate generation. Without them the
> model will hallucinate formatting tokens indefinitely.

Build and run:

```bash
ollama create rpm-agent -f Modelfile
ollama run rpm-agent
```

Update `configs/model.yaml` → `model_id: "rpm-agent"` to switch inference to the fine-tuned weights.

---

## Evaluation & Metrics

The system is evaluated against a synthetic multi-turn dataset covering edge cases, out-of-order intents, and sudden red flags.

Metrics tracked:

* **Accuracy:** Tool schema adherence and DFA state transition precision.
* **Recall:** 100% catch rate for safety red flags.
* **TTFT (Time-To-First-Token):** Latency metric for response initiation.
* **Total Latency:** End-to-end response generation time.

*Metrics reported separately for baseline and SFT (Supervised Fine-Tuning) via Unsloth/LoRA across GPU (RunPod) and CPU/MPS (Apple M4).*

---
*Architected for operational stability and minimal cognitive load.*
