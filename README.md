# RPM-Agent: Neuro-Symbolic Remote Patient Monitoring

> **Status:** Active Development (72-Hour Sprint)
> **Hardware Target:** RunPod (NVIDIA RTX 4090/A100) / Apple Silicon (M4 Unified Memory)

Most healthcare LLM (Large Language Model) wrappers fail because they trust a probabilistic model to handle deterministic workflows. They hallucinate states, skip vital steps, and casually dispense unauthorized medical advice.

This repository solves that. **RPM-Agent** is a multi-turn chat assistant for Remote Patient Monitoring (RPM). It enforces strict safety and state progression by wrapping a lightweight open-source LLM (≤9B parameters) inside a Deterministic Finite Automaton (DFA) state machine and an out-of-band safety interceptor.

The LLM is treated as a natural language NLU/NLG (Natural Language Understanding / Generation) engine: not the system controller.

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
