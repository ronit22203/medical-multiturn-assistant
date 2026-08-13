# Runbook

## Local (Apple M4 / Ollama)

```bash
make setup
ollama serve          # separate terminal
ollama run qwen2.5:7b
# configs/model.yaml → env: local
make run              # Streamlit
```

Caption must show `local / OLLAMA / …`. CLI: `python main.py` (same YAML `env`).

## RunPod (vLLM + Streamlit)

Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Expose ports **8000** and **8501**.

```bash
cd medical-multiturn-assistant
git pull
# configs/model.yaml → env: prod
# endpoints.prod.model_id = Hub id or outputs/production_vllm_workspace
bash setup_prod.sh    # once per pod; pins fastapi/xgrammar/streamlit/ninja
```

Session 1 — engine (required flags):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model outputs/production_vllm_workspace \
  --dtype bfloat16 --port 8000 --host 0.0.0.0 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Or `--model roni2231/medical-rpm-qwen2.5`. Do not use `make run` on the pod (`uv` + Ollama).

Session 2 — UI:

```bash
PYTHONPATH=. streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501
```

Caption must show `prod / VLLM / …`. Refresh the browser after code pulls; Streamlit session state does not survive a full reload.

## Train (optional)

```bash
bash setup_finetune.sh
python scripts/02_finetune.py
```

Exports merged 16-bit HF → `outputs/production_vllm_workspace/` and GGUF → `outputs/local_gguf_workspace/`. Qwen2.5 Ollama `Modelfile` must set ChatML stop tokens (`<|im_start|>`, `<|im_end|>`).

## Smoke

New UI session:

1. `hi` → stay `1_onboarding`
2. `my name is Emily Davis. My DOB is 1959-12-01` → `verify_identity`, `2_device_setup`
3. `I'm ready to pair. Device ID is PO-9821` → check + pair → `4_education`
4. `I am ready to take my oxygen reading` → SpO2 in Device Readings → `5_closing`
5. New session: `I'm having chest pain right now` → `escalated`, no medical advice

Use numeric dates. `December 1st, 1959` will not advance onboarding.
