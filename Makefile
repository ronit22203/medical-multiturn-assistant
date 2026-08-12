# =============================================================================
# RPM-Agent — Makefile
# =============================================================================
#
# LOCAL (Apple M4 / any non-CUDA machine)
#   make setup          — install local dev dependencies via uv
#   make run            — launch Streamlit UI (requires Ollama running)
#   make generate       — generate synthetic SFT dataset (300 rows)
#   make evaluate       — run offline evaluation suite
#   make test           — run pytest test suite
#
# RUNPOD (CUDA GPU instance — run these INSIDE the pod over SSH)
#   make setup-finetune — bootstrap Unsloth SFT environment (run once)
#   make setup-prod     — bootstrap vLLM serving environment (run once)
#   make train          — execute LoRA fine-tuning pipeline
#   make serve          — launch vLLM OpenAI-compatible server on :8000
#
# =============================================================================

.DEFAULT_GOAL := help
SHELL         := /usr/bin/env bash

# ── Local paths ──────────────────────────────────────────────────────────────
PYTHON        := python3
UV            := uv
STREAMLIT_APP := src/app.py
VLLM_MODEL    := outputs/production_vllm_workspace

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m

# =============================================================================
# LOCAL TARGETS
# =============================================================================

.PHONY: setup
setup: ## Install local dev dependencies (uv, no CUDA)
	@echo -e "$(BOLD)[setup] Installing local dependencies via uv...$(RESET)"
	$(UV) sync
	@echo -e "$(BOLD)[setup] Done. Activate with: source .venv/bin/activate$(RESET)"

.PHONY: run
run: ## Launch the Streamlit UI (requires Ollama running locally)
	@echo -e "$(BOLD)[run] Starting RPM Control Center...$(RESET)"
	$(UV) run streamlit run $(STREAMLIT_APP)

.PHONY: generate
generate: ## Generate 300-row synthetic SFT dataset
	@echo -e "$(BOLD)[generate] Building synthetic dataset...$(RESET)"
	$(UV) run python scripts/01_generate_data.py
	@echo -e "$(BOLD)[generate] Dataset written to data/synthetic/$(RESET)"

.PHONY: evaluate
evaluate: ## Run offline evaluation suite
	@echo -e "$(BOLD)[evaluate] Running evaluation...$(RESET)"
	$(UV) run python scripts/03_evaluate.py

.PHONY: test
test: ## Run pytest test suite
	@echo -e "$(BOLD)[test] Running tests...$(RESET)"
	$(UV) run pytest tests/ -v

# =============================================================================
# RUNPOD TARGETS  (execute inside RunPod SSH session)
# =============================================================================

.PHONY: setup-finetune
setup-finetune: ## [RunPod] Bootstrap Unsloth SFT environment (run once after pod spin-up)
	@echo -e "$(BOLD)[setup-finetune] Bootstrapping finetune environment...$(RESET)"
	@bash setup_finetune.sh

.PHONY: setup-prod
setup-prod: ## [RunPod] Bootstrap vLLM production serving environment (run once)
	@echo -e "$(BOLD)[setup-prod] Bootstrapping production vLLM environment...$(RESET)"
	@bash setup_prod.sh

.PHONY: train
train: ## [RunPod] Run LoRA fine-tuning + dual artifact export
	@echo -e "$(BOLD)[train] Starting Unsloth SFT forge...$(RESET)"
	python scripts/02_finetune.py

.PHONY: serve
serve: ## [RunPod] Launch vLLM server on port 8000 (requires setup-prod + trained model)
	@echo -e "$(BOLD)[serve] Launching vLLM server on :8000...$(RESET)"
	python -m vllm.entrypoints.openai.api_server \
		--model $(VLLM_MODEL) \
		--dtype bfloat16 \
		--port 8000 \
		--host 0.0.0.0 \
		--enable-auto-tool-choice \
		--tool-call-parser hermes

# =============================================================================
# UTILITY
# =============================================================================

.PHONY: clean
clean: ## Remove Python cache files and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo -e "$(BOLD)RPM-Agent$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
