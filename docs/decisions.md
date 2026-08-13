# Decisions

Compressed from working notes in `temp/chats.txt`. The LLM never owns control.

**OpenAI-compatible adapter.** Ollama (Metal) and vLLM (CUDA) both speak `/v1/chat/completions`. [`configs/model.yaml`](../configs/model.yaml) is the only switch. No LangChain.

**Qwen2.5-7B-Instruct.** Assignment cap is ≤9B. Function-calling is usable; it is not reliable enough to drive a clinical DFA. SFT (Unsloth LoRA) improves phrasing, not safety.

**Python executes workflow tools.** Forced `tool_choice` looked attractive and failed in prod (Yes/No permission prompts, name-token loops). Identity, device IDs, and measurements are extracted with regex and executed in the controller. The model writes the sentence after the fact.

**Interceptor before the LLM.** Keyword + vital thresholds. Filler words (`level is`) and follow-up corrections (`its 80% confirmed`) must still escalate. If this layer misses, a 7B will call hypoxic SpO2 “normal.”

**DFA is a graph, not a one-way street.** Education and setup both allow `troubleshoot_step`. Pairing requires status-check first. Escalation is terminal.

**One tool NLG loop, not ReAct.** After a tool result, one un-tooled LLM turn explains the next step. Dumping raw tool JSON to the user broke UX; unbounded ReAct burns TTFT and re-hallucinates.

**SFT data must match runtime.** Tool calls belong in native `tool_calls` + `tool` messages, not JSON stuffed into `content`. Serving/training mismatch was a training blocker; the synthetic corpus was regenerated with ChatML `text` for Unsloth `dataset_text_field="text"`.

**Pin the pod stack.** `vllm==0.6.6.post1` needs `fastapi<0.113`, `starlette<0.38`, `xgrammar<0.1.15`, `ninja`, `streamlit<1.36`, and `--tool-call-parser hermes`. See [`requirements-prod.txt`](../requirements-prod.txt) and [`setup_prod.sh`](../setup_prod.sh).

**Telemetry is registry state.** The UI must not re-call `start_measurement` (rerolls random vitals) and must not key off “this turn’s `tool_call` name.” `ToolRegistry.recorded_readings` survives later escalation turns.
