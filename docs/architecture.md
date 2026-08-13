# Architecture

```text
User turn
  -> SafetyInterceptor          # keywords + SpO2/BP thresholds; bypasses LLM
  -> RPMStateMachine (DFA)      # allowed tools + transitions
  -> Controller extractors      # identity, device ID, measurement type
  -> ToolRegistry.execute_tool  # Pydantic validate; record readings
  -> LLM (optional)             # NLG only when tools are already done
  -> Streamlit                  # chat + telemetry from registry.recorded_readings
```

## Layers

**Interceptor** ([`src/engine/interceptor.py`](../src/engine/interceptor.py)) runs first. Red-flag keywords (`chest pain`, `dizzy`, …) or SpO2 `< 90` / BP `≥ 180/110` force `escalate_to_nurse` and freeze the DFA on `escalated`. SpO2 corrections without the word “SpO2” (`its 80% confirmed`) use previous-turn vital context.

**DFA** ([`configs/state_graph.yaml`](../configs/state_graph.yaml)): `1_onboarding` → `2_device_setup` ⇄ `3_troubleshooting` → `4_education` ⇄ troubleshooting → `5_closing`. `pair_device` requires a prior `check_device_status` for the same `device_id`. Device failures in setup/education call `troubleshoot_step` and walk the graph backward.

**Controller** ([`src/agent/inference.py`](../src/agent/inference.py)) parses identity, hardware IDs (`PO-9821`, `OXI-1023`), and measurement intent in Python, then executes tools. The 7B model is not trusted to emit JSON for those steps. After a controller tool, a second LLM call (no tools) writes the user-facing sentence.

**LLM adapter** is OpenAI-compatible. [`configs/model.yaml`](../configs/model.yaml) `env` (or `RPM_ENV`) selects Ollama `:11434` or vLLM `:8000`. Same `openai` client either way.

**UI** ([`src/app.py`](../src/app.py)) is not the source of truth for vitals. `ToolRegistry.recorded_readings` is; the pane merges that log, tool-role payloads, and quoted SpO2/HR in chat.

## Tools

| Tool | When |
|------|------|
| `verify_identity(first_name, last_name, dob)` | Name + numeric DOB present |
| `check_device_status(device_id)` then `pair_device(device_id)` | Hardware ID + pair/setup intent |
| `start_measurement(device_id, measurement_type)` | Education + ready/vitals language |
| `troubleshoot_step(step_id, resolved)` | Connection/pairing failure language |
| `escalate_to_nurse(reason)` | Interceptor hit |
