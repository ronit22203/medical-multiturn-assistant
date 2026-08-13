# Failure modes

What we actually hit. Do not “fix” these by giving the 7B more prompt.

| Symptom | Cause | Fix in tree |
|---------|--------|-------------|
| `streamlit: command not found`; Ollama `:11434` on the pod | YAML `env` ignored; `RPMAgent()` hardcoded `local`; Streamlit not in prod install | Honor `env` / `RPM_ENV`; [`requirements.txt`](../requirements.txt) on `setup_prod.sh` |
| `auto tool choice requires --enable-auto-tool-choice` | vLLM disables tools by default | `make serve` passes `--enable-auto-tool-choice --tool-call-parser hermes` |
| FastAPI `Router.on_startup` TypeError | `fastapi 0.115` vs `vllm 0.6.6.post1`; Streamlit wants newer Starlette | `fastapi<0.113`, `starlette<0.38`, `streamlit<1.36` |
| xgrammar crash / connection reset mid-JSON | `xgrammar>=0.1.15` dropped `from_huggingface` | `xgrammar<0.1.15` |
| `Ninja is required to load C++ extensions` | JIT after xgrammar pin | `pip install ninja` before vLLM |
| Incomplete tool call / empty name | Hermes omits name when `tool_choice` is forced | Inject known name + `forced-{name}` id |
| DFA stuck on `1_onboarding`; “second patient”; safety-rule vomit | Model ignored schema; verbose safety text collapsed attention | Deterministic `verify_identity`; short invariant; sanitize dumps |
| `check_device_status` × 512 tokens | Forced `tool_choice` → Yes/No confirm → name-token loop | Do not force device tools; parse `PO-9821` / `OXI-1023` in Python |
| Hallucinated last name (`Shah`) | Forcing JSON when last name missing | Never guess identity fields; ask for the rest |
| SpO2 21 / “its 80% confirmed” not escalated | Regex missed `level`; no prior-turn vital memory | Interceptor filler + correction patterns |
| LLM called 80% SpO2 “normal” | Interceptor miss → model interpreted vitals | Interceptor must catch; invariant forbids interpret |
| Device fail in education, DFA did not move | Education only allowed `start_measurement` | Graph: `troubleshoot_step` from setup and education |
| Chatty setup, no `pair_device` | Alignment bias over JSON | Controller pairing; status-before-pair in DFA |
| Tool fired, no spoken next step | Brutalist dump of tool JSON, no NLG turn | Tool result → `tool` message → one un-tooled NLG call |
| Device Readings empty after a real SpO2 line | Model quoted numbers; `start_measurement` never ran; UI keyed off current `tool_call` | Controller measurement; `registry.recorded_readings`; harvest every rerun |
| Empty readings after escalation | Later turn did not re-bind telemetry | Merge registry log on every Streamlit rerun, including `escalated` |
| SFT taught the model to print JSON in `content` | Dataset ≠ OpenAI `tool_calls` | Native tool messages + ChatML `text` field |

**Rule:** if a 7B can skip a step, Python must own that step.
