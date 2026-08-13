# Docs

Operational notes for RPM-Agent. The root [README](../README.md) is the assignment-facing overview; this folder is how the system actually runs.

| Doc | Use |
|-----|-----|
| [architecture.md](architecture.md) | Control plane vs LLM; request path |
| [runbook.md](runbook.md) | Local Ollama and RunPod vLLM |
| [decisions.md](decisions.md) | Why the stack looks like this |
| [failure-modes.md](failure-modes.md) | Bugs we hit in prod and the fix |

The LLM is NLU/NLG only. Workflow, safety, and tool execution live in Python.
