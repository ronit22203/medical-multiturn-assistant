import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml
from openai import OpenAI, OpenAIError

from src.engine.interceptor import SafetyInterceptor
from src.engine.state_machine import RPMStateMachine
from src.tools.registry import ToolRegistry


@dataclass(frozen=True)
class InferenceMetrics:
    """Measurements captured from one streamed LLM response."""

    ttft_ms: float | None
    tpot_ms: float | None
    total_latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    output_tokens_per_second: float | None
    estimated_mbu_percent: float | None


@dataclass(frozen=True)
class AgentTurnResult:
    """Agent output and any performance measurements available for the turn."""

    message: str
    metrics: InferenceMetrics | None = None
    metrics_note: str | None = None


class RPMAgent:
    def __init__(self, env: str = "local") -> None:
        """Initialize the configured backend and its telemetry metadata."""
        try:
            with open("configs/model.yaml", "r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            endpoint = config["endpoints"][env]
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise RuntimeError(f"Unable to load model configuration: {exc}") from exc

        self.backend: str = endpoint["backend"]
        self.client = OpenAI(
            base_url=endpoint["api_base"],
            api_key="sk-local-dev",
        )
        self.model_id: str = endpoint["model_id"]
        self.temperature: float = endpoint["temperature"]
        self.max_tokens: int = endpoint["max_tokens"]

        bandwidth = endpoint.get("memory_bandwidth_gbps")
        if bandwidth is not None and (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, (int, float))
            or bandwidth <= 0
        ):
            raise ValueError("memory_bandwidth_gbps must be a positive number")
        self.memory_bandwidth_gbps = (
            float(bandwidth) if bandwidth is not None else None
        )
        self.model_size_bytes = (
            self._get_ollama_model_size(endpoint["api_base"], self.model_id)
            if self.backend == "ollama" and self.memory_bandwidth_gbps is not None
            else None
        )
        self.messages: list[dict[str, str]] = []

    @staticmethod
    def _get_ollama_model_size(api_base: str, model_id: str) -> int:
        """Return installed model bytes for estimated bandwidth utilization."""
        ollama_root = api_base.rstrip("/").removesuffix("/v1")
        tags_url = f"{ollama_root}/api/tags"

        try:
            with urlopen(tags_url, timeout=5.0) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to read Ollama model metadata from {tags_url}: {exc}"
            ) from exc

        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError(f"Ollama returned invalid metadata from {tags_url}")

        for model in models:
            if not isinstance(model, dict):
                continue
            if model.get("name") == model_id or model.get("model") == model_id:
                size = model.get("size")
                if isinstance(size, int) and size > 0:
                    return size
                raise RuntimeError(f"Ollama reported an invalid size for '{model_id}'")

        installed = [
            model["name"]
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        ]
        raise RuntimeError(
            f"Ollama model '{model_id}' is not installed. Installed models: "
            f"{', '.join(installed) or 'none'}"
        )

    def _build_metrics(
        self,
        started_at: float,
        first_output_at: float | None,
        completed_at: float,
        prompt_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> InferenceMetrics:
        """Calculate latency, decode throughput, and estimated MBU."""
        ttft_ms = (
            (first_output_at - started_at) * 1_000
            if first_output_at is not None
            else None
        )
        tpot_ms: float | None = None
        output_tokens_per_second: float | None = None

        if (
            first_output_at is not None
            and output_tokens is not None
            and output_tokens > 1
        ):
            decode_seconds = completed_at - first_output_at
            if decode_seconds > 0:
                decode_intervals = output_tokens - 1
                tpot_ms = decode_seconds * 1_000 / decode_intervals
                output_tokens_per_second = decode_intervals / decode_seconds

        estimated_mbu_percent = None
        if (
            output_tokens_per_second is not None
            and self.model_size_bytes is not None
            and self.memory_bandwidth_gbps is not None
        ):
            peak_bytes_per_second = self.memory_bandwidth_gbps * 1_000_000_000
            estimated_mbu_percent = (
                self.model_size_bytes
                * output_tokens_per_second
                / peak_bytes_per_second
                * 100
            )

        return InferenceMetrics(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=(completed_at - started_at) * 1_000,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_tokens_per_second=output_tokens_per_second,
            estimated_mbu_percent=estimated_mbu_percent,
        )

    def process_turn(
        self,
        user_input: str,
        dfa: RPMStateMachine,
        registry: ToolRegistry,
        interceptor: SafetyInterceptor,
    ) -> AgentTurnResult:
        """Process one user turn and return its response with telemetry."""
        safety_check = interceptor.inspect(user_input)
        if safety_check.is_red_flag:
            dfa.force_escalation()
            escalation_result = registry.execute_tool(
                "escalate_to_nurse",
                {"reason": safety_check.reason},
            )
            return AgentTurnResult(
                message=f"EMERGENCY PROTOCOL ENGAGED: {escalation_result['message']}",
                metrics_note="LLM bypassed by the deterministic safety interceptor",
            )

        self.messages.append({"role": "user", "content": user_input})
        system_prompt, allowed_tools = dfa.get_context()
        tool_schemas = registry.get_tool_schemas(allowed_tools)
        messages_for_llm = [
            {"role": "system", "content": system_prompt},
            *self.messages,
        ]

        api_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages_for_llm,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tool_schemas:
            api_kwargs["tools"] = tool_schemas
            api_kwargs["tool_choice"] = "auto"

        started_at = perf_counter()
        first_output_at: float | None = None
        content_parts: list[str] = []
        streamed_tool_calls: dict[int, dict[str, str]] = {}
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None

        try:
            stream = self.client.chat.completions.create(**api_kwargs)
            for chunk in stream:
                received_at = perf_counter()
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                for choice in chunk.choices:
                    delta = choice.delta
                    if (delta.content or delta.tool_calls) and first_output_at is None:
                        first_output_at = received_at
                    if delta.content:
                        content_parts.append(delta.content)

                    for tool_delta in delta.tool_calls or []:
                        tool_call = streamed_tool_calls.setdefault(
                            tool_delta.index,
                            {"name": "", "arguments": ""},
                        )
                        if tool_delta.function is not None:
                            tool_call["name"] += tool_delta.function.name or ""
                            tool_call["arguments"] += (
                                tool_delta.function.arguments or ""
                            )
        except OpenAIError as exc:
            return AgentTurnResult(
                message=f"[System Error] LLM backend request failed: {exc}",
                metrics_note="Request failed before complete metrics were available",
            )

        completed_at = perf_counter()
        metrics = self._build_metrics(
            started_at=started_at,
            first_output_at=first_output_at,
            completed_at=completed_at,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        if streamed_tool_calls:
            for tool_call in streamed_tool_calls.values():
                tool_name = tool_call["name"]
                try:
                    tool_args = json.loads(tool_call["arguments"])
                except json.JSONDecodeError:
                    return AgentTurnResult(
                        message=(
                            "[System Error] LLM generated malformed JSON "
                            f"for tool '{tool_name}'."
                        ),
                        metrics=metrics,
                    )

                execution_result = registry.execute_tool(tool_name, tool_args)
                transition_msg = dfa.process_tool_execution(
                    tool_name,
                    execution_result,
                )
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Executed {tool_name}: {execution_result['message']}"
                        ),
                    }
                )
                return AgentTurnResult(
                    message=(
                        f"Tool Executed: {tool_name}\n"
                        f"Result: {execution_result['message']}\n"
                        f"{transition_msg}"
                    ),
                    metrics=metrics,
                )

        response_content = "".join(content_parts)
        if response_content:
            self.messages.append(
                {"role": "assistant", "content": response_content}
            )
            return AgentTurnResult(message=response_content, metrics=metrics)

        return AgentTurnResult(
            message="[System] No valid response generated by the LLM.",
            metrics=metrics,
        )
