import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict

from src.engine.interceptor import SafetyInterceptor
from src.engine.state_machine import RPMStateMachine
from src.paths import project_path
from src.tools.registry import ToolRegistry


GLOBAL_SAFETY_INVARIANT = (
    "GLOBAL SAFETY INVARIANT: Do not provide medical advice, diagnose, or "
    "characterize any vital sign as normal, safe, good, bad, high, or low. "
    "Never reassure a user about a vital sign. Vital-sign interpretation and "
    "escalation are exclusively handled by deterministic safety controls. "
    "If a vital sign reaches you, acknowledge receipt without interpreting it "
    "and continue only with the current workflow."
)
DEVICE_ISSUE_PATTERN = re.compile(
    r"\b(?:(?:cannot|can't|won't|not|isn't|stopped)\s+"
    r"(?:pair(?:ing)?|connect(?:ing)?|work(?:ing)?)|"
    r"failed|failure|error|offline|disconnected|broken|"
    r"blinking\s+red|red\s+(?:status\s+)?light)\b",
    re.IGNORECASE,
)
PUBLIC_STATE_NAMES = {
    "1_onboarding": "onboarding",
    "2_device_setup": "device_setup",
    "3_troubleshooting": "troubleshooting",
    "4_education": "education",
    "5_closing": "closing",
    "escalated": "escalated",
}


class ResponseToolCall(BaseModel):
    """Tool call exposed through the assignment response contract."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]


class StructuredAgentResponse(BaseModel):
    """Stable response envelope consumed by the evaluator and chat interface."""

    model_config = ConfigDict(extra="forbid")

    state: str
    assistant_message: str
    tool_call: ResponseToolCall | None = None


@dataclass(frozen=True)
class InferenceMetrics:
    """Aggregate measurements captured across one agent turn."""

    ttft_ms: float | None
    tpot_ms: float | None
    total_latency_ms: float
    llm_requests: int
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    output_tokens_per_second: float | None
    estimated_mbu_percent: float | None


@dataclass(frozen=True)
class AgentTurnResult:
    """Agent output and any performance measurements available for the turn."""

    response: StructuredAgentResponse
    metrics: InferenceMetrics | None = None
    metrics_note: str | None = None

    @property
    def message(self) -> str:
        """Return the natural-language assistant message."""
        return self.response.assistant_message

    def model_dump_json(self) -> str:
        """Serialize the exact assignment response envelope."""
        return self.response.model_dump_json()


class RPMAgent:
    def __init__(self, env: str = "local") -> None:
        """Initialize the configured backend and its telemetry metadata."""
        try:
            with project_path("configs/model.yaml").open(
                "r",
                encoding="utf-8",
            ) as config_file:
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
        self.messages: list[dict[str, Any]] = []

    @staticmethod
    def _turn_result(
        state: str,
        message: str,
        tool_call: ResponseToolCall | None = None,
        metrics: InferenceMetrics | None = None,
        metrics_note: str | None = None,
    ) -> AgentTurnResult:
        """Build and validate a structured response for one completed turn."""
        public_state = PUBLIC_STATE_NAMES.get(state, state)
        response = StructuredAgentResponse(
            state=public_state,
            assistant_message=message,
            tool_call=tool_call,
        )
        return AgentTurnResult(
            response=response,
            metrics=metrics,
            metrics_note=metrics_note,
        )

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
        decode_seconds: float,
        decode_intervals: int,
        llm_requests: int,
    ) -> InferenceMetrics:
        """Calculate latency, decode throughput, and estimated MBU."""
        ttft_ms = (
            (first_output_at - started_at) * 1_000
            if first_output_at is not None
            else None
        )
        tpot_ms: float | None = None
        output_tokens_per_second: float | None = None

        if decode_seconds > 0 and decode_intervals > 0:
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
            llm_requests=llm_requests,
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
        turn_state = dfa.current_state
        safety_check = interceptor.inspect(user_input)
        if safety_check.is_red_flag:
            dfa.force_escalation()
            escalation_args = {"reason": safety_check.reason}
            escalation_result = registry.execute_tool(
                "escalate_to_nurse",
                escalation_args,
            )
            return self._turn_result(
                state=dfa.current_state,
                message=(
                    "EMERGENCY PROTOCOL ENGAGED: "
                    f"{escalation_result['message']}"
                ),
                tool_call=ResponseToolCall(
                    name="escalate_to_nurse",
                    arguments=escalation_args,
                ),
                metrics_note="LLM bypassed by the deterministic safety interceptor",
            )

        if dfa.current_state == "escalated":
            return self._turn_result(
                state=dfa.current_state,
                message=(
                    "Emergency escalation remains active. The automated workflow "
                    "cannot continue until clinical support takes over."
                ),
                metrics_note="LLM bypassed because escalation is terminal",
            )

        self.messages.append({"role": "user", "content": user_input})
        controller_context: str | None = None
        response_tool_call: ResponseToolCall | None = None

        if (
            dfa.current_state in {"2_device_setup", "4_education"}
            and DEVICE_ISSUE_PATTERN.search(user_input)
        ):
            issue_result = registry.execute_tool(
                "troubleshoot_step",
                {
                    "step_id": "device_issue_reported",
                    "resolved": False,
                },
            )
            if issue_result.get("status") != "success":
                return self._turn_result(
                    state=turn_state,
                    message=(
                        "[System Error] Unable to log the reported device issue: "
                        f"{issue_result.get('message', 'unknown tool error')}"
                    ),
                    metrics_note="LLM bypassed because deterministic routing failed",
                )

            dfa.process_tool_execution("troubleshoot_step", issue_result)
            response_tool_call = ResponseToolCall(
                name="troubleshoot_step",
                arguments={
                    "step_id": "device_issue_reported",
                    "resolved": False,
                },
            )
            controller_context = (
                "DETERMINISTIC CONTROLLER EVENT: A device failure was detected "
                "and logged before this request. The workflow is now in "
                "troubleshooting. For this response only, do not call a tool. "
                "Give one concrete troubleshooting action and ask the user to "
                "report whether it resolved the issue."
            )

        turn_started_at = perf_counter()
        first_turn_output_at: float | None = None
        aggregate_prompt_tokens = 0
        aggregate_output_tokens = 0
        aggregate_total_tokens = 0
        usage_complete = True
        aggregate_decode_seconds = 0.0
        aggregate_decode_intervals = 0
        llm_requests = 0
        metrics: InferenceMetrics | None = None

        for _ in range(3):
            system_prompt, allowed_tools = dfa.get_context()
            if controller_context is not None:
                allowed_tools = []
            tool_schemas = registry.get_tool_schemas(allowed_tools)
            full_system_prompt = (
                f"{system_prompt.rstrip()}\n\n{GLOBAL_SAFETY_INVARIANT}"
            )
            if controller_context is not None:
                full_system_prompt = (
                    f"{full_system_prompt}\n\n{controller_context}"
                )
            messages_for_llm = [
                {"role": "system", "content": full_system_prompt},
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
                api_kwargs["parallel_tool_calls"] = False

            request_first_output_at: float | None = None
            content_parts: list[str] = []
            streamed_tool_calls: dict[int, dict[str, str]] = {}
            request_prompt_tokens: int | None = None
            request_output_tokens: int | None = None
            request_total_tokens: int | None = None
            llm_requests += 1

            try:
                stream = self.client.chat.completions.create(**api_kwargs)
                for chunk in stream:
                    received_at = perf_counter()
                    if chunk.usage is not None:
                        request_prompt_tokens = chunk.usage.prompt_tokens
                        request_output_tokens = chunk.usage.completion_tokens
                        request_total_tokens = chunk.usage.total_tokens

                    for choice in chunk.choices:
                        delta = choice.delta
                        if delta.content or delta.tool_calls:
                            if request_first_output_at is None:
                                request_first_output_at = received_at
                            if first_turn_output_at is None:
                                first_turn_output_at = received_at

                        if delta.content:
                            content_parts.append(delta.content)

                        for tool_delta in delta.tool_calls or []:
                            tool_call = streamed_tool_calls.setdefault(
                                tool_delta.index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            tool_call["id"] += tool_delta.id or ""
                            if tool_delta.function is not None:
                                tool_call["name"] += (
                                    tool_delta.function.name or ""
                                )
                                tool_call["arguments"] += (
                                    tool_delta.function.arguments or ""
                                )
            except OpenAIError as exc:
                return self._turn_result(
                    state=turn_state,
                    message=f"[System Error] LLM backend request failed: {exc}",
                    tool_call=response_tool_call,
                    metrics=metrics,
                    metrics_note=(
                        "Request failed before complete metrics were available"
                    ),
                )

            request_completed_at = perf_counter()
            if (
                request_prompt_tokens is None
                or request_output_tokens is None
                or request_total_tokens is None
            ):
                usage_complete = False
            else:
                aggregate_prompt_tokens += request_prompt_tokens
                aggregate_output_tokens += request_output_tokens
                aggregate_total_tokens += request_total_tokens

                if (
                    request_first_output_at is not None
                    and request_output_tokens > 1
                ):
                    aggregate_decode_seconds += (
                        request_completed_at - request_first_output_at
                    )
                    aggregate_decode_intervals += request_output_tokens - 1

            metrics = self._build_metrics(
                started_at=turn_started_at,
                first_output_at=first_turn_output_at,
                completed_at=request_completed_at,
                prompt_tokens=(
                    aggregate_prompt_tokens if usage_complete else None
                ),
                output_tokens=(
                    aggregate_output_tokens if usage_complete else None
                ),
                total_tokens=aggregate_total_tokens if usage_complete else None,
                decode_seconds=aggregate_decode_seconds,
                decode_intervals=aggregate_decode_intervals,
                llm_requests=llm_requests,
            )

            if streamed_tool_calls:
                if len(streamed_tool_calls) > 1:
                    return self._turn_result(
                        state=turn_state,
                        message=(
                            "[System Error] LLM generated multiple tool calls; "
                            "only one tool call is permitted per turn."
                        ),
                        metrics=metrics,
                    )

                parsed_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
                assistant_tool_calls: list[dict[str, Any]] = []

                for tool_call in streamed_tool_calls.values():
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    if not tool_call_id or not tool_name:
                        return self._turn_result(
                            state=turn_state,
                            message=(
                                "[System Error] LLM generated an incomplete "
                                "tool call."
                            ),
                            metrics=metrics,
                        )

                    try:
                        raw_tool_args = json.loads(tool_call["arguments"])
                    except json.JSONDecodeError:
                        return self._turn_result(
                            state=turn_state,
                            message=(
                                "[System Error] LLM generated malformed JSON "
                                f"for tool '{tool_name}'."
                            ),
                            metrics=metrics,
                        )
                    if not isinstance(raw_tool_args, dict):
                        return self._turn_result(
                            state=turn_state,
                            message=(
                                "[System Error] LLM generated non-object "
                                f"arguments for tool '{tool_name}'."
                            ),
                            metrics=metrics,
                        )

                    parsed_tool_calls.append(
                        (tool_call_id, tool_name, raw_tool_args)
                    )
                    response_tool_call = ResponseToolCall(
                        name=tool_name,
                        arguments=raw_tool_args,
                    )
                    assistant_tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call["arguments"],
                            },
                        }
                    )

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(content_parts) or None,
                        "tool_calls": assistant_tool_calls,
                    }
                )

                for tool_call_id, tool_name, tool_args in parsed_tool_calls:
                    workflow_error = dfa.validate_tool_call(tool_name, tool_args)
                    if workflow_error is not None:
                        execution_result = {
                            "status": "error",
                            "message": workflow_error,
                        }
                    else:
                        execution_result = registry.execute_tool(
                            tool_name,
                            tool_args,
                        )
                    dfa.process_tool_execution(tool_name, execution_result)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(execution_result),
                        }
                    )

                controller_context = (
                    "DETERMINISTIC CONTROLLER EVENT: The requested tool calls "
                    "have already executed and the workflow state has been "
                    "updated. For this response only, do not call another tool. "
                    "Explain the result and the current next step, then wait for "
                    "a new user response before taking another action."
                )
                continue

            response_content = "".join(content_parts)
            if response_content:
                self.messages.append(
                    {"role": "assistant", "content": response_content}
                )
                return self._turn_result(
                    state=turn_state,
                    message=response_content,
                    tool_call=response_tool_call,
                    metrics=metrics,
                )

            return self._turn_result(
                state=turn_state,
                message="[System] No valid response generated by the LLM.",
                tool_call=response_tool_call,
                metrics=metrics,
            )

        return self._turn_result(
            state=turn_state,
            message="[System] LLM tool iteration limit reached.",
            tool_call=response_tool_call,
            metrics=metrics,
        )
