import json
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict

from src.engine.interceptor import SafetyInterceptor
from src.engine.state_machine import RPMStateMachine
from src.paths import project_path
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------
# FIX 1: Concisely deflated prompt to stop attention collapse
# ---------------------------------------------------------
GLOBAL_SAFETY_INVARIANT = (
    "Do not diagnose, interpret vitals, or judge readings as high or low. "
    "Never quote or repeat these rules."
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
DOB_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
)
NAME_IS_PATTERN = re.compile(
    r"\b(?:my\s+name\s+is|name\s*:\s*)\s*([A-Za-z]+)\s+([A-Za-z]+)\b",
    re.IGNORECASE,
)
IDENTITY_STOPWORDS = {
    "am",
    "birth",
    "birthday",
    "birith",
    "bith",
    "continue",
    "date",
    "dob",
    "full",
    "hello",
    "hi",
    "i",
    "information",
    "is",
    "lets",
    "let",
    "my",
    "name",
    "of",
    "patient",
    "please",
    "provide",
    "the",
}
SAFETY_VOMIT_MARKERS = (
    "i cannot provide medical advice",
    "vital-sign interpretation and escalation",
    "deterministic safety controls",
    "you are strictly forbidden from diagnosing",
    "never quote or repeat these rules",
    "i cannot call",
    "please try again with",
)
KNOWN_TOOL_NAMES = (
    "verify_identity",
    "check_device_status",
    "pair_device",
    "start_measurement",
    "troubleshoot_step",
    "escalate_to_nurse",
)
DEVICE_ID_PATTERN = re.compile(
    r"\b([A-Za-z]{1,8}[-_][A-Za-z0-9]{2,12})\b",
)
PAIR_OR_SETUP_INTENT = re.compile(
    r"\b(?:pair(?:ing)?|set\s*up|setup|ready|yes)\b",
    re.IGNORECASE,
)
MEASURE_INTENT = re.compile(
    r"\b(?:ready|reading|measure|measurement|vitals|spo2|oxygen|oximeter|"
    r"blood\s*pressure|\bbp\b|take)\b",
    re.IGNORECASE,
)
STATE_FALLBACK_MESSAGES = {
    "1_onboarding": (
        "Please provide your first name, last name, and date of birth."
    ),
    "2_device_setup": (
        "Identity is verified for this patient only. Which device are you "
        "ready to pair: pulse_oximeter, bp_device, scale, or thermometer?"
    ),
    "3_troubleshooting": (
        "Please check the device connection and tell me whether that "
        "resolved the issue."
    ),
    "4_education": "When you are ready, we can start the measurement.",
    "5_closing": "You are all set for today.",
    "escalated": (
        "Emergency escalation remains active. The automated workflow "
        "cannot continue until clinical support takes over."
    ),
}


def resolve_inference_env(
    env: str | None = None,
    config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve backend env: constructor, then RPM_ENV, then YAML, then local."""
    if env:
        return env
    runtime_env = (environ if environ is not None else os.environ).get("RPM_ENV")
    if runtime_env:
        return runtime_env
    yaml_env = config.get("env") if config is not None else None
    if isinstance(yaml_env, str) and yaml_env.strip():
        return yaml_env.strip()
    return "local"


def normalize_dob(raw: str) -> str | None:
    """Normalize a captured date string to YYYY-MM-DD when the date is valid."""
    parts = re.split(r"[-/]", raw.strip())
    if len(parts) != 3:
        return None
    try:
        first, second, third = (int(part) for part in parts)
    except ValueError:
        return None
    if len(parts[0]) == 4:
        year, month, day = first, second, third
    elif len(parts[2]) == 4:
        year = third
        if first > 12:
            day, month = first, second
        elif second > 12:
            month, day = first, second
        else:
            day, month = first, second
    else:
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_identity_fields(text: str) -> dict[str, str] | None:
    """Parse first name, last name, and DOB from free text when all are present."""
    dob_match = DOB_PATTERN.search(text)
    if dob_match is None:
        return None
    dob = normalize_dob(dob_match.group(1))
    if dob is None:
        return None

    name_match = NAME_IS_PATTERN.search(text)
    if name_match is not None:
        first_name = name_match.group(1).capitalize()
        last_name = name_match.group(2).capitalize()
        return {"first_name": first_name, "last_name": last_name, "dob": dob}

    remainder = f"{text[:dob_match.start()]} {text[dob_match.end():]}"
    name_words = [
        word
        for word in re.findall(r"[A-Za-z]{2,}", remainder)
        if word.lower() not in IDENTITY_STOPWORDS
    ]
    if len(name_words) < 2:
        return None
    return {
        "first_name": name_words[0].capitalize(),
        "last_name": " ".join(word.capitalize() for word in name_words[1:]),
        "dob": dob,
    }


def extract_identity_from_history(
    messages: list[dict[str, Any]],
    current_user_text: str,
) -> dict[str, str] | None:
    """Prefer the current turn, then concatenated prior user turns."""
    extracted = extract_identity_fields(current_user_text)
    if extracted is not None:
        return extracted
    prior = " ".join(
        str(message["content"])
        for message in messages
        if message.get("role") == "user" and message.get("content")
    )
    return extract_identity_fields(prior)


def user_turn_has_identity_fields(user_text: str) -> bool:
    """Return True when the user turn appears to include a name and date of birth."""
    return extract_identity_fields(user_text) is not None


def infer_measurement_type(text: str, device_id: str | None = None) -> str | None:
    """Infer spo2/bp/weight/temperature from user text or a device ID prefix."""
    lowered = text.lower()
    if re.search(r"\b(?:spo2|sp\s*o2|oxygen|oximeter|pulse\s*ox)\b", lowered):
        return "spo2"
    if re.search(r"\b(?:blood\s*pressure|bp)\b", lowered):
        return "bp"
    if re.search(r"\b(?:weight|scale)\b", lowered):
        return "weight"
    if re.search(r"\b(?:temp(?:erature)?|thermometer)\b", lowered):
        return "temperature"
    prefix = (device_id or "").split("-", 1)[0].split("_", 1)[0].upper()
    if prefix in {"PO", "OXI", "OX", "PULSE"}:
        return "spo2"
    if prefix in {"BP"}:
        return "bp"
    if prefix in {"WT", "SCALE"}:
        return "weight"
    if prefix in {"TEMP", "TH"}:
        return "temperature"
    return None


def extract_device_id(text: str) -> str | None:
    """Return a hardware-style device ID when one is present in free text."""
    match = DEVICE_ID_PATTERN.search(text)
    if match is None:
        return None
    device_id = match.group(1)
    if device_id in KNOWN_TOOL_NAMES:
        return None
    return device_id


def extract_device_id_from_history(
    messages: list[dict[str, Any]],
    current_user_text: str,
) -> str | None:
    """Prefer a device ID in the current turn, then prior user turns."""
    extracted = extract_device_id(current_user_text)
    if extracted is not None:
        return extracted
    for message in reversed(messages):
        if message.get("role") != "user" or not message.get("content"):
            continue
        extracted = extract_device_id(str(message["content"]))
        if extracted is not None:
            return extracted
    return None


def collapse_repeated_tool_name(name: str) -> str:
    """Collapse autoregressive repeats like check_device_statuscheck_device_status."""
    if name in KNOWN_TOOL_NAMES:
        return name
    for tool in sorted(KNOWN_TOOL_NAMES, key=len, reverse=True):
        if name.startswith(tool) and name.replace(tool, "") == "":
            return tool
    return name


def recover_tool_arguments(
    tool_name: str,
    arguments: str,
    messages: list[dict[str, Any]],
    user_text: str,
) -> dict[str, Any] | None:
    """Parse tool JSON, or rebuild args from conversation when the stream loops."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if tool_name in {"check_device_status", "pair_device"}:
        device_id = extract_device_id_from_history(messages, user_text)
        if device_id is not None:
            return {"device_id": device_id}
    if tool_name == "verify_identity":
        return extract_identity_from_history(messages, user_text)
    return None


def is_safety_regurgitation(text: str) -> bool:
    """Detect copied safety rules or a request for a nonexistent second patient."""
    lowered = text.lower()
    if "second patient" in lowered:
        return True
    if any(marker in lowered for marker in SAFETY_VOMIT_MARKERS):
        return True
    collapsed = collapse_repeated_tool_name(re.sub(r"[^A-Za-z_]", "", text))
    if collapsed in KNOWN_TOOL_NAMES and text.count(collapsed) > 2:
        return True
    return False


def fallback_assistant_message(state: str) -> str:
    """Return a short workflow prompt when the model output is unusable."""
    return STATE_FALLBACK_MESSAGES.get(
        state,
        "Please continue with the current workflow step.",
    )


def sanitize_assistant_message(text: str, state: str) -> str:
    """Replace safety-rule dumps with a concise next-step prompt."""
    stripped = text.strip()
    if not stripped or is_safety_regurgitation(stripped):
        return fallback_assistant_message(state)
    return stripped


def select_forced_tool(
    current_state: str,
    allowed_tools: list[str],
    user_text: str,
    checked_devices: set[str] | None = None,
) -> str | None:
    """Choose a DFA-safe tool to force, or None to leave routing on auto."""
    if not allowed_tools:
        return None
    allowed = set(allowed_tools)
    if current_state == "1_onboarding":
        if "verify_identity" in allowed and user_turn_has_identity_fields(user_text):
            return "verify_identity"
        return None
    # Device setup / education / troubleshooting are executed by the controller.
    # Forcing tool_choice here makes Qwen-7B emit Yes/No confirmations and then
    # repeat the tool name until max_tokens.
    return None


def complete_streamed_tool_call(
    tool_call: dict[str, str],
    forced_tool: str | None,
) -> dict[str, str] | None:
    """Fill Hermes-dropped name/id when a tool was forced; else require both."""
    name = collapse_repeated_tool_name(tool_call.get("name") or "")
    call_id = tool_call.get("id") or ""
    arguments = tool_call.get("arguments") or ""
    if forced_tool:
        if not name:
            name = forced_tool
        if not call_id:
            call_id = f"forced-{name}"
    name = collapse_repeated_tool_name(name)
    if name in KNOWN_TOOL_NAMES and not call_id:
        call_id = f"forced-{name}"
    if not name or not call_id:
        return None
    return {"id": call_id, "name": name, "arguments": arguments}


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
    def __init__(self, env: str | None = None) -> None:
        """Initialize the configured backend and its telemetry metadata."""
        try:
            with project_path("configs/model.yaml").open(
                "r",
                encoding="utf-8",
            ) as config_file:
                config = yaml.safe_load(config_file)
            resolved_env = resolve_inference_env(env, config)
            endpoint = config["endpoints"][resolved_env]
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise RuntimeError(f"Unable to load model configuration: {exc}") from exc
        self.env: str = resolved_env

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

    def _execute_workflow_tool(
        self,
        dfa: RPMStateMachine,
        registry: ToolRegistry,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Validate, execute, and record one controller-owned tool call."""
        workflow_error = dfa.validate_tool_call(tool_name, tool_args)
        if workflow_error is not None:
            execution_result = {
                "status": "error",
                "message": workflow_error,
            }
        else:
            execution_result = registry.execute_tool(tool_name, tool_args)
        dfa.process_tool_execution(tool_name, execution_result)
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(execution_result),
            }
        )
        return execution_result

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
                    state=dfa.current_state,
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
                "report whether it resolved the issue. Do not repeat safety rules."
            )

        if dfa.current_state == "1_onboarding":
            identity_args = extract_identity_from_history(
                self.messages,
                user_input,
            )
            if identity_args is not None:
                identity_result = self._execute_workflow_tool(
                    dfa,
                    registry,
                    "verify_identity",
                    identity_args,
                    "forced-verify_identity",
                )
                if identity_result.get("status") != "success":
                    return self._turn_result(
                        state=dfa.current_state,
                        message=(
                            "[System Error] Unable to verify identity: "
                            f"{identity_result.get('message', 'unknown tool error')}"
                        ),
                        metrics_note=(
                            "LLM bypassed because deterministic routing failed"
                        ),
                    )
                response_tool_call = ResponseToolCall(
                    name="verify_identity",
                    arguments=identity_args,
                )
                controller_context = (
                    "DETERMINISTIC CONTROLLER EVENT: verify_identity already "
                    "succeeded for this single patient. Do not ask for a second "
                    "patient. Do not call a tool. Do not repeat safety rules. "
                    "Ask which device they want to pair: pulse_oximeter, "
                    "bp_device, scale, or thermometer."
                )

        if dfa.current_state == "2_device_setup" and controller_context is None:
            device_id = extract_device_id_from_history(self.messages, user_input)
            if device_id is not None:
                last_name: str | None = None
                last_args: dict[str, Any] | None = None
                if device_id not in dfa.checked_devices:
                    check_result = self._execute_workflow_tool(
                        dfa,
                        registry,
                        "check_device_status",
                        {"device_id": device_id},
                        "forced-check_device_status",
                    )
                    if check_result.get("status") != "success":
                        return self._turn_result(
                            state=dfa.current_state,
                            message=(
                                "[System Error] Unable to check device status: "
                                f"{check_result.get('message', 'unknown tool error')}"
                            ),
                            metrics_note=(
                                "LLM bypassed because deterministic routing failed"
                            ),
                        )
                    last_name = "check_device_status"
                    last_args = {"device_id": device_id}
                should_pair = (
                    device_id in dfa.checked_devices
                    and device_id not in dfa.paired_devices
                    and PAIR_OR_SETUP_INTENT.search(user_input) is not None
                )
                if should_pair:
                    pair_result = self._execute_workflow_tool(
                        dfa,
                        registry,
                        "pair_device",
                        {"device_id": device_id},
                        "forced-pair_device",
                    )
                    if pair_result.get("status") != "success":
                        return self._turn_result(
                            state=dfa.current_state,
                            message=(
                                "[System Error] Unable to pair device: "
                                f"{pair_result.get('message', 'unknown tool error')}"
                            ),
                            metrics_note=(
                                "LLM bypassed because deterministic routing failed"
                            ),
                        )
                    last_name = "pair_device"
                    last_args = {"device_id": device_id}
                if last_name is not None and last_args is not None:
                    response_tool_call = ResponseToolCall(
                        name=last_name,
                        arguments=last_args,
                    )
                    controller_context = (
                        "DETERMINISTIC CONTROLLER EVENT: Device status and any "
                        "requested pairing already ran. Do not call a tool. "
                        "Do not ask Yes/No permission to call check_device_status. "
                        "Tell the user the current next step. "
                        "Do not repeat safety rules."
                    )

        if dfa.current_state == "4_education" and controller_context is None:
            device_id = extract_device_id_from_history(self.messages, user_input)
            if device_id is None and dfa.paired_devices:
                device_id = next(iter(sorted(dfa.paired_devices)))
            measurement_type = infer_measurement_type(user_input, device_id)
            should_measure = (
                device_id is not None
                and measurement_type is not None
                and MEASURE_INTENT.search(user_input) is not None
            )
            if should_measure:
                measure_args = {
                    "device_id": device_id,
                    "measurement_type": measurement_type,
                }
                measure_result = self._execute_workflow_tool(
                    dfa,
                    registry,
                    "start_measurement",
                    measure_args,
                    "forced-start_measurement",
                )
                if measure_result.get("status") != "success":
                    return self._turn_result(
                        state=dfa.current_state,
                        message=(
                            "[System Error] Unable to start measurement: "
                            f"{measure_result.get('message', 'unknown tool error')}"
                        ),
                        metrics_note=(
                            "LLM bypassed because deterministic routing failed"
                        ),
                    )
                response_tool_call = ResponseToolCall(
                    name="start_measurement",
                    arguments=measure_args,
                )
                controller_context = (
                    "DETERMINISTIC CONTROLLER EVENT: start_measurement already "
                    f"ran for {device_id}. Readings: {measure_result.get('readings')}. "
                    "Do not call a tool. Acknowledge receipt without interpreting "
                    "the values. Do not repeat safety rules."
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

            # ---------------------------------------------------------
            # FIX 2: Dynamic tool-forcing logic for state transitions
            # ---------------------------------------------------------
            forced_tool: str | None = None
            if tool_schemas:
                api_kwargs["tools"] = tool_schemas
                api_kwargs["parallel_tool_calls"] = False
                active_tool_names = [
                    schema.get("function", {}).get("name")
                    for schema in tool_schemas
                    if isinstance(schema.get("function"), dict)
                ]
                forced_tool = select_forced_tool(
                    dfa.current_state,
                    [name for name in active_tool_names if isinstance(name, str)],
                    user_input,
                    dfa.checked_devices,
                )
                if forced_tool:
                    api_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": forced_tool},
                    }
                else:
                    api_kwargs["tool_choice"] = "auto"

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
                                incoming_name = tool_delta.function.name or ""
                                if incoming_name:
                                    current_name = collapse_repeated_tool_name(
                                        tool_call["name"]
                                    )
                                    if current_name not in KNOWN_TOOL_NAMES:
                                        tool_call["name"] = collapse_repeated_tool_name(
                                            tool_call["name"] + incoming_name
                                        )
                                    else:
                                        tool_call["name"] = current_name
                                tool_call["arguments"] += (
                                    tool_delta.function.arguments or ""
                                )
            except OpenAIError as exc:
                if controller_context is not None:
                    message = fallback_assistant_message(dfa.current_state)
                    self.messages.append(
                        {"role": "assistant", "content": message}
                    )
                    return self._turn_result(
                        state=dfa.current_state,
                        message=message,
                        tool_call=response_tool_call,
                        metrics=metrics,
                        metrics_note=(
                            "LLM backend request failed after controller action: "
                            f"{exc}"
                        ),
                    )
                return self._turn_result(
                    state=dfa.current_state,
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
                        state=dfa.current_state,
                        message=(
                            "[System Error] LLM generated multiple tool calls; "
                            "only one tool call is permitted per turn."
                        ),
                        metrics=metrics,
                    )

                parsed_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
                assistant_tool_calls: list[dict[str, Any]] = []

                for tool_call in streamed_tool_calls.values():
                    completed = complete_streamed_tool_call(
                        tool_call,
                        forced_tool,
                    )
                    if completed is None:
                        return self._turn_result(
                            state=dfa.current_state,
                            message=(
                                "[System Error] LLM generated an incomplete "
                                "tool call."
                            ),
                            metrics=metrics,
                        )
                    tool_call_id = completed["id"]
                    tool_name = completed["name"]
                    tool_call["id"] = tool_call_id
                    tool_call["name"] = tool_name

                    raw_tool_args = recover_tool_arguments(
                        tool_name,
                        tool_call["arguments"],
                        self.messages,
                        user_input,
                    )
                    if not isinstance(raw_tool_args, dict):
                        return self._turn_result(
                            state=dfa.current_state,
                            message=fallback_assistant_message(dfa.current_state),
                            metrics=metrics,
                            metrics_note=(
                                "LLM tool arguments were malformed; "
                                "controller returned the workflow prompt"
                            ),
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
                    "Do not ask for a second patient. Do not repeat safety rules. "
                    "Explain the result and the current next step, then wait for "
                    "a new user response before taking another action."
                )
                continue

            response_content = sanitize_assistant_message(
                "".join(content_parts),
                dfa.current_state,
            )
            self.messages.append(
                {"role": "assistant", "content": response_content}
            )
            return self._turn_result(
                state=dfa.current_state,
                message=response_content,
                tool_call=response_tool_call,
                metrics=metrics,
            )

        return self._turn_result(
            state=dfa.current_state,
            message="[System] LLM tool iteration limit reached.",
            tool_call=response_tool_call,
            metrics=metrics,
        )
