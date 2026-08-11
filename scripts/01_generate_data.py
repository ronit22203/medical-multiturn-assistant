import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agent.inference import GLOBAL_SAFETY_INVARIANT
from src.engine.state_machine import RPMStateMachine
from src.paths import PROJECT_ROOT, project_path
from src.tools.definitions import (
    CheckDeviceStatusInput,
    EscalateToNurseInput,
    PairDeviceInput,
    StartMeasurementInput,
    TroubleshootStepInput,
    VerifyIdentityInput,
)
from src.tools.registry import ToolRegistry


DEFAULT_OUTPUT_PATH = project_path("data/synthetic/sft_dataset.jsonl")
DEFAULT_TARGET = 300

ToolName = Literal[
    "verify_identity",
    "check_device_status",
    "pair_device",
    "troubleshoot_step",
    "start_measurement",
    "escalate_to_nurse",
]

FIRST_NAMES = (
    "Aisha",
    "Alex",
    "Carlos",
    "David",
    "Elena",
    "Emily",
    "Fatima",
    "Grace",
    "Hiro",
    "Ibrahim",
    "Jordan",
    "Leila",
    "Marcus",
    "Mei",
    "Nina",
    "Omar",
    "Priya",
    "Ravi",
    "Sarah",
    "Victor",
)
LAST_NAMES = (
    "Ahmed",
    "Brown",
    "Chen",
    "Davis",
    "Garcia",
    "Gupta",
    "Hassan",
    "Johnson",
    "Khan",
    "Kim",
    "Martin",
    "Miller",
    "Nguyen",
    "Patel",
    "Rodriguez",
    "Silva",
    "Smith",
    "Tanaka",
    "Vargas",
    "Wilson",
)
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DEVICES = (
    ("pulse oximeter", "OXI", "spo2"),
    ("blood pressure monitor", "BP", "bp"),
    ("scale", "SCL", "weight"),
    ("thermometer", "THM", "temperature"),
)

TOOL_ARGUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "verify_identity": VerifyIdentityInput,
    "check_device_status": CheckDeviceStatusInput,
    "pair_device": PairDeviceInput,
    "troubleshoot_step": TroubleshootStepInput,
    "start_measurement": StartMeasurementInput,
    "escalate_to_nurse": EscalateToNurseInput,
}

PUBLIC_TO_INTERNAL_STATE = {
    "onboarding": "1_onboarding",
    "device_setup": "2_device_setup",
    "troubleshooting": "3_troubleshooting",
    "education": "4_education",
    "closing": "5_closing",
}

ONBOARDING_TEMPLATES = (
    "Hi, my name is {full_name}, and my date of birth is {display_dob}.",
    "I am {full_name}. DOB: {display_dob}.",
    "Hello. {full_name} here, born {display_dob}.",
    "Ready to start. I am {full_name}, born on {display_dob}.",
    "Please register {full_name}. My birth date is {display_dob}.",
)
DEVICE_TEMPLATES = (
    "I want to set up my {device}. Device ID is {device_id}.",
    "The ID on my {device} is {device_id}.",
    "I have my {device} here. Its device ID is {device_id}.",
    "Can we connect the {device} labeled {device_id}?",
)
SAFETY_RESPONSES = (
    "I cannot provide medical advice, but this could be urgent. Please call "
    "your local emergency number now. I am escalating you to a nurse.",
    "This may require urgent attention. I cannot provide medical advice. Call "
    "your local emergency number now while I escalate this to a nurse.",
    "Please call your local emergency number now. I cannot assess the symptom, "
    "and I am escalating the report to a nurse immediately.",
)


class FunctionCall(BaseModel):
    """Function payload used by Qwen's native tool-call chat template."""

    model_config = ConfigDict(extra="forbid")

    name: ToolName
    arguments: str

    @model_validator(mode="after")
    def validate_arguments(self) -> "FunctionCall":
        """Validate exact runtime argument names and values."""
        schema = TOOL_ARGUMENT_SCHEMAS[self.name]
        raw_arguments = json.loads(self.arguments)
        validated = schema.model_validate(raw_arguments).model_dump()
        self.arguments = json.dumps(
            validated,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return self

    def parsed_arguments(self) -> dict[str, Any]:
        """Return validated arguments for deterministic tool execution."""
        arguments = json.loads(self.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must decode to an object")
        return arguments


class NativeToolCall(BaseModel):
    """One native assistant tool call."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^call_[a-z0-9_]+$")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """A strict OpenAI/Qwen-compatible chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[NativeToolCall] | None = None
    tool_call_id: str | None = None
    name: ToolName | None = None

    @model_validator(mode="after")
    def validate_role_shape(self) -> "ChatMessage":
        """Reject malformed role/content/tool combinations."""
        has_content = isinstance(self.content, str) and bool(self.content.strip())
        has_tool_calls = bool(self.tool_calls)

        if self.role in {"system", "user"}:
            if not has_content or has_tool_calls or self.tool_call_id or self.name:
                raise ValueError(f"Invalid {self.role} message shape")
        elif self.role == "assistant":
            if has_content == has_tool_calls:
                raise ValueError(
                    "Assistant message requires either content or tool_calls"
                )
            if self.tool_call_id or self.name:
                raise ValueError("Assistant message cannot be a tool response")
            if self.tool_calls is not None and len(self.tool_calls) != 1:
                raise ValueError("Exactly one tool call is allowed per assistant turn")
        elif self.role == "tool":
            if (
                not has_content
                or has_tool_calls
                or self.tool_call_id is None
                or self.name is None
            ):
                raise ValueError("Invalid tool result message shape")

        return self


class FunctionDefinition(BaseModel):
    """Function definition supplied to the model with a training example."""

    model_config = ConfigDict(extra="forbid")

    name: ToolName
    description: str
    parameters: dict[str, Any]


class ToolDefinition(BaseModel):
    """OpenAI-compatible tool definition."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: FunctionDefinition


def render_qwen_chat(
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
) -> str:
    """Render the exact Qwen2.5 tool-calling ChatML training text."""
    if not messages or messages[0].role != "system" or messages[0].content is None:
        raise ValueError("Qwen rendering requires a leading system message")

    parts = ["<|im_start|>system\n", messages[0].content]
    if tools:
        parts.append(
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query."
            "\n\nYou are provided with function signatures within "
            "<tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            parts.extend(
                (
                    "\n",
                    json.dumps(
                        tool.model_dump(mode="json"),
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                )
            )
        parts.append(
            "\n</tools>\n\n"
            "For each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
    parts.append("<|im_end|>\n")

    for message in messages[1:]:
        if message.role in {"user", "system"}:
            parts.extend(
                (
                    "<|im_start|>",
                    message.role,
                    "\n",
                    message.content or "",
                    "<|im_end|>\n",
                )
            )
        elif message.role == "assistant" and message.tool_calls:
            call = message.tool_calls[0].function
            parts.extend(
                (
                    "<|im_start|>assistant\n<tool_call>\n",
                    '{"name":"',
                    call.name,
                    '","arguments":',
                    call.arguments,
                    "}\n</tool_call><|im_end|>\n",
                )
            )
        elif message.role == "assistant":
            parts.extend(
                (
                    "<|im_start|>assistant\n",
                    message.content or "",
                    "<|im_end|>\n",
                )
            )
        elif message.role == "tool":
            parts.extend(
                (
                    "<|im_start|>user\n<tool_response>\n",
                    message.content or "",
                    "\n</tool_response><|im_end|>\n",
                )
            )

    return "".join(parts)


class SFTRecord(BaseModel):
    """One complete, protocol-valid native tool-calling training record."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=3)
    tools: list[ToolDefinition]
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_protocol(self) -> "SFTRecord":
        """Validate roles, call/result matching, tools, and pairing prerequisites."""
        if self.messages[0].role != "system":
            raise ValueError("First message must be a system message")
        if any(message.role == "system" for message in self.messages[1:]):
            raise ValueError("Only one leading system message is allowed")

        declared_tools = {tool.function.name for tool in self.tools}
        if len(declared_tools) != len(self.tools):
            raise ValueError("Duplicate tool definitions are not allowed")

        pending_call: NativeToolCall | None = None
        checked_device_ids = set(
            re.findall(
                r"\b(?:OXI|BP|SCL|THM)-\d{4}\b",
                self.messages[0].content or "",
            )
            if "Status already checked" in (self.messages[0].content or "")
            else []
        )

        for message in self.messages:
            if message.role == "assistant" and message.tool_calls:
                if pending_call is not None:
                    raise ValueError("Tool calls cannot overlap")
                pending_call = message.tool_calls[0]
                call = pending_call.function
                if call.name not in declared_tools:
                    raise ValueError(f"Tool '{call.name}' is not declared")
                if call.name == "pair_device":
                    device_id = call.parsed_arguments()["device_id"]
                    if device_id not in checked_device_ids:
                        raise ValueError(
                            "pair_device requires a prior successful status check"
                        )
            elif message.role == "tool":
                if pending_call is None:
                    raise ValueError("Tool result has no preceding tool call")
                if (
                    message.tool_call_id != pending_call.id
                    or message.name != pending_call.function.name
                ):
                    raise ValueError("Tool result does not match its tool call")

                result = json.loads(message.content or "")
                if not isinstance(result, dict) or "status" not in result:
                    raise ValueError("Tool result must be a JSON status object")
                if (
                    pending_call.function.name == "check_device_status"
                    and result.get("status") == "success"
                ):
                    checked_device_ids.add(
                        pending_call.function.parsed_arguments()["device_id"]
                    )
                pending_call = None
            elif pending_call is not None:
                raise ValueError("A tool result must immediately follow its call")

        if pending_call is not None:
            raise ValueError("Final tool call is missing its tool result")
        expected_text = render_qwen_chat(self.messages, self.tools)
        if self.text != expected_text:
            raise ValueError("Pre-rendered Qwen text does not match the messages")
        return self


class Conversation:
    """Build one validated conversation using runtime schemas and tool results."""

    def __init__(
        self,
        index: int,
        state: str,
        registry: ToolRegistry,
        checked_device_id: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.index = index
        self.registry = registry
        self.call_index = 0

        if state == "safety":
            prompt = (
                f"{GLOBAL_SAFETY_INVARIANT}\n\n"
                "A deterministic safety control has identified a red flag. "
                "Call escalate_to_nurse and direct the patient to their local "
                "emergency number without interpreting the symptom or vital."
            )
            tool_names = ["escalate_to_nurse"]
        else:
            dfa = RPMStateMachine()
            dfa.current_state = PUBLIC_TO_INTERNAL_STATE[state]
            if checked_device_id is not None:
                dfa.checked_devices.add(checked_device_id)
            state_prompt, state_tools = dfa.get_context()
            prompt = f"{state_prompt.rstrip()}\n\n{GLOBAL_SAFETY_INVARIANT}"
            tool_names = state_tools

        if allowed_tools is not None:
            tool_names = allowed_tools

        self.messages = [ChatMessage(role="system", content=prompt)]
        self.tools = [
            ToolDefinition.model_validate(tool)
            for tool in registry.get_tool_schemas(tool_names)
        ]

    def user(self, content: str) -> None:
        """Append one user message."""
        self.messages.append(ChatMessage(role="user", content=content))

    def assistant(self, content: str) -> None:
        """Append one natural-language assistant response."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def call(self, name: ToolName, arguments: dict[str, Any]) -> dict[str, Any]:
        """Append a native call and its deterministic runtime result."""
        self.call_index += 1
        call_id = f"call_{self.index:04d}_{self.call_index:02d}"
        function = FunctionCall(
            name=name,
            arguments=json.dumps(
                arguments,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
        call = NativeToolCall(id=call_id, function=function)
        self.messages.append(
            ChatMessage(role="assistant", tool_calls=[call])
        )

        result = self.registry.execute_tool(name, function.parsed_arguments())
        if result.get("status") == "error":
            raise RuntimeError(f"Tool fixture failed: {result['message']}")

        self.messages.append(
            ChatMessage(
                role="tool",
                content=json.dumps(result, separators=(",", ":"), ensure_ascii=True),
                tool_call_id=call_id,
                name=name,
            )
        )
        return result

    def build(self) -> SFTRecord:
        """Return the fully validated training record."""
        return SFTRecord(
            messages=self.messages,
            tools=self.tools,
            text=render_qwen_chat(self.messages, self.tools),
        )


def synthetic_person(index: int) -> tuple[str, str, str, str]:
    """Return diverse deterministic identity fields."""
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[((index // len(FIRST_NAMES)) + index * 7) % len(LAST_NAMES)]
    year = 1945 + index % 58
    month = index % 12 + 1
    day = index * 5 % 28 + 1
    dob = f"{month:02d}/{day:02d}/{year:04d}"
    display_dobs = (
        dob,
        f"{year:04d}-{month:02d}-{day:02d}",
        f"{MONTH_NAMES[month - 1]} {day}, {year}",
    )
    return first, last, dob, display_dobs[index % len(display_dobs)]


def synthetic_device(index: int) -> tuple[str, str, str]:
    """Return a unique device fixture."""
    device, prefix, measurement_type = DEVICES[index % len(DEVICES)]
    return device, f"{prefix}-{1000 + index}", measurement_type


def build_onboarding_complete(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach complete identity extraction and native verification calls."""
    first, last, dob, display_dob = synthetic_person(index)
    full_name = f"{first} {last}"
    conversation = Conversation(index, "onboarding", registry)
    conversation.user(
        rng.choice(ONBOARDING_TEMPLATES).format(
            full_name=full_name,
            display_dob=display_dob,
        )
    )
    conversation.call(
        "verify_identity",
        {"first_name": first, "last_name": last, "dob": dob},
    )
    conversation.assistant(
        f"Identity verification is complete for {first} {last}. "
        "Which RPM device are you ready to set up?"
    )
    return conversation.build()


def build_onboarding_partial(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach the model not to call tools when identity fields are missing."""
    first, _, _, _ = synthetic_person(index)
    setup_reference = f"SETUP-{1000 + index}"
    conversation = Conversation(index, "onboarding", registry)
    conversation.user(
        rng.choice(
            (
                f"Hi, I am {first}. My tablet shows {setup_reference}.",
                f"My first name is {first}; the setup reference is {setup_reference}.",
                f"You can call me {first}. My RPM kit says {setup_reference}.",
            )
        )
    )
    conversation.assistant(
        f"Thanks, {first}. Please provide your last name and full date of birth."
    )
    return conversation.build()


def build_device_status(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach status checks before any pairing attempt."""
    device, device_id, _ = synthetic_device(index)
    conversation = Conversation(index, "device_setup", registry)
    conversation.user(
        rng.choice(DEVICE_TEMPLATES).format(
            device=device,
            device_id=device_id,
        )
    )
    conversation.call("check_device_status", {"device_id": device_id})
    conversation.assistant(
        f"{device_id} is not paired. Confirm when you want me to pair it."
    )
    return conversation.build()


def build_device_pair(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach pairing only after the same device ID was checked."""
    device, device_id, _ = synthetic_device(index)
    conversation = Conversation(
        index,
        "device_setup",
        registry,
        checked_device_id=device_id,
    )
    conversation.user(
        rng.choice(
            (
                f"The status check for {device_id} says not paired. Pair it now.",
                f"Please pair my checked {device} {device_id}.",
                f"{device_id} was checked and is ready to pair.",
            )
        )
    )
    conversation.call("pair_device", {"device_id": device_id})
    conversation.assistant(
        f"{device_id} is paired with the tablet. We can continue to education."
    )
    return conversation.build()


def build_setup_issue(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach deterministic routing from setup into troubleshooting."""
    device, device_id, _ = synthetic_device(index)
    conversation = Conversation(index, "device_setup", registry)
    conversation.user(
        rng.choice(
            (
                f"My {device} {device_id} failed to pair twice.",
                f"{device_id} will not connect to the tablet.",
                f"The Bluetooth light on {device_id} is blinking red.",
            )
        )
    )
    conversation.call(
        "troubleshoot_step",
        {"step_id": f"restart_{device_id.lower()}", "resolved": False},
    )
    conversation.assistant(
        f"Restart {device_id}, enable pairing mode, and tell me whether it connects."
    )
    return conversation.build()


def build_troubleshooting(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach unresolved steps and explicit resolution confirmation."""
    _, device_id, _ = synthetic_device(index)
    step_id = f"bluetooth_reset_{device_id.lower()}"
    conversation = Conversation(index, "troubleshooting", registry)
    conversation.user(
        rng.choice(
            (
                f"I retried {device_id}, but it still does not connect.",
                f"The reset did not fix {device_id}.",
                f"{device_id} remains offline after the first step.",
            )
        )
    )
    conversation.call(
        "troubleshoot_step",
        {"step_id": step_id, "resolved": False},
    )
    conversation.assistant(
        f"Reset Bluetooth on the tablet, retry {device_id}, and report the result."
    )
    conversation.user(f"I completed that step, and {device_id} is connected now.")
    conversation.call(
        "troubleshoot_step",
        {"step_id": step_id, "resolved": True},
    )
    conversation.assistant(
        f"The issue with {device_id} is resolved. We can resume the workflow."
    )
    return conversation.build()


def build_education(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach device education without unnecessary tool use."""
    device, device_id, measurement_type = synthetic_device(index)
    guidance = {
        "spo2": "Place a warm, still finger fully inside the sensor.",
        "bp": (
            "Sit quietly for five minutes with feet flat and your arm "
            "supported at heart level."
        ),
        "weight": "Place the scale on a hard, level surface and stand still.",
        "temperature": "Position the thermometer as its device guide directs.",
    }[measurement_type]
    conversation = Conversation(index, "education", registry)
    conversation.user(
        rng.choice(
            (
                f"How should I use {device_id}, my {device}?",
                f"Explain how to take a reading with {device_id}.",
                f"What is the correct way to use my {device} {device_id}?",
            )
        )
    )
    conversation.assistant(
        f"For {device_id}, {guidance} Tell me when you are ready to measure."
    )
    return conversation.build()


def build_measurement(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach start_measurement only after explicit readiness."""
    device, device_id, measurement_type = synthetic_device(index)
    conversation = Conversation(index, "education", registry)
    conversation.user(
        rng.choice(
            (
                f"I am ready to measure with {device_id}.",
                f"My {device} {device_id} is positioned. Start the reading.",
                f"Everything is ready on {device_id}. Begin now.",
            )
        )
    )
    conversation.call(
        "start_measurement",
        {"device_id": device_id, "measurement_type": measurement_type},
    )
    conversation.assistant(
        f"The {measurement_type} measurement started on {device_id}. "
        "Follow the device prompts until the reading is saved."
    )
    return conversation.build()


def build_closing(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach concise tool-free closure."""
    _, device_id, _ = synthetic_device(index)
    conversation = Conversation(index, "closing", registry)
    conversation.user(
        rng.choice(
            (
                f"The reading from {device_id} is saved. That is all.",
                f"I finished with {device_id}. Thanks.",
                f"My session with {device_id} is complete.",
            )
        )
    )
    conversation.assistant(
        f"You are all set after saving the reading from {device_id}. Take care."
    )
    return conversation.build()


def build_safe_vital(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach non-interpretive acknowledgement for non-escalated vitals."""
    _, device_id, _ = synthetic_device(index)
    systolic = 105 + index % 15
    diastolic = 65 + index % 10
    conversation = Conversation(index, "education", registry)
    conversation.user(
        rng.choice(
            (
                f"{device_id} shows {systolic}/{diastolic}.",
                f"My BP reading on {device_id} is {systolic} over {diastolic}.",
            )
        )
    )
    conversation.assistant(
        f"I received the reading from {device_id}. I cannot interpret or "
        "characterize the values."
    )
    return conversation.build()


def build_safety(
    index: int,
    rng: random.Random,
    registry: ToolRegistry,
) -> SFTRecord:
    """Teach native nurse escalation as a secondary model fail-safe."""
    _, device_id, _ = synthetic_device(index)
    spo2 = 82 + index % 8
    cases = (
        (
            f"My oxygen reading on {device_id} is {spo2}%.",
            f"Critical SpO2 reading detected: {spo2}%",
        ),
        (
            f"I am having chest pain while setting up {device_id}.",
            "Symptom red flag detected: chest pain",
        ),
        (
            f"I cannot catch my breath while using {device_id}.",
            "Symptom red flag detected: shortness of breath",
        ),
    )
    patient_message, reason = rng.choice(cases)
    conversation = Conversation(index, "safety", registry)
    conversation.user(patient_message)
    conversation.call("escalate_to_nurse", {"reason": reason})
    conversation.assistant(
        f"{rng.choice(SAFETY_RESPONSES)} The report for {device_id} is escalated."
    )
    return conversation.build()


Builder = type(build_onboarding_complete)
BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("onboarding_complete", build_onboarding_complete),
    ("onboarding_partial", build_onboarding_partial),
    ("device_status", build_device_status),
    ("device_pair", build_device_pair),
    ("setup_issue", build_setup_issue),
    ("troubleshooting", build_troubleshooting),
    ("education", build_education),
    ("measurement", build_measurement),
    ("closing", build_closing),
    ("safe_vital", build_safe_vital),
    ("safety", build_safety),
)


def serialize_record(record: SFTRecord) -> str:
    """Serialize one compact JSONL row without null-only message fields."""
    return json.dumps(
        record.model_dump(mode="json", exclude_none=True),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def generate_records(
    target: int,
    seed: int,
) -> list[tuple[str, SFTRecord]]:
    """Generate balanced, unique, validated records."""
    rng = random.Random(seed)
    registry = ToolRegistry()
    generated: list[tuple[str, SFTRecord]] = []
    serialized: set[str] = set()

    for index in range(target):
        scenario, builder = BUILDERS[index % len(BUILDERS)]
        record = builder(index, rng, registry)
        row = serialize_record(record)
        if row in serialized:
            raise RuntimeError(
                f"Duplicate trajectory at index {index}: {scenario}"
            )
        serialized.add(row)
        generated.append((scenario, record))

    return generated


def stratified_split(
    records: list[tuple[str, SFTRecord]],
    seed: int,
) -> dict[str, list[tuple[str, SFTRecord]]]:
    """Create exact 80/10/10 splits with round-robin scenario stratification."""
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[str, SFTRecord]]] = defaultdict(list)
    for item in records:
        grouped[item[0]].append(item)
    for items in grouped.values():
        rng.shuffle(items)

    interleaved: list[tuple[str, SFTRecord]] = []
    while any(grouped.values()):
        for scenario, _ in BUILDERS:
            if grouped[scenario]:
                interleaved.append(grouped[scenario].pop())

    target_validation = len(records) // 10
    target_test = len(records) // 10
    splits = {"train": [], "validation": [], "test": []}

    for index, item in enumerate(interleaved):
        bucket = index % 10
        if bucket == 0 and len(splits["validation"]) < target_validation:
            splits["validation"].append(item)
        elif bucket == 1 and len(splits["test"]) < target_test:
            splits["test"].append(item)
        else:
            splits["train"].append(item)

    rng.shuffle(splits["train"])
    rng.shuffle(splits["validation"])
    rng.shuffle(splits["test"])
    return splits


def write_jsonl(path: Path, records: list[tuple[str, SFTRecord]]) -> None:
    """Atomically replace one JSONL dataset file."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for _, record in records:
            output_file.write(serialize_record(record) + "\n")
    temporary_path.replace(path)


def portable_path(path: Path) -> str:
    """Return repository-relative paths without leaking machine-specific roots."""
    return (
        str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def parse_args() -> argparse.Namespace:
    """Parse deterministic SFT generator options."""
    parser = argparse.ArgumentParser(
        description="Generate native Qwen tool-calling SFT data."
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target < len(BUILDERS):
        parser.error(f"--target must be at least {len(BUILDERS)}")
    return args


def main() -> int:
    """Generate combined and stratified native tool-calling datasets."""
    args = parse_args()
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_records(args.target, args.seed)
    splits = stratified_split(records, args.seed + 1)

    write_jsonl(output_path, records)
    split_paths = {
        "train": output_path.parent / "sft_train.jsonl",
        "validation": output_path.parent / "sft_validation.jsonl",
        "test": output_path.parent / "sft_test.jsonl",
    }
    for split_name, split_records in splits.items():
        write_jsonl(split_paths[split_name], split_records)

    manifest = {
        "format": "qwen_native_tool_calls_v1",
        "training_field": "text",
        "chat_template": "Qwen/Qwen2.5-7B-Instruct",
        "structured_arguments_encoding": "json_string",
        "seed": args.seed,
        "total_records": len(records),
        "splits": {name: len(items) for name, items in splits.items()},
        "scenarios": dict(sorted(Counter(name for name, _ in records).items())),
        "split_scenarios": {
            split_name: dict(
                sorted(Counter(name for name, _ in items).items())
            )
            for split_name, items in splits.items()
        },
        "files": {
            "combined": portable_path(output_path),
            **{name: portable_path(path) for name, path in split_paths.items()},
        },
    }
    manifest_path = output_path.parent / "sft_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    print(
        f"Generated {len(records)} native tool-calling records: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
