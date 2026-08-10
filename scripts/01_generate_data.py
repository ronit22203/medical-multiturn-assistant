import argparse
import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.tools.definitions import (
    CheckDeviceStatusInput,
    EscalateToNurseInput,
    PairDeviceInput,
    StartMeasurementInput,
    TroubleshootStepInput,
    VerifyIdentityInput,
)


DEFAULT_OUTPUT_PATH = Path("data/synthetic/sft_dataset.jsonl")
DEFAULT_TARGET = 300

StateName = Literal[
    "onboarding",
    "device_setup",
    "troubleshooting",
    "education",
    "closing",
    "escalated",
]
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

ONBOARDING_TEMPLATES = (
    "Hi, my name is {full_name}, and my date of birth is {display_dob}.",
    "I am {full_name}. DOB: {display_dob}.",
    "Hello. {full_name} here, born {display_dob}.",
    "Ready to start. My name is {full_name} and I was born on {display_dob}.",
)
DEVICE_TEMPLATES = (
    "I want to set up my {device}. Device ID is {device_id}.",
    "The ID on my {device} is {device_id}.",
    "I have my {device} here. Its device ID is {device_id}.",
    "Can we connect the {device} labeled {device_id}?",
)

TOOL_ARGUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "verify_identity": VerifyIdentityInput,
    "check_device_status": CheckDeviceStatusInput,
    "pair_device": PairDeviceInput,
    "troubleshoot_step": TroubleshootStepInput,
    "start_measurement": StartMeasurementInput,
    "escalate_to_nurse": EscalateToNurseInput,
}
TOOLS_BY_STATE: dict[str, set[str]] = {
    "onboarding": {"verify_identity"},
    "device_setup": {
        "check_device_status",
        "pair_device",
        "troubleshoot_step",
    },
    "troubleshooting": {"troubleshoot_step"},
    "education": {"start_measurement", "troubleshoot_step"},
    "closing": set(),
    "escalated": {"escalate_to_nurse"},
}
STATE_TRANSITIONS: dict[str, set[str]] = {
    "onboarding": {"onboarding", "device_setup", "escalated"},
    "device_setup": {
        "device_setup",
        "troubleshooting",
        "education",
        "escalated",
    },
    "troubleshooting": {
        "troubleshooting",
        "device_setup",
        "education",
        "escalated",
    },
    "education": {"education", "troubleshooting", "closing", "escalated"},
    "closing": {"closing", "escalated"},
    "escalated": {"escalated"},
}


class ToolCall(BaseModel):
    """Validated tool call embedded in an assistant message."""

    model_config = ConfigDict(extra="forbid")

    name: ToolName
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolCall":
        """Validate arguments against the runtime tool schema."""
        schema = TOOL_ARGUMENT_SCHEMAS[self.name]
        self.arguments = schema.model_validate(self.arguments).model_dump()
        return self


class AssistantContent(BaseModel):
    """Structured assistant payload stored as a JSON string in ChatML."""

    model_config = ConfigDict(extra="forbid")

    state: StateName
    assistant_message: str = Field(min_length=1)
    tool_call: ToolCall | None

    @model_validator(mode="after")
    def validate_state_tool_compatibility(self) -> "AssistantContent":
        """Reject tools that are illegal in the declared workflow state."""
        if (
            self.tool_call is not None
            and self.tool_call.name not in TOOLS_BY_STATE[self.state]
        ):
            raise ValueError(
                f"Tool '{self.tool_call.name}' is not allowed in "
                f"state '{self.state}'"
            )
        return self


class ChatMessage(BaseModel):
    """One strict ChatML message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Trajectory(BaseModel):
    """Alternating ChatML trajectory with deterministic workflow validation."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_messages(self) -> "Trajectory":
        """Validate message roles, state flow, and device-tool ordering."""
        if len(self.messages) % 2 != 0:
            raise ValueError("Trajectory must contain complete user/assistant pairs")

        checked_device_ids: set[str] = set()
        previous_state: str | None = None

        for index, message in enumerate(self.messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.role != expected_role:
                raise ValueError(
                    f"Message {index} must have role '{expected_role}'"
                )
            if message.role == "user":
                continue

            content = AssistantContent.model_validate_json(message.content)
            if (
                previous_state is not None
                and content.state not in STATE_TRANSITIONS[previous_state]
            ):
                raise ValueError(
                    f"Invalid state transition: {previous_state} -> "
                    f"{content.state}"
                )

            tool_call = content.tool_call
            if tool_call is not None:
                device_id = tool_call.arguments.get("device_id")
                if (
                    tool_call.name == "check_device_status"
                    and isinstance(device_id, str)
                ):
                    checked_device_ids.add(device_id)
                elif (
                    tool_call.name == "pair_device"
                    and device_id not in checked_device_ids
                ):
                    raise ValueError(
                        "pair_device must follow check_device_status for "
                        "the same device_id"
                    )

            previous_state = content.state
            message.content = content.model_dump_json()

        return self


def synthetic_person(index: int) -> tuple[str, str, str, str]:
    """Return deterministic, varied identity fields for one record."""
    first_name = FIRST_NAMES[index % len(FIRST_NAMES)]
    last_name = LAST_NAMES[(index * 7 + 3) % len(LAST_NAMES)]
    year = 1945 + (index * 11) % 58
    month = index % 12 + 1
    day = index * 5 % 28 + 1
    dob = f"{year:04d}-{month:02d}-{day:02d}"

    display_formats = (
        dob,
        f"{month:02d}/{day:02d}/{year:04d}",
        f"{MONTH_NAMES[month - 1]} {day}, {year}",
    )
    display_dob = display_formats[index % len(display_formats)]
    return first_name, last_name, dob, display_dob


def synthetic_device(index: int) -> tuple[str, str, str]:
    """Return a deterministic device type, unique ID, and measurement type."""
    device, prefix, measurement_type = DEVICES[index % len(DEVICES)]
    device_id = f"{prefix}-{1000 + index}"
    return device, device_id, measurement_type


def assistant(
    state: StateName,
    message: str,
    tool_name: ToolName | None = None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Create one canonical assistant ChatML message."""
    tool_call = None
    if tool_name is not None:
        tool_call = ToolCall(name=tool_name, arguments=arguments or {})

    content = AssistantContent(
        state=state,
        assistant_message=message,
        tool_call=tool_call,
    )
    return {"role": "assistant", "content": content.model_dump_json()}


def user(message: str) -> dict[str, str]:
    """Create one user ChatML message."""
    return {"role": "user", "content": message}


def build_onboarding(index: int, rng: random.Random) -> Trajectory:
    """Build complete identity collection and verification."""
    first, last, dob, display_dob = synthetic_person(index)
    full_name = f"{first} {last}"
    prompt = rng.choice(ONBOARDING_TEMPLATES).format(
        full_name=full_name,
        display_dob=display_dob,
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(prompt),
                assistant(
                    "onboarding",
                    f"Thank you, {first}. Let me verify your identity now.",
                    "verify_identity",
                    {
                        "first_name": first,
                        "last_name": last,
                        "dob": dob,
                    },
                ),
            ]
        }
    )


def build_partial_onboarding(index: int, rng: random.Random) -> Trajectory:
    """Build missing-field recovery before identity verification."""
    first, last, dob, display_dob = synthetic_person(index)
    opening = rng.choice(
        (
            f"Hi, I am {first}.",
            f"My first name is {first}.",
            f"You can call me {first}.",
        )
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(opening),
                assistant(
                    "onboarding",
                    "Please provide your last name and full date of birth.",
                ),
                user(f"My last name is {last}, and I was born {display_dob}."),
                assistant(
                    "onboarding",
                    "Thank you. I will verify your identity now.",
                    "verify_identity",
                    {
                        "first_name": first,
                        "last_name": last,
                        "dob": dob,
                    },
                ),
            ]
        }
    )


def build_device_setup(index: int, rng: random.Random) -> Trajectory:
    """Build mandatory status-check-before-pairing behavior."""
    device, device_id, _ = synthetic_device(index)
    opening = rng.choice(DEVICE_TEMPLATES).format(
        device=device,
        device_id=device_id,
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(opening),
                assistant(
                    "device_setup",
                    f"I will check the status of {device_id} first.",
                    "check_device_status",
                    {"device_id": device_id},
                ),
                user("The tablet says it is not paired yet."),
                assistant(
                    "device_setup",
                    f"I will pair {device_id} with the tablet now.",
                    "pair_device",
                    {"device_id": device_id},
                ),
            ]
        }
    )


def build_troubleshooting(index: int, rng: random.Random) -> Trajectory:
    """Build an unresolved troubleshooting step followed by confirmation."""
    device, device_id, _ = synthetic_device(index)
    failure = rng.choice(
        (
            f"My {device} {device_id} failed to pair twice.",
            f"{device_id} is not connecting to the tablet.",
            f"The Bluetooth light on {device_id} is blinking red.",
        )
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(failure),
                assistant(
                    "device_setup",
                    "Restart the device, then enable pairing mode and try again.",
                    "troubleshoot_step",
                    {"step_id": "restart_and_pair", "resolved": False},
                ),
                user("I restarted it, and the device is connected now."),
                assistant(
                    "troubleshooting",
                    "The connection issue is resolved. We can resume setup.",
                    "troubleshoot_step",
                    {"step_id": "restart_and_pair", "resolved": True},
                ),
            ]
        }
    )


def build_education(index: int, rng: random.Random) -> Trajectory:
    """Build education followed by explicit measurement readiness."""
    device, device_id, measurement_type = synthetic_device(index)
    guidance = {
        "spo2": "Place a warm, still finger fully inside the pulse oximeter.",
        "bp": (
            "Sit quietly for five minutes with feet flat and your arm "
            "supported at heart level."
        ),
        "weight": "Place the scale on a hard, level surface and stand still.",
        "temperature": "Position the thermometer as its device guide directs.",
    }[measurement_type]
    readiness = rng.choice(
        (
            f"I followed those steps and I am ready. Device ID: {device_id}.",
            f"The {device} is positioned. Start the reading on {device_id}.",
            f"I am ready to measure now with {device_id}.",
        )
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(f"How should I use my {device} correctly?"),
                assistant("education", guidance),
                user(readiness),
                assistant(
                    "education",
                    f"I will start the {measurement_type} measurement now.",
                    "start_measurement",
                    {
                        "device_id": device_id,
                        "measurement_type": measurement_type,
                    },
                ),
            ]
        }
    )


def build_closing(index: int, rng: random.Random) -> Trajectory:
    """Build a tool-free closing exchange."""
    device, device_id, _ = synthetic_device(index)
    closing_template = rng.choice(
        (
            "That is all after saving the reading from {device_id}. Thank you.",
            "I am finished with my {device} {device_id} for today.",
            "The reading from {device_id} is saved. I do not need anything else.",
        )
    )
    closing = closing_template.format(device=device, device_id=device_id)
    return Trajectory.model_validate(
        {
            "messages": [
                user(closing),
                assistant(
                    "closing",
                    "You are all set for today. Take care.",
                ),
            ]
        }
    )


def build_safe_vital(index: int, rng: random.Random) -> Trajectory:
    """Build non-interpretive handling for a non-escalated vital."""
    systolic = 105 + index % 15
    diastolic = 65 + index % 10
    _, device_id, _ = synthetic_device(index)
    message_template = rng.choice(
        (
            "My blood pressure from {device_id} is {systolic} over {diastolic}.",
            "The monitor {device_id} shows {systolic}/{diastolic}.",
        )
    )
    message = message_template.format(
        device_id=device_id,
        systolic=systolic,
        diastolic=diastolic,
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(message),
                assistant(
                    "education",
                    "I received the measurement. I cannot interpret or "
                    "characterize the values.",
                ),
            ]
        }
    )


def build_safety_escalation(index: int, rng: random.Random) -> Trajectory:
    """Build deterministic red-flag escalation behavior."""
    spo2 = 82 + index % 8
    _, device_id, _ = synthetic_device(index)
    safety_cases = (
        (
            f"My oxygen reading on {device_id} is {spo2}%.",
            f"Critical SpO2 reading detected: {spo2}%",
        ),
        (
            f"I am having chest pain right now while using {device_id}.",
            "Symptom red flag detected: chest pain",
        ),
        (
            f"I cannot catch my breath while setting up {device_id}.",
            "Symptom red flag detected: shortness of breath",
        ),
    )
    patient_message, reason = rng.choice(safety_cases)
    return Trajectory.model_validate(
        {
            "messages": [
                user(patient_message),
                assistant(
                    "escalated",
                    "I cannot provide medical advice, but this could be urgent. "
                    "Please call your local emergency number now, and I am "
                    "escalating you to a nurse as well.",
                    "escalate_to_nurse",
                    {"reason": reason},
                ),
            ]
        }
    )


def build_golden_path(index: int, rng: random.Random) -> Trajectory:
    """Build one complete, correctly ordered RPM workflow."""
    first, last, dob, display_dob = synthetic_person(index)
    device, device_id, measurement_type = synthetic_device(index)
    full_name = f"{first} {last}"
    opening = rng.choice(ONBOARDING_TEMPLATES).format(
        full_name=full_name,
        display_dob=display_dob,
    )
    device_prompt = rng.choice(DEVICE_TEMPLATES).format(
        device=device,
        device_id=device_id,
    )
    return Trajectory.model_validate(
        {
            "messages": [
                user(opening),
                assistant(
                    "onboarding",
                    "Thank you. I will verify your identity now.",
                    "verify_identity",
                    {
                        "first_name": first,
                        "last_name": last,
                        "dob": dob,
                    },
                ),
                user(device_prompt),
                assistant(
                    "device_setup",
                    f"I will check the status of {device_id}.",
                    "check_device_status",
                    {"device_id": device_id},
                ),
                user("The status says not paired."),
                assistant(
                    "device_setup",
                    f"I will pair {device_id} now.",
                    "pair_device",
                    {"device_id": device_id},
                ),
                user(f"How do I use the {device}?"),
                assistant(
                    "education",
                    f"Position the {device} according to its device guide.",
                ),
                user(f"It is positioned, and I am ready. ID: {device_id}."),
                assistant(
                    "education",
                    f"I will start the {measurement_type} measurement.",
                    "start_measurement",
                    {
                        "device_id": device_id,
                        "measurement_type": measurement_type,
                    },
                ),
                user("The reading is saved. That is all."),
                assistant("closing", "You are all set for today. Take care."),
            ]
        }
    )


Builder = Callable[[int, random.Random], Trajectory]
BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("onboarding", build_onboarding),
    ("partial_onboarding", build_partial_onboarding),
    ("device_setup", build_device_setup),
    ("troubleshooting", build_troubleshooting),
    ("education", build_education),
    ("closing", build_closing),
    ("safe_vital", build_safe_vital),
    ("safety_escalation", build_safety_escalation),
    ("golden_path", build_golden_path),
)


def parse_args() -> argparse.Namespace:
    """Parse deterministic dataset generator options."""
    parser = argparse.ArgumentParser(
        description="Generate validated deterministic RPM ChatML trajectories."
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("--target must be at least 1")
    return args


def main() -> int:
    """Generate a balanced, unique, Pydantic-validated JSONL dataset."""
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    serialized_records: set[str] = set()
    scenario_counts: Counter[str] = Counter()

    for index in range(args.target):
        scenario_name, builder = BUILDERS[index % len(BUILDERS)]
        trajectory = builder(index, rng)
        record = trajectory.model_dump(mode="json")
        serialized = json.dumps(
            record,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if serialized in serialized_records:
            raise RuntimeError(
                f"Duplicate trajectory generated at index {index}: "
                f"{scenario_name}"
            )

        serialized_records.add(serialized)
        records.append(record)
        scenario_counts[scenario_name] += 1

    rng.shuffle(records)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            )

    distribution = " ".join(
        f"{name}={scenario_counts[name]}" for name, _ in BUILDERS
    )
    print(
        f"Generated {len(records)} unique trajectories at {args.output}\n"
        f"Distribution: {distribution}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
