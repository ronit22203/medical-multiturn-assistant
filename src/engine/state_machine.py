from pathlib import Path
from typing import Any

import yaml

from src.paths import project_path


class RPMStateMachine:
    def __init__(
        self,
        config_path: str | Path = "configs/state_graph.yaml",
    ) -> None:
        """Load the workflow graph and initialize deterministic session state."""
        with project_path(config_path).open("r", encoding="utf-8") as config_file:
            config: dict[str, Any] = yaml.safe_load(config_file)

        self.states: dict[str, dict[str, Any]] = config.get("states", {})
        self.current_state = "1_onboarding"
        self.checked_devices: set[str] = set()
        self.paired_devices: set[str] = set()
        self.required_devices = {
            "pulse_oximeter",
            "bp_device",
            "scale",
            "thermometer",
        }
        self._troubleshooting_return_state = "2_device_setup"

    def get_context(self) -> tuple[str, list[str]]:
        """Return the current state's system prompt and permitted tools."""
        state_data = self.states.get(self.current_state, {})
        prompt = state_data.get("system_prompt", "You are an RPM assistant.")
        allowed_tools = list(state_data.get("allowed_tools", []))

        if self.current_state == "2_device_setup":
            if self.checked_devices:
                checked_ids = ", ".join(sorted(self.checked_devices))
                prompt = (
                    f"{prompt.rstrip()}\nStatus already checked for device IDs: "
                    f"{checked_ids}."
                )
            else:
                allowed_tools = [
                    tool for tool in allowed_tools if tool != "pair_device"
                ]

        return prompt, allowed_tools

    def process_tool_execution(
        self,
        tool_name: str,
        result: dict[str, Any],
    ) -> str:
        """Apply a successful tool result to the deterministic workflow graph."""
        if result.get("status") != "success":
            return (
                f"[System] Tool {tool_name} failed. "
                f"Remaining in {self.current_state}."
            )

        previous_state = self.current_state

        if self.current_state == "1_onboarding" and tool_name == "verify_identity":
            self.current_state = "2_device_setup"

        elif self.current_state == "2_device_setup":
            if tool_name == "check_device_status":
                device_id = result.get("device_id", "")
                if isinstance(device_id, str) and device_id:
                    self.checked_devices.add(device_id)
            elif tool_name == "pair_device":
                device_id = result.get("device_id", "")
                if (
                    isinstance(device_id, str)
                    and device_id
                    and device_id in self.checked_devices
                ):
                    self.paired_devices.add(device_id)
                if self.paired_devices:
                    self.current_state = "4_education"
            elif tool_name == "troubleshoot_step":
                self._enter_troubleshooting(return_state="2_device_setup")

        elif self.current_state == "4_education":
            if tool_name == "start_measurement":
                self.current_state = "5_closing"
            elif tool_name == "troubleshoot_step":
                self._enter_troubleshooting(return_state="4_education")

        elif (
            self.current_state == "3_troubleshooting"
            and tool_name == "troubleshoot_step"
            and result.get("resolved") is True
        ):
            self.current_state = self._troubleshooting_return_state

        if previous_state != self.current_state:
            return (
                f"[System] Transitioned: {previous_state} -> "
                f"{self.current_state}"
            )
        return f"[System] State maintained: {self.current_state}"

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Return an error when a tool call violates workflow prerequisites."""
        if self.current_state == "2_device_setup" and tool_name == "pair_device":
            device_id = arguments.get("device_id")
            if not isinstance(device_id, str) or device_id not in self.checked_devices:
                return (
                    "pair_device requires a successful check_device_status call "
                    "for the same device_id"
                )
        return None

    def _enter_troubleshooting(self, return_state: str) -> None:
        """Enter troubleshooting while preserving the interrupted workflow state."""
        self._troubleshooting_return_state = return_state
        self.current_state = "3_troubleshooting"

    def force_escalation(self) -> None:
        """Move the session into its terminal clinical escalation state."""
        self.current_state = "escalated"
