import yaml
from typing import List, Tuple, Dict, Any

class RPMStateMachine:
    def __init__(self, config_path: str = "configs/state_graph.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.states = self.config.get("states", {})
        self.current_state = "1_onboarding"
        
        # Track paired devices to know when to move from Setup -> Education
        self.paired_devices = set()
        self.required_devices = {"pulse_oximeter", "bp_device", "scale", "thermometer"}

    def get_context(self) -> Tuple[str, List[str]]:
        """
        Returns the system prompt and allowed tools for the CURRENT state.
        The LLM is blind to anything outside this context.
        """
        state_data = self.states.get(self.current_state, {})
        prompt = state_data.get("system_prompt", "You are an RPM assistant.")
        allowed_tools = state_data.get("allowed_tools", [])
        return prompt, allowed_tools

    def process_tool_execution(self, tool_name: str, result: Dict[str, Any]) -> str:
        """
        Evaluates a successful tool execution and triggers DFA transitions.
        Returns a system message regarding the transition.
        """
        if result.get("status") != "success":
            return f"[System] Tool {tool_name} failed. Remaining in {self.current_state}."

        prev_state = self.current_state

        # State 1 -> State 2 (Onboarding Complete)
        if self.current_state == "1_onboarding" and tool_name == "verify_identity":
            self.current_state = "2_device_setup"
        
        # State 2 (Device Setup Logic)
        elif self.current_state == "2_device_setup" and tool_name == "pair_device":
            device_id = result.get("device_id", "")
            self.paired_devices.add(device_id)
            # Example condition: if at least 1 device is paired, we can proceed to education,
            # or you can enforce all 4. Let's allow transition to education for demonstration.
            if len(self.paired_devices) > 0: 
                self.current_state = "4_education"

        # State 3 (Troubleshooting) -> Back to Setup
        elif self.current_state == "3_troubleshooting" and tool_name == "troubleshoot_step":
            self.current_state = "2_device_setup"

        # State 4 -> State 5 (Measurement Complete)
        elif self.current_state == "4_education" and tool_name == "start_measurement":
            self.current_state = "5_closing"

        if prev_state != self.current_state:
            return f"[System] Transitioned: {prev_state} -> {self.current_state}"
        return f"[System] State maintained: {self.current_state}"
        
    def force_escalation(self):
        """Called by the Interceptor if a red flag is caught."""
        self.current_state = "escalated"
