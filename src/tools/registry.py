import json
from typing import Dict, Any, Callable, Tuple
from src.tools.definitions import (
    VerifyIdentityInput, CheckDeviceStatusInput, PairDeviceInput, StartMeasurementInput,
    TroubleshootStepInput, EscalateToNurseInput,
    verify_identity, check_device_status, pair_device, start_measurement,
    troubleshoot_step, escalate_to_nurse
)

class ToolRegistry:
    def __init__(self):
        # Map tool names to (function, pydantic_schema)
        self._registry: Dict[str, Tuple[Callable, Any]] = {
            "verify_identity": (verify_identity, VerifyIdentityInput),
            "check_device_status": (check_device_status, CheckDeviceStatusInput),
            "pair_device": (pair_device, PairDeviceInput),
            "start_measurement": (start_measurement, StartMeasurementInput),
            "troubleshoot_step": (troubleshoot_step, TroubleshootStepInput),
            "escalate_to_nurse": (escalate_to_nurse, EscalateToNurseInput),
        }

    def execute_tool(self, tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validates arguments via Pydantic and invokes the corresponding function."""
        if tool_name not in self._registry:
            return {
                "status": "error",
                "message": f"Unknown tool: '{tool_name}'"
            }

        func, schema_cls = self._registry[tool_name]

        try:
            # Validate input types dynamically
            validated_args = schema_cls(**raw_args)
            # Unpack validated Pydantic model into python kwarg dict
            return func(**validated_args.model_dump())
        except Exception as e:
            return {
                "status": "error",
                "message": f"Schema validation failed for tool '{tool_name}': {str(e)}"
            }

    def get_tool_schemas(self, allowed_tools: list[str]) -> list[Dict[str, Any]]:
        """Returns OpenAI-compatible function schemas strictly for the allowed tool list."""
        schemas = []
        for tool_name in allowed_tools:
            if tool_name in self._registry:
                _, schema_cls = self._registry[tool_name]
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": schema_cls.__doc__ or f"Execute {tool_name}",
                        "parameters": schema_cls.model_json_schema()
                    }
                })
        return schemas
