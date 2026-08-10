import json
import unittest

from src.agent.inference import RPMAgent, ResponseToolCall


class StructuredResponseTests(unittest.TestCase):
    def test_tool_response_matches_assignment_schema(self) -> None:
        result = RPMAgent._turn_result(
            state="1_onboarding",
            message="Thank you. Let me verify your identity now.",
            tool_call=ResponseToolCall(
                name="verify_identity",
                arguments={
                    "first_name": "Emily",
                    "last_name": "Davis",
                    "dob": "1959-12-01",
                },
            ),
        )

        self.assertEqual(
            json.loads(result.model_dump_json()),
            {
                "state": "onboarding",
                "assistant_message": (
                    "Thank you. Let me verify your identity now."
                ),
                "tool_call": {
                    "name": "verify_identity",
                    "arguments": {
                        "first_name": "Emily",
                        "last_name": "Davis",
                        "dob": "1959-12-01",
                    },
                },
            },
        )

    def test_text_response_uses_stable_nullable_tool_field(self) -> None:
        result = RPMAgent._turn_result(
            state="5_closing",
            message="You are all set for today.",
        )

        self.assertEqual(
            json.loads(result.model_dump_json()),
            {
                "state": "closing",
                "assistant_message": "You are all set for today.",
                "tool_call": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
