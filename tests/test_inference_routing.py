import unittest
from unittest.mock import patch

from src.agent.inference import (
    RPMAgent,
    collapse_repeated_tool_name,
    complete_streamed_tool_call,
    extract_device_id,
    extract_identity_fields,
    extract_identity_from_history,
    recover_tool_arguments,
    resolve_inference_env,
    sanitize_assistant_message,
    select_forced_tool,
    user_turn_has_identity_fields,
)


class ResolveInferenceEnvTests(unittest.TestCase):
    def test_constructor_arg_wins_over_env_and_yaml(self) -> None:
        resolved = resolve_inference_env(
            env="prod",
            config={"env": "local"},
            environ={"RPM_ENV": "local"},
        )
        self.assertEqual(resolved, "prod")

    def test_rpm_env_wins_over_yaml(self) -> None:
        resolved = resolve_inference_env(
            env=None,
            config={"env": "local"},
            environ={"RPM_ENV": "prod"},
        )
        self.assertEqual(resolved, "prod")

    def test_yaml_env_used_when_no_overrides(self) -> None:
        resolved = resolve_inference_env(
            env=None,
            config={"env": "prod"},
            environ={},
        )
        self.assertEqual(resolved, "prod")

    def test_defaults_to_local(self) -> None:
        resolved = resolve_inference_env(env=None, config={}, environ={})
        self.assertEqual(resolved, "local")

    @patch("src.agent.inference.OpenAI")
    def test_agent_constructor_override_selects_prod(self, mock_openai) -> None:
        agent = RPMAgent(env="prod")
        self.assertEqual(agent.env, "prod")
        self.assertEqual(agent.backend, "vllm")
        mock_openai.assert_called_once()
        self.assertEqual(
            mock_openai.call_args.kwargs["base_url"],
            "http://localhost:8000/v1",
        )

    @patch.dict("os.environ", {"RPM_ENV": ""})
    @patch.object(RPMAgent, "_get_ollama_model_size", return_value=1)
    @patch("src.agent.inference.OpenAI")
    def test_agent_honors_yaml_when_unspecified(
        self,
        _mock_openai,
        _mock_size,
    ) -> None:
        agent = RPMAgent()
        self.assertEqual(agent.env, "local")
        self.assertEqual(agent.backend, "ollama")


class ForcedToolSelectionTests(unittest.TestCase):
    def test_onboarding_does_not_force_on_greeting(self) -> None:
        self.assertIsNone(
            select_forced_tool("1_onboarding", ["verify_identity"], "Hi"),
        )

    def test_onboarding_forces_verify_identity_when_name_and_dob_present(
        self,
    ) -> None:
        self.assertTrue(
            user_turn_has_identity_fields(
                "my name is ronit saxena, dob 22-03-2002",
            ),
        )
        self.assertEqual(
            select_forced_tool(
                "1_onboarding",
                ["verify_identity"],
                "my name is ronit saxena, dob 2002-03-22",
            ),
            "verify_identity",
        )

    def test_device_setup_does_not_force_tool_choice(self) -> None:
        self.assertIsNone(
            select_forced_tool(
                "2_device_setup",
                ["check_device_status", "pair_device", "troubleshoot_step"],
                "I'm ready to pair my devices. I have a pulse oximeter with ID PO-9821",
                checked_devices=set(),
            ),
        )

    def test_education_does_not_force_tool_choice(self) -> None:
        self.assertIsNone(
            select_forced_tool(
                "4_education",
                ["start_measurement", "troubleshoot_step"],
                "lets continue",
            ),
        )

    def test_controller_owned_turn_does_not_force(self) -> None:
        self.assertIsNone(
            select_forced_tool("1_onboarding", [], "my name is ronit saxena"),
        )


class HermesToolCallCompletionTests(unittest.TestCase):
    def test_injects_forced_name_and_mock_id(self) -> None:
        completed = complete_streamed_tool_call(
            {"id": "", "name": "", "arguments": '{"first_name": "Ronit"}'},
            forced_tool="verify_identity",
        )
        self.assertEqual(
            completed,
            {
                "id": "forced-verify_identity",
                "name": "verify_identity",
                "arguments": '{"first_name": "Ronit"}',
            },
        )

    def test_incomplete_without_forced_tool_returns_none(self) -> None:
        completed = complete_streamed_tool_call(
            {"id": "", "name": "", "arguments": "{}"},
            forced_tool=None,
        )
        self.assertIsNone(completed)

    def test_preserves_streamed_name_and_id(self) -> None:
        completed = complete_streamed_tool_call(
            {
                "id": "call_abc",
                "name": "verify_identity",
                "arguments": "{}",
            },
            forced_tool="verify_identity",
        )
        self.assertEqual(completed["id"], "call_abc")
        self.assertEqual(completed["name"], "verify_identity")

    def test_collapses_repeated_check_device_status_name(self) -> None:
        completed = complete_streamed_tool_call(
            {
                "id": "",
                "name": "check_device_status" * 20,
                "arguments": "",
            },
            forced_tool=None,
        )
        self.assertIsNotNone(completed)
        self.assertEqual(completed["name"], "check_device_status")
        self.assertEqual(completed["id"], "forced-check_device_status")


class IdentityExtractionTests(unittest.TestCase):
    def test_extracts_european_dob_from_trace(self) -> None:
        extracted = extract_identity_fields(
            "my name is ronit saxena , dob 22-03-2002",
        )
        self.assertEqual(
            extracted,
            {
                "first_name": "Ronit",
                "last_name": "Saxena",
                "dob": "2002-03-22",
            },
        )

    def test_extracts_iso_dob_with_typo_birth(self) -> None:
        extracted = extract_identity_fields(
            "My name is ronit saxena my date of birith is 2002-03-22",
        )
        self.assertEqual(
            extracted,
            {
                "first_name": "Ronit",
                "last_name": "Saxena",
                "dob": "2002-03-22",
            },
        )

    def test_history_recovers_identity_when_current_turn_lacks_fields(
        self,
    ) -> None:
        extracted = extract_identity_from_history(
            [
                {
                    "role": "user",
                    "content": "my name is ronit saxena , dob 22-03-2002",
                }
            ],
            "i am the patient",
        )
        self.assertEqual(extracted["first_name"], "Ronit")
        self.assertEqual(extracted["dob"], "2002-03-22")


class SafetySanitizerTests(unittest.TestCase):
    def test_replaces_safety_rule_dump(self) -> None:
        dumped = (
            "I cannot provide medical advice, diagnose, or characterize any "
            "vital sign as normal. Vital-sign interpretation and escalation "
            "are exclusively handled by deterministic safety controls. "
        ) * 3
        message = sanitize_assistant_message(dumped, "1_onboarding")
        self.assertEqual(
            message,
            "Please provide your first name, last name, and date of birth.",
        )

    def test_replaces_second_patient_hallucination(self) -> None:
        message = sanitize_assistant_message(
            "Thank you, Ronit. Please provide the same information for a "
            "second patient.",
            "2_device_setup",
        )
        self.assertIn("pulse_oximeter", message)


class DeterministicOnboardingTests(unittest.TestCase):
    def test_name_and_dob_advance_dfa_without_model_tool_call(self) -> None:
        from openai import OpenAIError
        from unittest.mock import MagicMock

        from src.engine.interceptor import SafetyInterceptor
        from src.engine.state_machine import RPMStateMachine
        from src.tools.registry import ToolRegistry

        agent = RPMAgent.__new__(RPMAgent)
        agent.messages = []
        agent.model_id = "test-model"
        agent.temperature = 0.0
        agent.max_tokens = 64
        agent.memory_bandwidth_gbps = None
        agent.model_size_bytes = None
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = OpenAIError("down")

        dfa = RPMStateMachine()
        result = agent.process_turn(
            "my name is ronit saxena , dob 22-03-2002",
            dfa,
            ToolRegistry(),
            SafetyInterceptor(),
        )

        self.assertEqual(dfa.current_state, "2_device_setup")
        self.assertEqual(result.response.state, "device_setup")
        self.assertIsNotNone(result.response.tool_call)
        self.assertEqual(result.response.tool_call.name, "verify_identity")
        self.assertEqual(
            result.response.tool_call.arguments["dob"],
            "2002-03-22",
        )
        self.assertNotIn("second patient", result.message.lower())
        self.assertNotIn("i cannot provide medical advice", result.message.lower())


class DeviceSetupControllerTests(unittest.TestCase):
    def test_extracts_assignment_and_trace_device_ids(self) -> None:
        self.assertEqual(extract_device_id("Device ID is OXI-1023."), "OXI-1023")
        self.assertEqual(
            extract_device_id(
                "I'm ready to pair my devices. I have a pulse oximeter with ID PO-9821"
            ),
            "PO-9821",
        )

    def test_collapses_name_repetition_loop(self) -> None:
        self.assertEqual(
            collapse_repeated_tool_name("check_device_status" * 8),
            "check_device_status",
        )

    def test_recovers_device_id_when_arguments_are_empty(self) -> None:
        recovered = recover_tool_arguments(
            "check_device_status",
            "",
            [
                {
                    "role": "user",
                    "content": "pulse oximeter with ID PO-9821",
                }
            ],
            "Yes, check that device status.",
        )
        self.assertEqual(recovered, {"device_id": "PO-9821"})

    def test_replaces_tool_permission_prompt(self) -> None:
        message = sanitize_assistant_message(
            'I cannot call check_device_status. Please try again with "Yes, '
            'check that device status." or "No, do not check that device status."',
            "2_device_setup",
        )
        self.assertIn("device", message.lower())
        self.assertNotIn("i cannot call", message.lower())

    def test_device_id_with_pair_intent_checks_then_pairs(self) -> None:
        from openai import OpenAIError
        from unittest.mock import MagicMock

        from src.engine.interceptor import SafetyInterceptor
        from src.engine.state_machine import RPMStateMachine
        from src.tools.registry import ToolRegistry

        agent = RPMAgent.__new__(RPMAgent)
        agent.messages = []
        agent.model_id = "test-model"
        agent.temperature = 0.0
        agent.max_tokens = 64
        agent.memory_bandwidth_gbps = None
        agent.model_size_bytes = None
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = OpenAIError("down")

        dfa = RPMStateMachine()
        dfa.current_state = "2_device_setup"
        result = agent.process_turn(
            "I'm ready to pair my devices. I have a pulse oximeter with ID PO-9821",
            dfa,
            ToolRegistry(),
            SafetyInterceptor(),
        )

        self.assertIn("PO-9821", dfa.checked_devices)
        self.assertIn("PO-9821", dfa.paired_devices)
        self.assertEqual(dfa.current_state, "4_education")
        self.assertEqual(result.response.tool_call.name, "pair_device")
        self.assertNotIn("i cannot call", result.message.lower())


if __name__ == "__main__":
    unittest.main()
