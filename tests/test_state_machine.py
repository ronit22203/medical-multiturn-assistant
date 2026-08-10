import unittest

from src.agent.inference import DEVICE_ISSUE_PATTERN
from src.engine.state_machine import RPMStateMachine


class RPMStateMachineTests(unittest.TestCase):
    def test_device_issue_language_is_detected(self) -> None:
        issue_messages = (
            "The Bluetooth light is blinking red.",
            "My monitor is not connecting.",
            "The scale is offline.",
        )

        for message in issue_messages:
            with self.subTest(message=message):
                self.assertIsNotNone(DEVICE_ISSUE_PATTERN.search(message))

    def test_ready_device_language_is_not_an_issue(self) -> None:
        self.assertIsNone(
            DEVICE_ISSUE_PATTERN.search(
                "The blood pressure monitor is connected and ready."
            )
        )

    def test_device_setup_exposes_troubleshooting(self) -> None:
        dfa = RPMStateMachine()
        dfa.current_state = "2_device_setup"

        _, allowed_tools = dfa.get_context()

        self.assertEqual(
            allowed_tools,
            ["pair_device", "troubleshoot_step"],
        )

    def test_setup_troubleshooting_resumes_device_setup(self) -> None:
        dfa = RPMStateMachine()
        dfa.current_state = "2_device_setup"

        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": False},
        )
        self.assertEqual(dfa.current_state, "3_troubleshooting")

        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": True},
        )
        self.assertEqual(dfa.current_state, "2_device_setup")

    def test_education_troubleshooting_resumes_education(self) -> None:
        dfa = RPMStateMachine()
        dfa.current_state = "4_education"

        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": False},
        )
        self.assertEqual(dfa.current_state, "3_troubleshooting")

        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": False},
        )
        self.assertEqual(dfa.current_state, "3_troubleshooting")

        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": True},
        )
        self.assertEqual(dfa.current_state, "4_education")

    def test_measurement_still_transitions_to_closing(self) -> None:
        dfa = RPMStateMachine()
        dfa.current_state = "4_education"

        dfa.process_tool_execution(
            "start_measurement",
            {"status": "success"},
        )

        self.assertEqual(dfa.current_state, "5_closing")

    def test_troubleshooting_resolution_does_not_start_measurement(self) -> None:
        dfa = RPMStateMachine()
        dfa.current_state = "4_education"
        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": False},
        )

        dfa.process_tool_execution(
            "troubleshoot_step",
            {"status": "success", "resolved": True},
        )

        self.assertEqual(dfa.current_state, "4_education")


if __name__ == "__main__":
    unittest.main()
