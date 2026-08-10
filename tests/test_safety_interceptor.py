import unittest

from src.agent.inference import RPMAgent
from src.engine.interceptor import SafetyInterceptor
from src.engine.state_machine import RPMStateMachine
from src.tools.registry import ToolRegistry


class SafetyInterceptorTests(unittest.TestCase):
    def test_spo2_with_filler_words_escalates(self) -> None:
        interceptor = SafetyInterceptor()

        result = interceptor.inspect("my SpO2 level is 21")

        self.assertTrue(result.is_red_flag)
        self.assertEqual(result.extracted_vitals, {"spo2": 21})

    def test_safe_spo2_does_not_escalate(self) -> None:
        interceptor = SafetyInterceptor()

        result = interceptor.inspect("my oxygen reading is 92%")

        self.assertFalse(result.is_red_flag)

    def test_immediate_spo2_correction_uses_previous_turn_context(self) -> None:
        interceptor = SafetyInterceptor()
        interceptor.inspect("my SpO2 level is 92%")

        result = interceptor.inspect("Ure right my bad, its 80% confirmed")

        self.assertTrue(result.is_red_flag)
        self.assertEqual(result.extracted_vitals, {"spo2": 80})

    def test_unrelated_percentage_is_not_treated_as_spo2(self) -> None:
        interceptor = SafetyInterceptor()
        interceptor.inspect("my SpO2 level is 92%")

        result = interceptor.inspect("the device battery is 80%")

        self.assertFalse(result.is_red_flag)

    def test_bp_with_filler_words_and_over_escalates(self) -> None:
        interceptor = SafetyInterceptor()

        result = interceptor.inspect("my BP reading is 190 over 115")

        self.assertTrue(result.is_red_flag)
        self.assertEqual(
            result.extracted_vitals,
            {"systolic": 190, "diastolic": 115},
        )

    def test_escalated_state_never_calls_llm(self) -> None:
        agent = RPMAgent.__new__(RPMAgent)
        dfa = RPMStateMachine()
        dfa.force_escalation()

        result = agent.process_turn(
            "Ure right my bad, its 80% confirmed",
            dfa,
            ToolRegistry(),
            SafetyInterceptor(),
        )

        self.assertIsNone(result.metrics)
        self.assertEqual(result.metrics_note, "LLM bypassed because escalation is terminal")


if __name__ == "__main__":
    unittest.main()
