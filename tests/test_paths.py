import os
import tempfile
import unittest
from pathlib import Path

from src.engine.interceptor import SafetyInterceptor
from src.engine.state_machine import RPMStateMachine
from src.paths import PROJECT_ROOT, project_path


class ProjectPathTests(unittest.TestCase):
    def test_relative_paths_resolve_from_project_root(self) -> None:
        self.assertEqual(
            project_path("configs/state_graph.yaml"),
            PROJECT_ROOT / "configs/state_graph.yaml",
        )

    def test_default_configs_load_outside_project_directory(self) -> None:
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                dfa = RPMStateMachine()
                interceptor = SafetyInterceptor()
            finally:
                os.chdir(original_directory)

        self.assertEqual(dfa.current_state, "1_onboarding")
        self.assertFalse(interceptor.inspect("Hello").is_red_flag)


if __name__ == "__main__":
    unittest.main()
