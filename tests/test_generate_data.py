import importlib.util
import unittest
from pathlib import Path
from types import ModuleType


def load_generator() -> ModuleType:
    """Load the numerically named generator script as a module."""
    path = Path(__file__).resolve().parents[1] / "scripts/01_generate_data.py"
    spec = importlib.util.spec_from_file_location("generate_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SFTGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()

    def test_records_are_unique_and_cover_every_scenario(self) -> None:
        records = self.generator.generate_records(22, seed=7)

        rows = [self.generator.serialize_record(record) for _, record in records]
        scenarios = {scenario for scenario, _ in records}

        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(scenarios, {name for name, _ in self.generator.BUILDERS})

    def test_tool_calls_use_native_messages_not_json_content(self) -> None:
        records = self.generator.generate_records(22, seed=7)
        tool_messages = [
            message
            for _, record in records
            for message in record.messages
            if message.role == "assistant" and message.tool_calls
        ]

        self.assertTrue(tool_messages)
        for message in tool_messages:
            self.assertIsNone(message.content)
            self.assertEqual(len(message.tool_calls or []), 1)

    def test_qwen_text_renders_tool_arguments_as_objects(self) -> None:
        records = self.generator.generate_records(22, seed=7)
        tool_record = next(
            record
            for _, record in records
            if any(message.tool_calls for message in record.messages)
        )

        self.assertIn("<tool_call>", tool_record.text)
        self.assertIn('"arguments":{', tool_record.text)
        self.assertNotIn('"arguments":"{', tool_record.text)

    def test_splits_are_disjoint_and_complete(self) -> None:
        records = self.generator.generate_records(110, seed=7)
        splits = self.generator.stratified_split(records, seed=8)
        rows_by_split = {
            name: {
                self.generator.serialize_record(record)
                for _, record in split_records
            }
            for name, split_records in splits.items()
        }

        self.assertEqual(
            sum(len(rows) for rows in rows_by_split.values()),
            len(records),
        )
        self.assertTrue(rows_by_split["train"].isdisjoint(rows_by_split["validation"]))
        self.assertTrue(rows_by_split["train"].isdisjoint(rows_by_split["test"]))
        self.assertTrue(
            rows_by_split["validation"].isdisjoint(rows_by_split["test"])
        )


if __name__ == "__main__":
    unittest.main()
