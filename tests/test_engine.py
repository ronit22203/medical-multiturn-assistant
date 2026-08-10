import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.registry import ToolRegistry
from src.engine.interceptor import SafetyInterceptor


def run_tests():
    print("--- INITIATING SYSTEM TESTS ---\n")
    
    # 1. Test the Safety Interceptor
    print("[TEST 1] Out-of-Band Safety Interceptor")
    interceptor = SafetyInterceptor()
    
    test_cases = [
        "Hello, I am ready to start my onboarding.",
        "I can't breathe very well right now.",
        "My scale says 150 lbs and my SpO2 is 88%.",
        "BP is 195/110, feeling fine otherwise."
    ]
    
    for case in test_cases:
        result = interceptor.inspect(case)
        status = "RED FLAG" if result.is_red_flag else "PASS"
        print(f"Input: '{case}'\n -> {status} | Reason: {result.reason}\n")


    # 2. Test the Tool Registry & Pydantic Validation
    print("--------------------------------------------------")
    print("[TEST 2] Pydantic Schema Validation & Execution")
    registry = ToolRegistry()
    
    # Case A: Valid Input
    valid_payload = {
        "first_name": "John",
        "last_name": "Doe",
        "dob": "1980-05-14"
    }
    print("Executing 'verify_identity' with VALID payload...")
    result_valid = registry.execute_tool("verify_identity", valid_payload)
    print(f" -> Output: {json.dumps(result_valid, indent=2)}\n")
    
    # Case B: Invalid Input (Missing 'dob')
    invalid_payload = {
        "first_name": "Jane",
        "last_name": "Smith"
    }
    print("Executing 'verify_identity' with INVALID payload (missing DOB)...")
    result_invalid = registry.execute_tool("verify_identity", invalid_payload)
    print(f" -> Output: {json.dumps(result_invalid, indent=2)}\n")


if __name__ == "__main__":
    run_tests()
