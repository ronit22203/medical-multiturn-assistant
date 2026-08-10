import re
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class InterceptResult:
    is_red_flag: bool
    reason: Optional[str] = None
    extracted_vitals: Optional[dict] = None


class SafetyInterceptor:
    def __init__(self, config_path: str = "configs/safety_rules.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.red_flag_keywords = self.config.get("red_flag_keywords", [])
        self.vitals_thresholds = self.config.get("vitals_thresholds", {})

    def inspect(self, user_text: str) -> InterceptResult:
        text_lower = user_text.lower()

        # 1. Keyword Scan (Direct Red Flags)
        for keyword in self.red_flag_keywords:
            if keyword.lower() in text_lower:
                return InterceptResult(
                    is_red_flag=True,
                    reason=f"Symptom red flag detected: '{keyword}'"
                )

        # 2. Vital Threshold Extraction via Regex
        # Scan for SpO2 values (e.g., "SpO2 88", "spo2: 88%")
        spo2_match = re.search(r"(?:spo2|oximeter|oxygen)\s*(?:is|:|=)?\s*(\d{2,3})%?", text_lower)
        if spo2_match:
            val = int(spo2_match.group(1))
            spo2_bounds = self.vitals_thresholds.get("spo2", {})
            if val < spo2_bounds.get("min", 90):
                return InterceptResult(
                    is_red_flag=True,
                    reason=f"Critical SpO2 reading detected: {val}% (Threshold: < {spo2_bounds.get('min')}%)",
                    extracted_vitals={"spo2": val}
                )

        # Scan for Systolic/Diastolic BP (e.g., "190/115", "BP is 185/100")
        bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text_lower)
        if bp_match:
            systolic, diastolic = int(bp_match.group(1)), int(bp_match.group(2))
            sys_max = self.vitals_thresholds.get("systolic_bp", {}).get("max", 180)
            dia_max = self.vitals_thresholds.get("diastolic_bp", {}).get("max", 110)

            if systolic >= sys_max or diastolic >= dia_max:
                return InterceptResult(
                    is_red_flag=True,
                    reason=f"Hypertensive crisis blood pressure reading: {systolic}/{diastolic}",
                    extracted_vitals={"systolic": systolic, "diastolic": diastolic}
                )

        return InterceptResult(is_red_flag=False)
