import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.paths import project_path


SPO2_PATTERN = re.compile(
    r"\b(?:spo2|sp\s*o2|o2|oxygen(?:\s+saturation)?|"
    r"(?:pulse\s+)?oximeter)\b"
    r"(?:\s+(?:level|reading|value|result|says|is|at|of))*"
    r"\s*[:=]?\s*(\d{1,3})\s*%?\b",
    re.IGNORECASE,
)
BP_PATTERN = re.compile(
    r"\b(?:bp|blood\s+pressure)\b"
    r"(?:\s+(?:level|reading|value|result|says|is|at|of))*"
    r"\s*[:=]?\s*(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b",
    re.IGNORECASE,
)
CORRECTED_PERCENTAGE_PATTERN = re.compile(
    r"\b(?:actually|correction|corrected|confirmed|sorry|my\s+bad|"
    r"it'?s|its)\b[^\d%]{0,40}(\d{1,3})\s*%",
    re.IGNORECASE,
)


@dataclass
class InterceptResult:
    is_red_flag: bool
    reason: str | None = None
    extracted_vitals: dict[str, int] | None = None


class SafetyInterceptor:
    def __init__(
        self,
        config_path: str | Path = "configs/safety_rules.yaml",
    ) -> None:
        """Load deterministic safety rules and initialize turn-local context."""
        with project_path(config_path).open("r", encoding="utf-8") as config_file:
            config: dict[str, Any] = yaml.safe_load(config_file)

        self.red_flag_keywords: list[str] = config.get("red_flag_keywords", [])
        self.vitals_thresholds: dict[str, dict[str, int]] = config.get(
            "vitals_thresholds",
            {},
        )
        self._previous_vital_type: str | None = None

    def inspect(self, user_text: str) -> InterceptResult:
        """Inspect one turn for symptom and vital-sign escalation triggers."""
        text_lower = user_text.lower()
        previous_vital_type = self._previous_vital_type
        self._previous_vital_type = None

        for keyword in self.red_flag_keywords:
            if keyword.lower() in text_lower:
                return InterceptResult(
                    is_red_flag=True,
                    reason=f"Symptom red flag detected: '{keyword}'",
                )

        spo2_match = SPO2_PATTERN.search(text_lower)
        if spo2_match is None and previous_vital_type == "spo2":
            spo2_match = CORRECTED_PERCENTAGE_PATTERN.search(text_lower)

        if spo2_match:
            self._previous_vital_type = "spo2"
            val = int(spo2_match.group(1))
            spo2_bounds = self.vitals_thresholds.get("spo2", {})
            minimum = spo2_bounds.get("min", 90)
            if val < minimum:
                return InterceptResult(
                    is_red_flag=True,
                    reason=(
                        f"Critical SpO2 reading detected: {val}% "
                        f"(Threshold: < {minimum}%)"
                    ),
                    extracted_vitals={"spo2": val},
                )

        bp_match = BP_PATTERN.search(text_lower)
        if bp_match:
            systolic, diastolic = int(bp_match.group(1)), int(bp_match.group(2))
            sys_max = self.vitals_thresholds.get("systolic_bp", {}).get("max", 180)
            dia_max = self.vitals_thresholds.get("diastolic_bp", {}).get("max", 110)

            if systolic >= sys_max or diastolic >= dia_max:
                return InterceptResult(
                    is_red_flag=True,
                    reason=(
                        "Hypertensive crisis blood pressure reading: "
                        f"{systolic}/{diastolic}"
                    ),
                    extracted_vitals={
                        "systolic": systolic,
                        "diastolic": diastolic,
                    },
                )

        return InterceptResult(is_red_flag=False)
