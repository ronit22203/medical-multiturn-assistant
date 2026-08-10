from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError

# --- Tool Schemas (Pydantic Models) ---

class VerifyIdentityInput(BaseModel):
    first_name: str = Field(..., description="First name of the patient")
    last_name: str = Field(..., description="Last name of the patient")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD or MM/DD/YYYY format")

class PairDeviceInput(BaseModel):
    device_id: str = Field(..., description="Unique hardware identifier or name (e.g., pulse_ox_01, bp_monitor)")

class StartMeasurementInput(BaseModel):
    device_id: str = Field(..., description="Target device identifier")
    measurement_type: str = Field(..., description="Type of measurement: spo2, bp, weight, or temperature")

class TroubleshootStepInput(BaseModel):
    step_id: str = Field(..., description="Troubleshooting step identifier or action taken")
    resolved: bool = Field(
        default=False,
        description=(
            "True only when the user explicitly confirms the device issue is fixed"
        ),
    )

class EscalateToNurseInput(BaseModel):
    reason: str = Field(..., description="Detailed clinical reason or red flag that triggered escalation")


# --- Mock Execution Functions ---

def verify_identity(first_name: str, last_name: str, dob: str) -> Dict[str, Any]:
    # Mock system check
    return {
        "status": "success",
        "verified": True,
        "message": f"Identity verified for {first_name} {last_name}, DOB: {dob}."
    }

def pair_device(device_id: str) -> Dict[str, Any]:
    return {
        "status": "success",
        "paired": True,
        "device_id": device_id,
        "message": f"Device '{device_id}' successfully paired with tablet."
    }

def start_measurement(device_id: str, measurement_type: str) -> Dict[str, Any]:
    return {
        "status": "success",
        "device_id": device_id,
        "measurement_type": measurement_type,
        "reading_status": "in_progress",
        "message": f"Started {measurement_type} measurement on {device_id}."
    }

def troubleshoot_step(step_id: str, resolved: bool = False) -> Dict[str, Any]:
    return {
        "status": "success",
        "step_id": step_id,
        "resolved": resolved,
        "message": f"Troubleshooting step '{step_id}' executed successfully."
    }

def escalate_to_nurse(reason: str) -> Dict[str, Any]:
    return {
        "status": "escalated",
        "action": "nurse_notified",
        "reason": reason,
        "message": f"CRITICAL: Escalated to on-call nurse. Reason: {reason}"
    }
