import json
import re
from datetime import datetime

import streamlit as st

from src.agent.inference import RPMAgent, InferenceMetrics
from src.engine.state_machine import RPMStateMachine
from src.engine.interceptor import SafetyInterceptor
from src.tools.registry import ToolRegistry

MEASUREMENT_TYPE_ALIASES = {
    "spo2": "spo2",
    "sp_o2": "spo2",
    "oxygen": "spo2",
    "oximeter": "spo2",
    "pulse_ox": "spo2",
    "pulse_oximeter": "spo2",
    "bp": "bp",
    "blood_pressure": "bp",
    "weight": "weight",
    "scale": "weight",
    "temperature": "temperature",
    "temp": "temperature",
}
CHAT_SPO2_PATTERN = re.compile(
    r"SpO2:\s*([\d.]+)\s*%?.{0,120}?Heart Rate:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
CHAT_BP_PATTERN = re.compile(
    r"(?:BP|blood pressure)[^\d]{0,20}(\d{2,3})\s*/\s*(\d{2,3})",
    re.IGNORECASE,
)


def normalize_measurement_type(mtype: str, readings: dict) -> str:
    """Map tool aliases onto the telemetry card keys."""
    key = re.sub(r"[\s-]+", "_", str(mtype or "unknown").strip().lower())
    if key in MEASUREMENT_TYPE_ALIASES:
        return MEASUREMENT_TYPE_ALIASES[key]
    if "spo2_percent" in readings:
        return "spo2"
    if "systolic_mmhg" in readings:
        return "bp"
    if "weight_kg" in readings:
        return "weight"
    if "temp_celsius" in readings:
        return "temperature"
    return key or "unknown"


def harvest_device_readings(agent_messages: list) -> dict:
    """Copy every start_measurement tool payload into persistent UI state."""
    found: dict = {}
    for message in agent_messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        readings = payload.get("readings")
        if not isinstance(readings, dict) or not readings:
            continue
        mtype = normalize_measurement_type(
            payload.get("measurement_type", ""),
            readings,
        )
        found[mtype] = readings
    return found


def harvest_readings_from_chat(messages: list) -> dict:
    """Backfill the pane when the model quoted a saved reading in prose."""
    found: dict = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        spo2_match = CHAT_SPO2_PATTERN.search(content)
        if spo2_match:
            found["spo2"] = {
                "spo2_percent": float(spo2_match.group(1)),
                "pulse_bpm": int(spo2_match.group(2)),
            }
        bp_match = CHAT_BP_PATTERN.search(content)
        if bp_match:
            found.setdefault(
                "bp",
                {
                    "systolic_mmhg": int(bp_match.group(1)),
                    "diastolic_mmhg": int(bp_match.group(2)),
                },
            )
    return found


st.set_page_config(page_title="RPM Control Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    div[data-testid="column"]:first-of-type { border-right: 1px solid #334155; padding-right: 1.5rem; }
    div[data-testid="column"]:last-of-type { padding-left: 1.5rem; }
    .status-badge-safe { background-color: #064E3B; color: #34D399; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .status-badge-alert { background-color: #7F1D1D; color: #FCA5A5; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .chat-wrap { height: 520px; overflow-y: auto; padding: 0.5rem 0; }
    .msg-row { display: flex; margin-bottom: 0.75rem; }
    .msg-row.user  { justify-content: flex-end; }
    .msg-row.assistant { justify-content: flex-start; }
    .msg-label { font-size: 0.68rem; color: #64748B; margin-bottom: 0.2rem; }
    .msg-bubble { max-width: 82%; padding: 0.6rem 0.9rem; border-radius: 6px; font-size: 0.91rem; line-height: 1.55; word-wrap: break-word; }
    .msg-bubble.user { background-color: #1E3A5F; color: #E2E8F0; }
    .msg-bubble.assistant { background-color: #1E293B; color: #CBD5E1; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-level singletons
# ---------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.dfa = RPMStateMachine()
    st.session_state.interceptor = SafetyInterceptor()
    st.session_state.registry = ToolRegistry()
    st.session_state.agent = RPMAgent()
    st.session_state.messages = [
        {"role": "assistant", "content": "System initialized. Awaiting patient interaction."}
    ]
    st.session_state.last_metrics: InferenceMetrics | None = None
    st.session_state.last_metrics_note: str | None = None
    st.session_state.dfa_state: str = st.session_state.dfa.current_state
    # Keyed by measurement_type, value is the readings dict
    st.session_state.device_readings: dict = {}

harvested = harvest_device_readings(st.session_state.agent.messages)
if harvested:
    st.session_state.device_readings.update(harvested)
elif not st.session_state.device_readings:
    st.session_state.device_readings.update(
        harvest_readings_from_chat(st.session_state.messages)
    )

# ---------------------------------------------------------------------------
chat_pane, telemetry_pane = st.columns([1.35, 0.85], gap="medium")

with chat_pane:
    st.subheader("Multi-Turn RPM Clinical Assistant")
    st.caption(
        f"Active endpoint: {st.session_state.agent.env} / "
        f"{st.session_state.agent.backend.upper()} / "
        f"{st.session_state.agent.model_id}"
    )

    # Plain HTML chat — no st.chat_message(), no icons, no avatars
    rows = ""
    for msg in st.session_state.messages:
        role = msg["role"]
        label = "Patient" if role == "user" else "Agent"
        content = (
            msg["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        rows += (
            f'<div class="msg-row {role}">'
            f'<div><div class="msg-label">{label}</div>'
            f'<div class="msg-bubble {role}">{content}</div></div></div>'
        )

    st.markdown(f'<div class="chat-wrap">{rows}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Enter clinical query or patient telemetry context..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Processing..."):
            result = st.session_state.agent.process_turn(
                prompt,
                st.session_state.dfa,
                st.session_state.registry,
                st.session_state.interceptor,
            )
        st.session_state.messages.append({"role": "assistant", "content": result.message})
        st.session_state.last_metrics = result.metrics
        st.session_state.last_metrics_note = result.metrics_note
        st.session_state.dfa_state = st.session_state.dfa.current_state
        st.session_state.device_readings.update(
            harvest_device_readings(st.session_state.agent.messages)
        )
        if not st.session_state.device_readings:
            st.session_state.device_readings.update(
                harvest_readings_from_chat(st.session_state.messages)
            )

        st.rerun()

with telemetry_pane:
    st.subheader("Telemetry")

    with st.expander("Device Readings", expanded=True):
        readings = st.session_state.device_readings
        if not readings:
            st.caption("No measurements recorded yet.")
        else:
            # SpO2
            if "spo2" in readings:
                r = readings["spo2"]
                c1, c2 = st.columns(2)
                c1.metric("SpO2", f"{r.get('spo2_percent', '—')} %")
                c2.metric("Pulse (SpO2)", f"{r.get('pulse_bpm', '—')} bpm")
            # BP
            if "bp" in readings:
                r = readings["bp"]
                c1, c2 = st.columns(2)
                c1.metric("Systolic BP", f"{r.get('systolic_mmhg', '—')} mmHg")
                c2.metric("Diastolic BP", f"{r.get('diastolic_mmhg', '—')} mmHg")
            # Weight
            if "weight" in readings:
                r = readings["weight"]
                c1, c2 = st.columns(2)
                c1.metric("Weight", f"{r.get('weight_kg', '—')} kg")
                c2.metric("", f"{r.get('weight_lbs', '—')} lbs")
            # Temperature
            if "temperature" in readings:
                r = readings["temperature"]
                c1, c2 = st.columns(2)
                c1.metric("Temperature", f"{r.get('temp_celsius', '—')} °C")
                c2.metric("", f"{r.get('temp_fahrenheit', '—')} °F")
            known = {"spo2", "bp", "weight", "temperature"}
            extras = {key: value for key, value in readings.items() if key not in known}
            if extras:
                st.json(extras)

    with st.expander("Safety Interceptor & DFA State", expanded=True):
        st.markdown("##### DFA Current State")
        st.code(st.session_state.dfa_state, language=None)
        if st.session_state.dfa_state == "escalated":
            badge = '<span class="status-badge-alert">ESCALATED</span>'
        else:
            badge = '<span class="status-badge-safe">NOMINAL</span>'
        st.markdown(f"Status: {badge}", unsafe_allow_html=True)
        st.text("Last Check: " + datetime.now().strftime("%H:%M:%S"))

    with st.expander("Language Model Telemetry", expanded=True):
        m = st.session_state.last_metrics
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Tokens / Sec", f"{m.output_tokens_per_second:.1f}" if m and m.output_tokens_per_second else "—")
            st.metric("Prompt Tokens", str(m.prompt_tokens) if m and m.prompt_tokens else "—")
        with c2:
            st.metric("Latency (TTFT)", f"{m.ttft_ms:.0f} ms" if m and m.ttft_ms else "—")
            st.metric("Completion Tokens", str(m.output_tokens) if m and m.output_tokens else "—")
        if m:
            st.caption(f"Total: {m.total_latency_ms:.0f} ms | Requests: {m.llm_requests}")
        elif st.session_state.get("last_metrics_note"):
            st.caption(st.session_state.last_metrics_note)
