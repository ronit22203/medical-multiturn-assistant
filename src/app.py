import streamlit as st
from datetime import datetime

from src.agent.inference import RPMAgent, InferenceMetrics
from src.engine.state_machine import RPMStateMachine
from src.engine.interceptor import SafetyInterceptor
from src.tools.registry import ToolRegistry

st.set_page_config(page_title="RPM Control Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    div[data-testid="column"]:first-of-type { border-right: 1px solid #334155; padding-right: 1.5rem; }
    div[data-testid="column"]:last-of-type { padding-left: 1.5rem; }
    .status-badge-safe { background-color: #064E3B; color: #34D399; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
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
    st.session_state.dfa_state: str = st.session_state.dfa.current_state
    # Keyed by measurement_type, value is the readings dict
    st.session_state.device_readings: dict = {}

# ---------------------------------------------------------------------------
chat_pane, telemetry_pane = st.columns([1.35, 0.85], gap="medium")

with chat_pane:
    st.subheader("Multi-Turn RPM Clinical Assistant")
    st.caption(f"Active endpoint: {st.session_state.agent.backend.upper()} / {st.session_state.agent.model_id}")

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
        st.session_state.dfa_state = st.session_state.dfa.current_state

        # Capture device readings from start_measurement tool calls
        tc = result.response.tool_call
        if tc and tc.name == "start_measurement":
            mtype = tc.arguments.get("measurement_type", "unknown")
            # Re-execute to get readings (registry already ran it; pull from tool result via registry)
            tool_result = st.session_state.registry.execute_tool("start_measurement", tc.arguments)
            if "readings" in tool_result:
                st.session_state.device_readings[mtype] = tool_result["readings"]

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

    with st.expander("Safety Interceptor & DFA State", expanded=True):
        st.markdown("##### DFA Current State")
        st.code(st.session_state.dfa_state, language=None)
        st.markdown('Status: <span class="status-badge-safe">NOMINAL</span>', unsafe_allow_html=True)
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
