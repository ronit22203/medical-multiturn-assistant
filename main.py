import sys

from src.agent.inference import InferenceMetrics, RPMAgent
from src.engine.state_machine import RPMStateMachine
from src.engine.interceptor import SafetyInterceptor
from src.tools.registry import ToolRegistry


def format_metrics(metrics: InferenceMetrics) -> str:
    """Format per-turn LLM telemetry for terminal output."""
    ttft = f"{metrics.ttft_ms:.1f} ms" if metrics.ttft_ms is not None else "N/A"
    tpot = (
        f"{metrics.tpot_ms:.1f} ms/token"
        if metrics.tpot_ms is not None
        else "N/A"
    )
    throughput = (
        f"{metrics.output_tokens_per_second:.2f} tok/s"
        if metrics.output_tokens_per_second is not None
        else "N/A"
    )
    estimated_mbu = (
        f"{metrics.estimated_mbu_percent:.1f}%"
        if metrics.estimated_mbu_percent is not None
        else "N/A"
    )
    prompt_tokens = (
        str(metrics.prompt_tokens) if metrics.prompt_tokens is not None else "N/A"
    )
    output_tokens = (
        str(metrics.output_tokens) if metrics.output_tokens is not None else "N/A"
    )
    total_tokens = (
        str(metrics.total_tokens) if metrics.total_tokens is not None else "N/A"
    )

    return (
        f"Requests: {metrics.llm_requests} | TTFT: {ttft} | "
        f"TPOT: {tpot} | Decode: {throughput} | "
        f"Latency: {metrics.total_latency_ms:.1f} ms | "
        f"Tokens: {prompt_tokens} in / {output_tokens} out / "
        f"{total_tokens} total | Estimated MBU: {estimated_mbu}"
    )


def main() -> None:
    """Run the interactive RPM agent CLI."""
    print("Initializing RPM-Agent (Local M4 Execution)...")

    # Instantiate the Control Plane
    dfa = RPMStateMachine()
    interceptor = SafetyInterceptor()
    registry = ToolRegistry()

    # Instantiate the NLU Engine (routing to Ollama via model.yaml)
    try:
        agent = RPMAgent(env="local")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(
            "Failed to initialize Agent. Check model.yaml and Ollama. "
            f"Error: {exc}"
        )
        sys.exit(1)

    print(f"\nSystem Online. DFA Initialized at: [{dfa.current_state}]")
    print("Type 'exit' to terminate the session.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n[Patient]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Terminating session.")
                break

            # Route input through the architecture
            result = agent.process_turn(user_input, dfa, registry, interceptor)

            print(f"\n[Agent]: {result.model_dump_json()}")
            print(f"[DFA Tracker]: Currently in {dfa.current_state}")
            if result.metrics is not None:
                print(f"[Performance]: {format_metrics(result.metrics)}")
            elif result.metrics_note is not None:
                print(f"[Performance]: {result.metrics_note}")

        except KeyboardInterrupt:
            print("\nTerminating session.")
            break

if __name__ == "__main__":
    main()
