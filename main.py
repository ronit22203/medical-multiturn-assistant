import sys
from src.engine.state_machine import RPMStateMachine
from src.engine.interceptor import SafetyInterceptor
from src.tools.registry import ToolRegistry
from src.agent.inference import RPMAgent

def main():
    print("Initializing RPM-Agent (Local M4 Execution)...")

    # Instantiate the Control Plane
    dfa = RPMStateMachine()
    interceptor = SafetyInterceptor()
    registry = ToolRegistry()

    # Instantiate the NLU Engine (routing to Ollama via model.yaml)
    try:
        agent = RPMAgent(env="local")
    except Exception as e:
        print(f"Failed to initialize Agent. Check model.yaml and Ollama. Error: {e}")
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
            response = agent.process_turn(user_input, dfa, registry, interceptor)

            print(f"\n[Agent]: {response}")
            print(f"[DFA Tracker]: Currently in {dfa.current_state}")

        except KeyboardInterrupt:
            print("\nTerminating session.")
            break

if __name__ == "__main__":
    main()
