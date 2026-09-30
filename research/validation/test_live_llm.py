import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import BASE_DIR
from modules.llm_agents import LLMProvider, Agent1Analysis
from modules.feedback import get_feedback

def run_diagnostics():
    print("=== Phase 8 Diagnostic Test ===")
    
    # 1. Load .env explicitly
    env_path = BASE_DIR / ".env"
    env_loaded = load_dotenv(dotenv_path=env_path)
    print(f"Environment: {'PASS' if env_loaded else 'FAIL'} (loaded {env_path})")
    
    # 2. Verify key without printing
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        print(f"API initialization: PASS (Key length: {len(api_key)})")
    else:
        print("API initialization: FAIL (LLM_API_KEY not found)")
        return
        
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'gemini')}")
    print(f"Configured Model: {os.environ.get('LLM_MODEL', 'Not set')}")
    
    # 3. Initialize LLMProvider
    try:
        provider = LLMProvider()
    except Exception as e:
        print(f"Provider Initialization Failed: {e}")
        return
        
    # 4. Perform ONE Agent 1 structured request
    print("\nTesting single Agent 1 call for basic connectivity...")
    context = {
        "topic": "Java Core",
        "question": "What is OOP?",
        "initial_response": "Object oriented programming.",
        "followup_question": "Pillars?",
        "followup_response": "Abstraction, Encapsulation.",
        "score": 5.0,
        "learner_level": "Beginner"
    }
    
    try:
        from modules.llm_agents import _agent_1_analyzer
        _agent_1_analyzer(provider, context)
        print("Agent 1 Connectivity: PASS")
    except Exception as e:
        print(f"Agent 1 Connectivity: FAIL ({e})")
        return
        
    # 5. Run full 4-agent pipeline
    print("\nRunning full 4-agent pipeline...")
    start_time = time.time()
    
    # Reset tracking
    LLMProvider._total_requests = 0
    LLMProvider._working_model = None
    
    feedback = get_feedback(
        question="What is OOP?",
        initial_response="Object oriented programming.",
        followup_question="Pillars?",
        followup_response="Abstraction, Encapsulation.",
        score=5.0,
        learner_level="Beginner",
        topic="Java Core",
        use_llm=True
    )
    
    elapsed = time.time() - start_time
    mode = feedback.get("mode", "UNKNOWN")
    
    print("\n=== Final Results ===")
    print(f"Agent 1: {'PASS' if mode == 'Multi-Agent LLM' else 'FAIL'}")
    print(f"Agent 2: {'PASS' if mode == 'Multi-Agent LLM' else 'FAIL'}")
    print(f"Agent 3: {'PASS' if mode == 'Multi-Agent LLM' else 'FAIL'}")
    print(f"Agent 4: {'PASS' if mode == 'Multi-Agent LLM' else 'FAIL'}")
    print(f"Final mode: {mode}")
    print(f"Actual model used: {feedback.get('working_model', 'N/A')}")
    print(f"Total requests: {LLMProvider._total_requests}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    if "fallback_reason" in feedback:
        print(f"Fallback reason: {feedback['fallback_reason']}")

if __name__ == "__main__":
    run_diagnostics()
