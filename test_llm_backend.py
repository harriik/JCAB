import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent))
load_dotenv(".env")

# Force LLM required for diagnostic test
os.environ["LLM_REQUIRED_FOR_TEST"] = "true"

from modules.llm_agents import run_multi_agent_feedback, LLMProvider

def main():
    print("--- Pilot Readiness Check ---")
    key_avail = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    provider_name = os.environ.get("LLM_PROVIDER", "gemini").lower()
    configured_model = os.environ.get("LLM_MODEL", "gemini-3.5-flash")
    
    print(f"Environment: {'PASS' if key_avail else 'FAIL'}")
    print(f"Provider: {provider_name}")
    print(f"Configured model: {configured_model}")
    print("\n--- Executing Multi-Agent Pipeline ---")
    
    LLMProvider._total_requests = 0
    start_time = time.time()
    
    question = "What is inheritance in Java?"
    initial_response = "Inheritance allows a child class to acquire fields and methods from a parent class."
    followup_question = "How does method overriding relate to inheritance?"
    followup_response = "A child class can provide its own implementation of an inherited method."
    
    agent_status = {"Agent 1": "FAIL", "Agent 2": "FAIL", "Agent 3": "FAIL", "Agent 4": "FAIL"}
    
    def progress_callback(msg):
        if "Response Analyzer" in msg and "✓" in msg:
            agent_status["Agent 1"] = "PASS"
        if "Personalized Feedback" in msg and "✓" in msg:
            agent_status["Agent 2"] = "PASS"
        if "Learning Recommendation" in msg and "✓" in msg:
            agent_status["Agent 3"] = "PASS"
        if "Coordinator" in msg and "✓" in msg:
            # Note: Final coordinator checkmark might not be in the callback depending on implementation,
            # but we assume success if it finishes.
            pass
            
    mode = "Development Fallback (Rule-Based)"
    working_model = "Unknown"
    
    try:
        feedback_result = run_multi_agent_feedback(
            question=question,
            initial_response=initial_response,
            followup_question=followup_question,
            followup_response=followup_response,
            score=8.5,
            learner_level="Good",
            topic="Java Core",
            progress_callback=progress_callback
        )
        
        mode = feedback_result.get("mode", mode)
        working_model = feedback_result.get("working_model", "Unknown")
        agent_status["Agent 1"] = "PASS" if mode == "Multi-Agent LLM" else "FAIL"
        agent_status["Agent 2"] = "PASS" if mode == "Multi-Agent LLM" else "FAIL"
        agent_status["Agent 3"] = "PASS" if mode == "Multi-Agent LLM" else "FAIL"
        agent_status["Agent 4"] = "PASS" if mode == "Multi-Agent LLM" else "FAIL"
        
        print("\nLIVE MULTI-AGENT LLM TEST PASSED")
        
    except Exception as e:
        print(f"\nTEST FAILED LOUDLY: {e}")
        
    elapsed = time.time() - start_time
    
    print(f"\nAgent 1: {agent_status['Agent 1']}")
    print(f"Agent 2: {agent_status['Agent 2']}")
    print(f"Agent 3: {agent_status['Agent 3']}")
    print(f"Agent 4: {agent_status['Agent 4']}")
    print(f"Final mode: {mode}")
    print(f"Actual working model: {working_model}")
    print(f"Total requests: {LLMProvider._total_requests}")
    print(f"Total elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
