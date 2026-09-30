import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from modules.llm_agents import run_multi_agent_feedback, LLMProvider

def run_latency_test():
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    print("==================================================")
    print("      JCAB LLM PIPELINE LATENCY TEST")
    print("==================================================")
    
    question = "What is the Singleton design pattern?"
    initial = "It's a pattern that ensures only one instance of a class is created."
    followup = "How do you make a Singleton thread-safe in Java?"
    followup_resp = "You can use synchronized blocks or the double-checked locking idiom."
    
    print("\nStarting simulated assessment processing...\n")
    
    def ui_callback(msg):
        # Strip unicode emojis that crash Windows console print
        clean_msg = msg.replace('✓', '[DONE]').replace('⏳', '[WAIT]').replace(chr(10), ' | ')
        print(f"[UI UPDATE] {clean_msg}")
        
    try:
        feedback = run_multi_agent_feedback(
            question=question,
            initial_response=initial,
            followup_question=followup,
            followup_response=followup_resp,
            score=8.5,
            learner_level="Good",
            topic="Design Patterns (Singleton)",
            progress_callback=ui_callback
        )
        
        print("\n==================================================")
        print("      LATENCY RESULTS")
        print("==================================================")
        print(f"Final Feedback Mode: {feedback['mode']}")
        
        latencies = feedback.get('latencies', {})
        print(f"\nModel Used: {latencies.get('model_used', 'Unknown')}")
        
        total_requests = latencies.get('total_requests', 4)
        print(f"Total API Requests: {total_requests} (Retries: {total_requests - 4})")
        
        print(f"\nAgent 1 Latency: {latencies.get('agent_1_sec', 0.0)}s")
        print(f"Agent 2 Latency: {latencies.get('agent_2_sec', 0.0)}s")
        print(f"Agent 3 Latency: {latencies.get('agent_3_sec', 0.0)}s")
        print(f"Agent 4 Latency: {latencies.get('agent_4_sec', 0.0)}s")
        print(f"Total LLM Time:  {latencies.get('total_llm_sec', 0.0)}s")
        
    except Exception as e:
        print(f"PIPELINE FAILED: {e}")

if __name__ == "__main__":
    run_latency_test()
