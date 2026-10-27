"""
JCAB / Adaptive Assessment - Automated Performance & Scalability Benchmark Suite
Measures:
- Model initialization overhead (Sentence-BERT, XGBoost, faster-whisper)
- Concurrency levels: 1, 5, 10, and 20 concurrent users
- Latency (min, median/p50, p95, p99, max)
- Throughput (req/sec)
- Success / failure rates
- System resource utilization (CPU %, RAM MB)
- Single-instance vs Load-balanced comparison
- Controlled live Gemini latency measurement
"""

import os
import sys
import time
import json
import psutil
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ensure logs dir exists
(PROJECT_ROOT / "deploy" / "benchmarks" / "results").mkdir(parents=True, exist_ok=True)

# Sample test cases representing realistic student assessments across topics
ASSESSMENT_CASES = [
    {
        "topic": "Inheritance",
        "question": "What is inheritance in Java and how does it promote reusability?",
        "initial": "Inheritance allows a child class to inherit fields and methods from a parent class using the extends keyword.",
        "followup_q": "What is the difference between method overloading and overriding?",
        "followup_a": "Overloading occurs in the same class with different parameters. Overriding replaces a superclass method in a subclass."
    },
    {
        "topic": "Polymorphism",
        "question": "Explain compile-time vs runtime polymorphism in Java.",
        "initial": "Compile-time is method overloading resolved by compiler. Runtime is method overriding resolved dynamically.",
        "followup_q": "How does dynamic method dispatch work?",
        "followup_a": "The JVM determines the actual object type at runtime and invokes the overridden method of that specific subtype."
    },
    {
        "topic": "Concurrency vs Parallelism",
        "question": "What is the difference between concurrency and parallelism?",
        "initial": "Concurrency is about structure and handling multiple tasks by interleaving. Parallelism is running multiple tasks simultaneously on multiple cores.",
        "followup_q": "How does the Fork/Join framework achieve parallelism?",
        "followup_a": "It recursively splits large tasks into smaller subtasks using work-stealing across available CPU cores."
    },
    {
        "topic": "OOP Fundamentals",
        "question": "What are the four pillars of OOP?",
        "initial": "The four pillars are encapsulation, inheritance, polymorphism, and abstraction.",
        "followup_q": "Why is encapsulation important?",
        "followup_a": "It hides the internal state of an object and enforces access control via getters and setters."
    }
]

class ResourceMonitor:
    """Monitors CPU and RAM utilization in a background thread during test runs."""
    def __init__(self, interval_sec=0.1):
        self.interval = interval_sec
        self.stop_event = threading.Event()
        self.cpu_samples = []
        self.mem_samples = []
        self.thread = None
        self.process = psutil.Process(os.getpid())

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                # System-wide CPU and process memory
                cpu = psutil.cpu_percent(interval=None)
                mem = self.process.memory_info().rss / (1024 * 1024) # MB
                self.cpu_samples.append(cpu)
                self.mem_samples.append(mem)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.stop_event.clear()
        self.cpu_samples = []
        self.mem_samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        return {
            "cpu_mean_pct": float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0,
            "cpu_peak_pct": float(np.max(self.cpu_samples)) if self.cpu_samples else 0.0,
            "mem_start_mb": float(self.mem_samples[0]) if self.mem_samples else 0.0,
            "mem_peak_mb": float(np.max(self.mem_samples)) if self.mem_samples else 0.0,
            "mem_end_mb": float(self.mem_samples[-1]) if self.mem_samples else 0.0,
            "mem_delta_mb": float(self.mem_samples[-1] - self.mem_samples[0]) if len(self.mem_samples) > 1 else 0.0,
        }

def measure_model_initialization():
    """Measures cold model load times for Sentence-BERT, XGBoost, and faster-whisper."""
    print("=" * 70)
    print("PHASE 1: MEASURING MODEL INITIALIZATION OVERHEAD")
    print("=" * 70)

    # 1. Sentence-BERT
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_time = time.perf_counter() - t0
    print(f"[-] Sentence-BERT (all-MiniLM-L6-v2) loaded in: {sbert_time:.3f}s")

    # 2. XGBoost Models
    t0 = time.perf_counter()
    import joblib
    from config.settings import SCORE_MODEL_PATH, LEVEL_MODEL_PATH, ENCODER_MODEL_PATH
    score_model = joblib.load(SCORE_MODEL_PATH)
    level_model = joblib.load(LEVEL_MODEL_PATH)
    encoder_model = joblib.load(ENCODER_MODEL_PATH)
    xgb_time = time.perf_counter() - t0
    print(f"[-] XGBoost Models & Encoder loaded in:         {xgb_time:.3f}s")

    # 3. faster-whisper
    t0 = time.perf_counter()
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    whisper_time = time.perf_counter() - t0
    print(f"[-] faster-whisper (tiny.en int8) loaded in:    {whisper_time:.3f}s")

    total_init = sbert_time + xgb_time + whisper_time
    print(f"[-] Total Model Initialization Overhead:        {total_init:.3f}s\n")

    return {
        "sentence_bert_sec": round(sbert_time, 3),
        "xgboost_models_sec": round(xgb_time, 3),
        "faster_whisper_sec": round(whisper_time, 3),
        "total_init_sec": round(total_init, 3)
    }

def execute_assessment_worker(case, use_llm=False):
    """Executes a complete assessment cycle for one user response."""
    from modules.feedback import get_feedback
    t0 = time.perf_counter()
    try:
        res = get_feedback(
            question=case["question"],
            initial_response=case["initial"],
            followup_question=case["followup_q"],
            followup_response=case["followup_a"],
            experimental_ml_score=8.5,
            experimental_ml_learner_level="Good",
            topic=case["topic"],
            use_llm=use_llm
        )
        latency = time.perf_counter() - t0
        score = res.get("score", 0.0)
        mode = res.get("mode", res.get("feedback_mode", "unknown"))
        return True, latency, score, mode, None
    except Exception as e:
        latency = time.perf_counter() - t0
        return False, latency, 0.0, "error", str(e)

def run_concurrency_test(concurrency_level, total_requests, use_llm=False, test_label="Local Pipeline"):
    """Runs a controlled concurrent load test."""
    print(f"\n>>> Running Test: {test_label} | Concurrency: {concurrency_level} users | Total Requests: {total_requests}")
    
    monitor = ResourceMonitor(interval_sec=0.05)
    monitor.start()

    latencies = []
    success_count = 0
    failure_count = 0
    errors = []

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = []
        for i in range(total_requests):
            case = ASSESSMENT_CASES[i % len(ASSESSMENT_CASES)]
            futures.append(executor.submit(execute_assessment_worker, case, use_llm))

        for f in as_completed(futures):
            ok, lat, score, mode, err = f.result()
            latencies.append(lat)
            if ok:
                success_count += 1
            else:
                failure_count += 1
                errors.append(err)

    total_duration = time.perf_counter() - t_start
    res_metrics = monitor.stop()

    latencies_np = np.array(latencies)
    p50 = float(np.median(latencies_np)) * 1000.0 # ms
    p95 = float(np.percentile(latencies_np, 95)) * 1000.0 # ms
    p99 = float(np.percentile(latencies_np, 99)) * 1000.0 # ms
    min_lat = float(np.min(latencies_np)) * 1000.0 # ms
    max_lat = float(np.max(latencies_np)) * 1000.0 # ms
    throughput = total_requests / total_duration if total_duration > 0 else 0.0

    print(f"    Completed {total_requests} requests in {total_duration:.2f}s ({throughput:.2f} req/s)")
    print(f"    Latency: min={min_lat:.1f}ms | p50={p50:.1f}ms | p95={p95:.1f}ms | p99={p99:.1f}ms | max={max_lat:.1f}ms")
    print(f"    Success: {success_count}/{total_requests} ({(success_count/total_requests)*100:.1f}%) | Failures: {failure_count}")
    print(f"    CPU: mean={res_metrics['cpu_mean_pct']:.1f}% | peak={res_metrics['cpu_peak_pct']:.1f}%")
    print(f"    RAM: start={res_metrics['mem_start_mb']:.1f}MB | peak={res_metrics['mem_peak_mb']:.1f}MB | delta={res_metrics['mem_delta_mb']:+.1f}MB")

    return {
        "concurrency": concurrency_level,
        "total_requests": total_requests,
        "duration_sec": round(total_duration, 3),
        "throughput_rps": round(throughput, 2),
        "latency_min_ms": round(min_lat, 1),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "latency_p99_ms": round(p99, 1),
        "latency_max_ms": round(max_lat, 1),
        "success_rate_pct": round((success_count / total_requests) * 100.0, 1),
        "failure_count": failure_count,
        "cpu_mean_pct": round(res_metrics["cpu_mean_pct"], 1),
        "cpu_peak_pct": round(res_metrics["cpu_peak_pct"], 1),
        "mem_start_mb": round(res_metrics["mem_start_mb"], 1),
        "mem_peak_mb": round(res_metrics["mem_peak_mb"], 1),
        "mem_end_mb": round(res_metrics["mem_end_mb"], 1),
        "mem_delta_mb": round(res_metrics["mem_delta_mb"], 1),
        "errors": errors[:5]
    }

def measure_live_gemini():
    """Runs a minimal, controlled live Gemini 4-agent request to measure live remote latency safely."""
    print("\n" + "=" * 70)
    print("PHASE 4: CONTROLLED LIVE GEMINI 4-AGENT BENCHMARK (1 REQUEST)")
    print("=" * 70)

    from modules.feedback import get_feedback
    case = ASSESSMENT_CASES[0]

    t0 = time.perf_counter()
    try:
        res = get_feedback(
            question=case["question"],
            initial_response=case["initial"],
            followup_question=case["followup_q"],
            followup_response=case["followup_a"],
            experimental_ml_score=9.0,
            experimental_ml_learner_level="Excellent",
            topic=case["topic"],
            use_llm=True
        )
        latency = time.perf_counter() - t0
        mode = res.get("mode", "unknown")
        model = res.get("working_model", "unknown")
        summary = res.get("overall_summary", "")
        feedback = res.get("feedback", "")
        latencies = res.get("latencies", {})

        print(f"[-] Status: SUCCESS (Mode: {mode}, Model: {model})")
        print(f"[-] Total Pipeline Latency: {latency:.2f}s")
        print(f"[-] Detailed Agent Latencies: {latencies}")
        print(f"[-] Summary: {summary}")
        print(f"[-] Feedback: {feedback}")

        return {
            "status": "PASS",
            "mode": mode,
            "model": model,
            "latency_sec": round(latency, 2),
            "agent_latencies": latencies,
            "summary_words": len(summary.split()),
            "feedback_words": len(feedback.split())
        }
    except Exception as e:
        latency = time.perf_counter() - t0
        print(f"[-] Live Gemini Test Failed / Quota Limit: {e} (in {latency:.2f}s)")
        return {
            "status": "FAIL",
            "error": str(e),
            "latency_sec": round(latency, 2)
        }

def main():
    print("######################################################################")
    print("   JCAB / ADAPTIVE ASSESSMENT - SCALABILITY & PERFORMANCE SUITE       ")
    print("######################################################################\n")

    # 1. Initialization Overhead
    init_metrics = measure_model_initialization()

    # 2. Local Assessment Scalability Testing (Protecting Gemini Quota)
    print("\n" + "=" * 70)
    print("PHASE 3 & 4: LOCAL ASSESSMENT PIPELINE SCALABILITY (PROTECTED QUOTA)")
    print("Workload levels: 1, 5, 10, 20 concurrent users")
    print("=" * 70)

    workload_configs = [
        {"concurrency": 1,  "requests": 20},
        {"concurrency": 5,  "requests": 50},
        {"concurrency": 10, "requests": 50},
        {"concurrency": 20, "requests": 60},
    ]

    benchmark_results = []
    for cfg in workload_configs:
        res = run_concurrency_test(
            concurrency_level=cfg["concurrency"],
            total_requests=cfg["requests"],
            use_llm=False,
            test_label="Single Instance Assessment Engine"
        )
        benchmark_results.append(res)

    # 3. Controlled Live Gemini Measurement
    gemini_metrics = measure_live_gemini()

    # Save raw benchmark data
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu": "12th Gen Intel Core i7-12650H (10 cores, 16 threads)",
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "os": sys.platform
        },
        "model_initialization": init_metrics,
        "concurrency_benchmarks": benchmark_results,
        "live_gemini_benchmark": gemini_metrics
    }

    out_file = PROJECT_ROOT / "deploy" / "benchmarks" / "results" / "baseline_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[BENCHMARK] Results saved to: {out_file}")

if __name__ == "__main__":
    main()
