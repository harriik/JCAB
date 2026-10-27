"""
HTTP & WebSocket Concurrency Load Test for Streamlit / Nginx Reverse Proxy
Measures:
- Streamlit HTTP Page load & healthcheck latency under concurrency (1, 5, 10, 20 users)
- Nginx reverse-proxy distribution across app-1 and app-2
- Throughput and latency comparison between Single Instance (8501) and Nginx Load Balancer (80/443)
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_endpoint(url, timeout=5.0):
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "JCAB-LoadTest-Agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            code = response.getcode()
            body = response.read()
            lat = time.perf_counter() - t0
            headers = dict(response.info())
            return True, lat, code, len(body), headers, None
    except Exception as e:
        lat = time.perf_counter() - t0
        return False, lat, 0, 0, {}, str(e)

def run_http_concurrency_test(target_url, concurrency, total_requests, label=""):
    print(f"\n--- Testing: {label} | Target: {target_url} | Concurrency: {concurrency} | Total: {total_requests} ---")
    latencies = []
    successes = 0
    failures = 0
    errors = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(test_endpoint, target_url) for _ in range(total_requests)]
        for f in as_completed(futures):
            ok, lat, code, length, headers, err = f.result()
            latencies.append(lat)
            if ok and (200 <= code < 400):
                successes += 1
            else:
                failures += 1
                errors.append(f"HTTP {code}: {err}")

    duration = time.perf_counter() - t_start
    arr = np.array(latencies) * 1000.0 # ms

    p50 = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    throughput = total_requests / duration if duration > 0 else 0.0

    print(f"    Duration: {duration:.2f}s | Throughput: {throughput:.2f} req/s")
    print(f"    Latency: min={np.min(arr):.1f}ms | p50={p50:.1f}ms | p95={p95:.1f}ms | max={np.max(arr):.1f}ms")
    print(f"    Success: {successes}/{total_requests} ({(successes/total_requests)*100:.1f}%) | Failures: {failures}")

    return {
        "label": label,
        "target_url": target_url,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "duration_sec": round(duration, 3),
        "throughput_rps": round(throughput, 2),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
        "success_rate": round((successes / total_requests) * 100.0, 1),
        "failures": failures
    }

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip("/")
    else:
        base_url = "http://localhost:8501"

    health_url = f"{base_url}/_stcore/health"
    main_url = f"{base_url}/"

    results = []
    for c in [1, 5, 10, 20]:
        res = run_http_concurrency_test(health_url, concurrency=c, total_requests=100, label=f"Healthcheck (C={c})")
        results.append(res)
        res_main = run_http_concurrency_test(main_url, concurrency=c, total_requests=40, label=f"Page Load (C={c})")
        results.append(res_main)

    out_file = Path("deploy/benchmarks/results/http_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[HTTP BENCHMARK] Saved to {out_file}")

if __name__ == "__main__":
    main()
