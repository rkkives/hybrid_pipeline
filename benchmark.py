# =============================================================================
# benchmark.py
# Hybrid Toxic Language Detection Pipeline
# Component 5 — Performance Benchmarking (Polished)
# =============================================================================

import os
import sys
import time
import tracemalloc
import csv
from typing import Callable, List, Dict, Any

# -----------------------------------------------------------------------------
# Path setup (import project modules)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# Safe imports with fallbacks
# -----------------------------------------------------------------------------
try:
    from normalizer import normalize
except ImportError:
    def normalize(text: str) -> str:
        return text.lower().strip()

try:
    from boyer_moore import boyer_moore_detect
except ImportError:
    def boyer_moore_detect(text: str, *args, **kwargs) -> Dict[str, Any]:
        """Fallback — Dummy detector."""
        return {"is_toxic": any(w in text for w in ["gago", "bobo", "puta"]), "score": 0.5}

try:
    from aho_corasick import aho_corasick_scan
except ImportError:
    def aho_corasick_scan(text: str, dictionary: List[str]) -> List[str]:
        return [word for word in dictionary if word in text]

try:
    from hybrid import HybridPipeline
except ImportError:
    class HybridPipeline:  # type: ignore
        def __init__(self, dictionary: List[str]) -> None:
            self.dictionary = dictionary

        def process(self, text: str) -> Dict[str, Any]:
            return boyer_moore_detect(text)

try:
    from test_cases import test_cases
except ImportError:
    test_cases = [
        {"message": "ang gago mo", "type": "Filipino", "label": "toxic"},
        {"message": "good luck team", "type": "English", "label": "clean"},
    ]

# -----------------------------------------------------------------------------
# Optional libraries
# -----------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Note: pandas not found. Using fallback dictionary.")

try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not found. Skipping graph generation.")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATASET_PATH = os.path.join(BASE_DIR, "data", "toxic_word_dataset_final.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")

for d in (TABLES_DIR, GRAPHS_DIR):
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LOAD DICTIONARY
# =============================================================================
def load_dictionary(filepath: str) -> List[str]:
    """Load toxic terms from Excel or fallback list."""
    if not PANDAS_AVAILABLE:
        return ["gago", "bobo", "tanga", "ulol", "puta", "idiot", "trash"]
    try:
        df = pd.read_excel(filepath)
        col = df.columns[0]
        words = df[col].dropna().astype(str).tolist()
        return [w.lower().strip() for w in words if w.strip()]
    except Exception as e:
        print(f"Warning: Failed loading dataset ({e}) — using default list.")
        return ["gago", "bobo", "tanga", "ulol", "puta"]

# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================
def benchmark_approach(name: str, func: Callable[..., Any], *args: Any) -> Dict[str, Any]:
    """Measure time and peak memory for a given function call."""
    tracemalloc.start()
    start = time.perf_counter()
    _ = func(*args)
    end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "Approach": name,
        "Time_us": (end - start) * 1_000_000,
        "Memory_KB": peak_mem / 1024,
    }

def run_benchmarks(cases: List[Dict[str, Any]], dictionary: List[str]) -> List[Dict[str, Any]]:
    """Run all benchmark cases."""
    results: List[Dict[str, Any]] = []
    pipeline = HybridPipeline(dictionary)
    print("\nRunning performance benchmarks...\n")

    for idx, tc in enumerate(cases, start=1):
        raw: str = tc["message"]
        msg_type: str = tc.get("type", "Unknown")
        is_toxic = tc.get("label", "").lower() == "toxic"

        normalized = normalize(raw)

        bm_res = benchmark_approach("Boyer–Moore", boyer_moore_detect, normalized)
        ac_res = benchmark_approach("Aho–Corasick", aho_corasick_scan, normalized, dictionary)
        hy_res = benchmark_approach("Hybrid", pipeline.process, raw)

        for res in (bm_res, ac_res, hy_res):
            res.update({"Test_ID": idx, "Message_Type": msg_type, "Is_Toxic": is_toxic})
            results.append(res)
    return results

# =============================================================================
# EXPORT RESULTS
# =============================================================================
def save_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """Write benchmark results to CSV."""
    fields = ["Test_ID", "Message_Type", "Is_Toxic", "Approach", "Time_us", "Memory_KB"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {filepath}")

# =============================================================================
# VISUALS
# =============================================================================
def generate_graphs(results: List[Dict[str, Any]]) -> None:
    """Generate average execution time comparison graph."""
    if not MATPLOTLIB_AVAILABLE:
        return
    approaches = ["Boyer–Moore", "Aho–Corasick", "Hybrid"]
    avg_times: List[float] = []
    for app in approaches:
        times = [r["Time_us"] for r in results if r["Approach"] == app]
        avg_times.append(sum(times) / len(times) if times else 0.0)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(approaches, avg_times, color=["#EF8A8A", "#7FB3FF", "#8AFF9E"])
    plt.title("Average Execution Time per Algorithm (µs)")
    plt.ylabel("Execution Time (µs)")
    for b in bars:
        y = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, y + (y * 0.02), f"{y:.2f}", ha="center", va="bottom")
    img_path = os.path.join(GRAPHS_DIR, "average_execution_time.png")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    print(f"Graph generated: {img_path}")

# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    dictionary = load_dictionary(DATASET_PATH)

    print("=" * 70)
    print(" HYBRID PIPELINE PERFORMANCE BENCHMARK ".center(70, "="))
    print("=" * 70)
    print(f"Test Cases   : {len(test_cases)}")
    print(f"Dictionary   : {len(dictionary)} terms")
    print("-" * 70)

    if not dictionary:
        print("Error: Empty dictionary. Abort benchmark.")
        sys.exit(1)

    benchmark_data = run_benchmarks(test_cases, dictionary)

    print("\nAverage Results")
    print(f"{'Approach':<15} | {'Avg Time (µs)':<15} | {'Avg Memory (KB)':<18}")
    print("-" * 54)
    for app in ["Boyer–Moore", "Aho–Corasick", "Hybrid"]:
        times = [r["Time_us"] for r in benchmark_data if r["Approach"] == app]
        mems = [r["Memory_KB"] for r in benchmark_data if r["Approach"] == app]
        avg_time = sum(times) / len(times)
        avg_mem = sum(mems) / len(mems)
        print(f"{app:<15} | {avg_time:<15.2f} | {avg_mem:<18.2f}")
    print("=" * 70)

    csv_path = os.path.join(TABLES_DIR, "benchmark_results.csv")
    save_to_csv(benchmark_data, csv_path)
    generate_graphs(benchmark_data)
