"""
results_table.py - Markdown results table for the README.

One row per configuration: throughput, p50/p99 latency and peak GPU memory as
mean ± std across repeated runs, plus MMLU and TruthfulQA MC2 accuracy with
their sample sizes. Configurations listed in FAILURES.txt are shown as failed
rather than dropped. Also lists Pareto-front membership per objective pair.

Usage:
    python analysis/results_table.py --results-dir results
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysis.pareto import build_combined_df, compute_pareto  # noqa: E402
from evals.eval_utils import load_all_results  # noqa: E402

EXPECTED = [
    ("hf_transformers", "fp16"),
    ("hf_transformers", "int8"),
    ("hf_transformers", "int4"),
    ("vllm", "fp16"),
    ("llamacpp", "q4_k_m"),
    ("llamacpp", "q8_0"),
]

PARETO_OBJECTIVES = {
    "throughput vs memory": [("throughput_tokens_per_sec", "max"), ("peak_gpu_memory_mb", "min")],
    "latency vs memory": [("p50_latency_ms", "min"), ("peak_gpu_memory_mb", "min")],
    "MMLU vs efficiency": [("mmlu_accuracy", "max"), ("efficiency_tps_per_gb", "max")],
}


def mean_std(row, col, fmt="{:.1f}"):
    mean = row.get(col)
    if mean is None or pd.isna(mean):
        return "n/a"
    std = row.get(f"{col}_std")
    if std is None or pd.isna(std):
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def accuracy(value, n, backend):
    if value is None or pd.isna(value):
        # The eval wrappers support hf and vllm only; a missing hf/vllm score is a gap
        return "not evaluated" if backend == "llamacpp" else "missing"
    return f"{value * 100:.1f}% (n={n})"


def eval_sample_sizes(results_dir):
    sizes = {}
    for r in load_all_results(results_dir):
        if "accuracy" in r and "task" in r:
            backend = {"hf": "hf_transformers"}.get(r["backend"], r["backend"])
            task = "truthfulqa" if "truthfulqa" in r["task"] else r["task"]
            sizes[(backend, r["quantization"], task)] = r["num_samples"]
    return sizes


def failed_steps(results_dir):
    path = os.path.join(results_dir, "FAILURES.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Print the README results table")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    df = build_combined_df(args.results_dir)
    sizes = eval_sample_sizes(args.results_dir)
    failures = failed_steps(args.results_dir)

    print("| Backend | Precision | Runs | Tokens/s | p50 latency (ms) | p99 latency (ms) "
          "| Peak GPU memory (MB) | MMLU 5-shot (n per subtask, 57 subtasks) | TruthfulQA MC2 |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = {} if df.empty else {(r["backend"], r["quantization"]): r for _, r in df.iterrows()}
    for backend, quant in EXPECTED:
        row = rows.get((backend, quant))
        if row is None:
            print(f"| {backend} | {quant} | 0 | failed or not run | | | | | |")
            continue
        mmlu_n = sizes.get((backend, quant, "mmlu"), "?")
        tqa_n = sizes.get((backend, quant, "truthfulqa"), "?")
        print(
            f"| {backend} | {quant} | {int(row['n_runs'])} "
            f"| {mean_std(row, 'throughput_tokens_per_sec')} "
            f"| {mean_std(row, 'p50_latency_ms')} "
            f"| {mean_std(row, 'p99_latency_ms')} "
            f"| {mean_std(row, 'peak_gpu_memory_mb', '{:.0f}')} "
            f"| {accuracy(row.get('mmlu_accuracy'), mmlu_n, backend)} "
            f"| {accuracy(row.get('truthfulqa_accuracy'), tqa_n, backend)} |"
        )

    if not df.empty:
        print("\nPareto front (computed on run means):\n")
        for name, objectives in PARETO_OBJECTIVES.items():
            front = compute_pareto(df, objectives)
            members = front.loc[front["pareto_optimal"], "label"].tolist()
            print(f"- {name}: {', '.join(members) if members else 'no data'}")

    if failures:
        print("\nFailed steps:\n")
        for line in failures:
            print(f"- {line}")


if __name__ == "__main__":
    main()
