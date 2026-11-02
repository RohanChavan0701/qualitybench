# QualityBench

A framework for measuring LLM inference quality–efficiency tradeoffs across Hugging Face Transformers, vLLM, and llama.cpp. It collects generation latency, throughput, and GPU-memory data; wraps MMLU and TruthfulQA evaluation; and builds Pareto views from compatible result files.

**Current status:** the framework is implemented, but `results/` contains no committed benchmark outputs. This repository therefore makes no backend, quantization, latency, throughput, memory, or quality winner claims.

[Configuration](configs/benchmark_config.yaml) · [Colab notebook](notebooks/colab_benchmark.ipynb) · [Repository](https://github.com/RohanChavan0701/qualitybench)

## Implemented components

| Area | What exists |
|---|---|
| Profiling | Shared warmup/timing loop with CUDA synchronization, p50/p99/mean latency, tokens per second, and peak GPU memory as the `nvidia-smi` increase over a pre-load baseline (PyTorch allocator peak kept as a secondary field) |
| Backends | Runners for Hugging Face FP16/INT8/INT4, vLLM FP16, and llama.cpp GGUF models |
| Quality evaluation | `lm-evaluation-harness` command wrappers for MMLU and TruthfulQA MC2 on Hugging Face or vLLM |
| Analysis | Result merging, two- and three-objective Pareto selection, and Plotly HTML charts |
| Exploration | A Streamlit dashboard that reads local JSON results and reports when none are present |
| Tracking | Optional local MLflow or Weights & Biases logging selected in YAML |
| Automation | CI lint, import, config-schema, and dashboard syntax checks; GPU benchmarks are not run in CI |

The default configuration uses `mistralai/Mistral-7B-v0.1`, three warmup generations, 50 measured generations, 100 MMLU samples, and 50 TruthfulQA samples. These are experiment settings, not reported results.

## Install

Python 3.11 and a CUDA-capable environment are expected for the configured GPU runs.

```bash
git clone https://github.com/RohanChavan0701/qualitybench.git
cd qualitybench
python3 -m pip install -r requirements.txt
```

vLLM and `llama-cpp-python` have platform-specific installation requirements. The llama.cpp runner also requires a local GGUF model path; model weights are not stored in this repository.

## Run the benchmark matrix

```bash
# Hugging Face Transformers
python3 benchmarks/run_hf_transformers.py \
  --config configs/benchmark_config.yaml --quantization fp16
python3 benchmarks/run_hf_transformers.py \
  --config configs/benchmark_config.yaml --quantization int8
python3 benchmarks/run_hf_transformers.py \
  --config configs/benchmark_config.yaml --quantization int4

# vLLM
python3 benchmarks/run_vllm.py --config configs/benchmark_config.yaml

# llama.cpp
python3 benchmarks/run_llamacpp.py \
  --config configs/benchmark_config.yaml \
  --model-path /path/to/model.Q4_K_M.gguf \
  --quantization q4_k_m
```

Each profiler run writes a summary JSON file and a separate raw-latency JSON file under `results/`. Optional tracking is configured under `tracking` in `configs/benchmark_config.yaml`; local result files remain the analysis input.

## Run quality evaluation

```bash
python3 evals/run_mmlu.py \
  --config configs/benchmark_config.yaml --backend hf --quantization fp16

python3 evals/run_truthfulqa.py \
  --config configs/benchmark_config.yaml --backend hf --quantization fp16
```

The wrappers invoke `lm-evaluation-harness`, extract an accuracy value when available, and write a normalized evaluation record. MMLU uses the configured sample limit with five-shot prompting; TruthfulQA uses MC2 with zero-shot prompting.

## Analyze local results

```bash
python3 analysis/pareto.py --results-dir results
streamlit run dashboard/app.py
```

The analysis joins profiler and evaluation records by backend and quantization. It can plot throughput vs memory, quality vs efficiency, latency comparisons, and multi-objective Pareto membership. Empty or incomplete result sets are handled explicitly.

## Run the full matrix on a GPU instance

The backends have conflicting dependencies (vLLM pins its own torch), so the scripts use three venvs: `.venv-hf`, `.venv-vllm`, `.venv-llamacpp`.

```bash
export HF_TOKEN=...   # needs access to mistralai/Mistral-7B-v0.1
INSTANCE_TYPE="1x A100 40GB" HOURLY_PRICE="1.29" scripts/setup_lambda.sh

# Smoke run: cheap, catches bugs, writes to results/smoke (git-ignored)
python3 scripts/make_config.py --out configs/smoke.yaml \
  --revision "$(cat results/model_revision.txt)" \
  --warmup 1 --runs 3 --mmlu-samples 5 --truthfulqa-samples 5
scripts/run_matrix.sh --config configs/smoke.yaml --out results/smoke --repeats 1

# Full run: 3 repeats per configuration, one backend at a time
python3 scripts/make_config.py --out configs/final.yaml \
  --revision "$(cat results/model_revision.txt)" \
  --warmup 3 --runs 100 --mmlu-samples 50 --truthfulqa-samples 200
scripts/run_matrix.sh --config configs/final.yaml --out results --repeats 3 --backend hf

python3 analysis/results_table.py --results-dir results
```

`setup_lambda.sh` records the environment in `results/ENVIRONMENT.md`, pins the model revision, saves `pip freeze` per venv, and downloads the GGUF files with their SHA256. `run_matrix.sh` waits for GPU memory to return to idle before each step, logs every step, and lists failures in `FAILURES.txt` instead of dropping them. `results/smoke/timings.tsv` gives per-step durations for estimating the full run. MMLU `num_samples` applies per subtask (57 subtasks).

## Before making performance claims

Do not cite a comparative result from this project until the evidence is committed or linked and the run is reproducible. At minimum:

1. Pin the model revision, backend versions, CUDA stack, quantization method, and hardware.
2. Hold prompt, token budget, batch size, concurrency, warmup, and measured-run count constant across the configurations being compared.
3. Preserve summary files and raw latency samples, and report run-to-run variability instead of a single best run.
4. Pair efficiency measurements with quality evaluations produced from the same model artifact and quantization.
5. Distinguish framework overhead, model load time, time to first token, and steady-state token generation if the claim depends on latency semantics.
6. Publish the exact config and result artifacts behind every chart or headline.

Until those conditions are met, this repository should be described as a benchmarking framework, not as evidence that one backend or precision is superior.
