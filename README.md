# QualityBench

A framework for measuring LLM inference quality–efficiency tradeoffs across Hugging Face Transformers, vLLM, and llama.cpp. It collects generation latency, throughput, and GPU-memory data; wraps MMLU and TruthfulQA evaluation; and builds Pareto views from compatible result files.

**Current status (2026-11-05):** first full run committed in `results/`, covering one model on one GPU. See [Results](#results) and its limitations before citing any number.

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

The default configuration uses `mistralai/Mistral-7B-v0.1`, three warmup generations, 50 measured generations, 100 MMLU samples, and 50 TruthfulQA samples. These are experiment settings, not reported results. The run below used [`configs/final.yaml`](configs/final.yaml).

## Results

**Setup:** NVIDIA A100-SXM4 40 GB on Lambda Cloud, driver 570.148.08, Ubuntu 22.04, Python 3.10.12. Model `mistralai/Mistral-7B-v0.1` at revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`. GGUF files from `TheBloke/Mistral-7B-v0.1-GGUF`, with hashes in [`results/ENVIRONMENT.md`](results/ENVIRONMENT.md).

**Versions:** transformers 4.40.2, bitsandbytes 0.43.1, vLLM 0.4.2, llama-cpp-python 0.2.77, lm-eval 0.4.3, torch 2.2.2 (2.3.0 for vLLM).

**Config:** same prompt, exactly 128 new tokens, batch size 1, 3 warmup generations and 100 measured generations, repeated 3 times in separate processes. MMLU: 50 questions per subtask, 2,850 in total, 5-shot. TruthfulQA MC2: 200 questions. Each eval ran once per configuration.

| Backend | Precision | Runs | Tokens/s | p50 latency (ms) | p99 latency (ms) | Peak GPU memory (MB) | MMLU | TruthfulQA MC2 |
|---|---|---|---|---|---|---|---|---|
| hf_transformers | fp16 | 3 | 36.9 ± 0.3 | 3461.8 ± 28.5 | 3511.5 ± 38.5 | 15163 ± 0 | 63.9% | 40.6% |
| hf_transformers | int8 | 3 | 4.7 ± 0.0 | 27350.4 ± 108.3 | 27608.1 ± 104.2 | 11005 ± 0 | 63.9% | 40.6% |
| hf_transformers | int4 | 3 | 17.8 ± 0.1 | 7166.5 ± 56.9 | 7270.4 ± 32.0 | 5295 ± 0 | 62.8% | 40.3% |
| vllm | fp16 | 3 | 73.9 ± 0.1 | 1732.0 ± 1.8 | 1737.0 ± 0.9 | 36555 ± 0 | 64.1% | 40.6% |
| llamacpp | q4_k_m | 3 | 130.7 ± 0.1 | 979.4 ± 0.9 | 985.1 ± 2.6 | 5079 ± 0 | not evaluated | not evaluated |
| llamacpp | q8_0 | 3 | 106.6 ± 0.1 | 1200.1 ± 1.7 | 1204.3 ± 1.3 | 8189 ± 0 | not evaluated | not evaluated |

Values are mean ± std across the 3 repeats. Latency is the full 128-token generation, not time to first token. No configuration failed.

**Pareto front** (on run means; charts: [throughput vs memory](results/pareto_throughput_memory.html), [quality vs efficiency](results/pareto_quality_efficiency.html), [latency](results/latency_comparison.html)):

- Throughput vs memory: llama.cpp Q4_K_M
- Latency vs memory: llama.cpp Q4_K_M
- MMLU vs tokens/s per GB: HF FP16, HF INT4, vLLM FP16. llama.cpp is left out because it has no quality score.

### Limitations

- **Single GPU type and one model.** These numbers do not transfer to other hardware or models.
- **One prompt at batch size 1.** Nothing here measures concurrent serving, which is the workload vLLM is built for.
- **Sample sizes.** The MMLU margin is about ±1.8 points and the TruthfulQA margin about ±7 points (95%, binomial approximation). The accuracy gaps between configurations are smaller than these margins, so this run does not show that any configuration is more accurate, and it does not show that quantization has no quality cost.
- **GGUF is a different artifact.** llama.cpp runs use separately converted community files, not the same weights as the HF and vLLM runs.
- **llama.cpp has no quality evals.** Its speed and memory cannot be weighed against accuracy from this data.
- **vLLM memory is an allocation, not a requirement.** vLLM preallocates a share of the GPU for its KV cache, so its figure reflects this GPU's size.
- **vLLM uses a different context limit.** `max_model_len` is 4096 for vLLM only, so that it starts on smaller GPUs. Prompts are far below this limit.
- **vLLM evals use a patched harness.** One line of lm-eval is changed at install so vLLM 0.4.2 returns the prompt logprobs. See `scripts/setup_lambda.sh`.
- **Version deviations.** peft 0.10.0, and transformers, accelerate and datasets in the vLLM venv, are pinned beyond `requirements.txt`; vLLM uses torch 2.3.0; Python is 3.10.12, not the 3.11 stated under Install. See [`results/ENVIRONMENT.md`](results/ENVIRONMENT.md).
- **Eval scores are not portable across GPUs.** In smoke tests, HF INT8 scored differently on an A10 and an A100 on the same 285 questions, cause unknown. Do not mix these results with runs from other hardware.

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
