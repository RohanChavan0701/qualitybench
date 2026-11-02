#!/usr/bin/env bash
# run_matrix.sh - Run the benchmark and eval matrix, one process per run.
#
# Each benchmark configuration runs REPEATS times in separate processes (model
# reloaded every time), tagged run1..runN. Before every step the script waits
# until GPU memory is back at the idle baseline, so no run measures another's
# leftovers. A failed step is logged and recorded, never dropped silently.
#
# Usage (from the repo root, after scripts/setup_lambda.sh):
#   scripts/run_matrix.sh --config configs/smoke.yaml --out results/smoke --repeats 1
#   scripts/run_matrix.sh --config configs/final.yaml --out results --repeats 3 --backend hf
#
# Options:
#   --backend hf|vllm|llamacpp|all   (default all)
#   --stage bench|eval|all           (default all)
#
# Writes into --out: benchmark/eval JSON, logs/<step>.log,
# timings.tsv (step, seconds, exit code) and FAILURES.txt.

set -uo pipefail

CONFIG="" OUT="" REPEATS=3 BACKEND=all STAGE=all
GGUF_DIR="models/gguf"
IDLE_SLACK_MB=300
IDLE_TIMEOUT_S=180

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    --stage) STAGE=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$CONFIG" ] && [ -n "$OUT" ] || { echo "need --config and --out" >&2; exit 2; }

mkdir -p "$OUT/logs"
TIMINGS="$OUT/timings.tsv"
FAILURES="$OUT/FAILURES.txt"
[ -f "$TIMINGS" ] || printf 'step\tseconds\texit_code\n' > "$TIMINGS"

gpu_used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

IDLE_MB=$(gpu_used_mb)
echo "Idle GPU memory: ${IDLE_MB} MB"

wait_gpu_idle() {
  local waited=0
  while [ "$(gpu_used_mb)" -gt $((IDLE_MB + IDLE_SLACK_MB)) ]; do
    if [ "$waited" -ge "$IDLE_TIMEOUT_S" ]; then
      echo "GPU memory did not return to idle (${IDLE_MB} MB) within ${IDLE_TIMEOUT_S}s:" >&2
      nvidia-smi >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

# run_step <name> <command...>
run_step() {
  local name=$1; shift
  if ! wait_gpu_idle; then
    echo "$name: skipped, GPU not idle" >> "$FAILURES"
    return
  fi
  echo "--> $name"
  local start=$SECONDS
  "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local secs=$((SECONDS - start))
  printf '%s\t%s\t%s\n' "$name" "$secs" "$rc" >> "$TIMINGS"
  if [ "$rc" -ne 0 ]; then
    echo "$name: exit $rc (see logs/$name.log)" >> "$FAILURES"
    echo "    FAILED in ${secs}s, exit $rc"
  else
    echo "    ok in ${secs}s"
  fi
}

want() { [ "$BACKEND" = all ] || [ "$BACKEND" = "$1" ]; }
want_stage() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

COMMON=(--config "$CONFIG" --output-dir "$OUT")

if want_stage bench; then
  if want hf; then
    for q in fp16 int8 int4; do
      for r in $(seq 1 "$REPEATS"); do
        run_step "bench_hf_${q}_run${r}" .venv-hf/bin/python benchmarks/run_hf_transformers.py \
          "${COMMON[@]}" --quantization "$q" --tag "run$r"
      done
    done
  fi
  if want vllm; then
    for r in $(seq 1 "$REPEATS"); do
      run_step "bench_vllm_fp16_run${r}" .venv-vllm/bin/python benchmarks/run_vllm.py \
        "${COMMON[@]}" --tag "run$r"
    done
  fi
  if want llamacpp; then
    for q in q4_k_m q8_0; do
      # q4_k_m -> mistral-7b-v0.1.Q4_K_M.gguf (names from scripts/setup_lambda.sh)
      gguf="$GGUF_DIR/mistral-7b-v0.1.$(echo "$q" | tr '[:lower:]' '[:upper:]').gguf"
      for r in $(seq 1 "$REPEATS"); do
        run_step "bench_llamacpp_${q}_run${r}" .venv-llamacpp/bin/python benchmarks/run_llamacpp.py \
          "${COMMON[@]}" --model-path "$gguf" --quantization "$q" --tag "run$r"
      done
    done
  fi
fi

if want_stage eval; then
  if want hf; then
    for q in fp16 int8 int4; do
      run_step "eval_mmlu_hf_${q}" .venv-hf/bin/python evals/run_mmlu.py \
        "${COMMON[@]}" --backend hf --quantization "$q"
      run_step "eval_truthfulqa_hf_${q}" .venv-hf/bin/python evals/run_truthfulqa.py \
        "${COMMON[@]}" --backend hf --quantization "$q"
    done
  fi
  if want vllm; then
    run_step "eval_mmlu_vllm_fp16" .venv-vllm/bin/python evals/run_mmlu.py \
      "${COMMON[@]}" --backend vllm --quantization fp16
    run_step "eval_truthfulqa_vllm_fp16" .venv-vllm/bin/python evals/run_truthfulqa.py \
      "${COMMON[@]}" --backend vllm --quantization fp16
  fi
  # llama.cpp: the eval wrappers support hf and vllm only; reported as "not evaluated"
fi

echo
echo "Timings: $TIMINGS"
if [ -s "$FAILURES" ]; then
  echo "Failures:"
  cat "$FAILURES"
  exit 1
fi
echo "No failures."
