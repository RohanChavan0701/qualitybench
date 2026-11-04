#!/usr/bin/env bash
# setup_lambda.sh - Preflight and environment setup for a fresh GPU instance.
#
# Records the environment, checks disk / HF access / git push, pins the model
# revision, builds the three venvs (.venv-hf, .venv-vllm, .venv-llamacpp), and
# downloads the GGUF files with their SHA256.
#
# Usage (from the repo root):
#   export HF_TOKEN=...            # never commit or print it
#   INSTANCE_TYPE="1x A100 40GB" HOURLY_PRICE="1.29" scripts/setup_lambda.sh
#
# Writes:
#   results/ENVIRONMENT.md         environment, pinned revision, GGUF hashes
#   results/freeze-<venv>.txt      pip freeze per venv
#   results/model_revision.txt     pinned Hub commit hash
#   models/gguf/*.gguf             GGUF files (not committed)
#
# Re-running is safe: existing venvs and GGUF files are reused.

set -euo pipefail

MODEL="mistralai/Mistral-7B-v0.1"
GGUF_REPO="TheBloke/Mistral-7B-v0.1-GGUF"
GGUF_FILES=("mistral-7b-v0.1.Q4_K_M.gguf" "mistral-7b-v0.1.Q8_0.gguf")
MIN_FREE_GB=60
PYTHON="${PYTHON:-python3}"

ENV_MD="results/ENVIRONMENT.md"
mkdir -p results models/gguf

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

[ -f benchmarks/profiler.py ] || die "run from the repo root"

# ---------------------------------------------------------------------------
# Phase 0: preflight
# ---------------------------------------------------------------------------

log "Checking disk"
free_gb=$(df -BG --output=avail ~ | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "only ${free_gb} GB free in ~, need ${MIN_FREE_GB} GB"

# Set SKIP_PUSH_CHECK=1 when results are copied off the instance instead of pushed from it
if [ -z "${SKIP_PUSH_CHECK:-}" ]; then
  log "Checking git push access"
  git push --dry-run origin HEAD >/dev/null 2>&1 || die "git push --dry-run failed; set up GitHub auth"
fi

log "Recording environment"
{
  echo "# Environment"
  echo
  echo "Recorded by \`scripts/setup_lambda.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
  echo
  echo "## Instance"
  echo
  echo "- Instance type: ${INSTANCE_TYPE:-TODO}"
  echo "- Hourly price (USD): ${HOURLY_PRICE:-TODO}"
  echo "- Repo commit: $(git rev-parse HEAD)"
  echo
  echo "## GPU"
  echo
  echo '```'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo '```'
  echo
  echo '```'
  nvidia-smi
  echo '```'
  echo
  echo "## CUDA toolkit"
  echo
  echo '```'
  if command -v nvcc >/dev/null; then nvcc --version; else echo "nvcc not found"; fi
  echo '```'
  echo
  echo "## OS and Python"
  echo
  echo '```'
  grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
  "$PYTHON" --version
  echo '```'
  echo
  echo "## Disk"
  echo
  echo '```'
  df -h ~
  echo '```'
} > "$ENV_MD"

# ---------------------------------------------------------------------------
# Phase 1: venvs
# ---------------------------------------------------------------------------

make_venv() {
  local name=$1
  if [ ! -x "$name/bin/python" ]; then
    "$PYTHON" -m venv "$name"
  fi
  "$name/bin/python" -m pip install -q --upgrade pip wheel
}

check_gpu() {
  local name=$1
  "$name/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('$name torch', torch.__version__, torch.cuda.get_device_name(0))"
}

log "Building .venv-hf"
make_venv .venv-hf
.venv-hf/bin/pip install -q \
  torch==2.2.2 transformers==4.40.2 accelerate==0.29.3 bitsandbytes==0.43.1 \
  lm-eval==0.4.3 datasets==2.19.1 numpy==1.26.4 pyyaml==6.0.1 tqdm==4.66.2 \
  pandas==2.2.2 plotly==5.21.0 streamlit==1.34.0 mlflow==2.12.2 \
  nvidia-ml-py3==7.352.0 huggingface_hub
check_gpu .venv-hf

log "Building .venv-vllm"
make_venv .venv-vllm
# vLLM pins its own torch; install it first and let it choose
.venv-vllm/bin/pip install -q vllm==0.4.2 transformers==4.40.2
.venv-vllm/bin/pip install -q lm-eval==0.4.3 pyyaml==6.0.1 numpy mlflow==2.12.2
check_gpu .venv-vllm

log "Building .venv-llamacpp (CUDA build)"
make_venv .venv-llamacpp
# The profiler imports torch for CUDA sync, so this venv needs it too
.venv-llamacpp/bin/pip install -q torch==2.2.2 numpy==1.26.4 pyyaml==6.0.1 mlflow==2.12.2
if ! .venv-llamacpp/bin/python -c "import llama_cpp" 2>/dev/null; then
  CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 \
    .venv-llamacpp/bin/pip install -q --no-cache-dir llama-cpp-python==0.2.77
fi
check_gpu .venv-llamacpp
.venv-llamacpp/bin/python - <<'EOF'
import llama_cpp
ok = bool(llama_cpp.llama_supports_gpu_offload())
print("llama-cpp-python", llama_cpp.__version__, "GPU offload:", ok)
assert ok, "llama-cpp-python was built without CUDA"
EOF

log "Saving pip freeze"
for v in hf vllm llamacpp; do
  ".venv-$v/bin/pip" freeze > "results/freeze-venv-$v.txt"
done

# ---------------------------------------------------------------------------
# Model access, revision pin, GGUF download
# ---------------------------------------------------------------------------

# Checked here, not in preflight: the venvs do not need the token, and a rerun
# reuses them, so a missing token costs nothing already built
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set"

log "Checking access to $MODEL and pinning revision"
.venv-hf/bin/python - "$MODEL" <<'EOF' > results/model_revision.txt
import sys
from huggingface_hub import HfApi, hf_hub_download

model = sys.argv[1]
sha = HfApi().model_info(model).sha
# A gated model returns metadata without access; downloading a file proves access
hf_hub_download(model, "config.json", revision=sha)
print(sha)
EOF
REVISION=$(cat results/model_revision.txt)
echo "Pinned $MODEL @ $REVISION"

log "Downloading GGUF files from $GGUF_REPO"
GGUF_REVISION=$(.venv-hf/bin/python -c "from huggingface_hub import HfApi; print(HfApi().model_info('$GGUF_REPO').sha)")
for f in "${GGUF_FILES[@]}"; do
  if [ ! -f "models/gguf/$f" ]; then
    .venv-hf/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('$GGUF_REPO', '$f', revision='$GGUF_REVISION', local_dir='models/gguf')"
  fi
done

{
  echo
  echo "## Model"
  echo
  echo "- HF model: \`$MODEL\` @ \`$REVISION\`"
  echo "- GGUF repo: \`$GGUF_REPO\` @ \`$GGUF_REVISION\`"
  echo "- GGUF files are separately quantized artifacts, not the same weights as the HF runs."
  echo
  echo "| GGUF file | SHA256 |"
  echo "|---|---|"
  for f in "${GGUF_FILES[@]}"; do
    echo "| \`$f\` | \`$(sha256sum "models/gguf/$f" | cut -d' ' -f1)\` |"
  done
  echo
  echo "## Python packages"
  echo
  echo "Full lists: \`results/freeze-venv-{hf,vllm,llamacpp}.txt\`."
  echo
  echo "| venv | torch | key package |"
  echo "|---|---|---|"
  echo "| .venv-hf | $(.venv-hf/bin/python -c 'import torch;print(torch.__version__)') | transformers $(.venv-hf/bin/python -c 'import transformers;print(transformers.__version__)'), bitsandbytes $(.venv-hf/bin/python -c 'import bitsandbytes;print(bitsandbytes.__version__)'), lm-eval $(.venv-hf/bin/pip show lm-eval | awk '/^Version/{print $2}') |"
  echo "| .venv-vllm | $(.venv-vllm/bin/python -c 'import torch;print(torch.__version__)') | vllm $(.venv-vllm/bin/python -c 'import vllm;print(vllm.__version__)') |"
  echo "| .venv-llamacpp | $(.venv-llamacpp/bin/python -c 'import torch;print(torch.__version__)') | llama-cpp-python $(.venv-llamacpp/bin/python -c 'import llama_cpp;print(llama_cpp.__version__)') |"
  echo
  echo "## Version deviations from requirements.txt"
  echo
  echo "| Package | Pinned | Used | Reason |"
  echo "|---|---|---|---|"
  echo "| (none recorded) | | | |"
} >> "$ENV_MD"

log "Done"
echo "Revision:   $REVISION"
echo "Next:       scripts/make_config.py ... then scripts/run_matrix.sh"
