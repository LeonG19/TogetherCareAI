#!/usr/bin/env bash
# Serve the merged TogetherCare model with vLLM on loopback only.
#
# Reads from the environment (with defaults tuned for the djarin box):
#   VLLM_KEY   - static API key clients must send as "Authorization: Bearer ..."
#                Must match HF_TOKEN in backend/.env. Required.
#   MODEL_DIR  - merged model directory (NOT the LoRA adapter directory).
#   VLLM_PY    - python from the vLLM-only virtualenv.
#
# Loading an 8B model takes several minutes. vLLM does not bind the port until
# it is finished, so anything probing earlier gets connection refused, not 503.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/lagarza3/models/togethercare-llama3-8b}"
SERVED_NAME="${SERVED_NAME:-togethercare-llama3-8b}"
VLLM_PY="${VLLM_PY:-/mnt/lagarza3/.venv-vllm/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

# GPU 0 is the only card on this box with enough free VRAM (~21.9 GB of 48 GB;
# GPUs 1 and 2 are held by another user's long-running vLLM instances). It is
# also running someone else's compute job, so gpu-memory-utilization is capped
# well below the usual 0.85: 0.42 x 49140 MiB ~= 20.6 GB, which fits inside the
# free space without evicting anyone. Re-check `nvidia-smi` and raise this if
# the card frees up.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_UTIL="${GPU_UTIL:-0.42}"

# Llama-3.1-8B-Instruct is natively 131072 ctx. Pinned to 8192 explicitly:
# vLLM would otherwise size the KV cache for the full context and fail to fit
# in the memory carved out above.
MAX_LEN="${MAX_LEN:-8192}"

# The checkpoint is bfloat16 (base config torch_dtype: bfloat16, merged at
# bfloat16). Stated explicitly so nothing silently upcasts to fp32.
DTYPE="${DTYPE:-bfloat16}"

# Accept HF_TOKEN as an alias so the systemd unit can point EnvironmentFile
# straight at backend/.env (which stores the same secret under that name)
# instead of duplicating the key into a second file.
VLLM_KEY="${VLLM_KEY:-${HF_TOKEN:-}}"

if [[ -z "${VLLM_KEY:-}" ]]; then
  echo "VLLM_KEY is not set. Source it from backend/.env, e.g.:" >&2
  echo "  export VLLM_KEY=\$(grep '^HF_TOKEN=' backend/.env | cut -d= -f2-)" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "No config.json in $MODEL_DIR - is this a merged model?" >&2
  echo "LoRA adapter dirs will not work here; run scripts/merge_lora.py first." >&2
  exit 1
fi

echo "[run_vllm] model=$MODEL_DIR served-as=$SERVED_NAME"
echo "[run_vllm] $HOST:$PORT  gpu=$CUDA_VISIBLE_DEVICES util=$GPU_UTIL len=$MAX_LEN dtype=$DTYPE"

# Single GPU holds the model, so no --tensor-parallel-size.
# NB: vLLM 0.28 renamed --disable-log-requests to --no-enable-log-requests.
exec "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" --port "$PORT" \
  --api-key "$VLLM_KEY" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --no-enable-log-requests
