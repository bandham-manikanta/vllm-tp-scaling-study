#!/usr/bin/env bash
#
# vLLM Tensor Parallelism Macro Benchmark Runner
# Target: /gpfs/projects/MaffeiGroup/open-source-contributions/vllm-tp-scaling-study
#
# Usage:
#   bash scripts/run_benchmarks.sh [MODEL_NAME] [TP_SIZE] [PORT] [HOST]
#   bash scripts/run_benchmarks.sh Qwen/Qwen3.5-27B 4 8000 127.0.0.1
set -euo pipefail

export HF_HOME="/gpfs/projects/MaffeiGroup/cache/hf"
VLLM_BIN=${VLLM_BIN:-"/gpfs/projects/MaffeiGroup/venvs/tp_scaling_venv/bin/vllm"}

MODEL_NAME=${1:-"Qwen/Qwen3.5-27B"}
TP_SIZE=${2:-1}
PORT=${3:-8000}
HOST=${4:-"127.0.0.1"}
OUT_DIR="results/tp${TP_SIZE}"

mkdir -p "${OUT_DIR}"

echo "================================================================="
echo " Starting Macro Benchmark Suite (TP=${TP_SIZE})"
echo " Model:    ${MODEL_NAME}"
echo " Endpoint: http://${HOST}:${PORT}/v1/completions"
echo " Results:  ${OUT_DIR}"
echo "================================================================="

# Pre-flight check: ensure the server is live and responsive
echo "[*] Checking vLLM server health at http://${HOST}:${PORT}/health ..."
if ! curl -s -f "http://${HOST}:${PORT}/health" > /dev/null; then
  echo "[!] Error: Server at http://${HOST}:${PORT}/health did not respond."
  echo "    Ensure vllm serve is running with --tensor-parallel-size ${TP_SIZE} on port ${PORT}."
  exit 1
fi
echo "[✓] Server is healthy and ready for traffic."

# Warmup run to trigger CUDA Graph capture and stabilize memory pools
echo "-----------------------------------------------------------------"
echo "[*] Running warmup phase (16 requests, 512 in x 128 out)..."
echo "-----------------------------------------------------------------"
"${VLLM_BIN}" bench serve \
  --backend vllm \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_NAME}" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 16 \
  --request-rate inf > /dev/null 2>&1 || true

echo "[✓] Warmup complete. CUDA graphs and memory pools stabilized."

# Workload matrix definition: (name in_len out_len)
declare -a WORKLOADS=(
  "prefill_heavy 8192 128"
  "decode_heavy 256 1024"
)

# Concurrency levels for closed-loop evaluation
CONCURRENCIES=("1" "8" "32")

for wl in "${WORKLOADS[@]}"; do
  read -r name in_len out_len <<< "$wl"
  
  for c in "${CONCURRENCIES[@]}"; do
    if [ "$c" -eq 1 ]; then
      NUM_PROMPTS=32
    elif [ "$c" -eq 8 ]; then
      NUM_PROMPTS=48
    else
      NUM_PROMPTS=96
    fi
    
    OUT_FILE="${name}_${in_len}x${out_len}_c${c}_tp${TP_SIZE}.json"
    
    echo "-----------------------------------------------------------------"
    echo "[+] Running: ${name} (${in_len} in x ${out_len} out) | Concurrency: ${c} | Prompts: ${NUM_PROMPTS}"
    echo "-----------------------------------------------------------------"
    
    "${VLLM_BIN}" bench serve \
      --backend vllm \
      --host "${HOST}" \
      --port "${PORT}" \
      --model "${MODEL_NAME}" \
      --endpoint /v1/completions \
      --dataset-name random \
      --random-input-len "${in_len}" \
      --random-output-len "${out_len}" \
      --num-prompts "${NUM_PROMPTS}" \
      --request-rate inf \
      --max-concurrency "${c}" \
      --save-result \
      --result-dir "${OUT_DIR}" \
      --result-filename "${OUT_FILE}"
      
    echo "[✓] Completed: ${OUT_DIR}/${OUT_FILE}"
  done
done

echo "================================================================="
echo "[✓] Macro benchmarks completed successfully for TP=${TP_SIZE}!"
echo "    Artifacts saved to: ${OUT_DIR}/"
echo "================================================================="
