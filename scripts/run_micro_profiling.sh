#!/usr/bin/env bash
# ==============================================================================
# Micro Profiling Trace Runner for vLLM Tensor Parallelism Scaling Study
# Author: Manikanta Bandham
# Target: /gpfs/projects/MaffeiGroup/open-source-contributions/vllm-tp-scaling-study
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
TRACES_DIR="${REPO_ROOT}/results/traces"
mkdir -p "${TRACES_DIR}"

VENV_PATH="/gpfs/projects/MaffeiGroup/venvs/tp_scaling_venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

export HF_HOME="/gpfs/projects/MaffeiGroup/cache/hf"
export VLLM_CACHE_ROOT="/gpfs/projects/MaffeiGroup/cache/vllm"

run_trace() {
    local tp="$1"
    local mode="$2"
    local param="$3"
    local name="$4"
    local dev="$5"

    echo "================================================================="
    echo ">>> Running Micro Trace: ${name} (TP=${tp}, Mode=${mode})"
    echo "================================================================="

    if [ "${mode}" == "prefill" ]; then
        CUDA_VISIBLE_DEVICES="${dev}" nsys profile \
            --trace=cuda,nvtx \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --output="${TRACES_DIR}/${name}" \
            --force-overwrite=true \
            python3 "${SCRIPT_DIR}/profile_tp_step.py" \
                --tp "${tp}" \
                --mode prefill \
                --tokens "${param}"
    else
        CUDA_VISIBLE_DEVICES="${dev}" nsys profile \
            --trace=cuda,nvtx \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --output="${TRACES_DIR}/${name}" \
            --force-overwrite=true \
            python3 "${SCRIPT_DIR}/profile_tp_step.py" \
                --tp "${tp}" \
                --mode decode \
                --batch "${param}"
    fi

    echo "[✓] Saved trace to ${TRACES_DIR}/${name}.nsys-rep"
    echo ""
}

TARGET="${1:-all}"

# --- TP=1 Suite ---
if [ "${TARGET}" == "tp1" ] || [ "${TARGET}" == "all" ]; then
    echo "#################################################################"
    echo "# STARTING TP=1 MICRO PROFILING SUITE (GPU 0)                   #"
    echo "#################################################################"
    run_trace 1 prefill 512  "prefill_m512_tp1"  "0"
    run_trace 1 prefill 2048 "prefill_m2048_tp1" "0"
    run_trace 1 prefill 8192 "prefill_m8192_tp1" "0"
    run_trace 1 decode  1    "decode_b1_tp1"     "0"
    run_trace 1 decode  8    "decode_b8_tp1"     "0"
    run_trace 1 decode  32   "decode_b32_tp1"    "0"
fi

# --- TP=2 Suite ---
if [ "${TARGET}" == "tp2" ] || [ "${TARGET}" == "all" ]; then
    echo "#################################################################"
    echo "# STARTING TP=2 MICRO PROFILING SUITE (GPUs 0,1 - NVLink Bridge)#"
    echo "#################################################################"
    run_trace 2 prefill 512  "prefill_m512_tp2"  "0,1"
    run_trace 2 prefill 2048 "prefill_m2048_tp2" "0,1"
    run_trace 2 prefill 8192 "prefill_m8192_tp2" "0,1"
    run_trace 2 decode  1    "decode_b1_tp2"     "0,1"
    run_trace 2 decode  8    "decode_b8_tp2"     "0,1"
    run_trace 2 decode  32   "decode_b32_tp2"    "0,1"
fi

# --- TP=4 Suite ---
if [ "${TARGET}" == "tp4" ] || [ "${TARGET}" == "all" ]; then
    echo "#################################################################"
    echo "# STARTING TP=4 MICRO PROFILING SUITE (GPUs 0,1,2,3 - SYS Cross)#"
    echo "#################################################################"
    run_trace 4 prefill 512  "prefill_m512_tp4"  "0,1,2,3"
    run_trace 4 prefill 2048 "prefill_m2048_tp4" "0,1,2,3"
    run_trace 4 prefill 8192 "prefill_m8192_tp4" "0,1,2,3"
    run_trace 4 decode  1    "decode_b1_tp4"     "0,1,2,3"
    run_trace 4 decode  8    "decode_b8_tp4"     "0,1,2,3"
    run_trace 4 decode  32   "decode_b32_tp4"    "0,1,2,3"
fi

echo "================================================================="
echo "[✓] All requested micro profiling traces completed successfully!"
echo "================================================================="
