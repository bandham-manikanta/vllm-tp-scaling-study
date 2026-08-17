#!/bin/bash
# Script to capture complete hardware and software stack metadata for TP scaling study
# Run from repository root on target compute node (e.g. a100-11): bash scripts/capture_env.sh

set -e

ENV_DIR="env_info"
PYTHON_BIN=${PYTHON_BIN:-"/gpfs/projects/MaffeiGroup/venvs/tp_scaling_venv/bin/python"}

mkdir -p ${ENV_DIR}

echo "[*] Capturing CPU info..."
lscpu > ${ENV_DIR}/cpu_info.txt

echo "[*] Capturing GPU topology and bus topology..."
nvidia-smi topo -m > ${ENV_DIR}/topo_matrix.txt

echo "[*] Capturing GPU hardware specs (PCIe ID, Memory, Driver)..."
nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv > ${ENV_DIR}/gpu_info.csv

echo "[*] Capturing NUMA node topology..."
numactl --hardware > ${ENV_DIR}/numa_info.txt 2>/dev/null || lscpu --extended=NODE,CPU,CORE,SOCKET > ${ENV_DIR}/numa_info.txt

echo "[*] Capturing software stack using ${PYTHON_BIN}..."
${PYTHON_BIN} -c '
import sys, json, torch
try:
    import vllm
    vllm_version = vllm.__version__
except Exception:
    vllm_version = "unknown"

nccl_ver = torch.cuda.nccl.version() if torch.cuda.is_available() else "N/A"

stack = {
    "python_version": sys.version.split()[0],
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "nccl_version": nccl_ver,
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "vllm_version": vllm_version,
}

with open("'"${ENV_DIR}"'/software_stack.json", "w") as f:
    json.dump(stack, f, indent=2)
print(json.dumps(stack, indent=2))
'

echo "[✓] Environment capture complete! Stored in ${ENV_DIR}/"
