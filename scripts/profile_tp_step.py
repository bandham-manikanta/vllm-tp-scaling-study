#!/usr/bin/env python3
"""
Targeted Step Profiler for vLLM Tensor Parallelism Scaling Study.
Author: Manikanta Bandham
Target: /gpfs/projects/MaffeiGroup/open-source-contributions/vllm-tp-scaling-study

Instruments isolated single-step Prefill and multi-step Decode phases
using clean outer NVTX range markers and CUDA Profiler API for pristine Nsight Systems tracing.

Usage:
  # Profile Prefill Chunk of 512 tokens on TP=1
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output=results/traces/prefill_m512_tp1 --force-overwrite=true \
    python3 scripts/profile_tp_step.py --tp 1 --mode prefill --tokens 512

  # Profile Decode Batch of 8 streams on TP=2
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output=results/traces/decode_b8_tp2 --force-overwrite=true \
    python3 scripts/profile_tp_step.py --tp 2 --mode decode --batch 8
"""

import argparse
import sys
import os
import torch

# Ensure Hugging Face and vLLM use the shared GPFS cache directory
os.environ["HF_HOME"] = "/gpfs/projects/MaffeiGroup/cache/hf"
os.environ["VLLM_CACHE_ROOT"] = "/gpfs/projects/MaffeiGroup/cache/vllm"

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("[!] Error: vLLM is not installed in the current Python environment.")
    sys.exit(1)


DEFAULT_MODEL_PATH = "/gpfs/projects/MaffeiGroup/cache/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"


def parse_args():
    parser = argparse.ArgumentParser(description="Profile isolated vLLM forward steps with NVTX and CUDA Profiler.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model name or path.")
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2, 4], help="Tensor parallel size.")
    parser.add_argument("--mode", type=str, required=True, choices=["prefill", "decode"], help="Profiling mode.")
    parser.add_argument("--tokens", type=int, default=512, choices=[512, 2048, 8192], help="Prefill prompt token count.")
    parser.add_argument("--batch", type=int, default=8, choices=[1, 8, 32], help="Decode concurrent batch size.")
    parser.add_argument("--max-model-len", type=int, default=16384, help="Maximum context length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory pool fraction.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=================================================================")
    print(f" Initializing vLLM Profiling Step | TP={args.tp} | Mode={args.mode.upper()}")
    print(f" Model: {args.model}")
    if args.mode == "prefill":
        print(f" Prefill Chunk Size: {args.tokens} tokens")
    else:
        print(f" Decode Batch Size:  {args.batch} streams (5 decode steps)")
    print("=================================================================")

    # Initialize offline vLLM engine with CUDA Graphs enabled
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )

    # -------------------------------------------------------------------------
    # 1. Warmup Phase (Stabilize CUDA memory allocator & capture CUDA graphs)
    # -------------------------------------------------------------------------
    print("[*] Running engine warmup passes...")
    warmup_prompts = [{"prompt_token_ids": [100] * 128} for _ in range(4)]
    warmup_sampling = SamplingParams(max_tokens=4, min_tokens=4, temperature=0.0)
    llm.generate(warmup_prompts, warmup_sampling, use_tqdm=False)
    torch.cuda.synchronize()
    print("[✓] Warmup complete. Engine ready for profiling.")

    # -------------------------------------------------------------------------
    # 2. Targeted Execution & NVTX Profiler Capture
    # -------------------------------------------------------------------------
    if args.mode == "prefill":
        prompt_data = [{"prompt_token_ids": [101 + (i % 500) for i in range(args.tokens)]}]
        sampling = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0)

        nvtx_tag = f"vllm_prefill_step_M{args.tokens}_TP{args.tp}"
        print(f"[*] Profiling 1 Prefill Forward Step: {nvtx_tag} ...")

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(nvtx_tag)
        torch.cuda.profiler.start()

        llm.generate(prompt_data, sampling, use_tqdm=False)

        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        torch.cuda.nvtx.range_pop()

    elif args.mode == "decode":
        batch_prompts = [
            {"prompt_token_ids": [101 + (i % 500) for i in range(256)]}
            for _ in range(args.batch)
        ]
        sampling = SamplingParams(max_tokens=5, min_tokens=5, temperature=0.0)

        nvtx_tag = f"vllm_decode_phase_B{args.batch}_TP{args.tp}"
        print(f"[*] Profiling 5 Decode Forward Steps: {nvtx_tag} ...")

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(nvtx_tag)
        torch.cuda.profiler.start()

        llm.generate(batch_prompts, sampling, use_tqdm=False)

        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        torch.cuda.nvtx.range_pop()

    print(f"[✓] Trace capture finished successfully for {nvtx_tag}!")


if __name__ == "__main__":
    main()
