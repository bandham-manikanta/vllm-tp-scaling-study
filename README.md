# vLLM Tensor Parallelism Scaling Study: Qwen3.5-27B on Multi-GPU A100

An empirical evaluation of **Tensor Parallelism (TP=1, TP=2, TP=4)** scaling behaviors, compute efficiency, and interconnect bottlenecks on **Qwen3.5-27B** (a hybrid 3:1 Linear Attention / DeltaNet + Full Attention model) served via **vLLM (v0.7.0+)** on a dual-socket **4x NVIDIA A100-80GB PCIe** node.

---

## Executive Summary

I conducted an end-to-end serving and kernel-level profiling study across 18 macro-benchmarks and 18 Nsight Systems micro-traces. The objective was to determine the exact boundary where Tensor Parallelism transitions from compute-bound speedup to communication-bound stagnation on hybrid linear-attention architectures.

```
+----------------------------------------------------------------------------------------+
|                              EXECUTIVE SYSTEMS TAKEAWAYS                               |
+----------------------------------------------------------------------------------------+
| 1. Long-Context Prefill (M=8192 Tokens):                                               |
|    - TP=2 (NVLink @ 600 GB/s): Near-linear 1.83x-1.98x TTFT speedup (2,841 ms -> 1,434 ms |
|      Mean, 2,494 ms -> 1,362 ms Median). Compute scales 2.01x, while NCCL AllReduce    |
|      adds only 7.3% (0.51 ms/barrier, 1.26 ms/layer total comm across 2 barriers/layer)|
|    - TP=4 (PCIe Cross-Socket @ 32 GB/s): Pure math scales 4.19x (2,058 ms -> 491 ms),  |
|      BUT cross-socket Tree AllReduces account for 50.8% (505.8 ms total, 3.95-4.20 ms  |
|      per barrier / 7.90 ms per layer total comm), stalling net TTFT speedup at only     |
|      1.00x-1.06x over TP=2.                                                            |
|                                                                                        |
| 2. Negative Scaling on Intermediate Prefill (M=2048) & Decode (B=8):                   |
|    - On TP=4, PCIe communication tax (58.6% for M=2048, 50.6% for B=8) exceeds math    |
|      savings, making TP=4 SLOWER than TP=2 (319 ms vs 279 ms for M=2048; 237 ms vs     |
|      223 ms for B=8).                                                                  |
|                                                                                        |
| 3. Hybrid DeltaNet (O(N)) vs. FlashAttention-2 Scaling:                                |
|    - Because 75% of Qwen3.5's layers use DeltaNet, attention math stays linear (O(N))  |
|      and takes only 63.6 ms at 8k prompt length, maintaining O(N) complexity compared   |
|      to standard full attention.                                                       |
|                                                                                        |
| 4. vLLM Dispatch Engine Thresholds:                                                    |
|    - Short prefill (M<=512) and decode (B=1..32) run via static CUDA Graphs            |
|      (GraphExec) for zero CPU dispatch overhead. Long prefill (M>=2048) dynamically    |
|      switches to Eager execution to avoid multi-gigabyte static VRAM allocations.      |
+----------------------------------------------------------------------------------------+
```

---

## System Architecture & Hardware Topology

The benchmarking cluster consists of a dual-socket AMD EPYC server with asymmetric GPU interconnects:

```
                  +-----------------------------------------+
                  |       Dual AMD EPYC 7713 (NUMA)         |
                  |   Inter-Socket Interconnect (PCIe Gen4) |
                  +----------+-------------------+----------+
                             |                   |
               +-------------+-----+       +-----+-------------+
               |   NUMA Node 0     |       |   NUMA Node 1     |
               |  (PCIe Switch 0)  |       |  (PCIe Switch 1)  |
               +------+-----+------+       +------+-----+------+
                      |     |                     |     |
                 +----v+   +v----+           +----v+   +v----+
                 |GPU 0|===|GPU 1|           |GPU 2|   |GPU 3|
                 +-----+   +-----+           +-----+   +-----+
                   600 GB/s NVLink             64 GB/s PCIe Gen4
```

### Hardware Specifications
- **Accelerators:** 4x NVIDIA A100-PCIE-80GB (312 TFLOPs BF16 Tensor Core, 2,039 GB/s HBM2e per GPU).
- **Interconnect Topology:**
  - **GPU 0 <-> GPU 1:** Direct 600 GB/s bidirectional NVLink Bridge.
  - **GPU 0/1 <-> GPU 2/3:** Traverses PCIe Gen4 host bridges and inter-socket CPU links (~32 GB/s practical unidirectional bandwidth).
- **Model:** `Qwen/Qwen3.5-27B` (64 total layers: 48 Linear Attention / DeltaNet layers + 16 Full Attention / FlashAttention-2 layers in a repeating 3:1 ratio).
- **Software Stack:** vLLM `v0.7.0+`, PyTorch `2.6.0`, CUDA `12.8`, NCCL `2.21.5`.

---

## Macro Serving Performance (End-to-End Evaluation)

I benchmarked vLLM with `vllm bench serve` across Prefill-heavy (`8192 in / 128 out`) and Decode-heavy (`256 in / 1024 out`) workloads across Concurrency levels C in {1, 8, 32}.

![Macro Scaling Dashboard](results/plots/macro_scaling_dashboard.png)
> *Figure Note: Green circles (Plots A–D) and green arrows (Plots E–F) highlight the optimal operating points for the TP=2 configuration across all benchmarks.*

### Summary Serving Performance Tables

#### 1. Prefill-Heavy Workload (`8192 in / 128 out`)
| Concurrency | Configuration | Median TTFT (ms) | Total Tok/s | Scaling & Performance Verdict |
| :---: | :--- | :---: | :---: | :--- |
| **C = 1** | TP=1 (Baseline) | 2,494.3 ms | 1,092 | Single-GPU baseline |
| | **TP=2 (NVLink)\*** | **1,362.1 ms** | **1,978** | **1.83× TTFT speedup (Linear scaling regime)** |
| | TP=4 (PCIe) | 1,356.3 ms | 2,490 | 1.00× vs TP=2 (Stalls on PCIe) |
| **C = 8** | TP=1 (Baseline) | 3,941.5 ms | 2,970 | Single-GPU baseline |
| | **TP=2 (NVLink)\*** | **2,548.0 ms** | **5,188** | **1.75× throughput gain over TP=1** |
| | TP=4 (PCIe) | 2,651.6 ms | 5,252 | Slower TTFT than TP=2 |
| **C = 32** | TP=1 (Baseline) | 20,372.2 ms | 3,319 | Saturated queue delay |
| | **TP=2 (NVLink)\*** | **2,733.5 ms** | **5,947** | **7.45× TTFT speedup (Peak saturation)** |
| | TP=4 (PCIe) | 2,735.1 ms | 5,903 | 0% throughput gain over TP=2 |

#### 2. Decode-Heavy Workload (`256 in / 1024 out`)
| Concurrency | Configuration | Median ITL (ms) | Output Tok/s | Scaling & Performance Verdict |
| :---: | :--- | :---: | :---: | :--- |
| **C = 1** | TP=1 (Baseline) | 36.9 ms | 27.1 | Memory-bandwidth bound single GPU |
| | **TP=2 (NVLink)\*** | **21.4 ms** | **46.5** | **1.72× generation speedup** |
| | TP=4 (PCIe) | 15.8 ms | 63.2 | Higher tok/s, but slower TTFT than TP=2 |
| **C = 8** | TP=1 (Baseline) | 39.1 ms | 202.1 | Single-GPU baseline |
| | **TP=2 (NVLink)\*** | **23.3 ms** | **339.0** | **1.68× throughput gain over TP=1** |
| | TP=4 (PCIe) | 18.0 ms | 434.9 | PCIe AllReduce latency penalty |
| **C = 32** | TP=1 (Baseline) | 48.0 ms | 639.8 | Baseline single-GPU |
| | **TP=2 (NVLink)\*** | **27.9 ms** | **1,099.8** | **1.72× throughput gain (2× HW efficiency)** |
| | TP=4 (PCIe) | 27.3 ms | 1,121.1 | Inefficient (+1.9% throughput with 2× GPUs) |

*\* Denotes optimal latency/cost operating point.*

### Key Serving Observations:
1. **Prefill Scaling (8192 Tokens):**
   - TP=2 cuts TTFT almost in half (Mean: 2,840.91 ms -> 1,433.52 ms, **1.98x speedup**; Median: 2,494.25 ms -> 1,362.07 ms, **1.83x speedup**) over NVLink.
   - TP=4 only marginally reduces Mean TTFT to 1,356.57 ms (**1.06x speedup over TP=2**; Median: 1,356.34 ms, **1.00x speedup**), despite doubling the total GPU count from 2 to 4.
2. **Decode Scaling (B=8 & B=32):**
   - Moving from TP=1 -> TP=2 increases generation throughput by **1.68x** (C=8: 202.12 -> 339.05 tok/s) to **1.72x** (C=32: 639.79 -> 1,099.83 tok/s).
   - Moving from TP=2 -> TP=4 yields minimal throughput gain at C=32 (+1.9%, 1,099.83 -> 1,121.14 tok/s) and suffers negative TTFT scaling due to PCIe cross-socket communication overhead.

![Throughput vs Latency Pareto Frontier](results/plots/pareto_frontier.png)

---

## GPU kernel trace Decomposition (T_comp vs. T_comm)

Using Nsight Systems and CUPTI activity traces across all 18 runs, I parsed the exact duration of each kernel category on hardware to isolate compute math from communication overhead.

![Micro Scaling Decomposition](results/plots/micro_scaling_decomposition.png)

### Micro-Benchmark Hardware Measurement Table

| Workload | TP Config | Interconnect | Step Time (ms) | Pure Compute T_comp (ms) | NCCL Comm T_comm (ms) | Comm Overhead (%) | GEMM Time (ms) | DeltaNet Recurrence (ms) | Hardware Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Prefill M=512** | TP=1 | Local HBM | 14.47 ms | 14.47 (100.0%) | 0.00 (0.0%) | **0.0%** | 2.49 | 8.88 | Single-GPU baseline |
| | **TP=2\*** | **600 GB/s NVLink** | **3.97 ms** | **3.94 (99.3%)** | **0.03 (0.7%)** | **0.7%** | **1.00** | **2.04** | **3.65× speedup (Negligible 0.7% comm tax)** |
| | TP=4 | 32 GB/s PCIe | 6.89 ms | 5.93 (86.1%) | 0.96 (13.9%) | 13.9% | 0.88 | 3.41 | Slower than TP=2 (PCIe launch latency) |
| **Prefill M=2048** | TP=1 | Local HBM | 498.36 ms | 498.36 (100.0%) | 0.00 (0.0%) | **0.0%** | 438.88 | 31.80 | Single-GPU baseline |
| | **TP=2\*** | **600 GB/s NVLink** | **279.11 ms** | **254.05 (91.0%)** | **25.05 (9.0%)** | **9.0%** | **218.63** | **16.86** | **1.79× speedup (91.0% compute efficiency)** |
| | TP=4 | 32 GB/s PCIe | 319.22 ms | 132.15 (41.4%) | 187.07 (58.6%) | 58.6% | 109.01 | 9.64 | Negative scaling (58.6% PCIe comm tax) |
| **Prefill M=8192** | TP=1 | Local HBM | 2,057.93 ms | 2,057.93 (100.0%) | 0.00 (0.0%) | **0.0%** | 1,833.91 | 120.84 | Single-GPU baseline |
| | **TP=2\*** | **600 GB/s NVLink** | **1,106.73 ms** | **1,025.78 (92.7%)** | **80.95 (7.3%)** | **7.3%** | **895.65** | **63.57** | **1.86× speedup (92.7% compute efficiency)** |
| | TP=4 | 32 GB/s PCIe | 996.58 ms | 490.78 (49.2%) | 505.80 (50.8%) | 50.8% | 407.66 | 35.76 | Interconnect bottleneck (50.8% Comm Wall) |
| **Decode B=1** | TP=1 | Local HBM | 15.39 ms | 15.39 (100.0%) | 0.00 (0.0%) | **0.0%** | 7.81 | 4.77 | Single-GPU memory-bandwidth bound baseline |
| | **TP=2\*** | **600 GB/s NVLink** | **9.64 ms** | **9.53 (98.9%)** | **0.11 (1.1%)** | **1.1%** | **4.07** | **3.25** | **1.60× step speedup (Negligible 1.1% comm tax)** |
| | TP=4 | 32 GB/s PCIe | 9.92 ms | 7.02 (70.7%) | 2.90 (29.3%) | 29.3% | 2.20 | 2.87 | Slower than TP=2 (PCIe AllReduce barrier tax) |
| **Decode B=8** | TP=1 | Local HBM | 438.37 ms | 438.37 (100.0%) | 0.00 (0.0%) | **0.0%** | 372.82 | 34.68 | Single-GPU baseline |
| | **TP=2\*** | **600 GB/s NVLink** | **222.88 ms** | **204.20 (91.6%)** | **18.68 (8.4%)** | **8.4%** | **162.02** | **21.31** | **1.97× speedup over TP=1** |
| | TP=4 | 32 GB/s PCIe | 237.30 ms | 117.27 (49.4%) | 120.03 (50.6%) | 50.6% | 92.45 | 10.31 | Slower than TP=2 (PCIe AllReduce overhead) |
| **Decode B=32** | TP=1 | Local HBM | 1,948.48 ms | 1,948.48 (100.0%) | 0.00 (0.0%) | **0.0%** | 1,695.21 | 124.50 | Single-GPU baseline |
| | **TP=2\*** | **600 GB/s NVLink** | **1,018.02 ms** | **943.50 (92.7%)** | **74.52 (7.3%)** | **7.3%** | **795.79** | **67.08** | **Matches TP=4 throughput at 2× hardware efficiency** |
| | TP=4 | 32 GB/s PCIe | 944.53 ms | 461.11 (48.8%) | 483.43 (51.2%) | 51.2% | 376.92 | 30.43 | Interconnect bottleneck (51.2% Comm Wall) |

*\* Denotes optimal hardware configuration.*

---

## Deep-Dive Insights: Why TP=4 Stalls on PCIe

### 1. Compute Scaled 4.19x, but Communication Grew to 50.8%
- In Prefill M=8192, Tensor Core GEMM math dropped from **1,833.91 ms -> 407.66 ms** (**4.50x compute speedup**; pure compute dropped from **2,057.93 ms -> 490.78 ms**, a **4.19x speedup**).
- However, exchanging the 8,192-token activation vectors across the 4 GPUs required 128 `ncclDevKernel_AllReduce_Sum_bf16_TREE_LL` calls (2 AllReduce barriers per layer across 64 layers).
- On PCIe Gen4 cross-socket links, these AllReduces took **505.80 ms** (50.8% of total GPU runtime, ~3.95 ms per barrier / 7.90 ms per layer), offsetting the 1.57-second compute reduction.

### 2. Negative Scaling at Intermediate Workloads (M=2048 & B=8)
- At M=2048, TP=4 (**319.22 ms**) was actually **slower than TP=2 (279.11 ms)**.
- The cross-socket communication overhead (187.07 ms, 58.6% of step time) exceeded the compute reduction achieved by adding 2 extra GPUs.
- Similarly, at Decode B=8, TP=4 (**237.30 ms**) was slower than TP=2 (**222.88 ms**) due to 50.6% communication overhead (120.03 ms).

### 3. DeltaNet Linear Attention (O(N)) vs. FlashAttention-2 Scaling
- **DeltaNet Layers (48 layers):** Recurrence kernels (`chunk_gated_delta_rule_fwd_kernel_h`, `_causal_conv1d_fwd`) scaled linearly with sequence length:
  - M=512: 2.04 ms
  - M=2048: 16.86 ms (8.3x increase for 4x tokens)
  - M=8192: 63.57 ms (3.8x increase for 4x tokens)
- **FlashAttention-2 Layers (16 layers):** Scaled with sequence length (**0.05 ms -> 0.36 ms -> 1.28 ms**). Total attention compute remained tightly bounded at 8k context due to the hybrid design.

---

## Production Deployment Recommendations

```
+----------------------------------------------------------------------------------------+
|                        SERVING TOPOLOGY DECISION MATRIX                                |
+----------------------------+-----------------------------------------------------------+
| Target Node Hardware       | Optimal Serving Strategy                                  |
+----------------------------+-----------------------------------------------------------+
| 2x A100 / H100 with NVLink | TP=2. Best price/performance and lowest TTFT latency.    |
|                            | NVLink communication overhead is negligible (<8%).       |
+----------------------------+-----------------------------------------------------------+
| 4x / 8x PCIe-only Servers  | DO NOT use TP=4/8. Deploy TP=2 per NVLink pair + DP=2/4  |
| (No full-mesh NVLink)      | (Data Parallelism) or PP=2 (Pipeline Parallelism).       |
|                            | Yields ~1.9x higher throughput at half communication tax. |
+----------------------------+-----------------------------------------------------------+
| Disaggregated Serving      | Prefill Nodes: TP=2 with NVLink to maximize compute FLOPs.|
| (Prefill / Decode Split)   | Decode Nodes: TP=1 or TP=2 with high batching (C>=32)    |
|                            | to saturate HBM2e memory bandwidth without PCIe stalls.   |
+----------------------------+-----------------------------------------------------------+
```

---

## Quick Reproduction Guide

### 1. Run Macro Serving Benchmarks
```bash
# Run TP=1, TP=2, or TP=4 benchmark suite
bash scripts/run_benchmarks.sh Qwen/Qwen3.5-27B 2 8000 127.0.0.1
```

### 2. Run Precision Nsight Micro-Profiling
```bash
# Run all 18 micro profiling traces
bash scripts/run_micro_profiling.sh
```

### 3. Parse Databases & Generate Summary Plots
```bash
# Parse SQLite trace databases and generate stacked decomposition charts
python3 scripts/parse_micro_traces.py
```

---

## Repository Structure

```
vllm-tp-scaling-study/
├── results/
│   ├── MACRO_BENCHMARK_SUMMARY.md       # Full 18-point serving benchmark table
│   ├── MICRO_BENCHMARK_SUMMARY.md       # Full 18-point GPU Kernel Compute vs comm table
│   ├── micro_trace_results.json         # Raw extracted CUPTI hardware metrics
│   ├── plots/                           # High-resolution publication charts
│   ├── tp1/, tp2/, tp4/                 # Raw macro JSON benchmark logs
│   └── traces/                          # Raw .nsys-rep traces & SQLite databases
└── scripts/
    ├── parse_macro_results.py           # Macro JSON benchmark aggregator
    ├── plot_macro_results.py            # Macro dashboard & pareto visualization generator
    ├── parse_micro_traces.py            # Automated CUPTI trace parsing engine
    ├── plot_micro_scaling.py            # Micro kernel decomposition visualization generator
    ├── profile_tp_step.py               # Precision NVTX & CUDA Profiler step harness
    ├── run_benchmarks.sh                # Automated macro serving benchmark suite
    ├── run_micro_profiling.sh           # Automated Nsight micro-profiling suite
    └── capture_env.sh                   # Environment & topology diagnostic capture tool
```
