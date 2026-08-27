#!/usr/bin/env python3
"""
Publication-Grade Visualizations for vLLM TP Scaling Study (Qwen3.5-27B on A100-80GB PCIe)
Author: Manikanta Bandham

Generates 5 standalone publication plots + 1 unified master dashboard:
1. roofline_model.png       - Empirical GPU Roofline with Prefill & Decode mapping
2. pareto_frontier.png      - Latency vs. Throughput Pareto frontier
3. ttft_scaling.png         - Time To First Token (TTFT) scaling & memory cliff
4. itl_scaling.png          - Inter-Token Latency (ITL) scaling & SYS interconnect plateau
5. mfu_efficiency.png       - Model FLOPs Utilization (MFU %) & parallel scaling efficiency
6. macro_dashboard.png      - Unified 2x3 publication master figure
"""

import json
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- Physical & Model Constants ---
MODEL_PARAMS = 27.2e9            # 27.2B params for Qwen3.5-27B
PEAK_TFLOPS = 311.96             # A100 PCIe BF16 Tensor Core Peak per GPU (TFLOPs/s)
PEAK_HBM_BW_GBS = 1935.36        # A100 PCIe HBM2e Memory Bandwidth (GB/s)
RIDGE_POINT = (PEAK_TFLOPS * 1e12) / (PEAK_HBM_BW_GBS * 1e9)  # 161.21 FLOPs/Byte

# Directory setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette: TP=1 (Blue), TP=2 (Green), TP=4 (Orange/Red)
COLORS = {
    1: "#1f77b4",  # Steel Blue
    2: "#2ca02c",  # Forest Green
    4: "#d62728",  # Crimson Red
}
MARKERS = {1: "o", 2: "s", 4: "D"}


def load_dataset():
    """Parses all result JSONs from results/tp* directories."""
    data = {}
    pattern = str(RESULTS_DIR / "tp*" / "*.json")
    for fpath in glob.glob(pattern):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                res = json.load(f)
        except Exception:
            continue
        
        fname = os.path.basename(fpath)
        m = re.match(r"([a-z_]+)_(\d+)x(\d+)_c(\d+)_tp(\d+)\.json", fname)
        if not m:
            continue
            
        profile, in_len, out_len, concurrency, tp = m.groups()
        in_len, out_len, concurrency, tp = int(in_len), int(out_len), int(concurrency), int(tp)
        
        ttft_med = res.get("median_ttft_ms") or res.get("mean_ttft_ms") or 0.0
        ttft_p99 = res.get("p99_ttft_ms") or res.get("p95_ttft_ms") or 0.0
        itl_med = res.get("median_itl_ms") or res.get("median_tpot_ms") or res.get("mean_tpot_ms") or 0.0
        itl_p99 = res.get("p99_itl_ms") or res.get("p99_tpot_ms") or res.get("p95_tpot_ms") or 0.0
        
        out_tp = res.get("output_throughput") or 0.0
        tot_tp = res.get("total_token_throughput") or res.get("tokens_per_second") or (out_tp * (in_len + out_len) / out_len if out_len else out_tp)
        
        key = (profile, in_len, out_len)
        if key not in data:
            data[key] = {}
        if concurrency not in data[key]:
            data[key][concurrency] = {}
            
        # Compute achieved TFLOPs per GPU
        achieved_tflops_per_gpu = (tot_tp * 2.0 * MODEL_PARAMS) / (tp * 1e12)
        mfu = (achieved_tflops_per_gpu / PEAK_TFLOPS) * 100.0
        
        data[key][concurrency][tp] = {
            "ttft_med": ttft_med,
            "ttft_p99": ttft_p99,
            "itl_med": itl_med,
            "itl_p99": itl_p99,
            "output_tp": out_tp,
            "total_tp": tot_tp,
            "achieved_tflops_gpu": achieved_tflops_per_gpu,
            "mfu": mfu
        }
    return data


# ==============================================================================
# 1. Plot: Empirical Roofline Model
# ==============================================================================
def plot_roofline(data, save_path=PLOTS_DIR / "roofline_model.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # 1. Draw Theoretical Roofline Envelope
    ai_range = np.logspace(-1, 3.5, 500)  # FLOPs/Byte from 0.1 to ~3000
    # Roofline formula: y = min(Peak_Compute, Bandwidth * AI)
    mem_bound = (PEAK_HBM_BW_GBS * ai_range) / 1e3  # In TFLOPs
    peak_compute = np.full_like(ai_range, PEAK_TFLOPS)
    roof = np.minimum(mem_bound, peak_compute)
    
    ax.plot(ai_range, roof, color="#333333", linewidth=2.5, label="A100-80GB PCIe Ceiling")
    ax.axvline(RIDGE_POINT, color="#777777", linestyle=":", alpha=0.8, label=f"Ridge Point ({RIDGE_POINT:.1f} FLOP/B)")
    
    # Fill operational regimes
    ax.fill_between(ai_range[ai_range <= RIDGE_POINT], 0.1, roof[ai_range <= RIDGE_POINT], color="#9ecae1", alpha=0.15, label="Memory-Bandwidth Bound")
    ax.fill_between(ai_range[ai_range >= RIDGE_POINT], 0.1, roof[ai_range >= RIDGE_POINT], color="#fcbba1", alpha=0.15, label="Compute Bound (Tensor Cores)")

    # 2. Plot Empirical Data Points
    # Decode Points: AI approx B (1.0, 8.0, 32.0 FLOPs/Byte)
    decode_key = ("decode_heavy", 256, 1024)
    if decode_key in data:
        for c, ai_val in [(1, 1.0), (8, 8.0), (32, 32.0)]:
            for tp in [1, 2, 4]:
                pt = data[decode_key].get(c, {}).get(tp)
                if pt:
                    ax.scatter(ai_val, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=90, zorder=5)
                    if tp == 1:
                        ax.annotate(f"Decode B={c}", (ai_val, pt["achieved_tflops_gpu"]), textcoords="offset points", xytext=(8, -4), fontsize=9, fontweight="bold")

    # Prefill Points: AI approx 360 FLOPs/Byte (for 8k tokens)
    prefill_key = ("prefill_heavy", 8192, 128)
    if prefill_key in data:
        for c in [32]:
            for tp in [1, 2, 4]:
                pt = data[prefill_key].get(c, {}).get(tp)
                if pt:
                    ax.scatter(360.0, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=110, zorder=5, label=f"TP={tp} Workload" if c == 32 else "")
                    ax.annotate(f"Prefill 8k (TP={tp})\n{pt['mfu']:.1f}% MFU", (360.0, pt["achieved_tflops_gpu"]), textcoords="offset points", xytext=(10, -5 if tp != 4 else -15), fontsize=8.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 2000)
    ax.set_ylim(0.5, 450)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte moved from HBM)")
    ax.set_ylabel("Per-GPU Achieved Performance (TFLOPs/s)")
    ax.set_title("Empirical GPU Roofline Model: Qwen3.5-27B on A100-PCIe", fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


# ==============================================================================
# 2. Plot: Latency-Throughput Pareto Frontier
# ==============================================================================
def plot_pareto(data, save_path=PLOTS_DIR / "pareto_frontier.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    decode_key = ("decode_heavy", 256, 1024)
    
    if decode_key in data:
        for tp in [1, 2, 4]:
            thpts = []
            itls = []
            concurrencies = [1, 8, 32]
            for c in concurrencies:
                pt = data[decode_key].get(c, {}).get(tp)
                if pt:
                    thpts.append(pt["output_tp"])
                    itls.append(pt["itl_med"])
            
            ax.plot(thpts, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.2, markersize=8, label=f"TP = {tp}")
            for c, x, y in zip(concurrencies, thpts, itls):
                ax.annotate(f"C={c}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color=COLORS[tp])
                
    ax.set_xlabel("Output Throughput (Generated Tokens / Sec) [Higher is better →]")
    ax.set_ylabel("Median Inter-Token Latency ITL (ms) [Lower is better ↓]")
    ax.set_title("Decode Pareto Frontier: Latency vs. Throughput", fontweight="bold", pad=12)
    ax.set_ylim(10, 55)
    ax.legend(loc="upper left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


# ==============================================================================
# 3. Plot: TTFT Prefill Scaling & Memory Cliff
# ==============================================================================
def plot_ttft_scaling(data, save_path=PLOTS_DIR / "ttft_scaling.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    prefill_key = ("prefill_heavy", 8192, 128)
    concurrencies = [1, 8, 32]
    
    if prefill_key in data:
        # Left Panel: Linear Scale (Exposing the TP=1 20s preemption cliff)
        for tp in [1, 2, 4]:
            ttfts = [data[prefill_key].get(c, {}).get(tp, {}).get("ttft_med", 0.0) for c in concurrencies]
            ax1.plot(concurrencies, ttfts, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.2, markersize=8, label=f"TP = {tp}")
        
        ax1.annotate("TP=1 Memory Saturation Cliff\n(20,372 ms @ C=32)", xy=(32, 20372), xytext=(12, 16000),
                     arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5), fontsize=8.5, fontweight="bold")
        ax1.set_xlabel("Concurrency (C)")
        ax1.set_ylabel("Median TTFT (ms)")
        ax1.set_title("Prefill TTFT Scaling (Linear Scale)", fontweight="bold")
        ax1.set_xticks(concurrencies)
        ax1.legend(loc="upper left")

        # Right Panel: Log Scale (Comparing TP=2 vs TP=4 scaling)
        for tp in [1, 2, 4]:
            ttfts = [data[prefill_key].get(c, {}).get(tp, {}).get("ttft_med", 0.0) for c in concurrencies]
            ax2.plot(concurrencies, ttfts, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.2, markersize=8, label=f"TP = {tp}")
            
        ax2.set_yscale("log")
        ax2.set_xlabel("Concurrency (C)")
        ax2.set_ylabel("Median TTFT (ms, Log Scale)")
        ax2.set_title("Prefill TTFT (Log Scale: TP=2 vs TP=4)", fontweight="bold")
        ax2.set_xticks(concurrencies)
        ax2.legend(loc="lower right")

    plt.suptitle("Prefill Scaling: 8192 Prompt Tokens (Qwen3.5-27B)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


# ==============================================================================
# 4. Plot: Decode ITL Scaling & Interconnect Plateau
# ==============================================================================
def plot_itl_scaling(data, save_path=PLOTS_DIR / "itl_scaling.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    decode_key = ("decode_heavy", 256, 1024)
    concurrencies = [1, 8, 32]
    
    if decode_key in data:
        for tp in [1, 2, 4]:
            itls = [data[decode_key].get(c, {}).get(tp, {}).get("itl_med", 0.0) for c in concurrencies]
            ax.plot(concurrencies, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.2, markersize=8, label=f"TP = {tp}")
            for c, itl in zip(concurrencies, itls):
                ax.annotate(f"{itl:.1f} ms", (c, itl), textcoords="offset points", xytext=(0, -14 if tp==2 and c==32 else 8), ha="center", fontsize=8.5, fontweight="bold")

    ax.annotate("Cross-Socket SYS Interconnect Bottleneck\nTP=4 (27.6ms) matches TP=2 (28.3ms)", xy=(32, 27.64), xytext=(10, 36),
                 arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5), fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Concurrency (C)")
    ax.set_ylabel("Median Inter-Token Latency ITL (ms)")
    ax.set_title("Decode ITL Scaling: 256 in x 1024 out", fontweight="bold", pad=12)
    ax.set_xticks(concurrencies)
    ax.set_ylim(10, 55)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


# ==============================================================================
# 5. Plot: MFU % & Scaling Efficiency
# ==============================================================================
def plot_mfu_efficiency(data, save_path=PLOTS_DIR / "mfu_efficiency.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    prefill_key = ("prefill_heavy", 8192, 128)
    decode_key = ("decode_heavy", 256, 1024)
    tp_sizes = [1, 2, 4]
    
    # Panel 1: Saturated MFU % at C=32
    if prefill_key in data and decode_key in data:
        p_mfu = [data[prefill_key][32][tp]["mfu"] for tp in tp_sizes]
        d_mfu = [data[decode_key][32][tp]["mfu"] for tp in tp_sizes]
        
        x = np.arange(len(tp_sizes))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, p_mfu, width, label="Prefill-Heavy (8k)", color="#2b5c8f")
        rects2 = ax1.bar(x + width/2, d_mfu, width, label="Decode-Heavy (1k)", color="#d95f02")
        
        ax1.set_ylabel("Model FLOPs Utilization (MFU %)")
        ax1.set_title("Hardware Utilization Ceiling (C = 32)", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"TP = {tp}" for tp in tp_sizes])
        ax1.set_ylim(0, 70)
        ax1.legend(loc="upper right")
        
        # Add labels on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

        # Panel 2: Total Throughput (tok/s) across C=1, 8, 32
        concurrencies = [1, 8, 32]
        x_c = np.arange(len(concurrencies))
        w = 0.25
        
        for i, tp in enumerate(tp_sizes):
            thpt = [data[prefill_key][c][tp]["total_tp"] for c in concurrencies]
            ax2.bar(x_c + (i - 1)*w, thpt, w, label=f"TP = {tp}", color=COLORS[tp])
            
        ax2.set_ylabel("Total Tokens / Second")
        ax2.set_title("Prefill Throughput Scaling vs. Concurrency", fontweight="bold")
        ax2.set_xticks(x_c)
        ax2.set_xticklabels([f"C = {c}" for c in concurrencies])
        ax2.legend(loc="upper left")

    plt.suptitle("Compute vs. Memory Bottlenecks & MFU Efficiency", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


# ==============================================================================
# 6. Master 2x3 Dashboard Figure
# ==============================================================================
def plot_master_dashboard(data, save_path=PLOTS_DIR / "macro_scaling_dashboard.png"):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)
    
    prefill_key = ("prefill_heavy", 8192, 128)
    decode_key = ("decode_heavy", 256, 1024)
    concurrencies = [1, 8, 32]
    tp_sizes = [1, 2, 4]
    
    # 1. Roofline (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ai_range = np.logspace(-1, 3.5, 400)
    roof = np.minimum((PEAK_HBM_BW_GBS * ai_range) / 1e3, PEAK_TFLOPS)
    ax1.plot(ai_range, roof, color="#333333", linewidth=2.2)
    ax1.axvline(RIDGE_POINT, color="#777777", linestyle=":", alpha=0.7)
    for c, ai_val in [(1, 1.0), (32, 32.0)]:
        for tp in [1, 2, 4]:
            pt = data[decode_key].get(c, {}).get(tp)
            if pt: ax1.scatter(ai_val, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=60)
    for tp in [1, 2, 4]:
        pt = data[prefill_key].get(32, {}).get(tp)
        if pt: ax1.scatter(360.0, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=70)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("A. Empirical Roofline Model", fontweight="bold")
    ax1.set_xlabel("Arithmetic Intensity (FLOPs/B)")
    ax1.set_ylabel("TFLOPs/s per GPU")

    # 2. Pareto Frontier (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    for tp in [1, 2, 4]:
        thpts = [data[decode_key][c][tp]["output_tp"] for c in concurrencies]
        itls = [data[decode_key][c][tp]["itl_med"] for c in concurrencies]
        ax2.plot(thpts, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.0, label=f"TP = {tp}")
    ax2.set_title("B. Decode Pareto Frontier", fontweight="bold")
    ax2.set_xlabel("Output Tok/s")
    ax2.set_ylabel("Median ITL (ms)")
    ax2.legend(loc="upper left", fontsize=8.5)

    # 3. TTFT Log Scaling (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    for tp in [1, 2, 4]:
        ttfts = [data[prefill_key][c][tp]["ttft_med"] for c in concurrencies]
        ax3.plot(concurrencies, ttfts, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.0, label=f"TP = {tp}")
    ax3.set_yscale("log")
    ax3.set_title("C. Prefill TTFT (Log Scale)", fontweight="bold")
    ax3.set_xlabel("Concurrency (C)")
    ax3.set_ylabel("Median TTFT (ms)")
    ax3.set_xticks(concurrencies)
    ax3.legend(loc="lower right", fontsize=8.5)

    # 4. ITL Decode Scaling (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0])
    for tp in [1, 2, 4]:
        itls = [data[decode_key][c][tp]["itl_med"] for c in concurrencies]
        ax4.plot(concurrencies, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.0, label=f"TP = {tp}")
    ax4.set_title("D. Decode ITL & Interconnect Plateau", fontweight="bold")
    ax4.set_xlabel("Concurrency (C)")
    ax4.set_ylabel("Median ITL (ms)")
    ax4.set_xticks(concurrencies)
    ax4.legend(loc="upper left", fontsize=8.5)

    # 5. MFU Utilization Bar (Bottom Center)
    ax5 = fig.add_subplot(gs[1, 1])
    p_mfu = [data[prefill_key][32][tp]["mfu"] for tp in tp_sizes]
    d_mfu = [data[decode_key][32][tp]["mfu"] for tp in tp_sizes]
    x = np.arange(len(tp_sizes))
    w = 0.35
    ax5.bar(x - w/2, p_mfu, w, label="Prefill (8k)", color="#2b5c8f")
    ax5.bar(x + w/2, d_mfu, w, label="Decode (1k)", color="#d95f02")
    ax5.set_title("E. Peak Hardware Utilization (C=32)", fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"TP={tp}" for tp in tp_sizes])
    ax5.set_ylabel("MFU (%)")
    ax5.legend(loc="upper right", fontsize=8.5)

    # 6. Total Throughput Comparison (Bottom Right)
    ax6 = fig.add_subplot(gs[1, 2])
    x_c = np.arange(len(concurrencies))
    w = 0.25
    for i, tp in enumerate(tp_sizes):
        thpt = [data[prefill_key][c][tp]["total_tp"] for c in concurrencies]
        ax6.bar(x_c + (i - 1)*w, thpt, w, label=f"TP={tp}", color=COLORS[tp])
    ax6.set_title("F. Prefill Throughput Ceiling", fontweight="bold")
    ax6.set_xticks(x_c)
    ax6.set_xticklabels([f"C={c}" for c in concurrencies])
    ax6.set_ylabel("Total Tokens / Sec")
    ax6.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("vLLM Tensor Parallelism Scaling Study (Qwen3.5-27B on 4x A100-80GB PCIe)", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


def main():
    print("[*] Loading macro benchmark dataset...")
    data = load_dataset()
    if not data:
        print("[!] Error: No benchmark JSON files found in results/tp*/")
        return
        
    print("[*] Generating publication plots...")
    plot_roofline(data)
    plot_pareto(data)
    plot_ttft_scaling(data)
    plot_itl_scaling(data)
    plot_mfu_efficiency(data)
    plot_master_dashboard(data)
    print(f"\n[✓] All 6 publication figures successfully generated in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
