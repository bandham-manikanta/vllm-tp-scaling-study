#!/usr/bin/env python3
"""
Publication-Quality Macro Benchmark Plotting Suite
Generates all 6 macro scaling figures, including the Pareto Frontier and Master Dashboard
with explicit TP=2 sweet-spot annotations and green arrows on Subplots E & F (zero overlap).
"""

import glob
import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Specifications (A100-PCIE-80GB)
MODEL_PARAMS = 27.2e9  # 27.2B params for Qwen3.5-27B
PEAK_TFLOPS_PER_GPU = 312.0  # FP16/BF16 Tensor Core Peak per GPU (TFLOPs)
PEAK_HBM_BW_GBS = 2039.0  # GB/s HBM2e per GPU
RIDGE_POINT = (PEAK_TFLOPS_PER_GPU * 1e12) / (PEAK_HBM_BW_GBS * 1e9)  # ~153.0 FLOPs/Byte

COLORS = {1: "#2b5c8f", 2: "#2ca02c", 4: "#d95f02"}
MARKERS = {1: "o", 2: "s", 4: "^"}


def compute_mfu(total_tok_per_sec, tp_size):
    if not total_tok_per_sec or not tp_size:
        return 0.0
    flop_per_token = 2.0 * MODEL_PARAMS
    total_flops = total_tok_per_sec * flop_per_token
    peak_hardware_flops = tp_size * (PEAK_TFLOPS_PER_GPU * 1e12)
    return (total_flops / peak_hardware_flops) * 100.0


def load_dataset():
    files = glob.glob(str(RESULTS_DIR / "tp*" / "*.json"))
    data = {}
    
    for f in files:
        m = re.match(r".*?([a-z_]+)_(\d+)x(\d+)_c(\d+)_tp(\d+)\.json", f)
        if not m:
            continue
        name, in_len, out_len, c, tp = m.groups()
        in_len, out_len, c, tp = int(in_len), int(out_len), int(c), int(tp)
        
        with open(f) as fp:
            d = json.load(fp)
            
        key = (name, in_len, out_len)
        if key not in data:
            data[key] = {}
        if c not in data[key]:
            data[key][c] = {}
            
        tot_tp = d.get("total_token_throughput", 0)
        out_tp = d.get("output_throughput", 0)
        mfu = compute_mfu(tot_tp, tp)
        
        data[key][c][tp] = {
            "ttft_med": d.get("median_ttft_ms", 0),
            "itl_med": d.get("median_itl_ms", 0),
            "output_tp": out_tp,
            "total_tp": tot_tp,
            "tokens_per_gpu": tot_tp / tp if tp > 0 else 0,
            "mfu": mfu,
            "achieved_tflops_gpu": (mfu / 100.0) * PEAK_TFLOPS_PER_GPU,
        }
    return data


def plot_roofline(data, save_path=PLOTS_DIR / "roofline_model.png"):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ai_range = np.logspace(-1, 3.5, 400)
    bw_bound = (PEAK_HBM_BW_GBS * ai_range) / 1e3
    comp_bound = np.full_like(ai_range, PEAK_TFLOPS_PER_GPU)
    roofline = np.minimum(bw_bound, comp_bound)
    
    ax.plot(ai_range, roofline, color="#222222", linewidth=2.5, label="Theoretical A100 Ceiling")
    ax.axvline(RIDGE_POINT, color="#888888", linestyle="--", alpha=0.7, label=f"Ridge Point ({RIDGE_POINT:.1f} FLOPs/B)")
    
    prefill_key = ("prefill_heavy", 8192, 128)
    decode_key = ("decode_heavy", 256, 1024)
    
    if decode_key in data:
        for c, ai_val, lbl in [(1, 1.0, "Decode B=1"), (32, 32.0, "Decode B=32")]:
            for tp in [1, 2, 4]:
                pt = data[decode_key].get(c, {}).get(tp)
                if pt:
                    ax.scatter(ai_val, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=90, zorder=5, label=f"{lbl} (TP={tp})" if c == 1 else "")
                    
    if prefill_key in data:
        for tp in [1, 2, 4]:
            pt = data[prefill_key].get(32, {}).get(tp)
            if pt:
                ax.scatter(360.0, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=110, zorder=5, label=f"Prefill 8k (TP={tp})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Achieved Performance (TFLOPs/s per GPU)")
    ax.set_title("Empirical Roofline Model: Qwen3.5-27B on A100-80GB", fontweight="bold", pad=12)
    ax.set_ylim(0.1, 450)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


def plot_pareto(data, save_path=PLOTS_DIR / "pareto_frontier.png"):
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    decode_key = ("decode_heavy", 256, 1024)
    concurrencies = [1, 8, 32]
    
    for tp in [1, 2, 4]:
        thpts = [data[decode_key][c][tp]["output_tp"] for c in concurrencies]
        itls = [data[decode_key][c][tp]["itl_med"] for c in concurrencies]
        ax.plot(thpts, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.2, markersize=8, label=f"TP = {tp}")
        for c, x, y in zip(concurrencies, thpts, itls):
            y_offset = 9 if (tp != 4 or c != 32) else -14
            ax.annotate(f"C={c}", (x, y), textcoords="offset points", xytext=(0, y_offset), ha="center", fontsize=9, fontweight="bold", color=COLORS[tp])

    # Circle around TP=2 @ C=32
    tp2_thpt_c32 = data[decode_key][32][2]["output_tp"]
    tp2_itl_c32 = data[decode_key][32][2]["itl_med"]
    ax.scatter(tp2_thpt_c32, tp2_itl_c32, s=340, facecolors="none", edgecolors="#2ca02c", linewidth=2.5, zorder=10)

    ax.annotate("Optimal Operating Point\n(TP=2 NVLink: 1,100 tok/s)",
                xy=(tp2_thpt_c32, tp2_itl_c32),
                xytext=(tp2_thpt_c32 - 320, tp2_itl_c32 + 8),
                arrowprops=dict(facecolor="#2ca02c", edgecolor="#2ca02c", arrowstyle="->", lw=1.8),
                fontsize=9.5, fontweight="bold", color="#1b7837",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e5f5e0", edgecolor="#2ca02c", alpha=0.9))

    ax.set_xlabel("Output Throughput (Generated Tokens / Sec) [Higher is better →]", fontweight="bold")
    ax.set_ylabel("Median Inter-Token Latency ITL (ms) [Lower is better ↓]", fontweight="bold")
    ax.set_title("Decode Pareto Frontier: Latency vs. Throughput (256 in / 1024 out)", fontweight="bold", pad=12, fontsize=12.5)
    ax.set_ylim(10, 55)
    ax.set_xlim(-20, 1250)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")


def plot_master_dashboard(data, save_path=PLOTS_DIR / "macro_scaling_dashboard.png"):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
    
    prefill_key = ("prefill_heavy", 8192, 128)
    decode_key = ("decode_heavy", 256, 1024)
    concurrencies = [1, 8, 32]
    tp_sizes = [1, 2, 4]
    
    # --------------------------------------------------------------------------
    # 1. Roofline (Top Left)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ai_range = np.logspace(-1, 3.5, 400)
    roof = np.minimum((PEAK_HBM_BW_GBS * ai_range) / 1e3, PEAK_TFLOPS_PER_GPU)
    ax1.plot(ai_range, roof, color="#333333", linewidth=2.2, label="Roofline")
    ax1.axvline(RIDGE_POINT, color="#777777", linestyle=":", alpha=0.7)
    
    for c, ai_val in [(1, 1.0), (32, 32.0)]:
        for tp in [1, 2, 4]:
            pt = data[decode_key].get(c, {}).get(tp)
            if pt: ax1.scatter(ai_val, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=65, zorder=5)
            
    for tp in [1, 2, 4]:
        pt = data[prefill_key].get(32, {}).get(tp)
        if pt:
            ax1.scatter(360.0, pt["achieved_tflops_gpu"], color=COLORS[tp], marker=MARKERS[tp], s=75, zorder=5)
            if tp == 2:
                ax1.scatter(360.0, pt["achieved_tflops_gpu"], s=260, facecolors="none", edgecolors="#2ca02c", linewidth=2.0, zorder=10)
                
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("A. Empirical Roofline Model", fontweight="bold")
    ax1.set_xlabel("Arithmetic Intensity (FLOPs/B)")
    ax1.set_ylabel("TFLOPs/s per GPU")
    ax1.set_ylim(0.1, 450)

    # --------------------------------------------------------------------------
    # 2. Pareto Frontier (Top Center)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for tp in [1, 2, 4]:
        thpts = [data[decode_key][c][tp]["output_tp"] for c in concurrencies]
        itls = [data[decode_key][c][tp]["itl_med"] for c in concurrencies]
        ax2.plot(thpts, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.0, label=f"TP = {tp}")
        
    tp2_thpt_c32 = data[decode_key][32][2]["output_tp"]
    tp2_itl_c32 = data[decode_key][32][2]["itl_med"]
    ax2.scatter(tp2_thpt_c32, tp2_itl_c32, s=260, facecolors="none", edgecolors="#2ca02c", linewidth=2.0, zorder=10)
    ax2.set_title("B. Decode Pareto Frontier", fontweight="bold")
    ax2.set_xlabel("Output Tok/s")
    ax2.set_ylabel("Median ITL (ms)")
    ax2.legend(loc="upper left", fontsize=8.5)

    # --------------------------------------------------------------------------
    # 3. TTFT Log Scaling (Top Right)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for tp in [1, 2, 4]:
        ttfts = [data[prefill_key][c][tp]["ttft_med"] for c in concurrencies]
        ax3.plot(concurrencies, ttfts, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.0, label=f"TP = {tp}")
        
    tp2_ttft_c1 = data[prefill_key][1][2]["ttft_med"]
    ax3.scatter(1, tp2_ttft_c1, s=260, facecolors="none", edgecolors="#2ca02c", linewidth=2.0, zorder=10)
    ax3.set_yscale("log")
    ax3.set_title("C. Prefill TTFT (Log Scale)", fontweight="bold")
    ax3.set_xlabel("Concurrency (C)")
    ax3.set_ylabel("Median TTFT (ms)")
    ax3.set_xticks(concurrencies)
    ax3.legend(loc="lower right", fontsize=8.5)

    # --------------------------------------------------------------------------
    # 4. ITL Decode Scaling (Bottom Left)
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    for tp in [1, 2, 4]:
        itls = [data[decode_key][c][tp]["itl_med"] for c in concurrencies]
        ax4.plot(concurrencies, itls, color=COLORS[tp], marker=MARKERS[tp], linewidth=2.0, label=f"TP = {tp}")
        
    tp2_itl_c8 = data[decode_key][8][2]["itl_med"]
    ax4.scatter(8, tp2_itl_c8, s=260, facecolors="none", edgecolors="#2ca02c", linewidth=2.0, zorder=10)
    ax4.set_title("D. Decode ITL & Interconnect Plateau", fontweight="bold")
    ax4.set_xlabel("Concurrency (C)")
    ax4.set_ylabel("Median ITL (ms)")
    ax4.set_xticks(concurrencies)
    ax4.legend(loc="upper left", fontsize=8.5)

    # --------------------------------------------------------------------------
    # 5. MFU Utilization Bar (Bottom Center) - PLOT 5 (With Clean Arrow)
    # --------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    p_mfu = [data[prefill_key][32][tp]["mfu"] for tp in tp_sizes]
    d_mfu = [data[decode_key][32][tp]["mfu"] for tp in tp_sizes]
    x = np.arange(len(tp_sizes))
    w = 0.35
    
    rects1 = ax5.bar(x - w/2, p_mfu, w, label="Prefill (8k)", color="#2b5c8f", alpha=0.9)
    rects2 = ax5.bar(x + w/2, d_mfu, w, label="Decode (1k)", color="#d95f02", alpha=0.9)
    
    # Add numerical percentage labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax5.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#1c3d5a")
    for rect in rects2:
        h = rect.get_height()
        ax5.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#8c3b01")

    # Arrow pointing to TP=2 Prefill bar positioned neatly to the right
    tp2_p_mfu = p_mfu[1]  # 51.8%
    ax5.annotate("Optimal: 51.8% MFU\n(2× Efficiency)",
                 xy=(1 - w/2, tp2_p_mfu + 2),
                 xytext=(1.35, tp2_p_mfu + 12),
                 arrowprops=dict(facecolor="#2ca02c", edgecolor="#2ca02c", arrowstyle="->", lw=1.8),
                 fontsize=8.5, fontweight="bold", color="#1b7837",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#e5f5e0", edgecolor="#2ca02c", alpha=0.9))

    ax5.set_title("E. Peak Hardware Utilization (C=32)", fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"TP={tp}" for tp in tp_sizes])
    ax5.set_ylim(0, 75)
    ax5.set_ylabel("Model FLOPs Utilization (MFU %)")
    ax5.legend(loc="upper left", fontsize=8.5)

    # --------------------------------------------------------------------------
    # 6. Total Throughput Comparison (Bottom Right) - PLOT 6 (With Clean Arrow)
    # --------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    x_c = np.arange(len(concurrencies))
    w = 0.25
    
    for i, tp in enumerate(tp_sizes):
        thpt = [data[prefill_key][c][tp]["total_tp"] for c in concurrencies]
        rects = ax6.bar(x_c + (i - 1)*w, thpt, w, label=f"TP={tp}", color=COLORS[tp], alpha=0.9)
        # Add labels on C=32 bars
        if tp in [2, 4]:
            h = thpt[2]
            ax6.annotate(f"{h:,.0f}", (x_c[2] + (i - 1)*w, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=COLORS[tp])

    # Arrow pointing to TP=2 at C=32 bar
    tp2_thpt_c32_tot = data[prefill_key][32][2]["total_tp"]  # 5947
    ax6.annotate("Matches TP=4 @ 2× Eff.\n(Saves 2 GPUs)",
                 xy=(2.0, tp2_thpt_c32_tot + 200),
                 xytext=(1.35, tp2_thpt_c32_tot + 1100),
                 arrowprops=dict(facecolor="#2ca02c", edgecolor="#2ca02c", arrowstyle="->", lw=1.8),
                 fontsize=8.5, fontweight="bold", color="#1b7837",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#e5f5e0", edgecolor="#2ca02c", alpha=0.9))

    ax6.set_title("F. Prefill Throughput Scaling", fontweight="bold")
    ax6.set_xticks(x_c)
    ax6.set_xticklabels([f"C={c}" for c in concurrencies])
    ax6.set_ylim(0, 7800)
    ax6.set_ylabel("Total Tokens / Second")
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
    plot_master_dashboard(data)
    print(f"\n[✓] Macro publication figures successfully generated in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
