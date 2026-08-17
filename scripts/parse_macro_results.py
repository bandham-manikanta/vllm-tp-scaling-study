#!/usr/bin/env python3
"""
Parses vLLM macro benchmark JSON results and outputs two comparison Markdown tables:
1. Prefill-Heavy (8192 in x 128 out)
2. Decode-Heavy (256 in x 1024 out)
"""

import json
import glob
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUTPUT_MD_PATH = os.path.join(BASE_RESULTS_DIR, "MACRO_BENCHMARK_SUMMARY.md")

MODEL_PARAMS = 27.2e9  # 27.2B params for Qwen3.5-27B
PEAK_TFLOPS_PER_GPU = 311.96e12  # A100 PCIe BF16 Tensor Core Peak (311.96 TFLOPs)

def compute_mfu(total_tok_per_sec, tp_size):
    if not total_tok_per_sec or not tp_size:
        return 0.0
    flop_per_token = 2.0 * MODEL_PARAMS
    total_flops = total_tok_per_sec * flop_per_token
    peak_hardware_flops = tp_size * PEAK_TFLOPS_PER_GPU
    return (total_flops / peak_hardware_flops) * 100.0

def calc_latency_speedup(val_tp1, val_tpn):
    """Lower is better: Speedup = TP1 / TPN"""
    if val_tp1 and val_tpn and val_tpn > 0:
        return f"{val_tp1 / val_tpn:.2f}x"
    return "N/A"

def calc_throughput_speedup(val_tp1, val_tpn):
    """Higher is better: Speedup = TPN / TP1"""
    if val_tp1 and val_tpn and val_tpn > 0:
        return f"{val_tpn / val_tp1:.2f}x"
    return "N/A"

def calc_mfu_efficiency(m1, mn):
    """Parallel scaling efficiency: Efficiency = MFU_TPN / MFU_TP1"""
    if m1 and mn and m1 > 0:
        return f"{mn / m1:.2f}x"
    return "N/A"

def load_results(base_dir=BASE_RESULTS_DIR):
    data = {}
    pattern = os.path.join(base_dir, "tp*", "*.json")
    files = glob.glob(pattern)
    
    for fpath in files:
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
        
        tpot_med = res.get("median_tpot_ms") or res.get("median_itl_ms") or res.get("mean_tpot_ms") or 0.0
        tpot_p99 = res.get("p99_tpot_ms") or res.get("p99_itl_ms") or res.get("p95_tpot_ms") or 0.0
        
        out_tp = res.get("output_throughput") or 0.0
        total_tp = res.get("total_token_throughput") or res.get("tokens_per_second") or (out_tp * (in_len + out_len) / out_len if out_len else out_tp)
        
        key = (profile, in_len, out_len)
        if key not in data:
            data[key] = {}
        if concurrency not in data[key]:
            data[key][concurrency] = {}
            
        data[key][concurrency][tp] = {
            "ttft_med": ttft_med,
            "ttft_p99": ttft_p99,
            "tpot_med": tpot_med,
            "tpot_p99": tpot_p99,
            "output_tp": out_tp,
            "total_tp": total_tp,
            "mfu": compute_mfu(total_tp, tp)
        }
        
    return data

def generate_markdown_tables(data, out_path=OUTPUT_MD_PATH):
    lines = []
    
    # 1. Prefill-Heavy Table
    prefill_key = ("prefill_heavy", 8192, 128)
    lines.append("### Table 1: Prefill-Heavy Scaling (`8192` in $\\times$ `128` out)\n")
    lines.append("| Concurrency ($C$) | Metric | TP = 1 | TP = 2 | TP = 4 | Speedup / Ratio (TP2 / TP1) | Speedup / Ratio (TP4 / TP1) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    if prefill_key in data:
        p_data = data[prefill_key]
        for c in sorted(p_data.keys()):
            c_tp = p_data[c]
            tp1 = c_tp.get(1, {})
            tp2 = c_tp.get(2, {})
            tp4 = c_tp.get(4, {})
            
            # Median TTFT (Lower is better)
            t1_m = tp1.get("ttft_med", 0.0)
            t2_m = tp2.get("ttft_med", 0.0)
            t4_m = tp4.get("ttft_med", 0.0)
            sp2 = calc_latency_speedup(t1_m, t2_m)
            sp4 = calc_latency_speedup(t1_m, t4_m)
            lines.append(f"| **$C = {c}$** | **Median TTFT (ms)** | {t1_m:,.1f} | {t2_m:,.1f} | {t4_m:,.1f} | {sp2} | {sp4} |")
            
            # p99 TTFT (Lower is better)
            t1_p99 = tp1.get("ttft_p99", 0.0)
            t2_p99 = tp2.get("ttft_p99", 0.0)
            t4_p99 = tp4.get("ttft_p99", 0.0)
            sp2_p99 = calc_latency_speedup(t1_p99, t2_p99)
            sp4_p99 = calc_latency_speedup(t1_p99, t4_p99)
            lines.append(f"| | **p99 TTFT (ms)** | {t1_p99:,.1f} | {t2_p99:,.1f} | {t4_p99:,.1f} | {sp2_p99} | {sp4_p99} |")
            
            # Total Throughput (Higher is better)
            tp1_t = tp1.get("total_tp", 0.0)
            tp2_t = tp2.get("total_tp", 0.0)
            tp4_t = tp4.get("total_tp", 0.0)
            sp2_t = calc_throughput_speedup(tp1_t, tp2_t)
            sp4_t = calc_throughput_speedup(tp1_t, tp4_t)
            lines.append(f"| | **Total Throughput (tok/s)** | {tp1_t:,.1f} | {tp2_t:,.1f} | {tp4_t:,.1f} | {sp2_t} | {sp4_t} |")
            
            # MFU & Parallel Scaling Efficiency
            m1 = tp1.get("mfu", 0.0)
            m2 = tp2.get("mfu", 0.0)
            m4 = tp4.get("mfu", 0.0)
            sp2_m = calc_mfu_efficiency(m1, m2)
            sp4_m = calc_mfu_efficiency(m1, m4)
            lines.append(f"| | **MFU (%)** | {m1:.1f}% | {m2:.1f}% | {m4:.1f}% | {sp2_m} | {sp4_m} |")
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        
    lines.append("\n---\n")
    
    # 2. Decode-Heavy Table
    decode_key = ("decode_heavy", 256, 1024)
    lines.append("### Table 2: Decode-Heavy Scaling (`256` in $\\times$ `1024` out)\n")
    lines.append("| Concurrency ($C$) | Metric | TP = 1 | TP = 2 | TP = 4 | Speedup / Ratio (TP2 / TP1) | Speedup / Ratio (TP4 / TP1) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    if decode_key in data:
        d_data = data[decode_key]
        for c in sorted(d_data.keys()):
            c_tp = d_data[c]
            tp1 = c_tp.get(1, {})
            tp2 = c_tp.get(2, {})
            tp4 = c_tp.get(4, {})
            
            # Median ITL (Lower is better)
            t1_m = tp1.get("tpot_med", 0.0)
            t2_m = tp2.get("tpot_med", 0.0)
            t4_m = tp4.get("tpot_med", 0.0)
            sp2 = calc_latency_speedup(t1_m, t2_m)
            sp4 = calc_latency_speedup(t1_m, t4_m)
            lines.append(f"| **$C = {c}$** | **Median ITL / TPOT (ms)** | {t1_m:.2f} | {t2_m:.2f} | {t4_m:.2f} | {sp2} | {sp4} |")
            
            # p99 ITL (Lower is better)
            t1_p99 = tp1.get("tpot_p99", 0.0)
            t2_p99 = tp2.get("tpot_p99", 0.0)
            t4_p99 = tp4.get("tpot_p99", 0.0)
            sp2_p99 = calc_latency_speedup(t1_p99, t2_p99)
            sp4_p99 = calc_latency_speedup(t1_p99, t4_p99)
            lines.append(f"| | **p99 ITL / TPOT (ms)** | {t1_p99:.2f} | {t2_p99:.2f} | {t4_p99:.2f} | {sp2_p99} | {sp4_p99} |")
            
            # Output Throughput (Higher is better)
            tp1_o = tp1.get("output_tp", 0.0)
            tp2_o = tp2.get("output_tp", 0.0)
            tp4_o = tp4.get("output_tp", 0.0)
            sp2_o = calc_throughput_speedup(tp1_o, tp2_o)
            sp4_o = calc_throughput_speedup(tp1_o, tp4_o)
            lines.append(f"| | **Output Throughput (tok/s)** | {tp1_o:,.1f} | {tp2_o:,.1f} | {tp4_o:,.1f} | {sp2_o} | {sp4_o} |")
            
            # Total Throughput (Higher is better)
            tp1_t = tp1.get("total_tp", 0.0)
            tp2_t = tp2.get("total_tp", 0.0)
            tp4_t = tp4.get("total_tp", 0.0)
            sp2_t = calc_throughput_speedup(tp1_t, tp2_t)
            sp4_t = calc_throughput_speedup(tp1_t, tp4_t)
            lines.append(f"| | **Total Throughput (tok/s)** | {tp1_t:,.1f} | {tp2_t:,.1f} | {tp4_t:,.1f} | {sp2_t} | {sp4_t} |")
            
            # MFU & Parallel Scaling Efficiency
            m1 = tp1.get("mfu", 0.0)
            m2 = tp2.get("mfu", 0.0)
            m4 = tp4.get("mfu", 0.0)
            sp2_m = calc_mfu_efficiency(m1, m2)
            sp4_m = calc_mfu_efficiency(m1, m4)
            lines.append(f"| | **MFU (%)** | {m1:.1f}% | {m2:.1f}% | {m4:.1f}% | {sp2_m} | {sp4_m} |")
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        
    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"[✓] Summary written to {out_path}")

if __name__ == "__main__":
    results = load_results(BASE_RESULTS_DIR)
    generate_markdown_tables(results, OUTPUT_MD_PATH)
