#!/usr/bin/env python3
"""
Automated CUPTI Trace Parsing Engine
Extracts GPU kernel execution, pure compute (T_comp), and communication (T_comm)
from all 18 Nsight Systems SQLite trace databases across TP=1, TP=2, and TP=4.
"""

import json
import os
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TRACES_DIR = RESULTS_DIR / "traces"
OUTPUT_MD = RESULTS_DIR / "MICRO_BENCHMARK_SUMMARY.md"
OUTPUT_JSON = RESULTS_DIR / "micro_trace_results.json"


def classify_kernel(name: str) -> str:
    name_lower = name.lower()
    if "nccl" in name_lower or "allreduce" in name_lower or "allgather" in name_lower:
        return "nccl"
    elif "gemm" in name_lower or "cutlass" in name_lower or "s16816" in name_lower:
        return "gemm"
    elif any(k in name_lower for k in ["causal_conv", "delta_rule", "merge_16x16", "chunk_fwd", "kkt", "recompute_w_u", "chunk_local"]):
        return "deltanet"
    elif "flash" in name_lower or "attention" in name_lower or "splitkv" in name_lower:
        return "flash_attn"
    elif "norm" in name_lower or "silu" in name_lower or "elementwise" in name_lower or "triton" in name_lower:
        return "norm_elem"
    else:
        return "other_compute"


def parse_sqlite_trace(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    
    stats = {
        "trace_name": db_path.stem,
        "num_devices": 1,
        "step_wall_clock_ms": 0.0,
        "total_gpu_time_ms": 0.0,
        "t_comp_ms": 0.0,
        "t_comm_ms": 0.0,
        "comm_pct": 0.0,
        "comp_pct": 100.0,
        "categories_ms": {
            "gemm": 0.0,
            "deltanet": 0.0,
            "flash_attn": 0.0,
            "norm_elem": 0.0,
            "nccl": 0.0,
            "other_compute": 0.0
        },
        "is_graph": False,
        "graph_count": 0,
        "kernel_count": 0
    }
    
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        cur.execute("SELECT count(*), min(start)/1e6, max(end)/1e6, (max(end)-min(start))/1e6 FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        g_cnt, g_start, g_end, g_span = cur.fetchone()
        if g_cnt and g_cnt > 0:
            stats["is_graph"] = True
            stats["graph_count"] = g_cnt
    
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        cur.execute("SELECT count(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        num_devs = max(1, cur.fetchone()[0])
        stats["num_devices"] = num_devs
        
        cur.execute("SELECT (max(end) - min(start))/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL")
        span = cur.fetchone()[0]
        stats["step_wall_clock_ms"] = span if span else 0.0
        
        cur.execute("""
            SELECT k.start, k.end, (k.end - k.start)/1e6 as dur_ms, s.value, k.deviceId
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start ASC
        """)
        rows = cur.fetchall()
        stats["kernel_count"] = len(rows)
        
        for start, end, dur_ms, name, dev_id in rows:
            cat = classify_kernel(name)
            stats["categories_ms"][cat] += (dur_ms / num_devs)
            
        t_comm = stats["categories_ms"]["nccl"]
        t_comp = sum(v for k, v in stats["categories_ms"].items() if k != "nccl")
        t_tot = t_comm + t_comp
        
        stats["t_comm_ms"] = t_comm
        stats["t_comp_ms"] = t_comp
        stats["total_gpu_time_ms"] = t_tot
        
        if t_tot > 0:
            stats["comm_pct"] = (t_comm / t_tot) * 100.0
            stats["comp_pct"] = (t_comp / t_tot) * 100.0
            
    conn.close()
    return stats


def main():
    print("=" * 80)
    print("PARSING ALL 18 NSIGHT TRACE DATABASES ACROSS TP=1, TP=2, TP=4")
    print("=" * 80)
    
    all_results = {"tp1": {}, "tp2": {}, "tp4": {}}
    
    sqlite_files = sorted(TRACES_DIR.glob("*.sqlite"))
    if not sqlite_files:
        print(f"[!] Error: No .sqlite files found in {TRACES_DIR}")
        return
        
    for db_file in sqlite_files:
        m = re.search(r"_(tp\d+)", db_file.stem)
        tp = m.group(1) if m else "tp1"
        if tp not in all_results:
            all_results[tp] = {}
        print(f"Parsing: {db_file.name} ...")
        stats = parse_sqlite_trace(db_file)
        all_results[tp][db_file.stem] = stats
            
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Saved raw parsed results to: {OUTPUT_JSON}")
    
    lines = []
    lines.append("# Micro-Trace Profiling Summary (GPU Kernel Compute vs. Communication Breakdown)\n")
    lines.append("Parsed from all 18 local Nsight Systems SQLite databases across NVIDIA A100-80GB GPUs.\n")
    
    lines.append("## 1. Prefill Workload Scaling ($T_{\\text{comp}}$ vs. $T_{\\text{comm}}$)\n")
    lines.append("| Workload | TP | Total Kernel Time (ms) | Compute $T_{\\text{comp}}$ (ms) | Comm $T_{\\text{comm}}$ (ms) | Comm Overhead (%) | GEMM (ms) | DeltaNet (ms) | FlashAttn (ms) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for m in [512, 2048, 8192]:
        for tp in ["tp1", "tp2", "tp4"]:
            key = f"prefill_m{m}_{tp}"
            data = all_results.get(tp, {}).get(key)
            if data:
                c = data["categories_ms"]
                lines.append(
                    f"| **Prefill M={m}** | {tp.upper()} | {data['total_gpu_time_ms']:8.2f} | "
                    f"{data['t_comp_ms']:8.2f} ({data['comp_pct']:5.1f}%) | "
                    f"{data['t_comm_ms']:8.2f} ({data['comm_pct']:5.1f}%) | "
                    f"**{data['comm_pct']:5.1f}%** | "
                    f"{c['gemm']:7.2f} | {c['deltanet']:7.2f} | {c['flash_attn']:7.2f} |"
                )
                
    lines.append("\n## 2. Decode Workload Scaling ($T_{\\text{comp}}$ vs. $T_{\\text{comm}}$)\n")
    lines.append("| Workload | TP | Total Kernel Time (ms) | Compute $T_{\\text{comp}}$ (ms) | Comm $T_{\\text{comm}}$ (ms) | Comm Overhead (%) | GEMM (ms) | DeltaNet (ms) | FlashAttn (ms) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for b in [1, 8, 32]:
        for tp in ["tp1", "tp2", "tp4"]:
            key = f"decode_b{b}_{tp}"
            data = all_results.get(tp, {}).get(key)
            if data:
                c = data["categories_ms"]
                lines.append(
                    f"| **Decode B={b}** | {tp.upper()} | {data['total_gpu_time_ms']:8.2f} | "
                    f"{data['t_comp_ms']:8.2f} ({data['comp_pct']:5.1f}%) | "
                    f"{data['t_comm_ms']:8.2f} ({data['comm_pct']:5.1f}%) | "
                    f"**{data['comm_pct']:5.1f}%** | "
                    f"{c['gemm']:7.2f} | {c['deltanet']:7.2f} | {c['flash_attn']:7.2f} |"
                )
                
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"[✓] Generated Markdown Summary: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
