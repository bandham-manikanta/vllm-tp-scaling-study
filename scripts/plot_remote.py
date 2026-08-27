import os, json, glob, sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path('/gpfs/projects/MaffeiGroup/open-source-contributions/vllm-tp-scaling-study')
TRACES_DIR = BASE_DIR / 'results' / 'traces'
PLOTS_DIR = BASE_DIR / 'results' / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

def classify_kernel(name: str) -> str:
    name_lower = name.lower()
    if 'nccl' in name_lower or 'allreduce' in name_lower or 'allgather' in name_lower:
        return 'nccl'
    elif 'gemm' in name_lower or 'cutlass' in name_lower or 's16816' in name_lower:
        return 'gemm'
    elif any(k in name_lower for k in ['causal_conv', 'delta_rule', 'merge_16x16', 'chunk_fwd', 'kkt', 'recompute_w_u', 'chunk_local']):
        return 'deltanet'
    elif 'flash' in name_lower or 'attention' in name_lower or 'splitkv' in name_lower:
        return 'flash_attn'
    elif 'norm' in name_lower or 'silu' in name_lower or 'elementwise' in name_lower or 'triton' in name_lower:
        return 'norm_elem'
    else:
        return 'other_compute'

all_results = {}
for tp in ['tp1', 'tp2', 'tp4']:
    all_results[tp] = {}
    for db_file in sorted(TRACES_DIR.glob(f'*_{tp}.sqlite')):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute('SELECT count(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL')
        num_devs = max(1, cur.fetchone()[0])
        
        stats = {'t_comp_ms': 0.0, 't_comm_ms': 0.0, 'categories_ms': {'gemm': 0.0, 'deltanet': 0.0, 'flash_attn': 0.0, 'norm_elem': 0.0, 'nccl': 0.0, 'other_compute': 0.0}}
        cur.execute('SELECT k.start, k.end, (k.end - k.start)/1e6, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id')
        for s, e, dur_ms, name in cur.fetchall():
            cat = classify_kernel(name)
            stats['categories_ms'][cat] += (dur_ms / num_devs)
        
        t_comm = stats['categories_ms']['nccl']
        t_comp = sum(v for k, v in stats['categories_ms'].items() if k != 'nccl')
        stats['t_comm_ms'] = t_comm
        stats['t_comp_ms'] = t_comp
        stats['total_gpu_time_ms'] = t_comm + t_comp
        all_results[tp][db_file.stem] = stats
        conn.close()

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
workloads_p = ['M=512', 'M=2048', 'M=8192']
x = np.arange(len(workloads_p))
width = 0.25

colors_comp = ['#1f77b4', '#2ca02c', '#d62728']
colors_comm = ['#aec7e8', '#98df8a', '#ff9896']

for i, tp in enumerate(['tp1', 'tp2', 'tp4']):
    t_comp = [all_results[tp][f'prefill_m{m}_{tp}']['t_comp_ms'] for m in [512, 2048, 8192]]
    t_comm = [all_results[tp][f'prefill_m{m}_{tp}']['t_comm_ms'] for m in [512, 2048, 8192]]
    pos = x + (i - 1) * width
    ax1.bar(pos, t_comp, width, label=tp.upper() + ' Compute', color=colors_comp[i], alpha=0.85)
    ax1.bar(pos, t_comm, width, bottom=t_comp, label=tp.upper() + ' Comm (NCCL)', color=colors_comm[i], hatch='//')

ax1.set_title('Prefill Micro Scaling: Compute vs. NCCL Communication', fontsize=13, fontweight='bold')
ax1.set_xlabel('Prefill Prompt Length (Tokens)', fontsize=11)
ax1.set_ylabel('Kernel Execution Time per Step (ms)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(workloads_p, fontsize=11)
ax1.legend(fontsize=9, loc='upper left')

ax2 = axes[1]
workloads_d = ['B=1', 'B=8', 'B=32']
x2 = np.arange(len(workloads_d))

for i, tp in enumerate(['tp1', 'tp2', 'tp4']):
    t_comp = [all_results[tp][f'decode_b{b}_{tp}']['t_comp_ms'] for b in [1, 8, 32]]
    t_comm = [all_results[tp][f'decode_b{b}_{tp}']['t_comm_ms'] for b in [1, 8, 32]]
    pos = x2 + (i - 1) * width
    ax2.bar(pos, t_comp, width, label=tp.upper() + ' Compute', color=colors_comp[i], alpha=0.85)
    ax2.bar(pos, t_comm, width, bottom=t_comp, label=tp.upper() + ' Comm (NCCL)', color=colors_comm[i], hatch='//')

ax2.set_title('Decode Micro Scaling: Compute vs. NCCL Communication', fontsize=13, fontweight='bold')
ax2.set_xlabel('Decode Batch Size (Users)', fontsize=11)
ax2.set_ylabel('Kernel Execution Time per Step (ms)', fontsize=11)
ax2.set_xticks(x2)
ax2.set_xticklabels(workloads_d, fontsize=11)
ax2.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'micro_scaling_decomposition.png', dpi=300)
print('Generated Plot:', PLOTS_DIR / 'micro_scaling_decomposition.png')
