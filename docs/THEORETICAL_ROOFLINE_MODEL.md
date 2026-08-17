# Theoretical Roofline: Qwen3.5-27B on 4x A100-PCIe-80GB

## 1. Hardware Specs (NVIDIA A100-PCIe-80GB)

| Metric | Per-GPU | 4x GPU Node |
| :--- | :--- | :--- |
| **Dense BF16 Compute** | 311.96 TFLOPs/s | 1,247.84 TFLOPs/s |
| **HBM2e Bandwidth** | 1,935.36 GB/s | 7,741.44 GB/s |
| **Ridge Point** | 161.21 FLOPs/Byte | 161.21 FLOPs/Byte |
| **NVLink-3 (`NV12`)** | 600 GB/s (bi-dir) | — |
| **Host Interconnect (`SYS`)** | ~32–64 GB/s | — |

---

## 2. Model Specs (`Qwen/Qwen3.5-27B`)

| Metric | Value |
| :--- | :--- |
| **Parameters ($P$)** | 27.2 Billion |
| **Weights Size (BF16)** | 54.4 GB |
| **Total Layers ($L$)** | 64 (48 Linear-Attn + 16 Full-Attn) |
| **Hidden Size ($d_{\text{model}}$)** | 5,120 |
| **Intermediate Size ($d_{\text{ffn}}$)** | 17,408 |
| **Attention Heads ($N_q / N_{\text{kv}}$)** | 24 / 4 ($d_k = 256$) |
| **Vocabulary Size** | 248,320 |
| **KV Cache Rate** | 64.0 KB / token |

---

## 3. TP Scaling Ceilings

| Metric | TP = 1 | TP = 2 | TP = 4 |
| :--- | :--- | :--- | :--- |
| **Peak Compute** | 312.0 TFLOPs/s | 624.0 TFLOPs/s | 1,248.0 TFLOPs/s |
| **Peak Memory Bandwidth** | 1,935.4 GB/s | 3,870.7 GB/s | 7,741.4 GB/s |
| **Model Weights / GPU** | 54.4 GB | 27.2 GB | 13.6 GB |
| **KV VRAM Pool** | 25.6 GB | 105.6 GB | 265.6 GB |
| **KV Rate / GPU** | 64.0 KB/token | 32.0 KB/token | 16.0 KB/token |
| **KV Heads / GPU** | 4 heads | 2 heads | 1 head |

---

## 4. Communication Parameters

| Metric | Value |
| :--- | :--- |
| **AllReduce Calls / Step** | 128 ($2 \times 64\text{ layers}$) |
| **Decode ($B=1$) Message Size** | 10.24 KB |
| **Decode ($B=32$) Message Size** | 327.68 KB |
| **Prefill ($M=2048$) Message Size** | 20.97 MB |
| **Prefill ($M=8192$) Message Size** | 83.89 MB |
| **TP = 2 Comm Bus** | NVLink-3 (`NV12`, 600 GB/s) |
| **TP = 4 Comm Bus** | Hybrid NVLink + `SYS` (~32–64 GB/s) |
