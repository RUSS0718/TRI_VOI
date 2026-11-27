# IJCAI 2027 Canonical Tables (Single Source of Truth from JSON)

Generated automatically by `generate_ijcai_tables.py` using canonical terminology (`Requester`, `Specialist`, `Benchmark v1-Clean`, `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri`).

## Table 1: `Benchmark v1-Clean` & `Benchmark v1-Noisy` Main Comparison (`Qwen/Qwen3.5-35B-A3B`, `N=360`)
| Method | v1-Clean (Cap=2, c=1.5) | v1-Clean (Cap=2, c=2.5) | v1-Clean (Cap=1/3, c=1.5) | v1-Noisy (Cap=2, c=1.5) | v1-Noisy (Cap=2, c=2.5) | v1-Noisy (Cap=1/3, c=1.5) |
|:---|---:|---:|---:|---:|---:|---:|
| `Direct-ACT` | 2.000 | 2.000 | 2.000 | 0.556 | 0.556 | 0.556 |
| `KL-PBD` | 4.061 | 4.061 | 2.621 | 3.389 | 3.389 | 1.268 |
| `KL-PBD+ (Warm)` | 4.450 | 4.450 | 3.106 | 3.665 | 3.665 | 1.517 |
| `ACL-2026-VoI (Calib)` | 2.182 | 2.333 | 2.182 | 1.142 | 0.715 | 1.142 |
| `ACL-2026-VoI+Post-Defer` | 3.633 | 2.800 | 3.609 | 2.081 | 1.071 | 2.031 |
| `Joint-Greedy+Naor` | 5.190 | 5.015 | 3.188 | 3.887 | 3.618 | 1.295 |
| `Calibrated-Cascade+Naor` | 4.867 | 4.470 | 3.798 | 3.413 | 2.952 | 1.614 |
| `Tri-VoI-L2D` | 5.220 | 5.142 | 3.792 | 3.503 | 3.511 | 1.525 |

## Table 2: Cross-Benchmark Generalization on `Benchmark v2 (Implicit-Override)` & `Public-Tri (MediQ)`
| Suite & Model | Miss-F1 / Collapse% | `KL-PBD+ (Warm)` | `ACL-2026-VoI (Live)` | `Joint-Greedy+Naor` | `Calib-Cascade+Naor` | `Tri-VoI-L2D` (Cap=2, c=1.5) | `Tri-VoI-L2D` (Cap=1/3) |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| v2 / `Qwen3.5-35B-A3B` | 0.870 / 78.9% | 3.366 | 2.203 | 4.421 | 4.996 | **4.657** | **3.847** |
| v2 / `DeepSeek-V3.2` | 0.935 / 86.7% | 3.620 | 2.550 | 4.850 | 4.754 | **5.031** | **4.334** |
| v2 / `Hy4-preview` | 1.000 / 91.1% | 3.426 | 3.006 | 5.408 | 6.271 | **6.053** | **5.313** |
| v2 / `Qwen3-8B` | 0.178 / 87.8% | 3.600 | 0.131 | 3.551 | 3.508 | **3.571** | **1.163** |
| Public-Tri / `Qwen3.5-35B-A3B` | 0.932 / 81.7% | 3.533 | -0.400 | 3.723 | 3.067 | **3.612** (c\_def=4: 1.999) | **1.259** |
| Public-Tri / `DeepSeek-V3.2` | 0.831 / 72.5% | 3.443 | -0.792 | 3.331 | 2.769 | **3.327** (c\_def=4: 1.634) | **0.417** |
| Public-Tri / `Qwen3-8B` | 0.802 / 91.7% | 3.133 | -3.402 | 2.809 | 2.013 | **2.809** (c\_def=4: 0.937) | **-1.611** |

## Table 3: 56-Cell 2D Minimax Regret (`v1-Clean` & `v2`) & Lightweight 2-Call Ablation
| Suite | Method | Mean Util (56) | Worst-Case Regret | Wilcoxon $p_{\text{Holm}}$ vs `Tri-VoI-L2D` |
|:---|:---|---:|---:|---:|
| v1-Clean (Qwen3.5) | `Tri-VoI-L2D` | 4.470 | **0.572** | — |
| v1-Clean (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.095 | **1.443** | 5.94e-07 |
| v1-Clean (Qwen3.5) | `Joint-Greedy+Naor` | 4.332 | **0.983** | 1.05e-06 |
| v1-Clean (Qwen3.5) | `KL-PBD+ (Warm)` | 4.049 | **1.179** | 1.28e-07 |
| v1-Clean (Qwen3.5) | `w/o Queue-Externality` | 4.119 | **1.727** | 5.91e-06 |
| v2 (DeepSeek-V3.2) | `Tri-VoI-L2D` | 4.258 | **0.937** | — |
| v2 (DeepSeek-V3.2) | `Calibrated-Cascade+Naor` | 4.117 | **0.983** | 1.85e-03 |
| v2 (DeepSeek-V3.2) | `Joint-Greedy+Naor` | 4.174 | **0.741** | 1.85e-03 |
| v2 (DeepSeek-V3.2) | `KL-PBD+ (Warm)` | 3.301 | **2.368** | 1.98e-08 |
| v2 (DeepSeek-V3.2) | `w/o Queue-Externality` | 4.001 | **1.939** | 7.58e-06 |
| v2 (Qwen3.5) | `Tri-VoI-L2D` | 4.124 | **0.511** | — |
| v2 (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.232 | **0.536** | 5.96e-04 |
| v2 (Qwen3.5) | `Joint-Greedy+Naor` | 3.903 | **0.954** | 8.42e-06 |
| v2 (Qwen3.5) | `KL-PBD+ (Warm)` | 3.144 | **2.086** | 4.52e-10 |
| v2 (Qwen3.5) | `w/o Queue-Externality` | 3.760 | **1.368** | 3.30e-07 |

### 2-Call Lightweight Deployment vs. Full 5-Call & Signal-Parity `KL-PBD+`
| Suite | Model | `Tri-VoI` Full | `Tri-VoI` 2-Call | Tokens Full -> 2-Call | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 | **5.416** | 5404 -> 2810 | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | 5.523 | **5.523** | 4846 -> 2520 | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | 5.842 | **5.842** | 5040 -> 2621 | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 | **4.750** | 5428 -> 2822 | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | 5.031 | **5.031** | 4901 -> 2549 | 3.620 | 2.965 |
| v2 | Hy4-preview | 6.053 | **6.053** | 5056 -> 2629 | 3.426 | 2.788 |
