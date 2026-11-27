# Reviewer Hardening Suite Report (0 Extra API Calls)

## 1. 56-Cell 2D Minimax Regret & Wilcoxon Holm Significance
### v1-Clean (Qwen3.5)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 4.470 | 0.078 | **0.572** | 18 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 4.095 | 0.453 | **1.443** | 13 | 42/56 | 5.940e-07 |
| `Joint-Greedy+Naor` | 4.332 | 0.216 | **0.983** | 9 | 44/56 | 1.051e-06 |
| `KL-PBD+ (Warm)` | 4.049 | 0.498 | **1.179** | 8 | 48/56 | 1.276e-07 |
| `ACL-2026-VoI+Post-Defer` | 3.063 | 1.484 | **2.788** | 3 | 53/56 | 1.361e-09 |
| `Congested-L2D-Only` | 3.375 | 1.173 | **2.747** | 0 | 56/56 | 4.522e-10 |
| `w/o Queue-Externality` | 4.119 | 0.428 | **1.727** | 18 | 49/56 | 5.906e-06 |

### v2 (Qwen3.5-35B-A3B)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 4.124 | 0.167 | **0.511** | 8 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 4.232 | 0.059 | **0.536** | 41 | 17/56 | 5.960e-04 |
| `Joint-Greedy+Naor` | 3.903 | 0.388 | **0.954** | 10 | 40/56 | 8.419e-06 |
| `KL-PBD+ (Warm)` | 3.144 | 1.147 | **2.086** | 1 | 55/56 | 4.522e-10 |
| `ACL-2026-VoI (Live)+Post-Defer` | 3.097 | 1.194 | **1.777** | 0 | 56/56 | 4.522e-10 |
| `Congested-L2D-Only` | 2.385 | 1.906 | **4.000** | 0 | 56/56 | 4.522e-10 |
| `w/o Queue-Externality` | 3.760 | 0.531 | **1.368** | 0 | 53/56 | 3.301e-07 |

### v2 (DeepSeek-V3.2)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 4.258 | 0.111 | **0.937** | 26 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 4.117 | 0.251 | **0.983** | 14 | 42/56 | 1.853e-03 |
| `Joint-Greedy+Naor` | 4.174 | 0.195 | **0.741** | 9 | 37/56 | 1.853e-03 |
| `KL-PBD+ (Warm)` | 3.301 | 1.067 | **2.368** | 4 | 48/56 | 1.980e-08 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.797 | 1.571 | **2.417** | 0 | 55/56 | 4.774e-10 |
| `Congested-L2D-Only` | 2.790 | 1.579 | **3.617** | 2 | 50/56 | 1.435e-09 |
| `w/o Queue-Externality` | 4.001 | 0.367 | **1.939** | 16 | 50/56 | 7.579e-06 |

### v2 (Hy4-preview)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 5.169 | 0.328 | **1.410** | 9 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 5.497 | -0.000 | **0.000** | 56 | 9/56 | 4.785e-09 |
| `Joint-Greedy+Naor` | 4.692 | 0.805 | **1.410** | 0 | 55/56 | 4.005e-09 |
| `KL-PBD+ (Warm)` | 3.255 | 2.241 | **3.354** | 0 | 49/56 | 1.344e-09 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.983 | 2.514 | **3.628** | 0 | 56/56 | 4.522e-10 |
| `Congested-L2D-Only` | 2.608 | 2.889 | **4.526** | 0 | 55/56 | 8.588e-10 |
| `w/o Queue-Externality` | 4.902 | 0.595 | **2.276** | 4 | 54/56 | 2.561e-07 |

## 2. Stage 3 (`Benchmark v1-Noisy` + Imperfect Requester) External Baselines
### `Qwen/Qwen3.5-35B-A3B`
| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|---:|
| `Direct-ACT` | 0.556 | 0.556 | 0.556 |
| `Naive-Cascade` | 2.285 | 2.048 | -5.637 |
| `KL-PBD` | 3.389 | 3.389 | 1.268 |
| `KL-PBD+` | 3.692 | 3.692 | 1.490 |
| `KL-PBD+ (Warm)` | 3.665 | 3.665 | 1.517 |
| `KL-PBD+ (SC-Only)` | 3.345 | 3.345 | 1.564 |
| `ACL-2026-VoI (Calib)` | 1.142 | 0.715 | 1.142 |
| `ACL-2026-VoI+Post-Defer` | 2.081 | 1.071 | 2.031 |
| `Joint-Greedy+Naor` | 3.887 | 3.618 | 1.295 |
| `Calibrated-Cascade+Naor` | 3.413 | 2.952 | 1.614 |
| `Tri-VoI-L2D` | 3.503 | 3.511 | 1.525 |

### `tencent/Hy4-preview`
| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|---:|
| `Direct-ACT` | 0.500 | 0.500 | 0.500 |
| `Naive-Cascade` | 2.666 | 2.557 | 0.130 |
| `KL-PBD` | 3.351 | 3.351 | 1.588 |
| `KL-PBD+` | 3.543 | 3.543 | 1.956 |
| `KL-PBD+ (Warm)` | 3.744 | 3.744 | 1.766 |
| `KL-PBD+ (SC-Only)` | 3.424 | 3.424 | 1.830 |
| `ACL-2026-VoI (Calib)` | 1.483 | 0.639 | 1.483 |
| `ACL-2026-VoI+Post-Defer` | 2.196 | 0.639 | 2.184 |
| `Joint-Greedy+Naor` | 3.944 | 3.863 | 1.974 |
| `Calibrated-Cascade+Naor` | 4.301 | 3.803 | 2.614 |
| `Tri-VoI-L2D` | 4.380 | 4.082 | 2.304 |

## 3. End-to-End Stream `ASK` Latency Sweep ($\tau_{\text{ask}} \in \{0..8\}$)
### Cap=2, c=1.5
| $\tau_{\text{ask}}$ | `Tri-VoI-L2D` | `Tri-VoI (Lat-Aware)` | `Calib-Casc+Naor` | `KL-PBD+ (Warm)` |
|---:|---:|---:|---:|---:|
| 0 | 5.220 (24.7%) | 5.220 (24.7%) | 4.867 | 4.450 |
| 1 | 5.200 (24.9%) | 5.178 (23.3%) | 4.727 | 4.450 |
| 2 | 5.106 (25.0%) | 5.232 (21.3%) | 4.615 | 4.450 |
| 4 | 4.940 (25.2%) | 5.088 (19.4%) | 4.391 | 4.450 |
| 8 | 4.676 (25.1%) | 4.975 (14.6%) | 3.925 | 4.450 |

### Cap=1/3, c=1.5
| $\tau_{\text{ask}}$ | `Tri-VoI-L2D` | `Tri-VoI (Lat-Aware)` | `Calib-Casc+Naor` | `KL-PBD+ (Warm)` |
|---:|---:|---:|---:|---:|
| 0 | 3.792 (38.9%) | 3.792 (38.9%) | 3.798 | 3.106 |
| 1 | 3.579 (38.9%) | 3.520 (37.9%) | 3.667 | 3.106 |
| 2 | 3.582 (38.9%) | 3.577 (37.2%) | 3.607 | 3.106 |
| 4 | 3.210 (38.9%) | 3.276 (36.0%) | 3.245 | 3.106 |
| 8 | 2.991 (38.6%) | 2.934 (32.0%) | 2.911 | 3.106 |

## 4. Real-LLM Heterogeneous Specialist (`Qwen3-8B` Agent -> Real `Hy4` / `DeepSeek` Specialist)
### v1_qwen8b_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=0.778, `KL-PBD+ (Warm)`=5.258, `ACL-2026-VoI (Calib)`=0.778, `Calibrated-Cascade+Naor`=5.324, `Joint-Greedy+Naor`=5.202, `Tri-VoI-L2D`=5.202
- **Cap=1/3, c=1.5**: `Direct-ACT`=0.778, `KL-PBD+ (Warm)`=2.800, `ACL-2026-VoI (Calib)`=0.778, `Calibrated-Cascade+Naor`=2.217, `Joint-Greedy+Naor`=2.095, `Tri-VoI-L2D`=2.095

### public_tri_qwen8b_to_deepseek
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-3.250, `KL-PBD+ (Warm)`=1.438, `ACL-2026-VoI (Live)`=-3.402, `Calibrated-Cascade+Naor`=-0.140, `Joint-Greedy+Naor`=1.051, `Tri-VoI-L2D`=0.821

## 5. 2-Call Lightweight Deployment & `KL-PBD+ (SC-Only)` Signal Parity
| Suite | Model | `Tri-VoI` (Full) | `Tri-VoI` (2-Call) | Tok Full -> 2-Call | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 (83.67%) | **5.416** (84.44%) | 5404 -> 2810 | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | 5.523 (85.44%) | **5.523** (85.44%) | 4846 -> 2520 | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | 5.842 (86.94%) | **5.842** (86.94%) | 5040 -> 2621 | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 (80.89%) | **4.750** (81.33%) | 5428 -> 2822 | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | 5.031 (83.0%) | **5.031** (83.0%) | 4901 -> 2549 | 3.620 | 2.965 |
| v2 | Hy4-preview | 6.053 (87.89%) | **6.053** (87.89%) | 5056 -> 2629 | 3.426 | 2.788 |
