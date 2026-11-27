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

### Public-Tri (Qwen3.5-35B-A3B (c_def=2.0)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 2.331 | 0.210 | **0.780** | 1 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 1.852 | 0.689 | **2.185** | 1 | 46/56 | 1.109e-07 |
| `Joint-Greedy+Naor` | 2.488 | 0.053 | **0.566** | 35 | 4/56 | 8.909e-10 |
| `KL-PBD+ (Warm)` | 2.189 | 0.352 | **1.456** | 11 | 32/56 | 8.521e-02 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.412 | 2.953 | **5.157** | 0 | 56/56 | 4.522e-10 |
| `Congested-L2D-Only` | 0.826 | 1.715 | **4.263** | 8 | 44/56 | 8.647e-07 |
| `w/o Queue-Externality` | 1.721 | 0.820 | **2.032** | 0 | 49/56 | 8.647e-07 |

### Public-Tri (Qwen3.5-35B-A3B (c_def=4.0)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 1.283 | 0.210 | **0.579** | 2 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 0.859 | 0.635 | **1.885** | 0 | 45/56 | 1.207e-07 |
| `Joint-Greedy+Naor` | 1.472 | 0.022 | **0.411** | 46 | 2/56 | 5.169e-10 |
| `KL-PBD+ (Warm)` | 0.977 | 0.516 | **1.792** | 8 | 41/56 | 6.452e-06 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.699 | 2.192 | **3.790** | 0 | 56/56 | 4.522e-10 |
| `Congested-L2D-Only` | -0.175 | 1.669 | **3.970** | 0 | 56/56 | 4.522e-10 |
| `w/o Queue-Externality` | 0.878 | 0.616 | **1.775** | 0 | 53/56 | 2.201e-07 |

### Public-Tri (DeepSeek-V3.2 (c_def=2.0)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 2.084 | 0.217 | **0.709** | 8 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 1.546 | 0.755 | **2.495** | 11 | 37/56 | 4.783e-05 |
| `Joint-Greedy+Naor` | 2.070 | 0.231 | **0.737** | 12 | 33/56 | 1.894e-01 |
| `KL-PBD+ (Warm)` | 2.184 | 0.118 | **0.842** | 19 | 23/56 | 2.943e-02 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.896 | 3.198 | **5.650** | 0 | 56/56 | 4.522e-10 |
| `Congested-L2D-Only` | 1.234 | 1.067 | **3.034** | 11 | 42/56 | 3.185e-06 |
| `w/o Queue-Externality` | 1.533 | 0.768 | **2.097** | 14 | 48/56 | 2.491e-06 |

### Public-Tri (DeepSeek-V3.2 (c_def=4.0)
| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\text{Holm}}$ |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 0.946 | 0.219 | **0.543** | 7 | 56/56 | — |
| `Calibrated-Cascade+Naor` | 0.610 | 0.554 | **2.179** | 13 | 35/56 | 1.195e-02 |
| `Joint-Greedy+Naor` | 0.934 | 0.231 | **0.557** | 4 | 35/56 | 2.330e-01 |
| `KL-PBD+ (Warm)` | 0.985 | 0.179 | **1.111** | 33 | 20/56 | 2.271e-01 |
| `ACL-2026-VoI (Live)+Post-Defer` | -1.277 | 2.442 | **4.343** | 0 | 56/56 | 4.522e-10 |
| `Congested-L2D-Only` | 0.130 | 1.034 | **2.920** | 0 | 42/56 | 1.850e-06 |
| `w/o Queue-Externality` | 0.467 | 0.698 | **1.765** | 0 | 52/56 | 6.374e-07 |

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

## 4. Real-LLM Heterogeneous Specialist (Healthy Agent vs. Diagnostic-Floor Agent)
### v1_qwen35_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=2.000, `KL-PBD+ (Warm)`=5.592, `ACL-2026-VoI (Calib)`=2.182, `Calibrated-Cascade+Naor`=5.662, `Joint-Greedy+Naor`=6.060, `Tri-VoI-L2D`=5.893
- **Cap=1/3, c=1.5**: `Direct-ACT`=2.000, `KL-PBD+ (Warm)`=3.633, `ACL-2026-VoI (Calib)`=2.182, `Calibrated-Cascade+Naor`=4.216, `Joint-Greedy+Naor`=3.602, `Tri-VoI-L2D`=4.175

### v2_qwen35_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=1.333, `KL-PBD+ (Warm)`=4.652, `ACL-2026-VoI (Live)`=2.203, `Calibrated-Cascade+Naor`=4.772, `Joint-Greedy+Naor`=5.044, `Tri-VoI-L2D`=4.795
- **Cap=1/3, c=1.5**: `Direct-ACT`=1.333, `KL-PBD+ (Warm)`=2.563, `ACL-2026-VoI (Live)`=2.203, `Calibrated-Cascade+Naor`=3.906, `Joint-Greedy+Naor`=3.525, `Tri-VoI-L2D`=3.908

### public_tri_qwen35_to_deepseek
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=1.847, `ACL-2026-VoI (Live)`=-0.400, `Calibrated-Cascade+Naor`=1.639, `Joint-Greedy+Naor`=2.322, `Tri-VoI-L2D`=1.927
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=0.707, `ACL-2026-VoI (Live)`=-0.400, `Calibrated-Cascade+Naor`=1.441, `Joint-Greedy+Naor`=1.483, `Tri-VoI-L2D`=1.471

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
