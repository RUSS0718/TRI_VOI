# Reviewer Hardening Suite Report (0 Extra API Calls)

## 1. 56-Cell 2D Minimax Regret (Descriptive Envelope — No Pseudoreplication p-values)
### v1-Clean (Qwen3.5)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 4.470 | — | 0.083 | **0.572** | 18 | — |
| `Calibrated-Cascade+Naor` | 4.095 | -0.375 | 0.458 | **1.443** | 10 | 39 / 3 / 14 |
| `Joint-Greedy+Naor` | 4.332 | -0.137 | 0.220 | **0.983** | 8 | 44 / 0 / 12 |
| `KL-PBD+ (Warm)` | 4.049 | -0.420 | 0.503 | **1.179** | 8 | 47 / 1 / 8 |
| `ACL-2026-VoI+Post-Defer` | 3.063 | -1.406 | 1.489 | **2.788** | 3 | 53 / 0 / 3 |
| `ACL-2026-VoI+Full-Defer+Naor` | 4.277 | -0.193 | 0.276 | **0.741** | 11 | 33 / 3 / 20 |
| `Congested-L2D-Only` | 3.375 | -1.094 | 1.178 | **2.747** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 4.119 | -0.350 | 0.433 | **1.727** | 18 | 33 / 16 / 7 |

### v2 (Qwen3.5-35B-A3B)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 4.124 | — | 0.204 | **0.511** | 8 | — |
| `Calibrated-Cascade+Naor` | 4.232 | +0.108 | 0.096 | **0.536** | 32 | 13 / 4 / 39 |
| `Joint-Greedy+Naor` | 3.903 | -0.221 | 0.425 | **0.954** | 4 | 40 / 0 / 16 |
| `KL-PBD+ (Warm)` | 3.144 | -0.980 | 1.184 | **2.086** | 0 | 55 / 0 / 1 |
| `ACL-2026-VoI (Live)+Post-Defer` | 3.097 | -1.027 | 1.231 | **1.777** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.289 | +0.166 | 0.039 | **0.228** | 16 | 14 / 0 / 42 |
| `Congested-L2D-Only` | 2.385 | -1.738 | 1.943 | **4.000** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 3.760 | -0.363 | 0.568 | **1.461** | 0 | 37 / 16 / 3 |

### v2 (DeepSeek-V3.2)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 4.258 | — | 0.118 | **1.021** | 25 | — |
| `Calibrated-Cascade+Naor` | 4.117 | -0.141 | 0.259 | **0.983** | 12 | 39 / 3 / 14 |
| `Joint-Greedy+Naor` | 4.174 | -0.084 | 0.203 | **0.826** | 9 | 37 / 0 / 19 |
| `KL-PBD+ (Warm)` | 3.301 | -0.957 | 1.075 | **2.368** | 2 | 48 / 0 / 8 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.797 | -1.461 | 1.579 | **2.417** | 0 | 55 / 0 / 1 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.193 | -0.065 | 0.183 | **0.568** | 11 | 37 / 3 / 16 |
| `Congested-L2D-Only` | 2.790 | -1.468 | 1.587 | **3.617** | 1 | 50 / 0 / 6 |
| `w/o Queue-Externality` | 4.001 | -0.256 | 0.375 | **2.024** | 16 | 32 / 18 / 6 |

### v2 (Hy4-preview)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 5.169 | — | 0.328 | **1.410** | 9 | — |
| `Calibrated-Cascade+Naor` | 5.497 | +0.328 | -0.000 | **0.000** | 56 | 0 / 9 / 47 |
| `Joint-Greedy+Naor` | 4.692 | -0.477 | 0.805 | **1.410** | 0 | 48 / 7 / 1 |
| `KL-PBD+ (Warm)` | 3.255 | -1.913 | 2.241 | **3.354** | 0 | 49 / 0 / 7 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.983 | -2.186 | 2.514 | **3.628** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 5.005 | -0.163 | 0.491 | **1.240** | 0 | 40 / 0 / 16 |
| `Congested-L2D-Only` | 2.608 | -2.561 | 2.889 | **4.526** | 0 | 53 / 2 / 1 |
| `w/o Queue-Externality` | 4.902 | -0.267 | 0.595 | **2.276** | 4 | 34 / 20 / 2 |

### Public-Tri (Qwen3.5-35B-A3B (c_def=2.0)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 2.331 | — | 0.210 | **0.780** | 1 | — |
| `Calibrated-Cascade+Naor` | 1.852 | -0.479 | 0.689 | **2.185** | 1 | 46 / 0 / 10 |
| `Joint-Greedy+Naor` | 2.488 | +0.157 | 0.053 | **0.566** | 35 | 4 / 0 / 52 |
| `KL-PBD+ (Warm)` | 2.189 | -0.142 | 0.352 | **1.456** | 11 | 32 / 0 / 24 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.412 | -2.743 | 2.953 | **5.157** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.956 | -0.375 | 0.585 | **1.442** | 0 | 50 / 0 / 6 |
| `Congested-L2D-Only` | 0.826 | -1.505 | 1.715 | **4.263** | 8 | 44 / 0 / 12 |
| `w/o Queue-Externality` | 1.721 | -0.610 | 0.820 | **2.032** | 0 | 33 / 16 / 7 |

### Public-Tri (Qwen3.5-35B-A3B (c_def=4.0)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 1.283 | — | 0.212 | **0.579** | 2 | — |
| `Calibrated-Cascade+Naor` | 0.859 | -0.425 | 0.636 | **1.885** | 0 | 45 / 0 / 11 |
| `Joint-Greedy+Naor` | 1.472 | +0.189 | 0.023 | **0.411** | 45 | 2 / 0 / 54 |
| `KL-PBD+ (Warm)` | 0.977 | -0.306 | 0.518 | **1.792** | 8 | 41 / 0 / 15 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.699 | -1.982 | 2.193 | **3.790** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.948 | -0.335 | 0.546 | **1.250** | 1 | 48 / 0 / 8 |
| `Congested-L2D-Only` | -0.175 | -1.458 | 1.670 | **3.970** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 0.878 | -0.405 | 0.617 | **1.775** | 0 | 37 / 16 / 3 |

### Public-Tri (DeepSeek-V3.2 (c_def=2.0)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 2.084 | — | 0.222 | **0.709** | 8 | — |
| `Calibrated-Cascade+Naor` | 1.546 | -0.538 | 0.760 | **2.495** | 7 | 37 / 0 / 19 |
| `Joint-Greedy+Naor` | 2.070 | -0.014 | 0.236 | **0.737** | 12 | 24 / 9 / 23 |
| `KL-PBD+ (Warm)` | 2.184 | +0.099 | 0.123 | **0.842** | 18 | 23 / 0 / 33 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.896 | -2.981 | 3.203 | **5.650** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.748 | -0.336 | 0.558 | **1.385** | 5 | 40 / 0 / 16 |
| `Congested-L2D-Only` | 1.234 | -0.850 | 1.072 | **3.034** | 11 | 38 / 4 / 14 |
| `w/o Queue-Externality` | 1.533 | -0.551 | 0.773 | **2.097** | 14 | 32 / 16 / 8 |

### Public-Tri (DeepSeek-V3.2 (c_def=4.0)
| Method | Mean Util (56) | $\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-L2D` | 0.946 | — | 0.229 | **0.559** | 4 | — |
| `Calibrated-Cascade+Naor` | 0.610 | -0.336 | 0.565 | **2.179** | 6 | 35 / 0 / 21 |
| `Joint-Greedy+Naor` | 0.934 | -0.012 | 0.241 | **0.582** | 4 | 33 / 2 / 21 |
| `KL-PBD+ (Warm)` | 0.985 | +0.039 | 0.190 | **1.111** | 30 | 20 / 0 / 36 |
| `ACL-2026-VoI (Live)+Post-Defer` | -1.277 | -2.223 | 2.452 | **4.343** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.776 | -0.170 | 0.399 | **1.217** | 13 | 33 / 0 / 23 |
| `Congested-L2D-Only` | 0.130 | -0.816 | 1.044 | **2.920** | 0 | 42 / 0 / 14 |
| `w/o Queue-Externality` | 0.467 | -0.479 | 0.708 | **1.765** | 0 | 36 / 16 / 4 |

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
| 1 | 3.579 (38.9%) | 3.526 (37.6%) | 3.667 | 3.106 |
| 2 | 3.582 (38.9%) | 3.582 (36.8%) | 3.607 | 3.106 |
| 4 | 3.210 (38.9%) | 3.259 (36.0%) | 3.245 | 3.106 |
| 8 | 2.991 (38.6%) | 2.933 (31.3%) | 2.911 | 3.106 |

## 4. Real-LLM Clinical Specialist (`Public-Tri`, Full-Quadrant) vs. Semi-Synthetic Specialist (`v1/v2`)
### real_public_tri_qwen35_to_deepseek_stationary
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=3.020, `ACL-2026-VoI (Live)+Full-Defer+Naor`=2.635, `Calibrated-Cascade+Naor`=2.680, `Joint-Greedy+Naor`=3.127, `Tri-VoI-L2D`=3.124
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=1.471, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.963, `Calibrated-Cascade+Naor`=1.785, `Joint-Greedy+Naor`=1.355, `Tri-VoI-L2D`=1.477
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=0.279, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.276, `Calibrated-Cascade+Naor`=1.367, `Joint-Greedy+Naor`=1.420, `Tri-VoI-L2D`=0.960

### real_public_tri_qwen35_to_ds_to_qwen8b_shift
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=1.581, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.827, `Calibrated-Cascade+Naor`=1.888, `Joint-Greedy+Naor`=2.121, `Tri-VoI-L2D`=2.057
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=0.245, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.428, `Calibrated-Cascade+Naor`=1.373, `Joint-Greedy+Naor`=1.322, `Tri-VoI-L2D`=1.094
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=-0.449, `ACL-2026-VoI (Live)+Full-Defer+Naor`=0.636, `Calibrated-Cascade+Naor`=0.499, `Joint-Greedy+Naor`=0.870, `Tri-VoI-L2D`=0.367

### real_public_tri_qwen8b_to_deepseek_stationary
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-3.250, `KL-PBD+ (Warm)`=0.773, `ACL-2026-VoI (Live)+Full-Defer+Naor`=-1.172, `Calibrated-Cascade+Naor`=-1.269, `Joint-Greedy+Naor`=0.147, `Tri-VoI-L2D`=0.234
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-3.250, `KL-PBD+ (Warm)`=-1.226, `ACL-2026-VoI (Live)+Full-Defer+Naor`=-1.369, `Calibrated-Cascade+Naor`=-1.309, `Joint-Greedy+Naor`=-1.329, `Tri-VoI-L2D`=-1.444

### semi_synthetic_v1_qwen35_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=2.000, `KL-PBD+ (Warm)`=5.592, `ACL-2026-VoI+Full-Defer+Naor`=5.662, `Calibrated-Cascade+Naor`=5.662, `Joint-Greedy+Naor`=6.060, `Tri-VoI-L2D`=5.893
- **Cap=1/3, c=1.5**: `Direct-ACT`=2.000, `KL-PBD+ (Warm)`=3.633, `ACL-2026-VoI+Full-Defer+Naor`=4.216, `Calibrated-Cascade+Naor`=4.216, `Joint-Greedy+Naor`=3.602, `Tri-VoI-L2D`=4.175

### semi_synthetic_v2_qwen35_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=1.333, `KL-PBD+ (Warm)`=4.652, `ACL-2026-VoI (Live)+Full-Defer+Naor`=4.789, `Calibrated-Cascade+Naor`=4.772, `Joint-Greedy+Naor`=5.044, `Tri-VoI-L2D`=4.795
- **Cap=1/3, c=1.5**: `Direct-ACT`=1.333, `KL-PBD+ (Warm)`=2.563, `ACL-2026-VoI (Live)+Full-Defer+Naor`=4.062, `Calibrated-Cascade+Naor`=3.906, `Joint-Greedy+Naor`=3.525, `Tri-VoI-L2D`=3.908

### semi_synthetic_v1_qwen8b_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=0.778, `KL-PBD+ (Warm)`=5.258, `ACL-2026-VoI+Full-Defer+Naor`=5.202, `Calibrated-Cascade+Naor`=5.324, `Joint-Greedy+Naor`=5.202, `Tri-VoI-L2D`=5.202
- **Cap=1/3, c=1.5**: `Direct-ACT`=0.778, `KL-PBD+ (Warm)`=2.800, `ACL-2026-VoI+Full-Defer+Naor`=2.095, `Calibrated-Cascade+Naor`=2.217, `Joint-Greedy+Naor`=2.095, `Tri-VoI-L2D`=2.095

## 5. 2-Call Measured SQLite Token Cost & `KL-PBD+ (SC-Only)` Signal Parity
| Suite | Model | `Tri-VoI` (Full) | `Tri-VoI` (2-Call) | T1 Tok (Full->2C) | Realized Tok (Full->2C, Δ%) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 (83.67%) | **5.416** (84.44%) | 4732 -> 1895 | 4952 -> 2113 (-57.3%) | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | 5.523 (85.44%) | **5.523** (85.44%) | 4095 -> 1706 | 4441 -> 2052 (-53.8%) | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | 5.842 (86.94%) | **5.842** (86.94%) | 4256 -> 1753 | 4622 -> 2119 (-54.1%) | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 (80.89%) | **4.750** (81.33%) | 4763 -> 1897 | 4967 -> 2111 (-57.5%) | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | 5.031 (83.0%) | **5.031** (83.0%) | 4079 -> 1701 | 4373 -> 1995 (-54.4%) | 3.620 | 2.965 |
| v2 | Hy4-preview | 6.053 (87.89%) | **6.053** (87.89%) | 4248 -> 1755 | 4562 -> 2070 (-54.6%) | 3.426 | 2.788 |

## 6. Explicit 2x2 Decomposition (`±EVOI × ±Naor`) & Fair `ACL-2026-VoI+Full-Defer+Naor`
| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI` | `Smooth-Naor` | Naor Δ | EVOI vs CC+N |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 5.142 | 5.190 | 4.877 | 4.867 | 4.867 | 5.274 | 5.220 | 5.249 | -0.054 | +0.353 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 2.212 | 3.188 | 3.152 | 3.798 | 3.798 | 2.732 | 3.792 | 3.792 | +1.060 | -0.006 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.294 | 4.421 | 4.996 | 4.996 | 4.993 | 4.615 | 4.657 | 4.829 | +0.042 | -0.339 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 2.321 | 3.169 | 3.580 | 3.808 | 3.805 | 2.931 | 3.847 | 3.847 | +0.916 | +0.039 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.841 | 4.850 | 4.754 | 4.754 | 4.754 | 5.020 | 5.031 | 5.028 | +0.011 | +0.277 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 3.182 | 4.014 | 4.333 | 4.509 | 4.509 | 3.957 | 4.334 | 4.334 | +0.377 | -0.175 |
| v2 / Hy4-preview | Cap=2, c=1.5 | 5.390 | 5.408 | 6.271 | 6.271 | 5.986 | 6.024 | 6.053 | 6.194 | +0.029 | -0.218 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 3.860 | 4.711 | 5.378 | 5.512 | 5.021 | 5.252 | 5.313 | 5.313 | +0.061 | -0.199 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=2, c=1.5 | 3.693 | 3.723 | 3.140 | 3.067 | 3.105 | 3.545 | 3.612 | 3.673 | +0.067 | +0.545 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=1/3, c=1.5 | 0.224 | 1.450 | 0.053 | 1.347 | 1.251 | -0.057 | 1.259 | 1.259 | +1.316 | -0.088 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=2, c=1.5 | 2.229 | 2.237 | 1.626 | 1.684 | 1.797 | 1.966 | 1.999 | 1.935 | +0.033 | +0.315 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=1/3, c=1.5 | 0.187 | 1.082 | -0.017 | 0.853 | 0.784 | -0.247 | 0.780 | 0.764 | +1.027 | -0.073 |
| Public-Tri / DeepSeek (c_def=2) | Cap=2, c=1.5 | 3.520 | 3.331 | 2.905 | 2.769 | 2.871 | 3.505 | 3.327 | 3.288 | -0.178 | +0.558 |
| Public-Tri / DeepSeek (c_def=2) | Cap=1/3, c=1.5 | -0.908 | 0.444 | -0.444 | 0.752 | 0.784 | -0.755 | 0.417 | 0.417 | +1.172 | -0.335 |

## 7. 20 Stratified Random `(cal, test)` Resplits & 6 Disjoint-Episode Check
| Suite | Regime | `Tri-VoI` (mean±sd) | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | Tri Wins/20 vs (KL, ACL, JG+N, CC+N) |
|:---|:---|---:|---:|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | Cap=2, c=1.5 | 4.869±0.48 | 4.424 | 4.619 | 4.626 | 4.822 | 18/16/19/11 |
| v1-Clean (Qwen3.5) | Cap=1/3, c=1.5 | 3.766±0.55 | 3.240 | 3.688 | 3.391 | 4.055 | 17/13/20/6 |
| v2 (Qwen3.5) | Cap=2, c=1.5 | 4.628±0.54 | 3.840 | 4.749 | 4.380 | 4.737 | 18/8/18/10 |
| v2 (Qwen3.5) | Cap=1/3, c=1.5 | 3.693±0.55 | 2.333 | 3.905 | 3.190 | 3.920 | 20/6/19/6 |
| v2 (DeepSeek-V3.2) | Cap=2, c=1.5 | 4.974±0.34 | 4.097 | 4.744 | 4.914 | 4.744 | 20/16/11/16 |
| v2 (DeepSeek-V3.2) | Cap=1/3, c=1.5 | 4.088±0.53 | 2.535 | 4.267 | 3.823 | 4.267 | 20/3/18/3 |
| Public-Tri (Qwen3.5) | Cap=2, c=1.5 | 3.414±0.51 | 3.332 | 3.061 | 3.457 | 3.000 | 10/18/8/18 |
| Public-Tri (Qwen3.5) | Cap=1/3, c=1.5 | 0.589±0.48 | 0.649 | 0.885 | 0.721 | 0.909 | 10/5/4/4 |
| Public-Tri (DeepSeek-V3.2) | Cap=2, c=1.5 | 3.208±0.45 | 3.283 | 2.856 | 3.217 | 2.805 | 7/19/8/19 |
| Public-Tri (DeepSeek-V3.2) | Cap=1/3, c=1.5 | 0.404±0.46 | 0.484 | 0.798 | 0.347 | 0.724 | 9/3/12/5 |
