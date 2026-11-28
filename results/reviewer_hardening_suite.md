# Reviewer Hardening Suite Report (0 Extra API Calls)

## 1. 56-Cell 2D Minimax Regret (Descriptive Envelope — No Pseudoreplication p-values)
### v1-Clean (Qwen3.5)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 4.616 | — | 0.016 | **0.404** | 43 | — |
| `Tri-VoI-L2D` | 4.470 | -0.147 | 0.163 | **0.596** | 7 | 49 / 0 / 7 |
| `Calibrated-Cascade+Naor` | 4.095 | -0.521 | 0.537 | **1.529** | 0 | 56 / 0 / 0 |
| `Joint-Greedy+Naor` | 4.332 | -0.284 | 0.300 | **0.987** | 0 | 56 / 0 / 0 |
| `KL-PBD+ (Warm)` | 4.049 | -0.567 | 0.583 | **1.388** | 4 | 52 / 0 / 4 |
| `ACL-2026-VoI+Post-Defer` | 3.063 | -1.553 | 1.569 | **2.874** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI+Full-Defer+Naor` | 4.277 | -0.340 | 0.356 | **0.827** | 0 | 56 / 0 / 0 |
| `Congested-L2D-Only` | 3.375 | -1.241 | 1.257 | **3.046** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 4.119 | -0.497 | 0.513 | **1.786** | 8 | 48 / 0 / 8 |

### v2 (Qwen3.5-35B-A3B)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 4.144 | — | 0.211 | **0.451** | 13 | — |
| `Tri-VoI-L2D` | 4.124 | -0.021 | 0.232 | **0.511** | 2 | 30 / 0 / 26 |
| `Calibrated-Cascade+Naor` | 4.232 | +0.087 | 0.124 | **0.536** | 24 | 20 / 0 / 36 |
| `Joint-Greedy+Naor` | 3.903 | -0.242 | 0.452 | **0.975** | 3 | 46 / 0 / 10 |
| `KL-PBD+ (Warm)` | 3.144 | -1.001 | 1.211 | **2.086** | 0 | 55 / 0 / 1 |
| `ACL-2026-VoI (Live)+Post-Defer` | 3.097 | -1.047 | 1.258 | **1.777** | 0 | 55 / 0 / 1 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.289 | +0.145 | 0.066 | **0.292** | 14 | 15 / 0 / 41 |
| `Congested-L2D-Only` | 2.385 | -1.759 | 1.970 | **4.000** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 3.760 | -0.384 | 0.595 | **1.461** | 0 | 34 / 0 / 22 |

### v2 (DeepSeek-V3.2)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 4.428 | — | 0.040 | **0.178** | 24 | — |
| `Tri-VoI-L2D` | 4.258 | -0.170 | 0.211 | **1.529** | 20 | 32 / 2 / 22 |
| `Calibrated-Cascade+Naor` | 4.117 | -0.311 | 0.351 | **1.042** | 11 | 44 / 0 / 12 |
| `Joint-Greedy+Naor` | 4.174 | -0.254 | 0.295 | **1.271** | 2 | 51 / 0 / 5 |
| `KL-PBD+ (Warm)` | 3.301 | -1.127 | 1.167 | **2.368** | 2 | 54 / 0 / 2 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.797 | -1.631 | 1.671 | **2.476** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.193 | -0.235 | 0.276 | **0.848** | 5 | 49 / 0 / 7 |
| `Congested-L2D-Only` | 2.790 | -1.638 | 1.679 | **3.617** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 4.001 | -0.427 | 0.467 | **2.366** | 12 | 39 / 2 / 15 |

### v2 (Hy4-preview)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 5.254 | — | 0.262 | **1.307** | 13 | — |
| `Tri-VoI-L2D` | 5.169 | -0.086 | 0.347 | **1.410** | 7 | 32 / 12 / 12 |
| `Calibrated-Cascade+Naor` | 5.497 | +0.243 | 0.019 | **0.373** | 47 | 9 / 4 / 43 |
| `Joint-Greedy+Naor` | 4.692 | -0.562 | 0.824 | **1.410** | 0 | 56 / 0 / 0 |
| `KL-PBD+ (Warm)` | 3.255 | -1.999 | 2.260 | **3.354** | 0 | 50 / 0 / 6 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.983 | -2.271 | 2.533 | **3.628** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 5.005 | -0.249 | 0.511 | **1.408** | 0 | 39 / 0 / 17 |
| `Congested-L2D-Only` | 2.608 | -2.646 | 2.908 | **4.526** | 0 | 56 / 0 / 0 |
| `w/o Queue-Externality` | 4.902 | -0.353 | 0.614 | **2.276** | 4 | 43 / 13 / 0 |

### Public-Tri (Qwen3.5-35B-A3B (c_def=2.0)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 2.456 | — | 0.131 | **0.405** | 18 | — |
| `Tri-VoI-L2D` | 2.331 | -0.125 | 0.256 | **0.780** | 1 | 31 / 0 / 25 |
| `Calibrated-Cascade+Naor` | 1.852 | -0.603 | 0.734 | **2.185** | 1 | 54 / 0 / 2 |
| `Joint-Greedy+Naor` | 2.488 | +0.033 | 0.098 | **0.566** | 19 | 21 / 0 / 35 |
| `KL-PBD+ (Warm)` | 2.189 | -0.267 | 0.398 | **1.554** | 9 | 32 / 0 / 24 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.412 | -2.868 | 2.999 | **5.157** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.956 | -0.500 | 0.631 | **1.442** | 0 | 54 / 0 / 2 |
| `Congested-L2D-Only` | 0.826 | -1.630 | 1.760 | **4.362** | 8 | 36 / 0 / 20 |
| `w/o Queue-Externality` | 1.721 | -0.735 | 0.865 | **2.032** | 0 | 37 / 0 / 19 |

### Public-Tri (Qwen3.5-35B-A3B (c_def=4.0)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 1.298 | — | 0.214 | **0.679** | 13 | — |
| `Tri-VoI-L2D` | 1.283 | -0.015 | 0.229 | **0.579** | 0 | 36 / 0 / 20 |
| `Calibrated-Cascade+Naor` | 0.859 | -0.440 | 0.654 | **1.885** | 0 | 56 / 0 / 0 |
| `Joint-Greedy+Naor` | 1.472 | +0.173 | 0.041 | **0.411** | 36 | 14 / 0 / 42 |
| `KL-PBD+ (Warm)` | 0.977 | -0.321 | 0.535 | **1.792** | 7 | 38 / 0 / 18 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.699 | -1.997 | 2.211 | **3.790** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.948 | -0.350 | 0.564 | **1.250** | 0 | 56 / 0 / 0 |
| `Congested-L2D-Only` | -0.175 | -1.474 | 1.687 | **3.970** | 0 | 45 / 0 / 11 |
| `w/o Queue-Externality` | 0.878 | -0.420 | 0.634 | **1.775** | 0 | 39 / 0 / 17 |

### Public-Tri (DeepSeek-V3.2 (c_def=2.0)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 2.284 | — | 0.126 | **0.682** | 20 | — |
| `Tri-VoI-L2D` | 2.084 | -0.200 | 0.326 | **1.011** | 8 | 35 / 0 / 21 |
| `Calibrated-Cascade+Naor` | 1.546 | -0.738 | 0.865 | **2.507** | 2 | 54 / 0 / 2 |
| `Joint-Greedy+Naor` | 2.070 | -0.214 | 0.340 | **0.984** | 8 | 35 / 0 / 21 |
| `KL-PBD+ (Warm)` | 2.184 | -0.101 | 0.227 | **1.309** | 13 | 23 / 0 / 33 |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.896 | -3.181 | 3.307 | **5.650** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.748 | -0.536 | 0.663 | **1.385** | 0 | 54 / 0 / 2 |
| `Congested-L2D-Only` | 1.234 | -1.050 | 1.177 | **3.563** | 10 | 36 / 0 / 20 |
| `w/o Queue-Externality` | 1.533 | -0.751 | 0.878 | **2.267** | 13 | 37 / 0 / 19 |

### Public-Tri (DeepSeek-V3.2 (c_def=4.0)
| Method | Mean Util (56) | $\Delta$ vs MDP | Mean Regret | Worst-Case Regret | Best Cells | MDP W / T / L |
|:---|---:|---:|---:|---:|---:|---:|
| `Tri-VoI-MDP` | 1.009 | — | 0.241 | **0.792** | 18 | — |
| `Tri-VoI-L2D` | 0.946 | -0.063 | 0.304 | **0.753** | 4 | 31 / 0 / 25 |
| `Calibrated-Cascade+Naor` | 0.610 | -0.398 | 0.639 | **2.179** | 1 | 49 / 0 / 7 |
| `Joint-Greedy+Naor` | 0.934 | -0.075 | 0.315 | **0.714** | 2 | 33 / 0 / 23 |
| `KL-PBD+ (Warm)` | 0.985 | -0.024 | 0.264 | **1.264** | 25 | 26 / 0 / 30 |
| `ACL-2026-VoI (Live)+Post-Defer` | -1.277 | -2.286 | 2.526 | **4.343** | 0 | 56 / 0 / 0 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.776 | -0.233 | 0.473 | **1.217** | 7 | 47 / 0 / 9 |
| `Congested-L2D-Only` | 0.130 | -0.878 | 1.119 | **3.009** | 0 | 40 / 0 / 16 |
| `w/o Queue-Externality` | 0.467 | -0.542 | 0.783 | **1.765** | 0 | 43 / 0 / 13 |

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
| `Tri-VoI-MDP` | 3.811 | 3.500 | 2.398 |

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
| `Tri-VoI-MDP` | 4.468 | 4.104 | 3.480 |

## 3. End-to-End Stream `ASK` Latency Sweep ($\tau_{\text{ask}} \in \{0..8\}$)
### Cap=2, c=1.5
| $\tau_{\text{ask}}$ | `Tri-VoI-MDP` | `Tri-VoI-L2D` | `Tri-VoI (Lat-Aware)` | `Calib-Casc+Naor` | `KL-PBD+ (Warm)` |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.341 (24.3%) | 5.220 (24.7%) | 5.220 (24.7%) | 4.867 | 4.450 |
| 1 | 5.207 (22.2%) | 5.200 (24.9%) | 5.178 (23.3%) | 4.727 | 4.450 |
| 2 | 5.209 (20.4%) | 5.106 (25.0%) | 5.232 (21.3%) | 4.615 | 4.450 |
| 4 | 5.165 (18.0%) | 4.940 (25.2%) | 5.088 (19.4%) | 4.391 | 4.450 |
| 8 | 4.999 (13.8%) | 4.676 (25.1%) | 4.975 (14.6%) | 3.925 | 4.450 |

### Cap=1/3, c=1.5
| $\tau_{\text{ask}}$ | `Tri-VoI-MDP` | `Tri-VoI-L2D` | `Tri-VoI (Lat-Aware)` | `Calib-Casc+Naor` | `KL-PBD+ (Warm)` |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.048 (39.2%) | 3.792 (38.9%) | 3.792 (38.9%) | 3.798 | 3.106 |
| 1 | 3.953 (39.2%) | 3.579 (38.9%) | 3.526 (37.6%) | 3.667 | 3.106 |
| 2 | 3.744 (38.4%) | 3.582 (38.9%) | 3.582 (36.8%) | 3.607 | 3.106 |
| 4 | 3.618 (38.3%) | 3.210 (38.9%) | 3.259 (36.0%) | 3.245 | 3.106 |
| 8 | 3.146 (37.5%) | 2.991 (38.6%) | 2.933 (31.3%) | 2.911 | 3.106 |

## 4. Real-LLM Clinical Specialist (`Public-Tri`, Full-Quadrant + Pure Capability-Gap) vs. Semi-Synthetic (`v1/v2`)
### real_public_tri_qwen35_to_deepseek_stationary
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=3.020, `ACL-2026-VoI (Live)+Full-Defer+Naor`=2.635, `Calibrated-Cascade+Naor`=2.680, `Joint-Greedy+Naor`=3.127, `Tri-VoI-L2D`=3.124, `Tri-VoI-MDP`=2.957
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=1.471, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.963, `Calibrated-Cascade+Naor`=1.785, `Joint-Greedy+Naor`=1.355, `Tri-VoI-L2D`=1.477, `Tri-VoI-MDP`=1.830
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=0.279, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.276, `Calibrated-Cascade+Naor`=1.367, `Joint-Greedy+Naor`=1.420, `Tri-VoI-L2D`=0.960, `Tri-VoI-MDP`=1.651

### real_public_tri_qwen35_split_metrics
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250 (acc=72.1%), `KL-PBD+ (Warm)`=3.020 (acc=77.9%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=2.635 (acc=77.5%), `Calibrated-Cascade+Naor`=2.680 (acc=77.9%), `Joint-Greedy+Naor`=3.127 (acc=77.9%), `Tri-VoI-L2D`=3.124 (acc=77.9%), `Tri-VoI-MDP`=2.957 (acc=77.5%)
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.250 (acc=72.1%), `KL-PBD+ (Warm)`=1.471 (acc=77.1%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.963 (acc=79.2%), `Calibrated-Cascade+Naor`=1.785 (acc=79.2%), `Joint-Greedy+Naor`=1.355 (acc=79.2%), `Tri-VoI-L2D`=1.477 (acc=78.8%), `Tri-VoI-MDP`=1.830 (acc=78.8%)
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250 (acc=72.1%), `KL-PBD+ (Warm)`=0.279 (acc=72.1%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.276 (acc=80.0%), `Calibrated-Cascade+Naor`=1.367 (acc=80.0%), `Joint-Greedy+Naor`=1.420 (acc=80.8%), `Tri-VoI-L2D`=0.960 (acc=80.4%), `Tri-VoI-MDP`=1.651 (acc=79.6%)

### real_public_tri_qwen35_to_ds_to_qwen8b_shift
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=1.581, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.827, `Calibrated-Cascade+Naor`=1.888, `Joint-Greedy+Naor`=2.121, `Tri-VoI-L2D`=2.057, `Tri-VoI-MDP`=1.993
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=0.245, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.428, `Calibrated-Cascade+Naor`=1.373, `Joint-Greedy+Naor`=1.322, `Tri-VoI-L2D`=1.094, `Tri-VoI-MDP`=1.017
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.250, `KL-PBD+ (Warm)`=-0.449, `ACL-2026-VoI (Live)+Full-Defer+Naor`=0.636, `Calibrated-Cascade+Naor`=0.499, `Joint-Greedy+Naor`=0.870, `Tri-VoI-L2D`=0.367, `Tri-VoI-MDP`=0.834

### real_public_tri_qwen8b_to_deepseek_stationary
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-3.250, `KL-PBD+ (Warm)`=0.773, `ACL-2026-VoI (Live)+Full-Defer+Naor`=-1.172, `Calibrated-Cascade+Naor`=-1.269, `Joint-Greedy+Naor`=0.147, `Tri-VoI-L2D`=0.234, `Tri-VoI-MDP`=-0.188
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-3.250, `KL-PBD+ (Warm)`=-1.226, `ACL-2026-VoI (Live)+Full-Defer+Naor`=-1.369, `Calibrated-Cascade+Naor`=-1.309, `Joint-Greedy+Naor`=-1.329, `Tri-VoI-L2D`=-1.444, `Tri-VoI-MDP`=-0.331

### pure_cap_gap_qwen8b_to_qwen35
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=1.083 (acc=55.4%), `KL-PBD+ (Warm)`=3.679 (acc=78.8%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=2.997 (acc=77.5%), `Calibrated-Cascade+Naor`=2.642 (acc=76.7%), `Joint-Greedy+Naor`=3.641 (acc=77.5%), `Tri-VoI-L2D`=3.641 (acc=77.5%), `Tri-VoI-MDP`=3.474 (acc=77.1%)
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=1.083 (acc=55.4%), `KL-PBD+ (Warm)`=1.866 (acc=62.5%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.389 (acc=65.8%), `Calibrated-Cascade+Naor`=1.290 (acc=66.2%), `Joint-Greedy+Naor`=1.455 (acc=62.5%), `Tri-VoI-L2D`=1.134 (acc=60.8%), `Tri-VoI-MDP`=2.196 (acc=65.0%)

### pure_cap_gap_qwen8b_to_deepseek
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=1.083 (acc=55.4%), `KL-PBD+ (Warm)`=3.046 (acc=74.6%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.108 (acc=60.4%), `Calibrated-Cascade+Naor`=0.679 (acc=59.2%), `Joint-Greedy+Naor`=2.794 (acc=72.5%), `Tri-VoI-L2D`=2.794 (acc=72.5%), `Tri-VoI-MDP`=0.957 (acc=57.5%)
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=1.083 (acc=55.4%), `KL-PBD+ (Warm)`=1.336 (acc=59.6%), `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.208 (acc=62.9%), `Calibrated-Cascade+Naor`=1.092 (acc=63.3%), `Joint-Greedy+Naor`=1.557 (acc=63.3%), `Tri-VoI-L2D`=1.557 (acc=63.3%), `Tri-VoI-MDP`=1.855 (acc=63.3%)

### semi_synthetic_v1_qwen35_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=2.000, `KL-PBD+ (Warm)`=5.592, `ACL-2026-VoI+Full-Defer+Naor`=5.662, `Calibrated-Cascade+Naor`=5.662, `Joint-Greedy+Naor`=6.060, `Tri-VoI-L2D`=5.893, `Tri-VoI-MDP`=5.727
- **Cap=1/3, c=1.5**: `Direct-ACT`=2.000, `KL-PBD+ (Warm)`=3.633, `ACL-2026-VoI+Full-Defer+Naor`=4.216, `Calibrated-Cascade+Naor`=4.216, `Joint-Greedy+Naor`=3.602, `Tri-VoI-L2D`=4.175, `Tri-VoI-MDP`=4.309

### semi_synthetic_v2_qwen35_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=1.333, `KL-PBD+ (Warm)`=4.652, `ACL-2026-VoI (Live)+Full-Defer+Naor`=4.789, `Calibrated-Cascade+Naor`=4.772, `Joint-Greedy+Naor`=5.044, `Tri-VoI-L2D`=4.795, `Tri-VoI-MDP`=4.659
- **Cap=1/3, c=1.5**: `Direct-ACT`=1.333, `KL-PBD+ (Warm)`=2.563, `ACL-2026-VoI (Live)+Full-Defer+Naor`=4.062, `Calibrated-Cascade+Naor`=3.906, `Joint-Greedy+Naor`=3.525, `Tri-VoI-L2D`=3.908, `Tri-VoI-MDP`=3.930

### semi_synthetic_v1_qwen8b_to_hy4
- **Cap=2, c=1.5**: `Direct-ACT`=0.778, `KL-PBD+ (Warm)`=5.258, `ACL-2026-VoI+Full-Defer+Naor`=5.202, `Calibrated-Cascade+Naor`=5.324, `Joint-Greedy+Naor`=5.202, `Tri-VoI-L2D`=5.202, `Tri-VoI-MDP`=5.284
- **Cap=1/3, c=1.5**: `Direct-ACT`=0.778, `KL-PBD+ (Warm)`=2.800, `ACL-2026-VoI+Full-Defer+Naor`=2.095, `Calibrated-Cascade+Naor`=2.217, `Joint-Greedy+Naor`=2.095, `Tri-VoI-L2D`=2.095, `Tri-VoI-MDP`=3.485

## 5. 2-Call Measured SQLite Token Cost & `KL-PBD+ (SC-Only)` Signal Parity
| Suite | Model | `Tri-VoI` (Full) | `Tri-VoI-MDP` | T1 Tok (Full->2C) | Realized Tok (Full->MDP, Δ%) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 (83.67%) | **5.341** (84.11%) | 4732 -> 1895 | 4952 -> 2112 (-57.4%) | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | 5.523 (85.44%) | **5.520** (85.5%) | 4095 -> 1706 | 4441 -> 2055 (-53.7%) | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | 5.842 (86.94%) | **5.863** (87.06%) | 4256 -> 1753 | 4622 -> 2125 (-54.0%) | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 (80.89%) | **4.690** (81.11%) | 4763 -> 1897 | 4967 -> 2094 (-57.8%) | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | 5.031 (83.0%) | **5.135** (83.56%) | 4079 -> 1701 | 4373 -> 1992 (-54.4%) | 3.620 | 2.965 |
| v2 | Hy4-preview | 6.053 (87.89%) | **6.085** (88.0%) | 4248 -> 1755 | 4562 -> 2076 (-54.5%) | 3.426 | 2.788 |

## 6. Explicit 2x2 Decomposition (`±EVOI × ±Naor` & Exact MDP Shadow Pricing)
| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI` | `Tri-VoI-MDP` | MDP vs Tri | MDP vs CC+N |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 5.142 | 5.190 | 4.877 | 4.867 | 4.867 | 5.274 | 5.220 | **5.341** | +0.121 | +0.474 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 2.212 | 3.188 | 3.152 | 3.798 | 3.798 | 2.732 | 3.792 | **4.048** | +0.256 | +0.250 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.294 | 4.421 | 4.996 | 4.996 | 4.993 | 4.615 | 4.657 | **4.690** | +0.033 | -0.306 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 2.321 | 3.169 | 3.580 | 3.808 | 3.805 | 2.931 | 3.847 | **4.041** | +0.194 | +0.233 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.841 | 4.850 | 4.754 | 4.754 | 4.754 | 5.020 | 5.031 | **5.135** | +0.104 | +0.381 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 3.182 | 4.014 | 4.333 | 4.509 | 4.509 | 3.957 | 4.334 | **4.406** | +0.072 | -0.103 |
| v2 / Hy4-preview | Cap=2, c=1.5 | 5.390 | 5.408 | 6.271 | 6.271 | 5.986 | 6.024 | 6.053 | **6.085** | +0.032 | -0.186 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 3.860 | 4.711 | 5.378 | 5.512 | 5.021 | 5.252 | 5.313 | **5.551** | +0.238 | +0.039 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=2, c=1.5 | 3.693 | 3.723 | 3.140 | 3.067 | 3.105 | 3.545 | 3.612 | **3.445** | -0.167 | +0.378 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=1/3, c=1.5 | 0.224 | 1.450 | 0.053 | 1.347 | 1.251 | -0.057 | 1.259 | **1.720** | +0.461 | +0.373 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=2, c=1.5 | 2.229 | 2.237 | 1.626 | 1.684 | 1.797 | 1.966 | 1.999 | **2.083** | +0.084 | +0.399 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=1/3, c=1.5 | 0.187 | 1.082 | -0.017 | 0.853 | 0.784 | -0.247 | 0.780 | **1.154** | +0.374 | +0.301 |
| Public-Tri / DeepSeek (c_def=2) | Cap=2, c=1.5 | 3.520 | 3.331 | 2.905 | 2.769 | 2.871 | 3.505 | 3.327 | **3.382** | +0.055 | +0.613 |
| Public-Tri / DeepSeek (c_def=2) | Cap=1/3, c=1.5 | -0.908 | 0.444 | -0.444 | 0.752 | 0.784 | -0.755 | 0.417 | **1.428** | +1.011 | +0.676 |

## 7. 20 Stratified Random `(cal, test)` Resplits & 6 Disjoint-Episode Check
| Suite | Regime | `Tri-VoI-MDP` (mean±sd) | `Tri-VoI-L2D` | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | MDP Wins/20 vs (Tri, KL, ACL, JG+N, CC+N) |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | Cap=2, c=1.5 | **4.874±0.41** | 4.869 | 4.424 | 4.619 | 4.626 | 4.822 | 12/20/17/16/13 |
| v1-Clean (Qwen3.5) | Cap=1/3, c=1.5 | **3.972±0.56** | 3.766 | 3.240 | 3.688 | 3.391 | 4.055 | 15/19/16/20/10 |
| v2 (Qwen3.5) | Cap=2, c=1.5 | **4.696±0.40** | 4.628 | 3.840 | 4.749 | 4.380 | 4.737 | 8/20/9/16/10 |
| v2 (Qwen3.5) | Cap=1/3, c=1.5 | **4.096±0.38** | 3.693 | 2.333 | 3.905 | 3.190 | 3.920 | 18/20/17/20/16 |
| v2 (DeepSeek-V3.2) | Cap=2, c=1.5 | **4.983±0.34** | 4.974 | 4.097 | 4.744 | 4.914 | 4.744 | 11/19/14/11/14 |
| v2 (DeepSeek-V3.2) | Cap=1/3, c=1.5 | **4.248±0.66** | 4.088 | 2.535 | 4.267 | 3.823 | 4.267 | 13/20/12/19/12 |
| Public-Tri (Qwen3.5) | Cap=2, c=1.5 | **3.407±0.48** | 3.414 | 3.332 | 3.061 | 3.457 | 3.000 | 9/11/18/7/18 |
| Public-Tri (Qwen3.5) | Cap=1/3, c=1.5 | **1.225±0.44** | 0.589 | 0.649 | 0.885 | 0.721 | 0.909 | 19/18/16/19/16 |
| Public-Tri (DeepSeek-V3.2) | Cap=2, c=1.5 | **3.204±0.42** | 3.208 | 3.283 | 2.856 | 3.217 | 2.805 | 10/8/17/9/18 |
| Public-Tri (DeepSeek-V3.2) | Cap=1/3, c=1.5 | **1.210±0.49** | 0.404 | 0.484 | 0.798 | 0.347 | 0.724 | 20/19/19/20/19 |
