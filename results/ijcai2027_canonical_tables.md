# IJCAI 2027 Canonical Tables (Audit-Verified Single Source of Truth)

Generated automatically by `generate_ijcai_tables.py` using canonical terminology (`Requester`, `Specialist`, `Benchmark v1-Clean`, `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri`, `Benchmark v3`, `MediQ-Holdout`). Primary method: **`Tri-VoI-MDP`** (2-Call coarse sufficient statistic + exact 3-outcome Stage-2 `no_info` alignment + periodic-drain finite-horizon Bellman DP shadow prices). **Bolding strictly marks the true row/column best non-Oracle metric.**

## Table 1: `Benchmark v1-Clean` & `Benchmark v1-Noisy` Main Comparison (`Qwen/Qwen3.5-35B-A3B`, `N=360`)
| Method | v1-Clean (Cap=2, c=1.5) | v1-Clean (Cap=2, c=2.5) | v1-Clean (Cap=1/3, c=1.5) | v1-Noisy (Cap=2, c=1.5) | v1-Noisy (Cap=2, c=2.5) | v1-Noisy (Cap=1/3, c=1.5) |
|:---|---:|---:|---:|---:|---:|---:|
| `Direct-ACT` | 2.000 | 2.000 | 2.000 | 0.556 | 0.556 | 0.556 |
| `KL-PBD` | 4.061 | 4.061 | 2.621 | 3.389 | 3.389 | 1.268 |
| `KL-PBD+ (Warm)` | 4.450 | 4.450 | 3.106 | 3.665 | **3.665** | 1.517 |
| `ACL-2026-VoI (Calib)` | 2.182 | 2.333 | 2.182 | 1.142 | 0.715 | 1.142 |
| `ACL-2026-VoI+Post-Defer` | 3.633 | 2.800 | 3.609 | 2.081 | 1.071 | 2.031 |
| `Joint-Greedy+Naor` | 5.190 | 5.015 | 3.188 | **3.887** | 3.618 | 1.295 |
| `Calibrated-Cascade+Naor` | 4.867 | 4.470 | 3.798 | 3.413 | 2.952 | 1.614 |
| `Tri-VoI-L2D` | 5.220 | 5.142 | 3.792 | 3.503 | 3.511 | 1.525 |
| `Tri-VoI-MDP` | **5.341** | **5.168** | **4.048** | 3.811 | 3.500 | **2.398** |

## Table 2: Cross-Benchmark Generalization on `Benchmark v2 (Implicit-Override)` & `Public-Tri (MediQ Scope-of-Practice Triage)`
> **Transparency Note (`Public-Tri`)**: `Public-Tri` models **Scope-of-Practice / Regulatory Governance Triage** (`require_defer_for_ok=True` on high-acuity/multi-specialty `Q3`/`Q4_escalate`), alongside raw USMLE accuracy reported separately in Section 7. Bold marks true row-wise best among methods shown at `Cap=2, c=1.5` (and separate best across `Cap=1/3, c=1.5` for `Tri-VoI-L2D` vs `Tri-VoI-MDP` noted cleanly).

| Suite & Model | Miss-F1 / Collapse% | `KL-PBD+ (Warm)` | `ACL-2026-VoI (Live)` | `Joint-Greedy+Naor` | `Calib-Cascade+Naor` | `Tri-VoI-L2D` (Cap=2) | `Tri-VoI-MDP` (Cap=2) | `Tri-VoI-L2D` (1/3) | `Tri-VoI-MDP` (1/3) |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 / `Qwen3.5-35B-A3B` | 0.870 / 78.9% | 3.366 | 2.203 | 4.421 | **4.996** | 4.657 | 4.690 | 3.847 | 4.041 |
| v2 / `DeepSeek-V3.2` | 0.935 / 86.7% | 3.620 | 2.550 | 4.850 | 4.754 | 5.031 | **5.135** | 4.334 | 4.406 |
| v2 / `Hy4-preview` | 1.000 / 91.1% | 3.426 | 3.006 | 5.408 | **6.271** | 6.053 | 6.085 | 5.313 | 5.551 |
| v2 / `Qwen3-8B` | 0.178 / 87.8% | **3.600** | 0.131 | 3.551 | 3.508 | 3.571 | — | 1.163 | — |
| Public-Tri / `Qwen3.5-35B-A3B` | 0.932 / 81.7% | 3.533 | -0.400 | **3.723** | 3.067 | 3.612 | 3.446 | 1.259 | 1.720 |
| Public-Tri / `DeepSeek-V3.2` | 0.831 / 72.5% | **3.443** | -0.792 | 3.331 | 2.769 | 3.327 | 3.382 | 0.417 | 1.428 |
| Public-Tri / `Qwen3-8B` | 0.802 / 91.7% | **3.133** | -3.402 | 2.809 | 2.013 | 2.809 | — | -1.611 | — |

## Table 3: 56-Cell 2D Minimax Regret (Descriptive Sensitivity Envelope — No Pseudoreplication $p$-values)
| Suite | Method | Mean Util (56) | $\Delta$ vs MDP | Worst-Case Regret | MDP W / T / L |
|:---|:---|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | `Tri-VoI-MDP` | **4.616** | — | **0.404** | — |
| v1-Clean (Qwen3.5) | `Tri-VoI-L2D` | 4.470 | -0.147 | 0.596 | 49/0/7 |
| v1-Clean (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.095 | -0.521 | 1.529 | 56/0/0 |
| v1-Clean (Qwen3.5) | `Joint-Greedy+Naor` | 4.332 | -0.284 | 0.987 | 56/0/0 |
| v1-Clean (Qwen3.5) | `KL-PBD+ (Warm)` | 4.049 | -0.567 | 1.388 | 52/0/4 |
| v1-Clean (Qwen3.5) | `ACL-2026-VoI+Full-Defer+Naor` | 4.277 | -0.340 | 0.827 | 56/0/0 |
| v1-Clean (Qwen3.5) | `w/o Queue-Externality` | 4.119 | -0.497 | 1.786 | 48/0/8 |
| v2 (DeepSeek-V3.2) | `Tri-VoI-MDP` | **4.428** | — | **0.178** | — |
| v2 (DeepSeek-V3.2) | `Tri-VoI-L2D` | 4.258 | -0.170 | 1.529 | 32/2/22 |
| v2 (DeepSeek-V3.2) | `Calibrated-Cascade+Naor` | 4.117 | -0.311 | 1.042 | 44/0/12 |
| v2 (DeepSeek-V3.2) | `Joint-Greedy+Naor` | 4.174 | -0.254 | 1.271 | 51/0/5 |
| v2 (DeepSeek-V3.2) | `KL-PBD+ (Warm)` | 3.301 | -1.127 | 2.368 | 54/0/2 |
| v2 (DeepSeek-V3.2) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.193 | -0.235 | 0.848 | 49/0/7 |
| v2 (DeepSeek-V3.2) | `w/o Queue-Externality` | 4.001 | -0.427 | 2.366 | 39/2/15 |
| v2 (Qwen3.5) | `Tri-VoI-MDP` | 4.144 | — | 0.451 | — |
| v2 (Qwen3.5) | `Tri-VoI-L2D` | 4.124 | -0.021 | 0.511 | 30/0/26 |
| v2 (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.232 | +0.087 | 0.536 | 20/0/36 |
| v2 (Qwen3.5) | `Joint-Greedy+Naor` | 3.903 | -0.242 | 0.975 | 46/0/10 |
| v2 (Qwen3.5) | `KL-PBD+ (Warm)` | 3.144 | -1.001 | 2.086 | 55/0/1 |
| v2 (Qwen3.5) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | **4.289** | +0.145 | **0.292** | 15/0/41 |
| v2 (Qwen3.5) | `w/o Queue-Externality` | 3.760 | -0.384 | 1.461 | 34/0/22 |
| v2 (Hy4-preview) | `Tri-VoI-MDP` | 5.254 | — | 1.307 | — |
| v2 (Hy4-preview) | `Tri-VoI-L2D` | 5.169 | -0.086 | 1.410 | 32/12/12 |
| v2 (Hy4-preview) | `Calibrated-Cascade+Naor` | **5.497** | +0.243 | **0.373** | 9/4/43 |
| v2 (Hy4-preview) | `Joint-Greedy+Naor` | 4.692 | -0.562 | 1.410 | 56/0/0 |
| v2 (Hy4-preview) | `KL-PBD+ (Warm)` | 3.255 | -1.999 | 3.354 | 50/0/6 |
| v2 (Hy4-preview) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 5.005 | -0.249 | 1.408 | 39/0/17 |
| v2 (Hy4-preview) | `w/o Queue-Externality` | 4.902 | -0.353 | 2.276 | 43/13/0 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Tri-VoI-MDP` | 2.456 | — | **0.405** | — |
| Public-Tri (Qwen3.5, c_def=2.0) | `Tri-VoI-L2D` | 2.331 | -0.125 | 0.780 | 31/0/25 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Calibrated-Cascade+Naor` | 1.852 | -0.603 | 2.185 | 54/0/2 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Joint-Greedy+Naor` | **2.488** | +0.033 | 0.566 | 21/0/35 |
| Public-Tri (Qwen3.5, c_def=2.0) | `KL-PBD+ (Warm)` | 2.189 | -0.267 | 1.554 | 32/0/24 |
| Public-Tri (Qwen3.5, c_def=2.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.956 | -0.500 | 1.442 | 54/0/2 |
| Public-Tri (Qwen3.5, c_def=2.0) | `w/o Queue-Externality` | 1.721 | -0.735 | 2.032 | 37/0/19 |
| Public-Tri (DeepSeek, c_def=2.0) | `Tri-VoI-MDP` | **2.284** | — | **0.682** | — |
| Public-Tri (DeepSeek, c_def=2.0) | `Tri-VoI-L2D` | 2.084 | -0.200 | 1.011 | 35/0/21 |
| Public-Tri (DeepSeek, c_def=2.0) | `Calibrated-Cascade+Naor` | 1.546 | -0.738 | 2.507 | 54/0/2 |
| Public-Tri (DeepSeek, c_def=2.0) | `Joint-Greedy+Naor` | 2.070 | -0.214 | 0.984 | 35/0/21 |
| Public-Tri (DeepSeek, c_def=2.0) | `KL-PBD+ (Warm)` | 2.184 | -0.101 | 1.309 | 23/0/33 |
| Public-Tri (DeepSeek, c_def=2.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.748 | -0.536 | 1.385 | 54/0/2 |
| Public-Tri (DeepSeek, c_def=2.0) | `w/o Queue-Externality` | 1.533 | -0.751 | 2.267 | 37/0/19 |
| Public-Tri (DeepSeek, c_def=4.0) | `Tri-VoI-MDP` | **1.009** | — | 0.792 | — |
| Public-Tri (DeepSeek, c_def=4.0) | `Tri-VoI-L2D` | 0.946 | -0.063 | 0.753 | 31/0/25 |
| Public-Tri (DeepSeek, c_def=4.0) | `Calibrated-Cascade+Naor` | 0.610 | -0.398 | 2.179 | 49/0/7 |
| Public-Tri (DeepSeek, c_def=4.0) | `Joint-Greedy+Naor` | 0.934 | -0.075 | **0.714** | 33/0/23 |
| Public-Tri (DeepSeek, c_def=4.0) | `KL-PBD+ (Warm)` | 0.985 | -0.024 | 1.264 | 26/0/30 |
| Public-Tri (DeepSeek, c_def=4.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.776 | -0.233 | 1.217 | 47/0/9 |
| Public-Tri (DeepSeek, c_def=4.0) | `w/o Queue-Externality` | 0.467 | -0.542 | 1.765 | 43/0/13 |

## Section 4: Explicit 2x2 Decomposition (`±EVOI × ±Naor`) & Exact Bellman Shadow Pricing (`Tri-VoI-MDP`)
| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI-L2D` | `Smooth-Naor` | `Tri-VoI-MDP` | MDP vs L2D | MDP vs CC+N |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 5.142 | 5.190 | 4.877 | 4.867 | 4.867 | 5.274 | 5.220 | 5.249 | **5.341** | +0.121 | +0.474 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 2.212 | 3.188 | 3.152 | 3.798 | 3.798 | 2.732 | 3.792 | 3.792 | **4.048** | +0.256 | +0.250 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.294 | 4.421 | **4.996** | **4.996** | 4.993 | 4.615 | 4.657 | 4.829 | 4.690 | +0.033 | -0.306 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 2.321 | 3.169 | 3.580 | 3.808 | 3.805 | 2.931 | 3.847 | 3.847 | **4.041** | +0.194 | +0.233 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.841 | 4.850 | 4.754 | 4.754 | 4.754 | 5.020 | 5.031 | 5.028 | **5.135** | +0.104 | +0.381 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 3.182 | 4.014 | 4.333 | **4.509** | **4.509** | 3.957 | 4.334 | 4.334 | 4.406 | +0.072 | -0.103 |
| v2 / Hy4-preview | Cap=2, c=1.5 | 5.390 | 5.408 | **6.271** | **6.271** | 5.986 | 6.024 | 6.053 | 6.194 | 6.085 | +0.032 | -0.186 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 3.860 | 4.711 | 5.378 | 5.512 | 5.021 | 5.252 | 5.313 | 5.313 | **5.551** | +0.238 | +0.039 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=2, c=1.5 | 3.693 | **3.723** | 3.140 | 3.067 | 3.105 | 3.545 | 3.612 | 3.673 | 3.445 | -0.167 | +0.378 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=1/3, c=1.5 | 0.224 | 1.450 | 0.053 | 1.347 | 1.251 | -0.057 | 1.259 | 1.259 | **1.720** | +0.461 | +0.373 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=2, c=1.5 | 2.229 | **2.237** | 1.626 | 1.684 | 1.797 | 1.966 | 1.999 | 1.935 | 2.083 | +0.084 | +0.399 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=1/3, c=1.5 | 0.187 | 1.082 | -0.017 | 0.853 | 0.784 | -0.247 | 0.780 | 0.764 | **1.154** | +0.374 | +0.301 |
| Public-Tri / DeepSeek (c_def=2) | Cap=2, c=1.5 | **3.520** | 3.331 | 2.905 | 2.769 | 2.871 | 3.505 | 3.327 | 3.288 | 3.382 | +0.055 | +0.613 |
| Public-Tri / DeepSeek (c_def=2) | Cap=1/3, c=1.5 | -0.908 | 0.444 | -0.444 | 0.752 | 0.784 | -0.755 | 0.417 | 0.417 | **1.428** | +1.011 | +0.676 |

## Section 5: 20 Stratified Random `(cal, test)` Resplits (Calibrator + Stream Uncertainty)
| Suite | Regime | `Tri-VoI-MDP` (mean±sd) | `Tri-VoI-L2D` | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | MDP Wins/20 (L2D/KL/ACL/JG/CC) |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | Cap=2, c=1.5 | **4.874**±0.41 | 4.869 | 4.424 | 4.619 | 4.626 | 4.822 | 12/20/17/16/13 |
| v1-Clean (Qwen3.5) | Cap=1/3, c=1.5 | 3.972±0.56 | 3.766 | 3.240 | 3.688 | 3.391 | **4.055** | 15/19/16/20/10 |
| v2 (Qwen3.5) | Cap=2, c=1.5 | 4.696±0.40 | 4.628 | 3.840 | **4.749** | 4.380 | 4.737 | 8/20/9/16/10 |
| v2 (Qwen3.5) | Cap=1/3, c=1.5 | **4.096**±0.38 | 3.693 | 2.333 | 3.905 | 3.190 | 3.920 | 18/20/17/20/16 |
| v2 (DeepSeek-V3.2) | Cap=2, c=1.5 | **4.983**±0.34 | 4.974 | 4.097 | 4.744 | 4.914 | 4.744 | 11/19/14/11/14 |
| v2 (DeepSeek-V3.2) | Cap=1/3, c=1.5 | 4.248±0.66 | 4.088 | 2.535 | **4.267** | 3.823 | **4.267** | 13/20/12/19/12 |
| Public-Tri (Qwen3.5) | Cap=2, c=1.5 | 3.407±0.48 | 3.414 | 3.332 | 3.061 | **3.457** | 3.000 | 9/11/18/7/18 |
| Public-Tri (Qwen3.5) | Cap=1/3, c=1.5 | **1.225**±0.44 | 0.589 | 0.649 | 0.885 | 0.721 | 0.909 | 19/18/16/19/16 |
| Public-Tri (DeepSeek-V3.2) | Cap=2, c=1.5 | 3.204±0.42 | 3.208 | **3.283** | 2.856 | 3.217 | 2.805 | 10/8/17/9/18 |
| Public-Tri (DeepSeek-V3.2) | Cap=1/3, c=1.5 | **1.210**±0.49 | 0.404 | 0.484 | 0.798 | 0.347 | 0.724 | 20/19/19/20/19 |

## Section 6: Exact Measured SQLite Token Cost (`Tri-VoI-MDP` vs `Full` vs `2-Call`) & `KL-PBD+ (SC-Only)`
| Suite | Model | `Tri-VoI` Full | `Tri-VoI` 2-Call | `Tri-VoI-MDP` | Realized Tok (Full -> MDP, % Saved) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 | **5.416** | 5.341 | 4952 -> 2112 (-57.4%) | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | **5.523** | **5.523** | 5.520 | 4441 -> 2055 (-53.7%) | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | 5.842 | 5.842 | **5.863** | 4622 -> 2125 (-54.0%) | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 | **4.750** | 4.690 | 4967 -> 2094 (-57.8%) | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | 5.031 | 5.031 | **5.135** | 4373 -> 1992 (-54.4%) | 3.620 | 2.965 |
| v2 | Hy4-preview | 6.053 | 6.053 | **6.085** | 4562 -> 2076 (-54.5%) | 3.426 | 2.788 |

## Section 7: Genuine Real-LLM Specialist (`Public-Tri` Scope-Compliance + Pure Capability-Gap) vs. Semi-Synthetic (`v1/v2`)
| Setup | Regime | `KL-PBD+ (Warm)` | `ACL+FD+Naor` | `Calib-Casc+Naor` | `Joint-Greedy+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.) | Cap=2, c_def=2.0, c=1.5 | 3.020 | 2.635 | 2.680 | **3.127** | 3.124 | 2.957 |
| Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.) | Cap=2, c_def=4.0, c=1.5 | 1.471 | **1.963** | 1.785 | 1.355 | 1.477 | 1.830 |
| Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.) | Cap=1/3, c_def=2.0, c=1.5 | 0.279 | 1.276 | 1.367 | 1.420 | 0.960 | **1.651** |
| Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift) | Cap=2, c_def=2.0, c=1.5 | 1.581 | 1.827 | 1.888 | **2.121** | 2.057 | 1.993 |
| Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift) | Cap=2, c_def=4.0, c=1.5 | 0.245 | **1.428** | 1.373 | 1.322 | 1.094 | 1.017 |
| Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift) | Cap=1/3, c_def=2.0, c=1.5 | -0.449 | 0.636 | 0.499 | **0.870** | 0.367 | 0.834 |
| Real Scope `Public-Tri`: `Q8B` -> `DeepSeek` (Floor) | Cap=2, c_def=4.0, c=1.5 | **0.773** | -1.172 | -1.269 | 0.147 | 0.234 | -0.188 |
| Real Scope `Public-Tri`: `Q8B` -> `DeepSeek` (Floor) | Cap=1/3, c_def=2.0, c=1.5 | -1.226 | -1.369 | -1.309 | -1.329 | -1.444 | **-0.331** |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `Q3.5` | Cap=2, c_def=2.0, c=1.5 | **3.679** | 2.997 | 2.642 | 3.641 | 3.641 | 3.474 |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `Q3.5` | Cap=1/3, c_def=2.0, c=1.5 | 1.866 | 1.389 | 1.290 | 1.455 | 1.134 | **2.196** |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `DeepSeek` | Cap=2, c_def=2.0, c=1.5 | **3.046** | 1.108 | 0.679 | 2.794 | 2.794 | 0.957 |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `DeepSeek` | Cap=1/3, c_def=2.0, c=1.5 | 1.336 | 1.208 | 1.092 | 1.557 | 1.557 | **1.855** |
| Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override | Cap=2, c=1.5 | 5.592 | 5.662 | 5.662 | **6.060** | 5.893 | 5.727 |
| Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override | Cap=1/3, c=1.5 | 3.633 | 4.216 | 4.216 | 3.602 | 4.175 | **4.309** |
| Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override | Cap=2, c=1.5 | 4.652 | 4.789 | 4.772 | **5.044** | 4.795 | 4.659 |
| Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override | Cap=1/3, c=1.5 | 2.563 | **4.062** | 3.906 | 3.525 | 3.908 | 3.930 |

### Section 7b: Split Medical Accuracy vs. Mandatory Scope-Escalation Recall (`Public-Tri`, `Q3.5` -> `DeepSeek`)
| Regime | `KL-PBD+ (W)` (Util / RawAcc / EscRec) | `ACL+FD+N` | `CC+Naor` | `JG+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` |
|:---|---:|---:|---:|---:|---:|---:|
| Cap=2, c_def=2.0, c=1.5 | 3.020 / 77.9% / 92.2% | 2.635 / 77.5% / 86.7% | 2.680 / 77.9% / 86.7% | **3.127** / 77.9% / 91.1% | 3.124 / 77.9% / 91.1% | 2.957 / 77.5% / 90.0% |
| Cap=2, c_def=4.0, c=1.5 | 1.471 / 77.1% / 92.2% | **1.963** / 79.2% / 85.6% | 1.785 / 79.2% / 82.2% | 1.355 / 79.2% / 72.2% | 1.477 / 78.8% / 77.8% | 1.830 / 78.8% / 84.4% |
| Cap=1/3, c_def=2.0, c=1.5 | 0.279 / 72.1% / 38.9% | 1.276 / 80.0% / 55.6% | 1.367 / 80.0% / 57.8% | 1.420 / 80.8% / 53.3% | 0.960 / 80.4% / 50.0% | **1.651** / 79.6% / 56.7% |

## Section 8: Pre-Registered Zero-Touch Blind Evaluation (Git Tag `prereg-blind-v3`, `PY_DIFF_LEN=0`, Spend `¥3.32`)

### Suite A: `Benchmark v3` (4 Unseen Length-Matched Domains, Zero Protocol Code IDs, `N_cal=48, N_test=144`)
| Model & Regime | `Direct-ACT` | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` | Realized Tok (`L2D` -> `MDP`) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3` / `Qwen3.5-35B-A3B` (Cap=2, c=1.5) | -0.556 | 3.401 | **4.599** | 4.416 | **4.599** | 4.364 | 4.470 | 4734 -> 2091 |
| `v3` / `Qwen3.5-35B-A3B` (Cap=2, c=2.5) | -0.556 | 3.401 | 4.149 | **4.176** | 4.162 | 4.139 | 4.059 | 4640 -> 2004 |
| `v3` / `Qwen3.5-35B-A3B` (Cap=1/3, c=1.5) | -0.556 | 2.657 | **3.458** | 3.417 | **3.458** | **3.458** | 3.334 | 4815 -> 2187 |
| `v3` / `DeepSeek-V3.2` (Cap=2, c=1.5) | 0.000 | 3.216 | 4.411 | 4.204 | 4.528 | 4.558 | **4.582** | 4466 -> 2036 |
| `v3` / `DeepSeek-V3.2` (Cap=2, c=2.5) | 0.000 | 3.216 | 3.939 | 4.007 | 4.028 | 4.058 | **4.170** | 4350 -> 1928 |
| `v3` / `DeepSeek-V3.2` (Cap=1/3, c=1.5) | 0.000 | 1.413 | 2.612 | 3.469 | 2.695 | 2.695 | **3.484** | 4545 -> 2139 |

### Suite B: Blind `MediQ-Holdout` Real-LLM Specialist (Scope-Compliance + Pure Capability-Gap, `N_cal=48, N_test=128`)
| Setup & Regime | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` | Split `MDP` (RawMedAcc% / EscRec%) |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| Scope `Q3.5`->`DS` (Cap=2, c_def=2.0, c=1.5) | 1.694 | 1.530 | 1.244 | **1.718** | 1.676 | 1.538 | 76.6% / 66.7% |
| Scope `Q3.5`->`DS` (Cap=2, c_def=4.0, c=1.5) | -1.006 | 0.229 | -0.321 | **0.950** | 0.817 | 0.407 | 76.6% / 60.4% |
| Scope `Q3.5`->`DS` (Cap=1/3, c_def=2.0, c=1.5) | **1.035** | 0.222 | 0.191 | 0.882 | 0.434 | 1.003 | 76.6% / 54.2% |
| Pure Cap-Gap `Q8B`->`Q3.5` (Cap=2, c_def=2.0, c=1.5) | **2.247** | 0.637 | 0.895 | 0.266 | 0.827 | 0.496 | 57.0% / N/A |
| Pure Cap-Gap `Q8B`->`Q3.5` (Cap=1/3, c_def=2.0, c=1.5) | **0.892** | 0.476 | 0.419 | 0.455 | 0.148 | 0.551 | 57.8% / N/A |
| Pure Cap-Gap `Q8B`->`DS` (Cap=2, c_def=2.0, c=1.5) | **2.068** | 0.340 | 0.546 | 0.266 | 0.414 | 0.496 | 57.0% / N/A |
| Pure Cap-Gap `Q8B`->`DS` (Cap=1/3, c_def=2.0, c=1.5) | **1.388** | 0.217 | 0.344 | 0.330 | 0.148 | 0.258 | 56.2% / N/A |

### Pre-Choice-2 Historical Zero-Touch Reference (`zero_touch_holdout.json`, `Tri-VoI-L2D` Series)
| Suite / Model | Miss-F1 / Scope-F1 | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `2-Call` | `Smooth-Naor` |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| MediQ-Holdout / `Qwen3.5-35B-A3B` (Cap=2, c=1.5) | 0.797 / 0.583 | 3.632 | 3.529 | **3.843** | 3.453 | 3.690 | 3.497 | 3.698 |
| MediQ-Holdout / `Qwen3.5-35B-A3B` (Cap=2, c_def=4.0, c=1.5) | 0.797 / 0.583 | 2.397 | 2.188 | 2.351 | 2.114 | 2.325 | 2.480 | **2.518** |
| MediQ-Holdout / `Qwen3.5-35B-A3B` (Cap=1/3, c=1.5) | 0.797 / 0.583 | 1.129 | 0.460 | 1.169 | 0.995 | 0.924 | **1.346** | 0.924 |
| MediQ-Holdout / `DeepSeek-V3.2` (Cap=2, c=1.5) | 0.725 / 0.671 | 3.563 | 3.444 | 3.728 | 3.320 | 4.064 | 4.007 | **4.089** |
| MediQ-Holdout / `DeepSeek-V3.2` (Cap=2, c_def=4.0, c=1.5) | 0.725 / 0.671 | 2.201 | 2.102 | 2.476 | 2.071 | 2.728 | 2.785 | **2.847** |
| MediQ-Holdout / `DeepSeek-V3.2` (Cap=1/3, c=1.5) | 0.725 / 0.671 | 0.839 | 1.209 | 1.662 | 1.615 | **1.814** | 1.721 | 1.811 |
| v2-UnseenIDs / `Qwen3.5` (Cap=2, c=1.5) | 0.838 / 0.929 | 2.711 | 4.418 | 4.193 | **4.434** | 4.121 | 3.989 | 4.157 |
| v2-UnseenIDs / `Qwen3.5` (Cap=1/3, c=1.5) | 0.838 / 0.929 | 1.286 | 2.902 | 2.566 | **2.994** | 2.757 | 2.701 | 2.757 |
