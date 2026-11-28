# IJCAI 2027 Canonical Tables (Zero-Oracle Unified Single Source of Truth)

Generated automatically by `generate_ijcai_tables.py` using canonical terminology (`Requester`, `Specialist`, `Benchmark v1-Clean`, `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri`, `Benchmark v3`, `MediQ-Holdout`, `MediQ-Test-Disjoint`). **Primary Method across Sections 0–10: `Tri-VoI-MDP` (`Deploy-Self`, 100% Zero-Oracle)** — 2-Call coarse sufficient statistic + strictly local `post_missing` self-signal (zero access to simulator `oracle_revealed`) + finite-horizon periodic-drain Bellman DP shadow prices (**Theorem 4 & Proposition 5**). The legacy simulator-oracle variant (`MDP (Oracle-NoInfo Diag)`) appears **only** in explicit ablation columns of Sections 9A, 9B, and 10. **Bolding marks true row/column best non-Oracle metric.**

## Section 0: Cross-Suite Quantitative Robustness Summary (13 Suites, 100% Zero-Oracle)

### 0A. Severe-Congestion (`Cap=1/3, c=1.5`) Dynamic Bellman Shadow-Pricing Gains (`+MDP` vs `+Naor` across All 13 Suites)
| Policy Family | Positive Suites | Min Gain | Median Gain | Mean Gain | Max Gain |
|:---|:---:|---:|---:|---:|---:|
| `Tri-VoI-L2D -> Tri-VoI-MDP (Zero-Oracle)` | **12/13** | -0.124 | +0.435 | +0.429 | +0.914 |
| `CC+Naor -> CC+MDP` | **13/13** | +0.063 | +0.280 | +0.355 | +0.804 |
| `JG+Naor -> JG+MDP` | **12/13** | -0.029 | +0.370 | +0.429 | +0.824 |
| `ACL+FD+Naor -> ACL+FD+MDP` | **13/13** | +0.063 | +0.447 | +0.443 | +0.943 |

### 0B. Strictly Matched 2-Call Zero-Oracle Minimax Regret (`26` Cells = `13 Suites × {Cap=2, Cap=1/3}` at `c=1.5`, plus `40`-Cell `c_ask` Sweep)
| Policy (`2C-Self`, Zero-Oracle, Equal DP Pricing) | 26-Cell Mean Util | 26-Cell Wins | 26-Cell Mean Regret | 26-Cell Worst Regret | 40-Cell `c_ask` Mean Util | 40-Cell `c_ask` Worst Regret |
|:---|---:|---:|---:|---:|---:|---:|
| `CC+MDP (2C-S)` | 3.512 | 6/26 | 0.203 | 0.625 | 3.120 | 1.330 |
| `JG+MDP (2C-S)` | 3.480 | 8/26 | 0.236 | 0.820 | 3.324 | 0.927 |
| `ACL+FD+MDP (2C-S)` | 3.472 | 5/26 | 0.243 | 0.764 | 3.179 | 0.830 |
| `Tri-VoI-MDP (Zero-Oracle)` | 3.617 | 10/26 | 0.099 | 0.464 | 3.380 | 0.526 |

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
| `Tri-VoI-MDP` | **5.343** | **5.196** | **4.048** | 3.696 | 3.420 | **2.319** |

## Table 2: Cross-Benchmark Generalization on `Benchmark v2 (Implicit-Override)` & `Public-Tri (MediQ Scope-of-Practice Triage)`
> **Transparency Note (`Public-Tri`)**: `Public-Tri` models **Scope-of-Practice / Regulatory Governance Triage** (`require_defer_for_ok=True` on high-acuity/multi-specialty `Q3`/`Q4_escalate`), alongside raw USMLE accuracy reported separately in Section 7. `Tri-VoI-MDP` reports strictly Zero-Oracle `Deploy-Self`. Bold marks true row-wise best at `Cap=2, c=1.5`.

| Suite & Model | Miss-F1 / Collapse% | `KL-PBD+ (Warm)` | `ACL-2026-VoI (Live)` | `Joint-Greedy+Naor` | `Calib-Cascade+Naor` | `Tri-VoI-L2D` (Cap=2) | `Tri-VoI-MDP` (Cap=2) | `Tri-VoI-L2D` (1/3) | `Tri-VoI-MDP` (1/3) |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 / `Qwen3.5-35B-A3B` | 0.870 / 78.9% | 3.366 | 2.203 | 4.421 | **4.996** | 4.657 | 4.699 | 3.847 | 4.041 |
| v2 / `DeepSeek-V3.2` | 0.935 / 86.7% | 3.620 | 2.550 | 4.850 | 4.754 | 5.031 | **5.138** | 4.334 | 4.492 |
| v2 / `Hy4-preview` | 1.000 / 91.1% | 3.426 | 3.006 | 5.408 | **6.271** | 6.053 | 6.085 | 5.313 | 5.551 |
| v2 / `Qwen3-8B` | 0.178 / 87.8% | **3.600** | 0.131 | 3.551 | 3.508 | 3.571 | — | 1.163 | — |
| Public-Tri / `Qwen3.5-35B-A3B` | 0.932 / 81.7% | 3.533 | -0.400 | **3.723** | 3.067 | 3.612 | 3.446 | 1.259 | 1.718 |
| Public-Tri / `DeepSeek-V3.2` | 0.831 / 72.5% | 3.443 | -0.792 | 3.331 | 2.769 | 3.327 | **3.590** | 0.417 | 1.438 |
| Public-Tri / `Qwen3-8B` | 0.802 / 91.7% | **3.133** | -3.402 | 2.809 | 2.013 | 2.809 | — | -1.611 | — |

## Table 3: 56-Cell 2D Minimax Regret (`Tri-VoI-MDP` = Zero-Oracle `Deploy-Self`, Descriptive Sensitivity Envelope — No Pseudoreplication $p$-values)
| Suite | Method | Mean Util (56) | $\Delta$ vs MDP | Worst-Case Regret | MDP W / T / L |
|:---|:---|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | `Tri-VoI-MDP` | **4.623** | — | **0.404** | — |
| v1-Clean (Qwen3.5) | `Tri-VoI-L2D` | 4.470 | -0.154 | 0.596 | 56/0/0 |
| v1-Clean (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.095 | -0.529 | 1.529 | 56/0/0 |
| v1-Clean (Qwen3.5) | `Joint-Greedy+Naor` | 4.332 | -0.291 | 0.987 | 56/0/0 |
| v1-Clean (Qwen3.5) | `KL-PBD+ (Warm)` | 4.049 | -0.574 | 1.388 | 52/0/4 |
| v1-Clean (Qwen3.5) | `ACL-2026-VoI+Full-Defer+Naor` | 4.277 | -0.347 | 0.827 | 56/0/0 |
| v1-Clean (Qwen3.5) | `w/o Queue-Externality` | 4.120 | -0.504 | 1.786 | 54/0/2 |
| v2 (DeepSeek-V3.2) | `Tri-VoI-MDP` | **4.494** | — | **0.195** | — |
| v2 (DeepSeek-V3.2) | `Tri-VoI-L2D` | 4.258 | -0.237 | 1.508 | 34/0/22 |
| v2 (DeepSeek-V3.2) | `Calibrated-Cascade+Naor` | 4.117 | -0.377 | 1.453 | 43/0/13 |
| v2 (DeepSeek-V3.2) | `Joint-Greedy+Naor` | 4.174 | -0.321 | 1.251 | 50/0/6 |
| v2 (DeepSeek-V3.2) | `KL-PBD+ (Warm)` | 3.301 | -1.193 | 2.368 | 56/0/0 |
| v2 (DeepSeek-V3.2) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.193 | -0.302 | 1.056 | 48/0/8 |
| v2 (DeepSeek-V3.2) | `w/o Queue-Externality` | 4.001 | -0.493 | 2.346 | 44/0/12 |
| v2 (Qwen3.5) | `Tri-VoI-MDP` | 4.151 | — | 0.451 | — |
| v2 (Qwen3.5) | `Tri-VoI-L2D` | 4.124 | -0.027 | 0.511 | 30/0/26 |
| v2 (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.232 | +0.080 | 0.536 | 20/0/36 |
| v2 (Qwen3.5) | `Joint-Greedy+Naor` | 3.903 | -0.248 | 0.975 | 46/0/10 |
| v2 (Qwen3.5) | `KL-PBD+ (Warm)` | 3.144 | -1.007 | 2.086 | 55/0/1 |
| v2 (Qwen3.5) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | **4.289** | +0.138 | **0.292** | 15/0/41 |
| v2 (Qwen3.5) | `w/o Queue-Externality` | 3.760 | -0.391 | 1.461 | 35/0/21 |
| v2 (Hy4-preview) | `Tri-VoI-MDP` | 5.254 | — | 1.307 | — |
| v2 (Hy4-preview) | `Tri-VoI-L2D` | 5.169 | -0.086 | 1.410 | 32/12/12 |
| v2 (Hy4-preview) | `Calibrated-Cascade+Naor` | **5.497** | +0.243 | **0.374** | 9/4/43 |
| v2 (Hy4-preview) | `Joint-Greedy+Naor` | 4.692 | -0.562 | 1.410 | 56/0/0 |
| v2 (Hy4-preview) | `KL-PBD+ (Warm)` | 3.255 | -1.999 | 3.355 | 50/0/6 |
| v2 (Hy4-preview) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 5.005 | -0.249 | 1.408 | 39/0/17 |
| v2 (Hy4-preview) | `w/o Queue-Externality` | 4.902 | -0.353 | 2.276 | 43/13/0 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Tri-VoI-MDP` | **2.491** | — | **0.291** | — |
| Public-Tri (Qwen3.5, c_def=2.0) | `Tri-VoI-L2D` | 2.331 | -0.160 | 0.780 | 36/0/20 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Calibrated-Cascade+Naor` | 1.852 | -0.638 | 2.185 | 54/0/2 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Joint-Greedy+Naor` | 2.488 | -0.002 | 0.566 | 24/0/32 |
| Public-Tri (Qwen3.5, c_def=2.0) | `KL-PBD+ (Warm)` | 2.189 | -0.301 | 1.689 | 32/0/24 |
| Public-Tri (Qwen3.5, c_def=2.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.956 | -0.534 | 1.442 | 55/0/1 |
| Public-Tri (Qwen3.5, c_def=2.0) | `w/o Queue-Externality` | 1.721 | -0.769 | 2.032 | 39/0/17 |
| Public-Tri (DeepSeek, c_def=2.0) | `Tri-VoI-MDP` | **2.419** | — | **0.699** | — |
| Public-Tri (DeepSeek, c_def=2.0) | `Tri-VoI-L2D` | 2.084 | -0.335 | 1.120 | 47/0/9 |
| Public-Tri (DeepSeek, c_def=2.0) | `Calibrated-Cascade+Naor` | 1.546 | -0.873 | 2.495 | 56/0/0 |
| Public-Tri (DeepSeek, c_def=2.0) | `Joint-Greedy+Naor` | 2.070 | -0.349 | 1.208 | 47/0/9 |
| Public-Tri (DeepSeek, c_def=2.0) | `KL-PBD+ (Warm)` | 2.184 | -0.236 | 1.693 | 41/0/15 |
| Public-Tri (DeepSeek, c_def=2.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.748 | -0.671 | 1.385 | 56/0/0 |
| Public-Tri (DeepSeek, c_def=2.0) | `w/o Queue-Externality` | 1.533 | -0.886 | 2.321 | 48/0/8 |
| Public-Tri (DeepSeek, c_def=4.0) | `Tri-VoI-MDP` | **1.115** | — | **0.791** | — |
| Public-Tri (DeepSeek, c_def=4.0) | `Tri-VoI-L2D` | 0.946 | -0.169 | 0.910 | 39/0/17 |
| Public-Tri (DeepSeek, c_def=4.0) | `Calibrated-Cascade+Naor` | 0.610 | -0.505 | 2.179 | 54/0/2 |
| Public-Tri (DeepSeek, c_def=4.0) | `Joint-Greedy+Naor` | 0.934 | -0.181 | 0.933 | 40/0/16 |
| Public-Tri (DeepSeek, c_def=4.0) | `KL-PBD+ (Warm)` | 0.985 | -0.130 | 1.568 | 31/0/25 |
| Public-Tri (DeepSeek, c_def=4.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.776 | -0.339 | 1.217 | 53/0/3 |
| Public-Tri (DeepSeek, c_def=4.0) | `w/o Queue-Externality` | 0.467 | -0.649 | 1.944 | 45/0/11 |

## Section 4: Explicit 2x2 Decomposition (`±EVOI × ±Naor`) & Exact Bellman Shadow Pricing (`Tri-VoI-MDP`, Zero-Oracle)
| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI-L2D` | `Smooth-Naor` | `Tri-VoI-MDP` | MDP vs L2D | MDP vs CC+N |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 5.142 | 5.190 | 4.877 | 4.867 | 4.867 | 5.274 | 5.220 | 5.249 | **5.343** | +0.123 | +0.476 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 2.212 | 3.188 | 3.152 | 3.798 | 3.798 | 2.732 | 3.792 | 3.792 | **4.048** | +0.256 | +0.250 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.294 | 4.421 | **4.996** | **4.996** | 4.993 | 4.615 | 4.657 | 4.829 | 4.699 | +0.042 | -0.297 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 2.321 | 3.169 | 3.580 | 3.808 | 3.805 | 2.931 | 3.847 | 3.847 | **4.041** | +0.194 | +0.233 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.841 | 4.850 | 4.754 | 4.754 | 4.754 | 5.020 | 5.031 | 5.028 | **5.138** | +0.107 | +0.384 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 3.182 | 4.014 | 4.333 | **4.509** | **4.509** | 3.957 | 4.334 | 4.334 | 4.492 | +0.158 | -0.017 |
| v2 / Hy4-preview | Cap=2, c=1.5 | 5.390 | 5.408 | **6.271** | **6.271** | 5.986 | 6.024 | 6.053 | 6.194 | 6.085 | +0.032 | -0.186 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 3.860 | 4.711 | 5.378 | 5.512 | 5.021 | 5.252 | 5.313 | 5.313 | **5.551** | +0.238 | +0.039 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=2, c=1.5 | 3.693 | **3.723** | 3.140 | 3.067 | 3.105 | 3.545 | 3.612 | 3.673 | 3.446 | -0.166 | +0.379 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=1/3, c=1.5 | 0.224 | 1.450 | 0.053 | 1.347 | 1.251 | -0.057 | 1.259 | 1.259 | **1.718** | +0.459 | +0.371 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=2, c=1.5 | 2.229 | **2.237** | 1.626 | 1.684 | 1.797 | 1.966 | 1.999 | 1.935 | 2.097 | +0.098 | +0.413 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=1/3, c=1.5 | 0.187 | 1.082 | -0.017 | 0.853 | 0.784 | -0.247 | 0.780 | 0.764 | **1.121** | +0.341 | +0.268 |
| Public-Tri / DeepSeek (c_def=2) | Cap=2, c=1.5 | 3.520 | 3.331 | 2.905 | 2.769 | 2.871 | 3.505 | 3.327 | 3.288 | **3.590** | +0.263 | +0.821 |
| Public-Tri / DeepSeek (c_def=2) | Cap=1/3, c=1.5 | -0.908 | 0.444 | -0.444 | 0.752 | 0.784 | -0.755 | 0.417 | 0.417 | **1.438** | +1.021 | +0.686 |

## Section 5: 20 Stratified Random `(cal, test)` Resplits (`Tri-VoI-MDP` = Zero-Oracle `Deploy-Self`)
| Suite | Regime | `Tri-VoI-MDP` (mean±sd) | `Tri-VoI-L2D` | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | MDP Wins/20 (L2D/KL/ACL/JG/CC) |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | Cap=2, c=1.5 | **4.905**±0.40 | 4.869 | 4.424 | 4.619 | 4.626 | 4.822 | 11/20/17/16/12 |
| v1-Clean (Qwen3.5) | Cap=1/3, c=1.5 | 3.982±0.56 | 3.766 | 3.240 | 3.688 | 3.391 | **4.055** | 15/19/17/20/11 |
| v2 (Qwen3.5) | Cap=2, c=1.5 | 4.685±0.39 | 4.628 | 3.840 | **4.749** | 4.380 | 4.737 | 8/20/11/16/11 |
| v2 (Qwen3.5) | Cap=1/3, c=1.5 | **4.096**±0.38 | 3.693 | 2.333 | 3.905 | 3.190 | 3.920 | 18/20/17/20/16 |
| v2 (DeepSeek-V3.2) | Cap=2, c=1.5 | **4.986**±0.30 | 4.974 | 4.097 | 4.744 | 4.914 | 4.744 | 11/20/17/11/17 |
| v2 (DeepSeek-V3.2) | Cap=1/3, c=1.5 | 4.255±0.66 | 4.088 | 2.535 | **4.267** | 3.823 | **4.267** | 13/20/11/19/11 |
| Public-Tri (Qwen3.5) | Cap=2, c=1.5 | **3.467**±0.44 | 3.414 | 3.332 | 3.061 | 3.457 | 3.000 | 12/13/19/8/18 |
| Public-Tri (Qwen3.5) | Cap=1/3, c=1.5 | **1.307**±0.43 | 0.589 | 0.649 | 0.885 | 0.721 | 0.909 | 19/19/18/19/17 |
| Public-Tri (DeepSeek-V3.2) | Cap=2, c=1.5 | 3.271±0.41 | 3.208 | **3.283** | 2.856 | 3.217 | 2.805 | 13/8/19/11/19 |
| Public-Tri (DeepSeek-V3.2) | Cap=1/3, c=1.5 | **1.175**±0.52 | 0.404 | 0.484 | 0.798 | 0.347 | 0.724 | 19/19/17/19/19 |

## Section 6: Exact Measured SQLite Token Cost (`Tri-VoI-MDP` Zero-Oracle vs `Full` vs `2-Call`) & `KL-PBD+ (SC-Only)`
| Suite | Model | `Tri-VoI` Full | `Tri-VoI` 2-Call | `Tri-VoI-MDP` | Realized Tok (Full -> MDP, % Saved) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 | **5.416** | 5.343 | 4952 -> 2111 (-57.4%) | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | **5.523** | **5.523** | 5.519 | 4441 -> 2056 (-53.7%) | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | 5.842 | 5.842 | **5.863** | 4622 -> 2125 (-54.0%) | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 | **4.750** | 4.699 | 4967 -> 2097 (-57.8%) | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | 5.031 | 5.031 | **5.138** | 4373 -> 2014 (-53.9%) | 3.620 | 2.965 |
| v2 | Hy4-preview | 6.053 | 6.053 | **6.085** | 4562 -> 2076 (-54.5%) | 3.426 | 2.788 |

## Section 7: Genuine Real-LLM Specialist (`Public-Tri` Scope-Compliance + Pure Capability-Gap) vs. Semi-Synthetic (`v1/v2`)
| Setup | Regime | `KL-PBD+ (Warm)` | `ACL+FD+Naor` | `Calib-Casc+Naor` | `Joint-Greedy+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` (Zero-Oracle) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.) | Cap=2, c_def=2.0, c=1.5 | 3.020 | 2.635 | 2.680 | **3.127** | 3.124 | 2.961 |
| Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.) | Cap=2, c_def=4.0, c=1.5 | 1.471 | **1.963** | 1.785 | 1.355 | 1.477 | 1.673 |
| Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.) | Cap=1/3, c_def=2.0, c=1.5 | 0.279 | 1.276 | 1.367 | 1.420 | 0.960 | **1.671** |
| Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift) | Cap=2, c_def=2.0, c=1.5 | 1.581 | 1.827 | 1.888 | **2.121** | 2.057 | 2.002 |
| Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift) | Cap=2, c_def=4.0, c=1.5 | 0.245 | **1.428** | 1.373 | 1.322 | 1.094 | 0.957 |
| Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift) | Cap=1/3, c_def=2.0, c=1.5 | -0.449 | 0.636 | 0.499 | **0.870** | 0.367 | 0.607 |
| Real Scope `Public-Tri`: `Q8B` -> `DeepSeek` (Floor) | Cap=2, c_def=4.0, c=1.5 | **0.773** | -1.172 | -1.269 | 0.147 | 0.234 | -0.085 |
| Real Scope `Public-Tri`: `Q8B` -> `DeepSeek` (Floor) | Cap=1/3, c_def=2.0, c=1.5 | -1.226 | -1.369 | -1.309 | -1.329 | -1.444 | **-0.110** |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `Q3.5` | Cap=2, c_def=2.0, c=1.5 | **3.679** | 2.997 | 2.642 | 3.641 | 3.641 | 3.485 |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `Q3.5` | Cap=1/3, c_def=2.0, c=1.5 | 1.866 | 1.389 | 1.290 | 1.455 | 1.134 | **2.190** |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `DeepSeek` | Cap=2, c_def=2.0, c=1.5 | **3.046** | 1.108 | 0.679 | 2.794 | 2.794 | 1.160 |
| Pure Cap-Gap `Public-Tri`: `Q8B` -> `DeepSeek` | Cap=1/3, c_def=2.0, c=1.5 | 1.336 | 1.208 | 1.092 | **1.557** | **1.557** | 1.546 |
| Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override | Cap=2, c=1.5 | 5.592 | 5.662 | 5.662 | **6.060** | 5.893 | 5.851 |
| Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override | Cap=1/3, c=1.5 | 3.633 | 4.216 | 4.216 | 3.602 | 4.175 | **4.309** |
| Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override | Cap=2, c=1.5 | 4.652 | 4.789 | 4.772 | **5.044** | 4.795 | 4.659 |
| Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override | Cap=1/3, c=1.5 | 2.563 | **4.062** | 3.906 | 3.525 | 3.908 | 3.930 |

### Section 7b: Split Medical Accuracy vs. Mandatory Scope-Escalation Recall (`Public-Tri`, `Q3.5` -> `DeepSeek`)
| Regime | `KL-PBD+ (W)` (Util / RawAcc / EscRec) | `ACL+FD+N` | `CC+Naor` | `JG+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` (Zero-Oracle) |
|:---|---:|---:|---:|---:|---:|---:|
| Cap=2, c_def=2.0, c=1.5 | 3.020 / 77.9% / 92.2% | 2.635 / 77.5% / 86.7% | 2.680 / 77.9% / 86.7% | **3.127** / 77.9% / 91.1% | 3.124 / 77.9% / 91.1% | 2.961 / 77.5% / 90.0% |
| Cap=2, c_def=4.0, c=1.5 | 1.471 / 77.1% / 92.2% | **1.963** / 79.2% / 85.6% | 1.785 / 79.2% / 82.2% | 1.355 / 79.2% / 72.2% | 1.477 / 78.8% / 77.8% | 1.673 / 78.8% / 80.0% |
| Cap=1/3, c_def=2.0, c=1.5 | 0.279 / 72.1% / 38.9% | 1.276 / 80.0% / 55.6% | 1.367 / 80.0% / 57.8% | 1.420 / 80.8% / 53.3% | 0.960 / 80.4% / 50.0% | **1.671** / 79.6% / 55.6% |

## Section 8: Pre-Registered Unseen-Domain Template Generalization (`Benchmark v3`) & Disjoint `MediQ-Holdout` (`prereg-blind-v3`, Zero-Oracle `Tri-VoI-MDP`, Spend `¥3.32`)

> **Transparency Note on `Benchmark v3` Content Overlap**: Because each unseen domain in `v3` spans `2×2×2=8` discrete attribute templates, de-numbered audit shows `72/144` (`50.0%`) test items share normalized prose with `cal` and `51/144` (`35.4%`) share exact `(prose, missing_val, gold)` triples (`uniq_test_prose=73`). Thus `v3` measures **template generalization** (zero-overlap slices in Section 9D), while disjoint **`MediQ-Holdout`** (`176` cases) and **`MediQ-Test-Disjoint`** (Section 10) provide sample- and organ-system-disjoint clinical evidence.

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
| Scope `Q3.5`->`DS` (Cap=2, c_def=2.0, c=1.5) | 1.694 | 1.530 | 1.244 | 1.718 | 1.676 | **2.072** | 75.8% / 81.2% |
| Scope `Q3.5`->`DS` (Cap=2, c_def=4.0, c=1.5) | -1.006 | 0.229 | -0.321 | 0.950 | 0.817 | **1.050** | 75.8% / 72.9% |
| Scope `Q3.5`->`DS` (Cap=1/3, c_def=2.0, c=1.5) | 1.035 | 0.222 | 0.191 | 0.882 | 0.434 | **1.348** | 75.8% / 60.4% |
| Pure Cap-Gap `Q8B`->`Q3.5` (Cap=2, c_def=2.0, c=1.5) | **2.247** | 0.637 | 0.895 | 0.266 | 0.827 | 0.474 | 57.0% / N/A |
| Pure Cap-Gap `Q8B`->`Q3.5` (Cap=1/3, c_def=2.0, c=1.5) | **0.892** | 0.476 | 0.419 | 0.455 | 0.148 | 0.297 | 56.2% / N/A |
| Pure Cap-Gap `Q8B`->`DS` (Cap=2, c_def=2.0, c=1.5) | **2.068** | 0.340 | 0.546 | 0.266 | 0.414 | 0.474 | 57.0% / N/A |
| Pure Cap-Gap `Q8B`->`DS` (Cap=1/3, c_def=2.0, c=1.5) | **1.388** | 0.217 | 0.344 | 0.330 | 0.148 | 0.422 | 57.0% / N/A |

### Historical Pre-MDP Zero-Touch Reference (`zero_touch_holdout.json`, `Tri-VoI-L2D` Legacy Series)
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

## Section 9: Controlled Attribution (`±EVOI` under Equal `MDP` Shadow Pricing), Deployable `post_missing` Signal & `Benchmark v3` Zero-Overlap Slices (`live=0`, `¥0.00`)

### 9A. Step-by-Step Controlled Ablation (`L2D` -> `+2-Call` -> `+2C+NoInfo` -> `MDP (5C, Oracle Diag)` -> `MDP (Oracle-NoInfo Diag)` -> **`Tri-VoI-MDP` (Zero-Oracle)**)
| Suite | Regime | `L2D (5C)` | `+2-Call` | `+2C+NoInfo` | `MDP (5C, Oracle Diag)` | `MDP (Oracle-NoInfo Diag)` | `Tri-VoI-MDP` (Zero-Oracle) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 5.220 | **5.416** | 5.401 | 5.199 | 5.341 | 5.343 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 3.792 | 3.681 | 3.681 | 4.024 | 4.048 | **4.048** |
| v1-Noisy / Qwen3.5 | Cap=2, c=1.5 | 3.503 | 3.538 | **3.753** | 3.857 | 3.811 | 3.696 |
| v1-Noisy / Qwen3.5 | Cap=1/3, c=1.5 | 1.525 | 1.616 | 1.552 | 2.303 | 2.398 | **2.319** |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.657 | **4.750** | 4.728 | 4.703 | 4.690 | 4.699 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 3.847 | 3.810 | 3.810 | 4.016 | 4.041 | **4.041** |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 5.031 | 5.031 | 5.035 | 5.135 | 5.135 | **5.138** |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 4.334 | 4.284 | 4.284 | 4.572 | 4.406 | **4.492** |
| v2 / Hy4-preview | Cap=2, c=1.5 | 6.053 | 6.053 | 6.053 | 6.085 | 6.085 | **6.085** |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 5.313 | 5.095 | 5.095 | 5.762 | 5.551 | **5.551** |
| Public-Tri / Qwen3.5 (strat=False) | Cap=2, c=1.5 | 3.404 | 3.448 | **3.474** | 3.407 | 3.445 | 3.446 |
| Public-Tri / Qwen3.5 (strat=False) | Cap=1/3, c=1.5 | 1.203 | 1.125 | 1.357 | 1.638 | 1.720 | **1.718** |
| Public-Tri / Qwen3.5 (strat=True) | Cap=2, c=1.5 | **3.612** | 3.579 | 3.452 | 3.501 | 3.521 | 3.544 |
| Public-Tri / Qwen3.5 (strat=True) | Cap=1/3, c=1.5 | 1.259 | 1.092 | 1.294 | 1.583 | 1.636 | **1.711** |
| Public-Tri / DS (RealSpec, strat=False) | Cap=2, c=1.5 | 3.136 | **3.303** | 3.088 | 2.806 | 2.957 | 2.961 |
| Public-Tri / DS (RealSpec, strat=False) | Cap=1/3, c=1.5 | 1.236 | 1.170 | 1.183 | 1.505 | 1.651 | **1.671** |
| v3 / Qwen3.5 | Cap=2, c=1.5 | 4.364 | 4.422 | 4.422 | 4.501 | 4.470 | **4.470** |
| v3 / Qwen3.5 | Cap=1/3, c=1.5 | **3.458** | 2.970 | 2.970 | 3.891 | 3.334 | 3.334 |
| v3 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.558 | 4.525 | 4.525 | 4.567 | 4.582 | **4.582** |
| v3 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 2.695 | 2.846 | 2.846 | 3.499 | 3.484 | **3.484** |
| MediQ-Holdout / Q3.5->RealDS | Cap=2, c=1.5 | 1.676 | **2.216** | 1.550 | 1.133 | 1.538 | 2.072 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3, c=1.5 | 0.434 | 1.031 | 0.892 | 0.494 | 1.003 | **1.348** |

### 9B. Cross-Call `Baseline+MDP` vs. `Tri-VoI-MDP` (Note: 5C `Base+MDP` vs 2C `Tri-VoI-MDP`; Equal-Budget 2C Comparison in 9E)
| Suite | Regime | `CC+N` | `CC+MDP` | `JG+N` | `JG+MDP` | `ACL+FD+N` | `ACL+FD+MDP` | `MDP (Oracle Diag)` | `Tri-VoI-MDP` (Zero-Oracle) | EVOI Net (Zero-Oracle vs Best 5C+MDP) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 4.867 | 4.878 | 5.190 | 5.166 | 4.867 | 4.878 | 5.341 | **5.343** | +0.177 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 3.798 | **4.064** | 3.188 | 3.616 | 3.798 | **4.064** | 4.048 | 4.048 | -0.016 |
| v1-Noisy / Qwen3.5 | Cap=2, c=1.5 | 3.413 | 3.388 | **3.887** | 3.708 | 3.413 | 3.388 | 3.811 | 3.696 | -0.012 |
| v1-Noisy / Qwen3.5 | Cap=1/3, c=1.5 | 1.614 | 2.309 | 1.295 | 2.074 | 1.614 | 2.309 | 2.398 | **2.319** | +0.010 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | **4.996** | 4.977 | 4.421 | 4.341 | 4.993 | 4.974 | 4.690 | 4.699 | -0.278 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 3.808 | **4.056** | 3.169 | 3.487 | 3.805 | 4.043 | 4.041 | 4.041 | -0.015 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.754 | 4.754 | 4.850 | 4.808 | 4.754 | 4.754 | 5.135 | **5.138** | +0.330 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 4.509 | **4.572** | 4.014 | 4.246 | 4.509 | **4.572** | 4.406 | 4.492 | -0.080 |
| v2 / Hy4-preview | Cap=2, c=1.5 | **6.271** | **6.271** | 5.408 | 5.412 | 5.986 | 6.007 | 6.085 | 6.085 | -0.186 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 5.512 | **5.804** | 4.711 | 5.081 | 5.021 | 5.468 | 5.551 | 5.551 | -0.253 |
| Public-Tri / Qwen3.5 (strat=False) | Cap=2, c=1.5 | 3.094 | 3.124 | **3.536** | 3.483 | 3.186 | 3.184 | 3.445 | 3.446 | -0.037 |
| Public-Tri / Qwen3.5 (strat=False) | Cap=1/3, c=1.5 | 1.371 | 1.649 | 1.648 | **2.012** | 1.128 | 1.594 | 1.720 | 1.718 | -0.294 |
| Public-Tri / Qwen3.5 (strat=True) | Cap=2, c=1.5 | 3.067 | 3.054 | **3.723** | 3.662 | 3.105 | 3.065 | 3.521 | 3.544 | -0.118 |
| Public-Tri / Qwen3.5 (strat=True) | Cap=1/3, c=1.5 | 1.347 | 1.708 | 1.450 | **1.826** | 1.251 | 1.611 | 1.636 | 1.711 | -0.115 |
| Public-Tri / DS (RealSpec, strat=False) | Cap=2, c=1.5 | 2.398 | 2.436 | **3.129** | 2.917 | 2.579 | 2.550 | 2.957 | 2.961 | +0.044 |
| Public-Tri / DS (RealSpec, strat=False) | Cap=1/3, c=1.5 | 1.298 | 1.422 | 1.043 | **1.867** | 1.028 | 1.545 | 1.651 | 1.671 | -0.196 |
| v3 / Qwen3.5 | Cap=2, c=1.5 | 4.599 | **4.619** | 4.416 | 4.478 | 4.599 | **4.619** | 4.470 | 4.470 | -0.149 |
| v3 / Qwen3.5 | Cap=1/3, c=1.5 | 3.458 | 3.907 | 3.417 | **4.117** | 3.458 | 3.907 | 3.334 | 3.334 | -0.783 |
| v3 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.528 | **4.626** | 4.204 | 4.178 | 4.411 | 4.487 | 4.582 | 4.582 | -0.044 |
| v3 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 2.695 | 3.499 | 3.469 | **3.682** | 2.612 | 3.385 | 3.484 | 3.484 | -0.198 |
| MediQ-Holdout / Q3.5->RealDS | Cap=2, c=1.5 | 1.718 | 1.718 | 1.244 | 1.096 | 1.530 | 1.352 | 1.538 | **2.072** | +0.354 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3, c=1.5 | 0.882 | 1.072 | 0.191 | 0.162 | 0.222 | 0.420 | 1.003 | **1.348** | +0.276 |

### 9C. Deployable LLM `post_missing` vs. Simulator `not oracle_revealed` (`obs_missing=True` Test Items)
- **Zero-Oracle Audit**: Across `4974` cached `post_ask_combined*` calls, `5` lacked `missing_user_info` (truncated JSON). Primary `Tri-VoI-MDP` (`Deploy-Self`) falls back strictly to `obs_missing` and never reads `oracle_revealed`.

| Suite | `obs_missing` N | True NoInfo | Pred `post_missing` | Unparsed (`obs_missing` Fallback) | Agreement% | Precision | Recall | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | 143 | 6 | 0 | 0 | 95.80% | 0.000 | 0.000 | 0.000 |
| v1-Noisy / Qwen3.5 | 166 | 32 | 24 | 0 | 95.18% | 1.000 | 0.750 | 0.857 |
| v2 / Qwen3.5 | 71 | 1 | 0 | 0 | 98.59% | 0.000 | 0.000 | 0.000 |
| v2 / DeepSeek-V3.2 | 94 | 8 | 0 | 0 | 91.49% | 0.000 | 0.000 | 0.000 |
| v2 / Hy4-preview | 90 | 0 | 0 | 0 | 100.00% | 0.000 | 0.000 | 0.000 |
| Public-Tri / Qwen3.5 (strat=False) | 116 | 10 | 18 | 3 | 82.76% | 0.222 | 0.400 | 0.286 |
| v3 / Qwen3.5 | 63 | 1 | 0 | 0 | 98.41% | 0.000 | 0.000 | 0.000 |
| v3 / DeepSeek-V3.2 | 72 | 0 | 0 | 0 | 100.00% | 0.000 | 0.000 | 0.000 |
| MediQ-Holdout / Q3.5->RealDS | 64 | 16 | 14 | 0 | 65.62% | 0.286 | 0.250 | 0.267 |

### 9D. `Benchmark v3` Strict Zero-Overlap Slices (`Zero-Cal-Prose-Overlap (72)` & `Unique-Prose (40)`)
| Model | Slice | Regime | `ACL+FD+MDP (2C-S)` | `JG+MDP (2C-S)` | `CC+MDP (2C-S)` | `CC+MDP (5C)` | `JG+MDP (5C)` | `Tri-VoI-MDP` (Zero-Oracle) | p (Zero-Oracle vs Best 2C-S) |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | Full (144) | Cap=2, c=1.5 | 4.607 | 4.398 | 4.607 | **4.619** | 4.478 | 4.470 | 0.7341 |
| Qwen3.5-35B-A3B | Full (144) | Cap=1/3, c=1.5 | 3.334 | 3.528 | 3.334 | 3.907 | **4.117** | 3.334 | 0.3136 |
| Qwen3.5-35B-A3B | Zero-Cal-Prose-Overlap (72) | Cap=2, c=1.5 | 5.404 | 5.123 | 5.404 | 5.404 | 5.119 | **5.559** | 0.5801 |
| Qwen3.5-35B-A3B | Zero-Cal-Prose-Overlap (72) | Cap=1/3, c=1.5 | 3.825 | 3.761 | 3.825 | 4.023 | **4.167** | 3.937 | 0.5718 |
| Qwen3.5-35B-A3B | Zero-Overlap-Unique-Prose (40) | Cap=2, c=1.5 | 4.935 | 4.644 | 4.935 | **4.949** | 4.659 | 4.897 | 0.8177 |
| Qwen3.5-35B-A3B | Zero-Overlap-Unique-Prose (40) | Cap=1/3, c=1.5 | 2.751 | 2.941 | 2.751 | 3.023 | **3.192** | 2.751 | 0.6828 |
| DeepSeek-V3.2 | Full (144) | Cap=2, c=1.5 | 4.458 | 4.283 | 4.591 | **4.626** | 4.178 | 4.582 | 0.9810 |
| DeepSeek-V3.2 | Full (144) | Cap=1/3, c=1.5 | 3.265 | **3.948** | 3.484 | 3.499 | 3.682 | 3.484 | 0.2636 |
| DeepSeek-V3.2 | Zero-Cal-Prose-Overlap (72) | Cap=2, c=1.5 | 5.114 | 4.833 | **5.373** | 5.327 | 4.950 | 5.222 | 0.6708 |
| DeepSeek-V3.2 | Zero-Cal-Prose-Overlap (72) | Cap=1/3, c=1.5 | 3.127 | 3.257 | **3.261** | 2.711 | 2.480 | **3.261** | 1.0000 |
| DeepSeek-V3.2 | Zero-Overlap-Unique-Prose (40) | Cap=2, c=1.5 | 4.473 | 4.535 | 4.830 | **4.841** | 4.467 | 4.793 | 0.8820 |
| DeepSeek-V3.2 | Zero-Overlap-Unique-Prose (40) | Cap=1/3, c=1.5 | 1.860 | **3.016** | 2.197 | 1.664 | 2.849 | 2.197 | 0.1546 |

### 9E. Strictly Matched 2-Call Zero-Oracle Attribution (Equal Probe Cost, Zero Oracle & Same DP Pricing)
| Suite | Regime | `CC+MDP (2C-S)` | `CC+OwnDP` | `JG+MDP (2C-S)` | `JG+OwnDP` | `ACL+FD+MDP (2C-S)` | `Tri-VoI-MDP` (Zero-Oracle) | Net vs Shared (`p`) | Net vs OwnDP (`p`) | Tok (`CC/JG/MDP`) |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 4.933 | 4.921 | 5.217 | 5.205 | 4.933 | **5.343** | +0.126 (`0.5105`) | +0.138 (`0.4798`) | 2250/2025/2111 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | **4.121** | 4.097 | 3.588 | 3.533 | **4.121** | 4.048 | -0.073 (`0.4668`) | -0.073 (`0.4668`) | 2250/2216/2245 |
| v1-Noisy / Qwen3.5 | Cap=2, c=1.5 | 3.558 | 3.588 | 3.654 | **3.735** | 3.558 | 3.696 | +0.042 (`0.8624`) | -0.039 (`0.8697`) | 6777/6777/6777 |
| v1-Noisy / Qwen3.5 | Cap=1/3, c=1.5 | **2.324** | 2.306 | 2.092 | 1.976 | **2.324** | 2.319 | -0.005 (`0.8587`) | -0.005 (`0.8587`) | 6777/6777/6777 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.993 | 4.928 | 4.310 | 4.310 | **5.010** | 4.699 | -0.311 (`0.2036`) | -0.311 (`0.2036`) | 2249/2020/2097 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 4.003 | 4.003 | 3.491 | 3.558 | 3.992 | **4.041** | +0.038 (`0.8251`) | +0.038 (`0.8251`) | 2249/2198/2244 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.754 | 4.754 | 4.898 | 4.898 | 4.754 | **5.138** | +0.240 (`0.4745`) | +0.240 (`0.4745`) | 2137/1933/2014 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 4.483 | 4.483 | 4.095 | 4.120 | 4.483 | **4.492** | +0.009 (`0.6161`) | +0.009 (`0.6161`) | 2137/2074/2132 |
| v2 / Hy4-preview | Cap=2, c=1.5 | **6.271** | **6.271** | 5.451 | 5.451 | 5.986 | 6.085 | -0.186 (`0.4695`) | -0.186 (`0.4695`) | 2181/1958/2076 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | **5.554** | **5.554** | 4.984 | 5.185 | 5.386 | 5.551 | -0.003 (`0.9637`) | -0.003 (`0.9637`) | 2181/1965/2140 |
| Public-Tri / Qwen3.5 (strat=False) | Cap=2, c=1.5 | 3.182 | 3.190 | **3.513** | **3.513** | 3.256 | 3.446 | -0.067 (`0.5718`) | -0.067 (`0.5718`) | 1959/1564/1588 |
| Public-Tri / Qwen3.5 (strat=False) | Cap=1/3, c=1.5 | 1.713 | 1.721 | **2.121** | 2.068 | 1.690 | 1.718 | -0.403 (`0.2156`) | -0.350 (`0.3162`) | 1959/1880/1927 |
| Public-Tri / Qwen3.5 (strat=True) | Cap=2, c=1.5 | 3.117 | 3.104 | **3.644** | **3.644** | 3.193 | 3.544 | -0.100 (`0.4502`) | -0.100 (`0.4502`) | 1959/1576/1593 |
| Public-Tri / Qwen3.5 (strat=True) | Cap=1/3, c=1.5 | 1.718 | 1.680 | 1.913 | **1.951** | 1.691 | 1.711 | -0.202 (`0.4122`) | -0.240 (`0.3642`) | 1959/1872/1917 |
| Public-Tri / DS (RealSpec, strat=False) | Cap=2, c=1.5 | 2.467 | 2.499 | **3.092** | **3.092** | 2.663 | 2.961 | -0.131 (`0.4608`) | -0.131 (`0.4608`) | 1959/1502/1548 |
| Public-Tri / DS (RealSpec, strat=False) | Cap=1/3, c=1.5 | 1.630 | 1.572 | **1.798** | **1.798** | 1.608 | 1.671 | -0.127 (`0.5748`) | -0.127 (`0.5748`) | 1959/1875/1921 |
| v3 / Qwen3.5 | Cap=2, c=1.5 | **4.607** | **4.607** | 4.398 | 4.398 | **4.607** | 4.470 | -0.137 (`0.7341`) | -0.137 (`0.7341`) | 2187/1983/2091 |
| v3 / Qwen3.5 | Cap=1/3, c=1.5 | 3.334 | 3.334 | **3.528** | 3.496 | 3.334 | 3.334 | -0.194 (`0.3136`) | -0.162 (`0.3042`) | 2187/2172/2187 |
| v3 / DeepSeek-V3.2 | Cap=2, c=1.5 | **4.591** | **4.591** | 4.283 | 4.283 | 4.458 | 4.582 | -0.009 (`0.9810`) | -0.009 (`0.9810`) | 2139/1915/2036 |
| v3 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 3.484 | 3.484 | 3.948 | **4.066** | 3.265 | 3.484 | -0.464 (`0.2636`) | -0.582 (`0.2319`) | 2139/2104/2139 |
| MediQ-Holdout / Q3.5->RealDS | Cap=2, c=1.5 | 1.907 | 1.907 | 1.900 | 1.900 | 1.589 | **2.072** | +0.165 (`0.2799`) | +0.165 (`0.2799`) | 1969/1721/1894 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3, c=1.5 | 1.320 | **1.393** | 0.761 | 0.902 | 0.584 | 1.348 | +0.028 (`0.9220`) | -0.045 (`0.9017`) | 1969/1765/1840 |

### 9F. Clarification-Cost `c_ask` Regime Map Under Matched 2-Call Zero-Oracle Pricing (Proposition 5)
| Suite | Cap | `c_ask` | `CC+MDP (2C-S)` | `JG+MDP (2C-S)` | `ACL+FD+MDP (2C-S)` | `Tri-VoI-MDP` (Zero-Oracle) | EVOI Net vs Best 2C-S |
|:---|:---|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2 | 0.5 | 5.330 | 5.285 | 5.330 | **5.401** | +0.071 |
| v1-Clean / Qwen3.5 | Cap=2 | 1.5 | 4.933 | 5.217 | 4.933 | **5.343** | +0.126 |
| v1-Clean / Qwen3.5 | Cap=2 | 3.0 | 4.337 | **5.151** | 4.871 | 5.107 | -0.044 |
| v1-Clean / Qwen3.5 | Cap=2 | 4.5 | 3.741 | **4.908** | 4.588 | 4.893 | -0.015 |
| v1-Clean / Qwen3.5 | Cap=1/3 | 0.5 | **4.494** | 3.948 | **4.494** | **4.494** | +0.000 |
| v1-Clean / Qwen3.5 | Cap=1/3 | 1.5 | **4.121** | 3.588 | **4.121** | 4.048 | -0.073 |
| v1-Clean / Qwen3.5 | Cap=1/3 | 3.0 | 3.549 | **3.856** | **3.856** | 3.451 | -0.405 |
| v1-Clean / Qwen3.5 | Cap=1/3 | 4.5 | 2.971 | 3.534 | **3.597** | 3.257 | -0.340 |
| v1-Noisy / Qwen3.5 | Cap=2 | 0.5 | **4.049** | 4.020 | **4.049** | **4.049** | +0.000 |
| v1-Noisy / Qwen3.5 | Cap=2 | 1.5 | 3.558 | 3.654 | 3.558 | **3.696** | +0.042 |
| v1-Noisy / Qwen3.5 | Cap=2 | 3.0 | 2.902 | **3.541** | 3.485 | 3.409 | -0.132 |
| v1-Noisy / Qwen3.5 | Cap=2 | 4.5 | 2.211 | **3.541** | 3.439 | 3.444 | -0.097 |
| v1-Noisy / Qwen3.5 | Cap=1/3 | 0.5 | **2.767** | 2.554 | **2.767** | **2.767** | +0.000 |
| v1-Noisy / Qwen3.5 | Cap=1/3 | 1.5 | **2.324** | 2.092 | **2.324** | 2.319 | -0.005 |
| v1-Noisy / Qwen3.5 | Cap=1/3 | 3.0 | **1.669** | 1.506 | 1.472 | 1.662 | -0.007 |
| v1-Noisy / Qwen3.5 | Cap=1/3 | 4.5 | 0.730 | 1.326 | 1.326 | **1.383** | +0.057 |
| v2 / DeepSeek-V3.2 | Cap=2 | 0.5 | 5.276 | 5.310 | 5.276 | **5.420** | +0.110 |
| v2 / DeepSeek-V3.2 | Cap=2 | 1.5 | 4.754 | 4.898 | 4.754 | **5.138** | +0.240 |
| v2 / DeepSeek-V3.2 | Cap=2 | 3.0 | 3.970 | **4.481** | 4.109 | 4.343 | -0.138 |
| v2 / DeepSeek-V3.2 | Cap=2 | 4.5 | 3.187 | **4.285** | 3.567 | 4.235 | -0.050 |
| v2 / DeepSeek-V3.2 | Cap=1/3 | 0.5 | **5.006** | 4.650 | **5.006** | **5.006** | +0.000 |
| v2 / DeepSeek-V3.2 | Cap=1/3 | 1.5 | 4.483 | 4.095 | 4.483 | **4.492** | +0.009 |
| v2 / DeepSeek-V3.2 | Cap=1/3 | 3.0 | 3.654 | **4.089** | 3.515 | 3.563 | -0.526 |
| v2 / DeepSeek-V3.2 | Cap=1/3 | 4.5 | 2.838 | **3.738** | 2.908 | 3.472 | -0.266 |
| v3 / DeepSeek-V3.2 | Cap=2 | 0.5 | **5.091** | 4.605 | 4.930 | **5.091** | +0.000 |
| v3 / DeepSeek-V3.2 | Cap=2 | 1.5 | **4.591** | 4.283 | 4.458 | 4.582 | -0.009 |
| v3 / DeepSeek-V3.2 | Cap=2 | 3.0 | 3.841 | 4.033 | 3.750 | **4.044** | +0.011 |
| v3 / DeepSeek-V3.2 | Cap=2 | 4.5 | 3.118 | **3.726** | 3.104 | 3.590 | -0.136 |
| v3 / DeepSeek-V3.2 | Cap=1/3 | 0.5 | 3.984 | **4.393** | 3.737 | 3.984 | -0.409 |
| v3 / DeepSeek-V3.2 | Cap=1/3 | 1.5 | 3.484 | **3.948** | 3.265 | 3.484 | -0.464 |
| v3 / DeepSeek-V3.2 | Cap=1/3 | 3.0 | 2.734 | **3.126** | 2.556 | 2.734 | -0.392 |
| v3 / DeepSeek-V3.2 | Cap=1/3 | 4.5 | 2.095 | **2.521** | 2.175 | 2.073 | -0.448 |
| MediQ-Holdout / Q3.5->RealDS | Cap=2 | 0.5 | 2.407 | 2.084 | 2.088 | **2.411** | +0.004 |
| MediQ-Holdout / Q3.5->RealDS | Cap=2 | 1.5 | 1.907 | 1.900 | 1.589 | **2.072** | +0.165 |
| MediQ-Holdout / Q3.5->RealDS | Cap=2 | 3.0 | 1.173 | **1.384** | 1.011 | 1.277 | -0.107 |
| MediQ-Holdout / Q3.5->RealDS | Cap=2 | 4.5 | 0.423 | **1.345** | 0.873 | **1.345** | +0.000 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3 | 0.5 | **1.893** | 0.966 | 1.315 | 1.882 | -0.011 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3 | 1.5 | 1.320 | 0.761 | 0.584 | **1.348** | +0.028 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3 | 3.0 | 0.401 | 0.244 | 0.147 | **0.528** | +0.127 |
| MediQ-Holdout / Q3.5->RealDS | Cap=1/3 | 4.5 | -0.516 | **0.369** | -0.243 | **0.369** | +0.000 |

## Section 10: Pre-Registered Organ-System-Disjoint, Length-Matched `MediQ-Test-Disjoint` (`prereg-mediq-test-disjoint`, `PY_DIFF_LEN=0`, `N_cal=60, N_test=180`, Spend `¥3.55`)
> Constructed from official `medqa_test_convo.jsonl` (0 dev-split overlap, 0 clinical organ-system overlap between Cal `{Cardio_Pulm, Endo_Renal, Other}` and Test `{Heme_Onc_Repro_MSK, Infectious_Immuno, GI_Hepatic, Neuro_Psych}`, Q1 vs Q2 prose gap `4.86%`, `Unparsed Fallback=0` on both models).

### 10A. Setup A: Cross-Group Real-LLM Specialist (`Qwen3.5-35B-A3B` -> `DeepSeek-V3.2`)
| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 | Tok (Cap=2, c=1.5) |
|:---|---:|---:|---:|---:|---:|
| `Direct-ACT` | 0.667 (77.8%/0.0%) | 0.667 | 0.667 (77.8%/0.0%) | 0.667 | 806 |
| `KL-PBD+ (Warm)` | 4.260 (81.1%/92.5%) | **4.260** | 3.196 (80.0%/59.7%) | 3.196 | 4118 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.997 (81.1%/89.5%) | 3.676 | 2.594 (79.4%/73.1%) | 2.583 | 2206 |
| `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` | 3.874 (80.6%/83.6%) | 3.559 | 3.537 (81.7%/71.6%) | 3.181 | 2206 |
| `Joint-Greedy+Naor` | 4.216 (81.7%/89.5%) | 4.112 | 2.681 (80.0%/73.1%) | 2.142 | 4283 |
| `Joint-Greedy+MDP (2C-Self)` | 4.175 (81.1%/88.1%) | 4.020 | 3.407 (80.6%/71.6%) | **3.342** | 1752 |
| `Joint-Greedy+OwnDP (2C-Self)` | 4.178 (81.1%/88.1%) | 4.020 | 3.560 (81.1%/73.1%) | 3.324 | 1757 |
| `Calibrated-Cascade+Naor` | 3.991 (81.1%/88.1%) | 3.541 | 2.909 (80.6%/77.6%) | 2.459 | 4391 |
| `Calibrated-Cascade+MDP (2C-Self)` | 3.884 (80.6%/80.6%) | 3.434 | 3.471 (81.7%/71.6%) | 3.021 | 1906 |
| `Calibrated-Cascade+OwnDP (2C-Self)` | 3.884 (80.6%/80.6%) | 3.434 | 3.471 (81.7%/71.6%) | 3.021 | 1906 |
| `Tri-VoI-L2D` | 3.862 (80.6%/88.1%) | 3.764 | 2.909 (80.6%/77.6%) | 2.459 | 4354 |
| `MDP (Oracle-NoInfo Diag)` | 3.969 (81.1%/85.1%) | 3.923 | 3.356 (81.7%/71.6%) | 2.906 | 1853 |
| `Tri-VoI-MDP (Zero-Oracle)` | **4.266** (81.7%/85.1%) | 4.091 | **3.588** (82.2%/71.6%) | 3.244 | 1823 |

- **Cap=2, c=1.5**: `Tri-VoI-MDP (Zero-Oracle)` vs `Joint-Greedy+MDP (2C-Self)` = `+0.091` (`p=0.6851`), vs `Tri-VoI-L2D` = `+0.404` (`p=0.3236`).
- **Cap=2, c=3.0**: `Tri-VoI-MDP (Zero-Oracle)` vs `Joint-Greedy+MDP (2C-Self)` = `+0.071` (`p=0.5985`), vs `Tri-VoI-L2D` = `+0.327` (`p=0.3812`).
- **Cap=1/3, c=1.5**: `Tri-VoI-MDP (Zero-Oracle)` vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `+0.051` (`p=0.8221`), vs `Tri-VoI-L2D` = `+0.679` (`p=0.0137`).
- **Cap=1/3, c=3.0**: `Tri-VoI-MDP (Zero-Oracle)` vs `Joint-Greedy+MDP (2C-Self)` = `-0.098` (`p=0.7124`), vs `Tri-VoI-L2D` = `+0.785` (`p=0.0057`).

### 10B. Setup B: Cross-Group Generalist (`DeepSeek-V3.2`)
| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 | Tok (Cap=2, c=1.5) |
|:---|---:|---:|---:|---:|---:|
| `Direct-ACT` | -0.111 (76.1%/0.0%) | -0.111 | -0.111 (76.1%/0.0%) | -0.111 | 659 |
| `KL-PBD+ (Warm)` | 3.901 (79.0%/97.3%) | 3.901 | 2.547 (77.0%/57.9%) | **2.547** | 3431 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.869 (82.4%/92.5%) | 3.429 | 2.519 (83.4%/71.3%) | 2.123 | 2105 |
| `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` | 3.521 (82.7%/88.7%) | 3.033 | **2.867** (83.3%/70.5%) | 2.477 | 2105 |
| `Joint-Greedy+Naor` | 3.978 (81.9%/90.7%) | **3.922** | 2.349 (82.8%/66.0%) | 1.675 | 3528 |
| `Joint-Greedy+MDP (2C-Self)` | 3.588 (81.7%/86.9%) | 3.620 | 2.629 (81.4%/67.8%) | 2.324 | 1542 |
| `Joint-Greedy+OwnDP (2C-Self)` | 3.588 (81.7%/86.9%) | 3.620 | 2.834 (81.2%/69.8%) | 2.286 | 1542 |
| `Calibrated-Cascade+Naor` | 3.784 (82.3%/90.4%) | 3.193 | 2.409 (82.7%/71.9%) | 1.817 | 3757 |
| `Calibrated-Cascade+MDP (2C-Self)` | 3.216 (81.8%/85.7%) | 2.624 | 2.689 (83.3%/70.5%) | 2.078 | 1761 |
| `Calibrated-Cascade+OwnDP (2C-Self)` | 3.272 (82.0%/86.0%) | 2.680 | 2.661 (83.2%/70.2%) | 2.070 | 1761 |
| `Tri-VoI-L2D` | **4.176** (82.6%/91.6%) | 3.782 | 2.428 (82.8%/71.0%) | 1.811 | 3545 |
| `MDP (Oracle-NoInfo Diag)` | 3.669 (82.3%/86.0%) | 3.560 | 2.866 (82.6%/67.8%) | 2.554 | 1554 |
| `Tri-VoI-MDP (Zero-Oracle)` | 3.721 (82.1%/87.5%) | 3.576 | 2.706 (83.3%/70.5%) | 2.322 | 1555 |

- **Cap=2, c=1.5**: `Tri-VoI-MDP (Zero-Oracle)` vs `Joint-Greedy+MDP (2C-Self)` = `+0.133` (`p=0.4768`), vs `Tri-VoI-L2D` = `-0.455` (`p=0.0816`).
- **Cap=2, c=3.0**: `Tri-VoI-MDP (Zero-Oracle)` vs `Joint-Greedy+MDP (2C-Self)` = `-0.044` (`p=0.7574`), vs `Tri-VoI-L2D` = `-0.206` (`p=0.5162`).
- **Cap=1/3, c=1.5**: `Tri-VoI-MDP (Zero-Oracle)` vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `-0.161` (`p=0.1173`), vs `Tri-VoI-L2D` = `+0.278` (`p=0.2756`).
- **Cap=1/3, c=3.0**: `Tri-VoI-MDP (Zero-Oracle)` vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `-0.155` (`p=0.5681`), vs `Tri-VoI-L2D` = `+0.511` (`p=0.1636`).

