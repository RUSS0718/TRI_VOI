# IJCAI 2027 Canonical Tables (Audit-Verified Single Source of Truth)

Generated automatically by `generate_ijcai_tables.py` using canonical terminology (`Requester`, `Specialist`, `Benchmark v1-Clean`, `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri`). **Bolding strictly marks the true row/column best non-Oracle metric; losing entries are never bolded.**

## Table 1: `Benchmark v1-Clean` & `Benchmark v1-Noisy` Main Comparison (`Qwen/Qwen3.5-35B-A3B`, `N=360`)
| Method | v1-Clean (Cap=2, c=1.5) | v1-Clean (Cap=2, c=2.5) | v1-Clean (Cap=1/3, c=1.5) | v1-Noisy (Cap=2, c=1.5) | v1-Noisy (Cap=2, c=2.5) | v1-Noisy (Cap=1/3, c=1.5) |
|:---|---:|---:|---:|---:|---:|---:|
| `Direct-ACT` | 2.000 | 2.000 | 2.000 | 0.556 | 0.556 | 0.556 |
| `KL-PBD` | 4.061 | 4.061 | 2.621 | 3.389 | 3.389 | 1.268 |
| `KL-PBD+ (Warm)` | 4.450 | 4.450 | 3.106 | 3.665 | **3.665** | 1.517 |
| `ACL-2026-VoI (Calib)` | 2.182 | 2.333 | 2.182 | 1.142 | 0.715 | 1.142 |
| `ACL-2026-VoI+Post-Defer` | 3.633 | 2.800 | 3.609 | 2.081 | 1.071 | **2.031** |
| `Joint-Greedy+Naor` | 5.190 | 5.015 | 3.188 | **3.887** | 3.618 | 1.295 |
| `Calibrated-Cascade+Naor` | 4.867 | 4.470 | **3.798** | 3.413 | 2.952 | 1.614 |
| `Tri-VoI-L2D` | **5.220** | **5.142** | 3.792 | 3.503 | 3.511 | 1.525 |

## Table 2: Cross-Benchmark Generalization on `Benchmark v2 (Implicit-Override)` & `Public-Tri (MediQ Scope-of-Practice Triage)`
> **Transparency Note (`Public-Tri`)**: `Public-Tri` models **Scope-of-Practice / Regulatory Governance Triage** (`require_defer_for_ok=True` on high-acuity/multi-specialty `Q3`/`Q4_escalate`), NOT raw medical-QA incompetence — LLMs achieve **71.7%–83.3%** raw USMLE accuracy on `Q3/Q4_escalate` when evaluated without escalation policy. Also note `v2` reports `Tri-VoI` vs `KL-PBD+ (Warm)`, `ACL-2026-VoI (Live)`, `JG+Naor`, and `CC+Naor` (bold marks row-wise best at `Cap=2, c=1.5`).

| Suite & Model | Miss-F1 / Collapse% | `KL-PBD+ (Warm)` | `ACL-2026-VoI (Live)` | `Joint-Greedy+Naor` | `Calib-Cascade+Naor` | `Tri-VoI-L2D` (Cap=2, c=1.5) | `Tri-VoI-L2D` (Cap=1/3) |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| v2 / `Qwen3.5-35B-A3B` | 0.870 / 78.9% | 3.366 | 2.203 | 4.421 | **4.996** | 4.657 | 3.847 |
| v2 / `DeepSeek-V3.2` | 0.935 / 86.7% | 3.620 | 2.550 | 4.850 | 4.754 | **5.031** | 4.334 |
| v2 / `Hy4-preview` | 1.000 / 91.1% | 3.426 | 3.006 | 5.408 | **6.271** | 6.053 | 5.313 |
| v2 / `Qwen3-8B` | 0.178 / 87.8% | **3.600** | 0.131 | 3.551 | 3.508 | 3.571 | 1.163 |
| Public-Tri / `Qwen3.5-35B-A3B` | 0.932 / 81.7% | 3.533 | -0.400 | **3.723** | 3.067 | 3.612 (c\_def=4: 1.999) | 1.259 |
| Public-Tri / `DeepSeek-V3.2` | 0.831 / 72.5% | **3.443** | -0.792 | 3.331 | 2.769 | 3.327 (c\_def=4: 1.634) | 0.417 |
| Public-Tri / `Qwen3-8B` | 0.802 / 91.7% | **3.133** | -3.402 | 2.809 | 2.013 | 2.809 (c\_def=4: 0.937) | -1.611 |

## Table 3: 56-Cell 2D Minimax Regret (Descriptive Sensitivity Envelope — No Pseudoreplication $p$-values)
| Suite | Method | Mean Util (56) | $\Delta$ vs Tri | Worst-Case Regret | Tri W / T / L |
|:---|:---|---:|---:|---:|:---:|
| v1-Clean (Qwen3.5) | `Tri-VoI-L2D` | **4.470** | — | **0.572** | — |
| v1-Clean (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.095 | -0.375 | 1.443 | 39/3/14 |
| v1-Clean (Qwen3.5) | `Joint-Greedy+Naor` | 4.332 | -0.137 | 0.983 | 44/0/12 |
| v1-Clean (Qwen3.5) | `KL-PBD+ (Warm)` | 4.049 | -0.420 | 1.179 | 47/1/8 |
| v1-Clean (Qwen3.5) | `ACL-2026-VoI+Full-Defer+Naor` | 4.277 | -0.193 | 0.741 | 33/3/20 |
| v1-Clean (Qwen3.5) | `w/o Queue-Externality` | 4.119 | -0.350 | 1.727 | 33/16/7 |
| v2 (DeepSeek-V3.2) | `Tri-VoI-L2D` | **4.258** | — | 1.021 | — |
| v2 (DeepSeek-V3.2) | `Calibrated-Cascade+Naor` | 4.117 | -0.141 | 0.983 | 39/3/14 |
| v2 (DeepSeek-V3.2) | `Joint-Greedy+Naor` | 4.174 | -0.084 | 0.826 | 37/0/19 |
| v2 (DeepSeek-V3.2) | `KL-PBD+ (Warm)` | 3.301 | -0.957 | 2.368 | 48/0/8 |
| v2 (DeepSeek-V3.2) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.193 | -0.065 | **0.568** | 37/3/16 |
| v2 (DeepSeek-V3.2) | `w/o Queue-Externality` | 4.001 | -0.256 | 2.024 | 32/18/6 |
| v2 (Qwen3.5) | `Tri-VoI-L2D` | 4.124 | — | 0.511 | — |
| v2 (Qwen3.5) | `Calibrated-Cascade+Naor` | 4.232 | +0.108 | 0.536 | 13/4/39 |
| v2 (Qwen3.5) | `Joint-Greedy+Naor` | 3.903 | -0.221 | 0.954 | 40/0/16 |
| v2 (Qwen3.5) | `KL-PBD+ (Warm)` | 3.144 | -0.980 | 2.086 | 55/0/1 |
| v2 (Qwen3.5) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | **4.289** | +0.166 | **0.228** | 14/0/42 |
| v2 (Qwen3.5) | `w/o Queue-Externality` | 3.760 | -0.363 | 1.461 | 37/16/3 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Tri-VoI-L2D` | 2.331 | — | 0.780 | — |
| Public-Tri (Qwen3.5, c_def=2.0) | `Calibrated-Cascade+Naor` | 1.852 | -0.479 | 2.185 | 46/0/10 |
| Public-Tri (Qwen3.5, c_def=2.0) | `Joint-Greedy+Naor` | **2.488** | +0.157 | **0.566** | 4/0/52 |
| Public-Tri (Qwen3.5, c_def=2.0) | `KL-PBD+ (Warm)` | 2.189 | -0.142 | 1.456 | 32/0/24 |
| Public-Tri (Qwen3.5, c_def=2.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.956 | -0.375 | 1.442 | 50/0/6 |
| Public-Tri (Qwen3.5, c_def=2.0) | `w/o Queue-Externality` | 1.721 | -0.610 | 2.032 | 33/16/7 |
| Public-Tri (Qwen3.5, c_def=4.0) | `Tri-VoI-L2D` | 1.283 | — | 0.579 | — |
| Public-Tri (Qwen3.5, c_def=4.0) | `Calibrated-Cascade+Naor` | 0.859 | -0.425 | 1.885 | 45/0/11 |
| Public-Tri (Qwen3.5, c_def=4.0) | `Joint-Greedy+Naor` | **1.472** | +0.189 | **0.411** | 2/0/54 |
| Public-Tri (Qwen3.5, c_def=4.0) | `KL-PBD+ (Warm)` | 0.977 | -0.306 | 1.792 | 41/0/15 |
| Public-Tri (Qwen3.5, c_def=4.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.948 | -0.335 | 1.250 | 48/0/8 |
| Public-Tri (Qwen3.5, c_def=4.0) | `w/o Queue-Externality` | 0.878 | -0.405 | 1.775 | 37/16/3 |
| Public-Tri (DeepSeek, c_def=4.0) | `Tri-VoI-L2D` | 0.946 | — | **0.559** | — |
| Public-Tri (DeepSeek, c_def=4.0) | `Calibrated-Cascade+Naor` | 0.610 | -0.336 | 2.179 | 35/0/21 |
| Public-Tri (DeepSeek, c_def=4.0) | `Joint-Greedy+Naor` | 0.934 | -0.012 | 0.582 | 33/2/21 |
| Public-Tri (DeepSeek, c_def=4.0) | `KL-PBD+ (Warm)` | **0.985** | +0.039 | 1.111 | 20/0/36 |
| Public-Tri (DeepSeek, c_def=4.0) | `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.776 | -0.170 | 1.217 | 33/0/23 |
| Public-Tri (DeepSeek, c_def=4.0) | `w/o Queue-Externality` | 0.467 | -0.479 | 1.765 | 36/16/4 |

## Section 4: Explicit 2x2 Decomposition (`±EVOI × ±Naor`), Fair `ACL-2026-VoI+Full-Defer+Naor`, and `Smooth-Naor`
| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI` | `Smooth-Naor` | Naor Δ | EVOI vs CC+N |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1-Clean / Qwen3.5 | Cap=2, c=1.5 | 5.142 | 5.190 | 4.877 | 4.867 | 4.867 | **5.274** | 5.220 | 5.249 | -0.054 | +0.353 |
| v1-Clean / Qwen3.5 | Cap=1/3, c=1.5 | 2.212 | 3.188 | 3.152 | **3.798** | **3.798** | 2.732 | 3.792 | 3.792 | +1.060 | -0.006 |
| v2 / Qwen3.5 | Cap=2, c=1.5 | 4.294 | 4.421 | **4.996** | **4.996** | 4.993 | 4.615 | 4.657 | 4.829 | +0.042 | -0.339 |
| v2 / Qwen3.5 | Cap=1/3, c=1.5 | 2.321 | 3.169 | 3.580 | 3.808 | 3.805 | 2.931 | **3.847** | **3.847** | +0.916 | +0.039 |
| v2 / DeepSeek-V3.2 | Cap=2, c=1.5 | 4.841 | 4.850 | 4.754 | 4.754 | 4.754 | 5.020 | **5.031** | 5.028 | +0.011 | +0.277 |
| v2 / DeepSeek-V3.2 | Cap=1/3, c=1.5 | 3.182 | 4.014 | 4.333 | **4.509** | **4.509** | 3.957 | 4.334 | 4.334 | +0.377 | -0.175 |
| v2 / Hy4-preview | Cap=2, c=1.5 | 5.390 | 5.408 | **6.271** | **6.271** | 5.986 | 6.024 | 6.053 | 6.194 | +0.029 | -0.218 |
| v2 / Hy4-preview | Cap=1/3, c=1.5 | 3.860 | 4.711 | 5.378 | **5.512** | 5.021 | 5.252 | 5.313 | 5.313 | +0.061 | -0.199 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=2, c=1.5 | 3.693 | **3.723** | 3.140 | 3.067 | 3.105 | 3.545 | 3.612 | 3.673 | +0.067 | +0.545 |
| Public-Tri / Qwen3.5 (c_def=2) | Cap=1/3, c=1.5 | 0.224 | **1.450** | 0.053 | 1.347 | 1.251 | -0.057 | 1.259 | 1.259 | +1.316 | -0.088 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=2, c=1.5 | 2.229 | **2.237** | 1.626 | 1.684 | 1.797 | 1.966 | 1.999 | 1.935 | +0.033 | +0.315 |
| Public-Tri / Qwen3.5 (c_def=4) | Cap=1/3, c=1.5 | 0.187 | **1.082** | -0.017 | 0.853 | 0.784 | -0.247 | 0.780 | 0.764 | +1.027 | -0.073 |
| Public-Tri / DeepSeek (c_def=2) | Cap=2, c=1.5 | **3.520** | 3.331 | 2.905 | 2.769 | 2.871 | 3.505 | 3.327 | 3.288 | -0.178 | +0.558 |
| Public-Tri / DeepSeek (c_def=2) | Cap=1/3, c=1.5 | -0.908 | 0.444 | -0.444 | 0.752 | **0.784** | -0.755 | 0.417 | 0.417 | +1.172 | -0.335 |

## Section 5: 20 Stratified Random `(cal, test)` Resplits (Calibrator + Stream Uncertainty)
| Suite | Regime | `Tri-VoI` (mean±sd) | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | Tri Wins/20 (KL/ACL/JG/CC) |
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

## Section 6: Exact Measured SQLite Token Cost (`2-Call` vs `Full`) & `KL-PBD+ (SC-Only)`
| Suite | Model | `Tri-VoI` Full | `Tri-VoI` 2-Call | Realized Tok (Full -> 2-Call, % Saved) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |
|:---|:---|---:|---:|---:|---:|---:|
| v1-Clean | Qwen3.5-35B-A3B | 5.220 | **5.416** | 4952 -> 2113 (-57.3%) | 4.450 | 3.767 |
| v1-Clean | DeepSeek-V3.2 | **5.523** | **5.523** | 4441 -> 2052 (-53.8%) | 4.553 | 3.654 |
| v1-Clean | Hy4-preview | **5.842** | **5.842** | 4622 -> 2119 (-54.1%) | 4.570 | 3.645 |
| v2 | Qwen3.5-35B-A3B | 4.657 | **4.750** | 4967 -> 2111 (-57.5%) | 3.366 | 3.221 |
| v2 | DeepSeek-V3.2 | **5.031** | **5.031** | 4373 -> 1995 (-54.4%) | 3.620 | 2.965 |
| v2 | Hy4-preview | **6.053** | **6.053** | 4562 -> 2070 (-54.6%) | 3.426 | 2.788 |

## Section 7: Genuine Full-Quadrant Real-LLM Specialist (`Public-Tri`) vs. Semi-Synthetic Specialist (`v1/v2`)
| Setup | Regime | `KL-PBD+ (Warm)` | `ACL+FD+Naor` | `Calib-Casc+Naor` | `Joint-Greedy+Naor` | `Tri-VoI-L2D` |
|:---|:---|---:|---:|---:|---:|---:|
| Real `Public-Tri`: `Q3.5` -> `DeepSeek` (Stationary) | Cap=2, c_def=2.0, c=1.5 | 3.020 | 2.635 | 2.680 | **3.127** | 3.124 |
| Real `Public-Tri`: `Q3.5` -> `DeepSeek` (Stationary) | Cap=2, c_def=4.0, c=1.5 | 1.471 | **1.963** | 1.785 | 1.355 | 1.477 |
| Real `Public-Tri`: `Q3.5` -> `DeepSeek` (Stationary) | Cap=1/3, c_def=2.0, c=1.5 | 0.279 | 1.276 | 1.367 | **1.420** | 0.960 |
| Real `Public-Tri`: `Q3.5` -> `DeepSeek->Qwen8B` (Shift) | Cap=2, c_def=2.0, c=1.5 | 1.581 | 1.827 | 1.888 | **2.121** | 2.057 |
| Real `Public-Tri`: `Q3.5` -> `DeepSeek->Qwen8B` (Shift) | Cap=2, c_def=4.0, c=1.5 | 0.245 | **1.428** | 1.373 | 1.322 | 1.094 |
| Real `Public-Tri`: `Q3.5` -> `DeepSeek->Qwen8B` (Shift) | Cap=1/3, c_def=2.0, c=1.5 | -0.449 | 0.636 | 0.499 | **0.870** | 0.367 |
| Real `Public-Tri`: `Qwen8B` -> `DeepSeek` (Floor) | Cap=2, c_def=4.0, c=1.5 | **0.773** | -1.172 | -1.269 | 0.147 | 0.234 |
| Real `Public-Tri`: `Qwen8B` -> `DeepSeek` (Floor) | Cap=1/3, c_def=2.0, c=1.5 | **-1.226** | -1.369 | -1.309 | -1.329 | -1.444 |
| Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override | Cap=2, c=1.5 | 5.592 | 5.662 | 5.662 | **6.060** | 5.893 |
| Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override | Cap=1/3, c=1.5 | 3.633 | **4.216** | **4.216** | 3.602 | 4.175 |
| Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override | Cap=2, c=1.5 | 4.652 | 4.789 | 4.772 | **5.044** | 4.795 |
| Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override | Cap=1/3, c=1.5 | 2.563 | **4.062** | 3.906 | 3.525 | 3.908 |

## Section 8: Single-Shot Zero-Touch Disjoint Holdout (`N_cal=48, N_test=128`, Natural Prose) & Unseen-ID `v2`
| Suite / Model (Unified Default Calibrator) | Miss-F1 / Scope-F1 | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `Tri-VoI (2-Call)` | `Smooth-Naor` |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| MediQ-Holdout / `Qwen3.5-35B-A3B` (Cap=2, c=1.5) | 0.797 / 0.583 | 3.632 | 3.529 | **3.843** | 3.453 | 3.690 | 3.497 | 3.698 |
| MediQ-Holdout / `Qwen3.5-35B-A3B` (Cap=2, c_def=4.0, c=1.5) | 0.797 / 0.583 | 2.397 | 2.188 | 2.351 | 2.114 | 2.325 | 2.480 | **2.518** |
| MediQ-Holdout / `Qwen3.5-35B-A3B` (Cap=1/3, c=1.5) | 0.797 / 0.583 | 1.129 | 0.460 | 1.169 | 0.995 | 0.924 | **1.346** | 0.924 |
| MediQ-Holdout / `DeepSeek-V3.2` (Cap=2, c=1.5) | 0.725 / 0.671 | 3.563 | 3.444 | 3.728 | 3.320 | 4.064 | 4.007 | **4.089** |
| MediQ-Holdout / `DeepSeek-V3.2` (Cap=2, c_def=4.0, c=1.5) | 0.725 / 0.671 | 2.201 | 2.102 | 2.476 | 2.071 | 2.728 | 2.785 | **2.847** |
| MediQ-Holdout / `DeepSeek-V3.2` (Cap=1/3, c=1.5) | 0.725 / 0.671 | 0.839 | 1.209 | 1.662 | 1.615 | **1.814** | 1.721 | 1.811 |
| v2-UnseenIDs / `Qwen3.5` (Cap=2, c=1.5) | 0.838 / 0.929 | 2.711 | 4.418 | 4.193 | **4.434** | 4.121 | 3.989 | 4.157 |
| v2-UnseenIDs / `Qwen3.5` (Cap=1/3, c=1.5) | 0.838 / 0.929 | 1.286 | 2.902 | 2.566 | **2.994** | 2.757 | 2.701 | 2.757 |
