# Pre-Registered Zero-Touch Blind Evaluation (`Benchmark v3` + Pure Capability-Gap `MediQ-Holdout`)
- Live Calls: `3622`, Cached Hits: `1964`, Spent: `¥3.32`

## Suite A: `Benchmark v3` (4 Unseen Length-Matched Domains, Zero Protocol Codes)
### `Qwen/Qwen3.5-35B-A3B` (Missing-F1=`0.9185`, Override-F1=`0.6423`)
| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 | Realized Tok (Cap=2) |
|:---|---:|---:|---:|---:|
| `Direct-ACT` | -0.556 (47.2%) | -0.556 (47.2%) | -0.556 (47.2%) | 862 |
| `Naive-Cascade` | 2.176 (66.0%) | 2.072 (66.0%) | 0.570 (66.0%) | 4454 |
| `KL-PBD+ (Warm)` | 3.401 (75.0%) | 3.401 (75.0%) | 2.657 (67.1%) | 4431 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.121 (66.0%) | 1.601 (65.4%) | 2.088 (66.0%) | 2675 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.599 (82.8%) | 4.149 (82.6%) | 3.458 (78.6%) | 2675 |
| `Joint-Greedy+Naor` | 4.416 (80.8%) | 4.176 (80.6%) | 3.417 (77.5%) | 4610 |
| `Calibrated-Cascade+Naor` | 4.599 (82.8%) | 4.162 (82.8%) | 3.458 (78.6%) | 4815 |
| `Tri-VoI-L2D` | 4.364 (81.4%) | 4.139 (81.2%) | 3.458 (78.6%) | 4734 |
| `Tri-VoI-MDP` | 4.470 (82.1%) | 4.059 (80.8%) | 3.334 (74.4%) | 2091 |
| `Oracle` | 6.640 (90.0%) | 6.265 (90.0%) | 5.600 (86.5%) | 4710 |

### `deepseek-ai/DeepSeek-V3.2` (Missing-F1=`1.0`, Override-F1=`0.8161`)
| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 | Realized Tok (Cap=2) |
|:---|---:|---:|---:|---:|
| `Direct-ACT` | 0.000 (50.0%) | 0.000 (50.0%) | 0.000 (50.0%) | 801 |
| `Naive-Cascade` | 2.644 (67.5%) | 2.631 (67.5%) | 1.505 (67.5%) | 4132 |
| `KL-PBD+ (Warm)` | 3.216 (74.6%) | 3.216 (74.6%) | 1.413 (60.4%) | 4120 |
| `ACL-2026-VoI (Live)+Post-Defer` | 2.742 (69.7%) | 2.270 (69.7%) | 2.610 (69.7%) | 2634 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.411 (82.4%) | 3.939 (82.4%) | 2.612 (73.5%) | 2634 |
| `Joint-Greedy+Naor` | 4.204 (80.0%) | 4.007 (80.0%) | 3.469 (76.8%) | 4325 |
| `Calibrated-Cascade+Naor` | 4.528 (82.9%) | 4.028 (82.9%) | 2.695 (74.0%) | 4545 |
| `Tri-VoI-L2D` | 4.558 (82.9%) | 4.058 (81.2%) | 2.695 (74.0%) | 4466 |
| `Tri-VoI-MDP` | 4.582 (82.9%) | 4.170 (81.7%) | 3.484 (74.7%) | 2036 |
| `Oracle` | 7.196 (92.8%) | 6.821 (92.8%) | 6.156 (89.3%) | 4437 |

## Suite B1: `MediQ-Holdout` Scope-Compliance Real-LLM Specialist (`Qwen3.5` -> `DeepSeek-V3.2`)
| Method | Cap=2 (Util / RawMedAcc% / EscRec%) | Cap=2 HighCost (Util / RawMedAcc% / EscRec%) | Cap=1/3 (Util / RawMedAcc% / EscRec%) |
|:---|---:|---:|---:|
| `Direct-ACT` | -1.406 / 70.3% / 0.0% | -1.406 / 70.3% / 0.0% | -1.406 / 70.3% / 0.0% |
| `Naive-Cascade` | 1.313 / 72.7% / 75.0% | 0.344 / 72.7% / 75.0% | -3.196 / 72.7% / 75.0% |
| `KL-PBD+ (Warm)` | 1.694 / 70.3% / 85.4% | -1.006 / 68.0% / 52.1% | 1.035 / 71.9% / 50.0% |
| `ACL-2026-VoI (Live)+Post-Defer` | -0.598 / 73.4% / 18.8% | -0.848 / 73.4% / 18.8% | -0.598 / 73.4% / 18.8% |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 1.530 / 73.4% / 75.0% | 0.229 / 71.9% / 68.8% | 0.222 / 72.7% / 58.3% |
| `Joint-Greedy+Naor` | 1.244 / 73.4% / 64.6% | -0.321 / 73.4% / 50.0% | 0.191 / 74.2% / 47.9% |
| `Calibrated-Cascade+Naor` | 1.718 / 75.8% / 77.1% | 0.950 / 76.6% / 77.1% | 0.882 / 77.3% / 62.5% |
| `Tri-VoI-L2D` | 1.676 / 75.0% / 75.0% | 0.817 / 75.0% / 70.8% | 0.434 / 75.0% / 56.2% |
| `Tri-VoI-MDP` | 1.538 / 76.6% / 66.7% | 0.407 / 76.6% / 60.4% | 1.003 / 76.6% / 54.2% |
| `Oracle` | 3.322 / 73.4% / 100.0% | 2.572 / 73.4% / 100.0% | 1.056 / 73.4% / 97.9% |

## Suite B2: `MediQ-Holdout` Pure Capability-Gap Real-LLM Specialist (No Forced Escalation Penalty)
### `Qwen3-8B` -> Real `Qwen3.5-35B-A3B`
| Method | Cap=2 (Util / Acc%) | Cap=1/3 (Util / Acc%) | Realized Tok (Cap=2) |
|:---|---:|---:|---:|
| `Direct-ACT` | -0.156 (49.2%) | -0.156 (49.2%) | 596 |
| `Naive-Cascade` | 0.902 (57.8%) | 0.762 (57.8%) | 3263 |
| `KL-PBD+ (Warm)` | 2.247 (68.0%) | 0.892 (58.6%) | 3107 |
| `ACL-2026-VoI (Live)+Post-Defer` | 0.371 (56.2%) | 0.371 (56.2%) | 2330 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.637 (58.6%) | 0.476 (57.8%) | 2330 |
| `Joint-Greedy+Naor` | 0.895 (60.2%) | 0.419 (57.0%) | 3446 |
| `Calibrated-Cascade+Naor` | 0.266 (56.2%) | 0.455 (57.8%) | 3649 |
| `Tri-VoI-L2D` | 0.827 (59.4%) | 0.148 (55.5%) | 3551 |
| `Tri-VoI-MDP` | 0.496 (57.0%) | 0.551 (57.8%) | 1826 |
| `Oracle` | 1.603 (64.8%) | 0.557 (63.3%) | 3382 |

### `Qwen3-8B` -> Real `DeepSeek-V3.2`
| Method | Cap=2 (Util / Acc%) | Cap=1/3 (Util / Acc%) | Realized Tok (Cap=2) |
|:---|---:|---:|---:|
| `Direct-ACT` | -0.156 (49.2%) | -0.156 (49.2%) | 596 |
| `Naive-Cascade` | 0.589 (56.2%) | 0.450 (56.2%) | 3263 |
| `KL-PBD+ (Warm)` | 2.068 (67.2%) | 1.388 (60.9%) | 3107 |
| `ACL-2026-VoI (Live)+Post-Defer` | 0.527 (57.0%) | 0.527 (57.0%) | 2330 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 0.340 (56.2%) | 0.217 (56.2%) | 2330 |
| `Joint-Greedy+Naor` | 0.546 (57.8%) | 0.344 (56.2%) | 3538 |
| `Calibrated-Cascade+Naor` | 0.266 (56.2%) | 0.330 (57.0%) | 3649 |
| `Tri-VoI-L2D` | 0.414 (57.0%) | 0.148 (55.5%) | 3604 |
| `Tri-VoI-MDP` | 0.496 (57.0%) | 0.258 (56.2%) | 1826 |
| `Oracle` | 1.291 (63.3%) | 0.245 (61.7%) | 3382 |

