# Pre-Registered Organ-System-Disjoint `MediQ-Test-Disjoint` Blind Evaluation (`N_cal=60, N_test=180`)
- Live Calls: `2871`, Cached Hits: `0`, Spent: `¥3.55`

## Diagnostic & `post_missing` Signal Audit (Zero Oracle Fallback)
- **`Qwen/Qwen3.5-35B-A3B`**: Miss-F1=`0.694`, Override-F1=`0.759`, `post_missing` Agree=`87.04%`, F1=`0.588`, Unparsed Fallback=`0`.
- **`deepseek-ai/DeepSeek-V3.2`**: Miss-F1=`0.758`, Override-F1=`0.753`, `post_missing` Agree=`80.28%`, F1=`0.300`, Unparsed Fallback=`0`.

## Setup A: Cross-Group Real-LLM Specialist (`Qwen3.5-35B-A3B` -> `DeepSeek-V3.2`)
| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 | Tok (Cap=2, c=1.5) |
|:---|---:|---:|---:|---:|---:|
| `Direct-ACT` | 0.667 (77.8%/0.0%) | 0.667 | 0.667 (77.8%/0.0%) | 0.667 | 806 |
| `KL-PBD+ (Warm)` | 4.260 (81.1%/92.5%) | 4.260 | 3.196 (80.0%/59.7%) | 3.196 | 4118 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.997 (81.1%/89.5%) | 3.676 | 2.594 (79.4%/73.1%) | 2.583 | 2206 |
| `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` | 3.874 (80.6%/83.6%) | 3.559 | 3.537 (81.7%/71.6%) | 3.181 | 2206 |
| `Joint-Greedy+Naor` | 4.216 (81.7%/89.5%) | 4.112 | 2.681 (80.0%/73.1%) | 2.142 | 4283 |
| `Joint-Greedy+MDP (2C-Self)` | 4.175 (81.1%/88.1%) | 4.020 | 3.407 (80.6%/71.6%) | 3.342 | 1752 |
| `Joint-Greedy+OwnDP (2C-Self)` | 4.178 (81.1%/88.1%) | 4.020 | 3.560 (81.1%/73.1%) | 3.324 | 1757 |
| `Calibrated-Cascade+Naor` | 3.991 (81.1%/88.1%) | 3.541 | 2.909 (80.6%/77.6%) | 2.459 | 4391 |
| `Calibrated-Cascade+MDP (2C-Self)` | 3.884 (80.6%/80.6%) | 3.434 | 3.471 (81.7%/71.6%) | 3.021 | 1906 |
| `Calibrated-Cascade+OwnDP (2C-Self)` | 3.884 (80.6%/80.6%) | 3.434 | 3.471 (81.7%/71.6%) | 3.021 | 1906 |
| `Tri-VoI-L2D` | 3.862 (80.6%/88.1%) | 3.764 | 2.909 (80.6%/77.6%) | 2.459 | 4354 |
| `Tri-VoI-MDP` | 3.969 (81.1%/85.1%) | 3.923 | 3.356 (81.7%/71.6%) | 2.906 | 1853 |
| `Tri-VoI-MDP (Deploy-Self)` | 4.266 (81.7%/85.1%) | 4.091 | 3.588 (82.2%/71.6%) | 3.244 | 1823 |
| `Oracle` | 5.053 (82.2%/100.0%) | 4.486 | 3.688 (82.2%/100.0%) | 3.122 | 4342 |

- **Cap=2, c=1.5**: `Deploy-Self` vs `Joint-Greedy+MDP (2C-Self)` = `+0.091` (`p=0.6851`), vs `Joint-Greedy+OwnDP (2C-Self)` = `+0.088` (`p=0.6928`), vs `Tri-VoI-L2D` = `+0.404` (`p=0.3236`).
- **Cap=2, c=3.0**: `Deploy-Self` vs `Joint-Greedy+MDP (2C-Self)` = `+0.071` (`p=0.5985`), vs `Joint-Greedy+OwnDP (2C-Self)` = `+0.071` (`p=0.5985`), vs `Tri-VoI-L2D` = `+0.327` (`p=0.3812`).
- **Cap=1/3, c=1.5**: `Deploy-Self` vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `+0.051` (`p=0.8221`), vs `Joint-Greedy+OwnDP (2C-Self)` = `+0.028` (`p=0.9170`), vs `Tri-VoI-L2D` = `+0.679` (`p=0.0137`).
- **Cap=1/3, c=3.0**: `Deploy-Self` vs `Joint-Greedy+MDP (2C-Self)` = `-0.098` (`p=0.7124`), vs `Joint-Greedy+OwnDP (2C-Self)` = `-0.080` (`p=0.8001`), vs `Tri-VoI-L2D` = `+0.785` (`p=0.0057`).

## Setup B: Cross-Group Generalist (`DeepSeek-V3.2`)
| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 | Tok (Cap=2, c=1.5) |
|:---|---:|---:|---:|---:|---:|
| `Direct-ACT` | -0.111 (76.1%/0.0%) | -0.111 | -0.111 (76.1%/0.0%) | -0.111 | 659 |
| `KL-PBD+ (Warm)` | 3.901 (79.0%/97.3%) | 3.901 | 2.547 (77.0%/57.9%) | 2.547 | 3431 |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.869 (82.4%/92.5%) | 3.429 | 2.519 (83.4%/71.3%) | 2.123 | 2105 |
| `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` | 3.521 (82.7%/88.7%) | 3.033 | 2.867 (83.3%/70.5%) | 2.477 | 2105 |
| `Joint-Greedy+Naor` | 3.978 (81.9%/90.7%) | 3.922 | 2.349 (82.8%/66.0%) | 1.675 | 3528 |
| `Joint-Greedy+MDP (2C-Self)` | 3.588 (81.7%/86.9%) | 3.620 | 2.629 (81.4%/67.8%) | 2.324 | 1542 |
| `Joint-Greedy+OwnDP (2C-Self)` | 3.588 (81.7%/86.9%) | 3.620 | 2.834 (81.2%/69.8%) | 2.286 | 1542 |
| `Calibrated-Cascade+Naor` | 3.784 (82.3%/90.4%) | 3.193 | 2.409 (82.7%/71.9%) | 1.817 | 3757 |
| `Calibrated-Cascade+MDP (2C-Self)` | 3.216 (81.8%/85.7%) | 2.624 | 2.689 (83.3%/70.5%) | 2.078 | 1761 |
| `Calibrated-Cascade+OwnDP (2C-Self)` | 3.272 (82.0%/86.0%) | 2.680 | 2.661 (83.2%/70.2%) | 2.070 | 1761 |
| `Tri-VoI-L2D` | 4.176 (82.6%/91.6%) | 3.782 | 2.428 (82.8%/71.0%) | 1.811 | 3545 |
| `Tri-VoI-MDP` | 3.669 (82.3%/86.0%) | 3.560 | 2.866 (82.6%/67.8%) | 2.554 | 1554 |
| `Tri-VoI-MDP (Deploy-Self)` | 3.721 (82.1%/87.5%) | 3.576 | 2.706 (83.3%/70.5%) | 2.322 | 1555 |
| `Oracle` | 5.031 (82.1%/100.0%) | 4.464 | 3.666 (82.1%/100.0%) | 3.099 | 3675 |

- **Cap=2, c=1.5**: `Deploy-Self` vs `Joint-Greedy+MDP (2C-Self)` = `+0.133` (`p=0.4768`), vs `Joint-Greedy+OwnDP (2C-Self)` = `+0.133` (`p=0.4768`), vs `Tri-VoI-L2D` = `-0.455` (`p=0.0816`).
- **Cap=2, c=3.0**: `Deploy-Self` vs `Joint-Greedy+MDP (2C-Self)` = `-0.044` (`p=0.7574`), vs `Joint-Greedy+OwnDP (2C-Self)` = `-0.044` (`p=0.7574`), vs `Tri-VoI-L2D` = `-0.206` (`p=0.5162`).
- **Cap=1/3, c=1.5**: `Deploy-Self` vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `-0.161` (`p=0.1173`), vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `-0.161` (`p=0.1173`), vs `Tri-VoI-L2D` = `+0.278` (`p=0.2756`).
- **Cap=1/3, c=3.0**: `Deploy-Self` vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `-0.155` (`p=0.5681`), vs `ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)` = `-0.155` (`p=0.5681`), vs `Tri-VoI-L2D` = `+0.511` (`p=0.1636`).
