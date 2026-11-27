# Single-Shot Zero-Touch Holdout & Unseen-ID Report (Spend: ¥3.5210)

## Part A: Disjoint MediQ Holdout (`N_cal=48`, `N_test=128`, Natural Prose — Zero Bullet Shortcut)
### `Qwen/Qwen3.5-35B-A3B` (Missing-F1=0.7969, Scope-F1=0.5827, Collapse=59.38%, Raw USMLE Q1=75.0%, Q3=71.88%, Q4res=81.25%->87.5%, Q4esc=75.0%->87.5%)
#### Calibrator: `unified_default`
| Method | Cap=2, c=1.5 | Cap=2, c_def=4.0, c=1.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|---:|
| `Direct-ACT` | -1.406 (43.0%) | -1.406 (43.0%) | -1.406 (43.0%) |
| `KL-PBD+ (Warm)` | 3.632 (76.9%) | 2.397 (77.2%) | 1.129 (59.7%) |
| `ACL-2026-VoI (Live)` | -1.910 (43.0%) | -1.910 (43.0%) | -1.910 (43.0%) |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.529 (77.0%) | 2.188 (75.2%) | 0.460 (64.1%) |
| `Joint-Greedy` | 3.781 (77.5%) | 2.406 (76.4%) | 0.249 (67.5%) |
| `Joint-Greedy+Naor` | 3.843 (77.8%) | 2.351 (75.9%) | 1.169 (65.9%) |
| `Calibrated-Cascade` | 3.441 (77.3%) | 2.114 (74.5%) | 0.112 (71.1%) |
| `Calibrated-Cascade+Naor` | 3.453 (77.2%) | 2.114 (74.5%) | 0.995 (68.3%) |
| `w/o Queue-Externality` | 3.814 (78.1%) | 2.261 (76.1%) | -0.302 (69.4%) |
| `Tri-VoI-L2D` | 3.690 (77.5%) | 2.325 (76.2%) | 0.924 (66.7%) |
| `Tri-VoI-L2D (2-Call)` | 3.497 (76.2%) | 2.480 (76.2%) | 1.346 (68.9%) |
| `Tri-VoI-L2D (Smooth-Naor)` | 3.698 (77.5%) | 2.518 (76.7%) | 0.924 (66.7%) |
| `Oracle` | 4.134 (77.5%) | 3.384 (77.5%) | 1.869 (76.7%) |

#### Calibrator: `stratify_post_clear`
| Method | Cap=2, c=1.5 | Cap=2, c_def=4.0, c=1.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|---:|
| `Direct-ACT` | -1.406 (43.0%) | -1.406 (43.0%) | -1.406 (43.0%) |
| `KL-PBD+ (Warm)` | 3.632 (76.9%) | 2.397 (77.2%) | 1.129 (59.7%) |
| `ACL-2026-VoI (Live)` | -1.898 (43.0%) | -1.898 (43.0%) | -1.898 (43.0%) |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.145 (75.3%) | 2.247 (75.5%) | 0.550 (64.7%) |
| `Joint-Greedy` | 3.809 (77.7%) | 2.430 (76.6%) | 0.268 (67.5%) |
| `Joint-Greedy+Naor` | 3.875 (78.0%) | 2.320 (75.8%) | 1.232 (65.9%) |
| `Calibrated-Cascade` | 3.435 (77.7%) | 2.209 (75.3%) | -0.234 (70.9%) |
| `Calibrated-Cascade+Naor` | 3.390 (77.2%) | 2.239 (75.5%) | 1.011 (68.6%) |
| `w/o Queue-Externality` | 3.793 (77.8%) | 2.287 (75.8%) | -0.012 (69.8%) |
| `Tri-VoI-L2D` | 3.746 (77.5%) | 2.417 (76.1%) | 1.039 (66.6%) |
| `Tri-VoI-L2D (2-Call)` | 3.404 (75.6%) | 2.613 (76.4%) | 1.030 (66.6%) |
| `Tri-VoI-L2D (Smooth-Naor)` | 3.722 (77.3%) | 2.432 (75.8%) | 1.039 (66.6%) |
| `Oracle` | 4.134 (77.5%) | 3.384 (77.5%) | 1.869 (76.7%) |

### `deepseek-ai/DeepSeek-V3.2` (Missing-F1=0.7246, Scope-F1=0.6715, Collapse=70.31%, Raw USMLE Q1=81.25%, Q3=62.5%, Q4res=75.0%->75.0%, Q4esc=75.0%->93.75%)
#### Calibrator: `unified_default`
| Method | Cap=2, c=1.5 | Cap=2, c_def=4.0, c=1.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|---:|
| `Direct-ACT` | -1.562 (42.2%) | -1.562 (42.2%) | -1.562 (42.2%) |
| `KL-PBD+ (Warm)` | 3.563 (77.3%) | 2.201 (76.9%) | 0.839 (58.3%) |
| `ACL-2026-VoI (Live)` | -1.246 (46.9%) | -1.246 (46.9%) | -1.246 (46.9%) |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.444 (77.8%) | 2.102 (74.4%) | 1.209 (67.7%) |
| `Joint-Greedy` | 3.672 (78.1%) | 2.515 (77.2%) | 0.454 (68.1%) |
| `Joint-Greedy+Naor` | 3.728 (78.0%) | 2.476 (76.4%) | 1.662 (68.6%) |
| `Calibrated-Cascade` | 3.522 (80.0%) | 2.071 (74.2%) | 0.958 (73.8%) |
| `Calibrated-Cascade+Naor` | 3.320 (78.0%) | 2.071 (74.2%) | 1.615 (70.9%) |
| `w/o Queue-Externality` | 3.955 (80.2%) | 2.675 (78.9%) | 0.512 (73.3%) |
| `Tri-VoI-L2D` | 4.064 (80.3%) | 2.728 (78.6%) | 1.814 (71.7%) |
| `Tri-VoI-L2D (2-Call)` | 4.007 (79.8%) | 2.785 (78.6%) | 1.721 (71.1%) |
| `Tri-VoI-L2D (Smooth-Naor)` | 4.089 (80.3%) | 2.847 (79.2%) | 1.811 (71.7%) |
| `Oracle` | 4.603 (79.8%) | 3.853 (79.8%) | 2.338 (79.1%) |

#### Calibrator: `stratify_post_clear`
| Method | Cap=2, c=1.5 | Cap=2, c_def=4.0, c=1.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|---:|
| `Direct-ACT` | -1.562 (42.2%) | -1.562 (42.2%) | -1.562 (42.2%) |
| `KL-PBD+ (Warm)` | 3.563 (77.3%) | 2.201 (76.9%) | 0.839 (58.3%) |
| `ACL-2026-VoI (Live)` | -1.246 (46.9%) | -1.246 (46.9%) | -1.246 (46.9%) |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 3.484 (78.1%) | 2.287 (75.6%) | 1.209 (67.7%) |
| `Joint-Greedy` | 3.731 (78.4%) | 2.543 (77.3%) | 0.267 (67.8%) |
| `Joint-Greedy+Naor` | 3.932 (79.1%) | 2.672 (77.5%) | 1.591 (68.4%) |
| `Calibrated-Cascade` | 3.602 (80.3%) | 2.198 (75.6%) | 0.958 (73.8%) |
| `Calibrated-Cascade+Naor` | 3.450 (78.8%) | 2.115 (74.8%) | 1.615 (70.9%) |
| `w/o Queue-Externality` | 3.967 (80.2%) | 2.902 (79.7%) | 0.461 (73.0%) |
| `Tri-VoI-L2D` | 4.048 (80.2%) | 2.950 (79.2%) | 1.821 (71.7%) |
| `Tri-VoI-L2D (2-Call)` | 3.983 (79.7%) | 3.048 (79.5%) | 1.728 (71.1%) |
| `Tri-VoI-L2D (Smooth-Naor)` | 4.069 (80.2%) | 2.778 (78.3%) | 1.821 (71.7%) |
| `Oracle` | 4.603 (79.8%) | 3.853 (79.8%) | 2.338 (79.1%) |

### Holdout Real-LLM Clinical Specialist (`Qwen3.5-35B` Agent -> Real `DeepSeek-V3.2` Full-Quadrant)
- **Cap=2, c_def=2.0, c=1.5**: `Direct-ACT`=-1.406, `KL-PBD+ (Warm)`=1.694, `ACL-2026-VoI (Live)+Full-Defer+Naor`=1.530, `Calibrated-Cascade+Naor`=1.718, `Joint-Greedy+Naor`=1.244, `Tri-VoI-L2D`=1.676, `Tri-VoI-L2D (Smooth-Naor)`=1.542
- **Cap=2, c_def=4.0, c=1.5**: `Direct-ACT`=-1.406, `KL-PBD+ (Warm)`=-1.006, `ACL-2026-VoI (Live)+Full-Defer+Naor`=0.229, `Calibrated-Cascade+Naor`=0.950, `Joint-Greedy+Naor`=-0.321, `Tri-VoI-L2D`=0.817, `Tri-VoI-L2D (Smooth-Naor)`=0.773
- **Cap=1/3, c_def=2.0, c=1.5**: `Direct-ACT`=-1.406, `KL-PBD+ (Warm)`=1.035, `ACL-2026-VoI (Live)+Full-Defer+Naor`=0.222, `Calibrated-Cascade+Naor`=0.882, `Joint-Greedy+Naor`=0.191, `Tri-VoI-L2D`=0.434, `Tri-VoI-L2D (Smooth-Naor)`=0.434

## Part B: `Benchmark v2` Unseen-Protocol-ID Sanity Check (`Qwen3.5-35B`, `N_cal=40`, `N_test=120`)
- Missing-F1=`0.8381`, Override-F1=`0.9286`, Collapse=`71.67%`
| Method | Cap=2, c=1.5 | Cap=1/3, c=1.5 |
|:---|---:|---:|
| `Direct-ACT` | -0.833 (45.8%) | -0.833 (45.8%) |
| `KL-PBD+ (Warm)` | 2.711 (72.7%) | 1.286 (60.5%) |
| `ACL-2026-VoI (Live)+Full-Defer+Naor` | 4.418 (80.8%) | 2.902 (73.7%) |
| `Joint-Greedy+Naor` | 4.193 (78.7%) | 2.566 (70.3%) |
| `Calibrated-Cascade+Naor` | 4.434 (81.0%) | 2.994 (74.2%) |
| `Tri-VoI-L2D` | 4.121 (78.8%) | 2.757 (72.2%) |
| `Tri-VoI-L2D (2-Call)` | 3.989 (78.7%) | 2.701 (72.5%) |
| `Tri-VoI-L2D (Smooth-Naor)` | 4.157 (78.8%) | 2.757 (72.2%) |
