# Overnight Paper-Complete Experiment Master Report — [COMPLETED_ALL_5_STAGES]

- **Budget & Telemetry**: Live Calls=`0`, Cache Hits=`22731`, Dry Calls=`0`, Prompt Tokens=`0`, Completion Tokens=`0`, **Estimated Spend=`¥0.00` / `¥0.01`**, Elapsed=`1.51 min`.

## Stage 1 & 2: Full-Scale Cross-Model Evaluation (5 Models Across 4 Families, `N=360`, 6 Independent Episodes)

| Backbone | N | Missing F1 | Ent Coll% | Direct (1.5) | Naive (1.5) | J-Greedy (1.5) | CalibCasc (1.5) | **Tri-VoI (1.5)** | CalibCasc (2.5) | **Tri-VoI (2.5)** | Severe Cap=1/3: Tri / Calib+Naor / w/oExt / Naive |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Qwen/Qwen3.5-35B-A3B | 360 | 0.8483 | 77.78 | 2.000 | 3.165 | 5.142 | 4.877 | **5.220** | 4.479 | **5.142** | 3.792 / 3.798 / 2.732 / 3.000 |
| deepseek-ai/DeepSeek-V3.2 | 360 | 0.8946 | 96.11 | 1.111 | 3.305 | 5.391 | 5.550 | **5.523** | 5.075 | **5.259** | 4.062 / 4.080 / 3.470 / 3.185 |
| zai-org/GLM-4.5-Air | 360 | 0.8883 | 94.44 | 0.611 | 2.597 | 3.750 | 4.054 | **4.317** | 3.459 | **3.910** | 2.348 / 3.284 / 1.966 / 2.540 |
| Qwen/Qwen3-8B | 360 | 0.2157 | 87.22 | 0.778 | 2.682 | 3.589 | 3.711 | **3.731** | 3.644 | **3.731** | 1.714 / 1.836 / 0.245 / 2.434 |
| tencent/Hy4-preview | 360 | 0.9831 | 95.0 | 1.111 | 3.441 | 5.390 | 5.833 | **5.842** | 5.350 | **5.468** | 4.477 / 4.548 / 3.846 / 3.349 |

## Stage 3: Realistic Noisy Multi-Paragraph Benchmark + Imperfect LLM User (`N=360`)

| Backbone | Missing F1 | Ent Collapse% | Naive-Casc (1.5) | Joint-Greedy (1.5) | Calib-Casc (1.5) | **Tri-VoI (1.5)** | Calib-Casc (2.5) | **Tri-VoI (2.5)** | Boot p_Holm vs Naive (1.5) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3.5-35B-A3B | 0.9364 | 71.67 | 2.285 | 3.633 | 3.409 | **3.503** | 2.948 | **3.511** | 0.0095982 |
| tencent/Hy4-preview | 0.9972 | 88.33 | 2.666 | 3.990 | 4.300 | **4.380** | 3.803 | **4.082** | 0.0017996 |

## Stage 4: Entropy-Collapse Stress Test Across Sampling `K ∈ {3, 5, 10}` × Temperature `T ∈ {0.7, 1.0}`

| Backbone | T | K | H=0 Collapse % | H<=0.45 Missed % | Mean Entropy (bits) | Majority Acc % | Disentangled Missing F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | 0.7 | 3 | 76.67 | 76.67 | 0.2143 | 53.33 | 0.8483 |
| Qwen3.5-35B-A3B | 0.7 | 5 | 70.83 | 70.83 | 0.2401 | 52.5 | 0.8483 |
| Qwen3.5-35B-A3B | 0.7 | 10 | 59.17 | 59.17 | 0.2838 | 53.33 | 0.8483 |
| Qwen3.5-35B-A3B | 1.0 | 3 | 57.5 | 57.5 | 0.3958 | 53.33 | 0.8483 |
| Qwen3.5-35B-A3B | 1.0 | 5 | 47.5 | 47.5 | 0.4455 | 55.0 | 0.8483 |
| Qwen3.5-35B-A3B | 1.0 | 10 | 37.5 | 37.5 | 0.4829 | 53.33 | 0.8483 |
| Hy4-preview | 0.7 | 3 | 92.5 | 92.5 | 0.0744 | 54.17 | 0.9831 |
| Hy4-preview | 0.7 | 5 | 90.0 | 90.0 | 0.088 | 55.83 | 0.9831 |
| Hy4-preview | 0.7 | 10 | 90.0 | 90.0 | 0.0673 | 55.83 | 0.9831 |
| Hy4-preview | 1.0 | 3 | 89.17 | 89.17 | 0.0995 | 55.83 | 0.9831 |
| Hy4-preview | 1.0 | 5 | 87.5 | 87.5 | 0.106 | 55.83 | 0.9831 |
| Hy4-preview | 1.0 | 10 | 76.67 | 76.67 | 0.1463 | 56.67 | 0.9831 |

## Stage 5: Deep Structural, Cross-Validation, LODO Transfer, Bursty/3-Expert, Subset & Specialist Stress

- **56-Point 2D Operating Phase Map (`7 loads × 8 c_ask`, 8 methods)**: `Tri-VoI-L2D` is strict overall winner on **27/56** cells; outperforms/ties `Calibrated-Cascade` (**53/56**), `Calibrated-Cascade+Naor` (**43/56**), `Joint-Greedy` (**46/56**), `Joint-Greedy+Naor` (**44/56**), and `w/o Queue-Externality` (**49/56**).

### 5-Fold Stratified Cross-Validation Across Available Models (420 items each)
| Backbone | Naive-Cascade | Joint-Greedy | Calibrated-Cascade | **Tri-VoI-L2D** |
| :--- | :---: | :---: | :---: | :---: |
| Qwen3.5-35B-A3B | 3.181±0.67 | 5.708±0.214 | 4.999±0.732 | **5.764±0.34** |
| DeepSeek-V3.2 | 3.443±0.667 | 5.751±0.642 | 5.555±0.649 | **5.581±0.688** |
| GLM-4.5-Air | 2.624±0.824 | 4.92±0.352 | 4.732±0.615 | **5.402±0.136** |
| Qwen3-8B | 2.485±0.792 | 4.846±0.135 | 4.912±0.228 | **4.852±0.478** |
| Hy4-preview | 3.513±0.6 | 5.573±0.562 | 5.886±0.618 | **5.844±0.519** |

### Bursty Poisson Arrivals & 3-Expert Heterogeneous Shift-Fatigue Pool
| Method | Bursty Util (Acc% / Wait) | 3-Expert Shift Util (Acc% / Wait) |
| :--- | :---: | :---: |
| Direct-ACT | 2.000 (60.0% / W=0.0) | 2.000 (60.0% / W=0.0) |
| Congested-L2D-Only | 3.782 (78.67% / W=1.6) | 4.961 (87.61% / W=1.12) |
| Naive-Cascade | 3.141 (70.64% / W=0.62) | 3.643 (73.5% / W=0.16) |
| Joint-Greedy | 4.494 (81.56% / W=1.16) | 5.449 (87.86% / W=0.75) |
| Joint-Greedy+Naor | 4.499 (80.92% / W=0.94) | 5.673 (87.39% / W=0.16) |
| Calibrated-Cascade | 4.452 (81.0% / W=0.86) | 5.233 (85.61% / W=0.43) |
| Calibrated-Cascade+Naor | 4.433 (80.72% / W=0.79) | 5.338 (85.78% / W=0.16) |
| Tri-VoI-L2D | 4.611 (81.58% / W=0.83) | 5.506 (86.89% / W=0.17) |

### Stage 5e: Decision-Relevant Subset (`cal=56/60`, `test=328/360`, Invariant Q2 Excluded)
| Regime | Naive-Casc | Joint-Greedy | Joint-Greedy+Naor | Calib-Casc | Calib-Casc+Naor | w/o Externality | **Tri-VoI-L2D** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cap=2, c=1.5 | 2.717 | 5.199 | 5.258 | 4.941 | 4.961 | 5.259 | **5.253** |
| Cap=2, c=2.5 | 2.550 | 5.044 | 5.139 | 4.557 | 4.577 | 5.111 | **5.141** |
| Cap=1/3, c=1.5 | 2.461 | 2.150 | 3.401 | 2.858 | 3.767 | 2.440 | **3.622** |

### Stage 5f: Realistic Specialist Stress (Delayed Feedback & Unclarified-Ticket Rework)
| Stress Condition | Congested-L2D | Joint-Greedy | Joint-Greedy+Naor | Calib-Casc | Calib-Casc+Naor | w/o Externality | **Tri-VoI-L2D** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Delayed Feedback (Cap=2) | 4.679 | 5.163 | 5.210 | 4.877 | 4.867 | 5.273 | **5.219** |
| Delayed Feedback (Cap=1/3) | 1.244 | 1.957 | 3.195 | 2.889 | 3.711 | 2.486 | **3.652** |
| Unclarified Rework (Cap=2) | 2.801 | 3.754 | 3.663 | 4.678 | 4.678 | 4.264 | **4.274** |
| Unclarified Rework (Cap=1/3) | 1.246 | 1.890 | 2.889 | 3.008 | 3.711 | 2.653 | **3.576** |
| Delayed + Rework (Cap=2) | 2.748 | 3.658 | 3.651 | 4.678 | 4.678 | 4.268 | **4.279** |
| Delayed + Rework (Cap=1/3) | 1.028 | 1.789 | 2.823 | 2.765 | 3.676 | 2.422 | **3.561** |

### Error Taxonomy (`N=360`, Errors=63 / 17.5%)
| Error Category | Count | % of Errors |
| :--- | ---: | ---: |
| Specialist_Fatigue_or_Noise_Error | 40 | 63.5% |
| PostASK_Public_Rubric_Reasoning_Error | 8 | 12.7% |
| Diagnoser_FN_Missed_Underspec | 6 | 9.5% |
| Clear_Easy_Direct_ACT_Error | 5 | 7.9% |
| PostASK_Override_Detection_FN | 4 | 6.3% |
