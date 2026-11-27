# Overnight Paper-Complete Experiment Master Report — [COMPLETED_ALL_5_STAGES]

- **Budget & Telemetry**: Live Calls=`3854`, Cache Hits=`6826`, Dry Calls=`0`, Prompt Tokens=`2,988,510`, Completion Tokens=`479,383`, **Estimated Spend=`¥3.65` / `¥5.50`**, Elapsed=`140.58 min`.

## Stage 1 & 2: Full-Scale 5-Family Cross-Model Evaluation (`N=360`, 6 Independent Episodes)

| Backbone | N | Missing F1 | Ent Collapse% | Direct-ACT (1.5) | Naive-Casc (1.5) | Joint-Greedy (1.5) | Calib-Casc (1.5) | **Tri-VoI (1.5)** | Calib-Casc (2.5) | **Tri-VoI (2.5)** | Severe Cap=1/3: Tri vs Calib-Casc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Qwen/Qwen3.5-35B-A3B | 360 | 0.8483 | 77.78 | 2.000 | 3.120 | 5.073 | 4.881 | **5.030** | 4.484 | **4.985** | 3.764 vs 3.168 |
| deepseek-ai/DeepSeek-V3.2 | 360 | 0.8946 | 96.11 | 1.111 | 3.330 | 5.391 | 5.550 | **5.514** | 5.075 | **5.238** | 4.086 vs 3.548 |
| zai-org/GLM-4.5-Air | 360 | 0.8883 | 94.44 | 0.611 | 2.680 | 3.776 | 4.062 | **4.333** | 3.467 | **3.906** | 2.421 vs 2.802 |
| Qwen/Qwen3-8B | 360 | 0.2157 | 87.22 | 0.778 | 2.937 | 3.589 | 3.616 | **3.742** | 3.549 | **3.742** | 1.714 vs 0.368 |
| tencent/Hy4-preview | 360 | 0.9831 | 95.0 | 1.111 | 3.441 | 5.390 | 5.833 | **5.859** | 5.350 | **5.411** | 4.476 vs 3.927 |

## Stage 3: Realistic Noisy Multi-Paragraph Benchmark + Imperfect LLM User (`N=360`)

| Backbone | Missing F1 | Ent Collapse% | Naive-Casc (1.5) | Joint-Greedy (1.5) | Calib-Casc (1.5) | **Tri-VoI (1.5)** | Calib-Casc (2.5) | **Tri-VoI (2.5)** | Boot p_Holm vs Naive (1.5) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3.5-35B-A3B | 0.9364 | 71.67 | 2.343 | 3.668 | 3.409 | **3.553** | 2.948 | **3.544** | 0.0047992 |
| tencent/Hy4-preview | 0.9972 | 88.33 | 2.666 | 3.935 | 4.300 | **4.397** | 3.803 | **4.104** | 0.0013997 |

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

## Stage 5: Deep Structural, Cross-Validation, LODO Transfer, Bursty/3-Expert, Pareto & Error Taxonomy

- **56-Point 2D Operating Phase Map (`7 loads × 8 c_ask`)**: `Tri-VoI-L2D` outperforms or ties `Calibrated-Cascade` on **51/56** cells and outperforms or ties `Joint-Greedy` on **38/56** cells.

### 5-Fold Stratified Cross-Validation Across Available Models (420 items each)
| Backbone | Naive-Cascade | Joint-Greedy | Calibrated-Cascade | **Tri-VoI-L2D** |
| :--- | :---: | :---: | :---: | :---: |
| Qwen3.5-35B-A3B | 3.156±0.479 | 5.805±0.275 | 4.84±0.374 | **5.564±0.551** |
| DeepSeek-V3.2 | 3.513±0.588 | 5.797±0.618 | 5.555±0.649 | **5.522±0.715** |
| GLM-4.5-Air | 2.693±0.76 | 5.041±0.341 | 4.168±0.415 | **5.52±0.381** |
| Qwen3-8B | 2.78±0.988 | 4.789±0.083 | 4.695±0.194 | **4.767±0.479** |
| Hy4-preview | 3.513±0.6 | 5.68±0.662 | 5.886±0.618 | **5.918±0.486** |

### Bursty Poisson Arrivals & 3-Expert Heterogeneous Shift-Fatigue Pool
| Method | Bursty Util (Acc% / Wait) | 3-Expert Shift Util (Acc% / Wait) |
| :--- | :---: | :---: |
| Direct-ACT | 2.000 (60.0% / W=0.0) | 2.000 (60.0% / W=0.0) |
| Congested-L2D-Only | 3.782 (78.67% / W=1.6) | 4.961 (87.61% / W=1.12) |
| Naive-Cascade | 3.074 (70.19% / W=0.62) | 3.624 (73.56% / W=0.17) |
| Joint-Greedy | 4.374 (80.47% / W=1.1) | 5.540 (88.25% / W=0.72) |
| Calibrated-Cascade | 4.569 (81.44% / W=0.87) | 5.201 (85.53% / W=0.43) |
| Tri-VoI-L2D | 4.629 (81.5% / W=0.79) | 5.426 (85.94% / W=0.13) |

### Error Taxonomy (`N=360`, Errors=67 / 18.61%)
| Error Category | Count | % of Errors |
| :--- | ---: | ---: |
| Specialist_Fatigue_or_Noise_Error | 38 | 56.7% |
| PostASK_Public_Rubric_Reasoning_Error | 14 | 20.9% |
| Diagnoser_FN_Missed_Underspec | 6 | 9.0% |
| Clear_Easy_Direct_ACT_Error | 5 | 7.5% |
| PostASK_Override_Detection_FN | 4 | 6.0% |
