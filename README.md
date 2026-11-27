# TRI-VOI: Three-Way Human-AI Cooperative Decision-Making

**Ask the User or Defer to the Expert? Information-Source Disentanglement and Queue-Relief Deferral for Three-Way Human-AI Cooperative Decision-Making**

---

## Unified Research Thesis

> Effective human-AI cooperation requires not only calibrated uncertainty estimation, but also identifying **who holds the information needed to resolve that uncertainty** (the frontline requester vs. an internal domain specialist) and **coupling the Value of Information (EVOI) of asking the user with dynamic specialist queue congestion and reliability drift**.

### Three Linked Motivations
1. **Binary Routing Fallacy**: Existing frameworks split into `{ACT, ASK}` (clarifying uncertainty without specialist escalation) vs. `{ACT, DEFER}` (deferring to experts while assuming complete user inputs). Real-world triage/dispatch flows face orthogonal **Specification Uncertainty** (only the requester knows) and **Policy/Override Uncertainty** (only internal specialists know).
2. **High-Consensus Entropy Collapse**: On natural-language underspecified tickets without explicit placeholder spoilers, modern instruction-tuned LLMs deterministically guess a default assumption ($H=0$, **77.8%–96.1%** across 5 LLM families). Even at $K=10$ samples and temperature $T=1.0$, **37.5%–76.7%** of underspecified cases still collapse to $H=0$, disabling entropy/consensus triggers (`ADEPT`, `DREAM`).
3. **Queue-Relief VoI Coupling Proposition**: Under Conditional Escalation (where user clarification resolves routine cases with probability $\hat{p}_{\text{resolve}}$ and confirms confidential escalation with $1-\hat{p}_{\text{resolve}}$), two-stage Bellman optimization yields exact monotone resource coupling:
   $$\frac{\partial\,\text{EVOI}(\text{ASK}; Q_t, \hat{p}_e)}{\partial Q_t} = \hat{p}_{\text{resolve}}\gamma > 0, \qquad \frac{\partial\,\text{EVOI}(\text{ASK}; Q_t, \hat{p}_e)}{\partial \hat{p}_e} = -\hat{p}_{\text{resolve}}(R_+ - R_-) < 0$$
   Asking the user acts as a **queue-relief valve** under congestion and specialist fatigue, further stabilized via first-order Naor queuing-externality shadow pricing and conjugate Discounted Beta-Bernoulli tracking.

---

## Method Architecture (`Tri-VoI-L2D`)
1. **Disentangled Source Diagnosis + Hierarchical Empirical Bayes Held-Out Calibration** (`HeldOutCalibrator`): extracts $(missing\_user\_info, potential\_override, H(x))$ in a single audit pass and calibrates a 3-outcome post-ASK partition $(p_{\text{resolve}}, p_{\text{ovr\_after}}, p_{\text{no\_info}})$ strictly on held-out calibration splits (zero test leakage).
2. **Resource-Coupled Two-Stage Bellman EVOI Controller + Queue Externality** (`controller.py`): jointly evaluates `ACT`, `ASK` (with Stage-2 `DEFER` lookahead), and `DEFER` priced with first-order Naor congestion externality $\gamma (Q_t + \hat{\rho}_t (Q_t + 1))$.
3. **Discounted Beta-Bernoulli Specialist Tracker** (`DiscountedBetaTracker`, $\lambda = 0.92$): conjugate closed-form adaptation to sudden, gradual, and shift-based specialist fatigue drift.

---

## Key Experimental Findings (26,039 Live API Calls Across 5 LLM Families)

### 1. Five-Model Generalization (`N=360` Test Tickets × 5 Seeds per Family)
| Backbone | Missing-Info F1 | Entropy Collapse % | `Naive-Cascade` (`c=1.5`) | `Joint-Greedy` (`c=1.5`) | `Calibrated-Cascade` (`c=1.5`) | **`Tri-VoI-L2D` (`c=1.5`)** | `Calibrated-Cascade` (`c=2.5`) | **`Tri-VoI-L2D` (`c=2.5`)** | Severe `Cap=1/3` (`Tri` vs `Calib`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `Qwen/Qwen3.5-35B-A3B` | 0.8483 | 77.78 | 3.120 | 5.073 | 4.881 | **5.030** | 4.484 | **4.985** | **3.764** vs 3.168 |
| `deepseek-ai/DeepSeek-V3.2` | 0.8946 | 96.11 | 3.330 | 5.391 | 5.550 | **5.514** | 5.075 | **5.238** | **4.086** vs 3.548 |
| `zai-org/GLM-4.5-Air` | 0.8883 | 94.44 | 2.680 | 3.776 | 4.062 | **4.333** | 3.467 | **3.906** | 2.421 vs 2.802 |
| `Qwen/Qwen3-8B` | 0.2157 | 87.22 | 2.937 | 3.589 | 3.616 | **3.742** | 3.549 | **3.742** | **1.714** vs 0.368 |
| `tencent/Hy4-preview` | 0.9831 | 95.00 | 3.441 | 5.390 | 5.833 | **5.859** | 5.350 | **5.411** | **4.476** vs 3.927 |

- **Statistical Significance (`Qwen3.5-35B-A3B`, `N=360`, Holm-Bonferroni corrected)**:
  - Beats `Direct-ACT`, `ADEPT`, `DREAM`, and `Naive-Cascade` with `Boot p_Holm = 0.0014` (`McNemar p_Holm <= 6.25e-4`) and **6/6 wins** across 6 disjoint 60-ticket episodes (`p_Holm <= 0.0086`).
  - Under moderate user burden (`c_ask=2.5`), beats `Calibrated-Cascade` by **+0.502** (`95% Block CI [+0.09, +0.89]`, `Boot p_Holm = 0.0426`; **5/6** episodes).
  - Under `High (Cap=1)` and `Severe (Cap=1/3)` congestion, `Tri-VoI-L2D` achieves **4.743 / 3.764**, outperforming `Calibrated-Cascade` (**4.618 / 3.168**), `Joint-Greedy` (**3.986 / 2.235**), and `w/o Queue-Externality` (**4.613 / 2.905**).

### 2. Multi-Paragraph Noisy Workplace Benchmark + Imperfect LLM User (`N=360`)
- **`Qwen/Qwen3.5-35B-A3B`**: `Tri-VoI-L2D` reaches **3.553** (`c_ask=1.5`, `Boot p_Holm = 0.0048` vs `Naive-Cascade` 2.343) and **3.544** (`c_ask=2.5` vs `Calibrated-Cascade` 2.948), winning **6/6** independent episodes.
- **`tencent/Hy4-preview`**: `Tri-VoI-L2D` reaches **4.397** (`c_ask=1.5`, `Boot p_Holm = 0.0014` vs `Naive-Cascade` 2.666) and **4.104** (`c_ask=2.5` vs `Calibrated-Cascade` 3.803).

### 3. Theory, Negative Control, $K \times T$ Stress Test & Structural Robustness
- **Fixed-State Slope & Negative Control**: At $p_{\text{resolve}} = 0.00$, empirical slope $\partial\text{EVOI}/\partial Q = \mathbf{0.0000}$ (0.0% Q4 ASK); at $p_{\text{resolve}} \in \{0.25, 0.50, 0.75, 1.00\}$, empirical slopes **`0.2125, 0.4250, 0.6375, 0.8500`** match $p_{\text{resolve}}\gamma$ ($\gamma=0.85$) to 4 decimal places. At $p_e=0.92$, Q4 ASK flips sharply from **0.0%** ($Q=0$) to **83.3%** ($Q \ge 1$).
- **Entropy-Collapse $K \in \{3,5,10\} \times T \in \{0.7, 1.0\}$ Stress Test**: Even at 10× sampling ($K=10, T=1.0$), **37.50%** (`Qwen3.5-35B-A3B`) and **76.67%** (`Hy4-preview`) of underspecified cases remain collapsed at $H=0$.
- **56-Point 2D Operating Phase Map (`7 service rates × 8 ask costs`)**: `Tri-VoI-L2D` matches or beats `Calibrated-Cascade` on **51/56** cells and `Joint-Greedy` on **38/56** cells.
- **5-Fold Cross-Validation (420 items/model), LODO Zero-Shot Domain Transfer & Bursty/3-Expert Shift Stress**: Consistent strong performance across 5-fold CV (`Hy4-preview` `5.918±0.486`, `Qwen3.5` `5.564±0.551`), bursty Poisson arrival waves (**4.629** vs `Calibrated-Cascade` 4.569 / `Joint-Greedy` 4.374), and 3-expert shift-fatigue pools (**5.426**, lowest queue wait **0.13**).

---

## Repository Structure
- `PAPER_STORY_AND_STATUS.md` — Full paper narrative, motivations, method, and experimental audit
- `benchmark.py` — 4-domain Cooperative Decision-Making benchmark (`Q1`–`Q4` Conditional Escalation, `N=60 cal + 360 test`)
- `controller.py` — `HeldOutCalibrator`, `DiscountedBetaTracker`, two-stage Bellman EVOI controller, Naor queuing externality, and fixed-state sweeps
- `llm_client.py` — SiliconFlow OpenAI-compatible client with SQLite incremental caching & backoff
- `run_all.py` — Phase 2 `N=360` + 6-episode benchmark runner, statistical tests (`Block Bootstrap`, `McNemar`, `Holm-Bonferroni`), and Figures 1–4
- `overnight/` — 5-stage extended evaluation suite (`run_overnight.py`, `overnight_lib.py`: 5-model `N=360`, noisy long-form benchmark + imperfect LLM user, $K \times T$ stress test, 2D phase map, 5-fold CV, LODO, bursty/3-expert stress, and Figures 5–9)
- `results/` — Formal metrics JSON/Markdown/LaTeX tables and PDF/PNG figures (`results/report.md`, `results/overnight/OVERNIGHT_MASTER_REPORT.md`, `results/figs/`, `results/overnight/figs/`)

## Quick Reproduction
```bash
# Offline structural & statistical dry-run verification (no API key required)
python run_all.py --dry-run
python overnight/run_overnight.py --dry-run --stages 1,2,3,4,5

# Live evaluation (reads SiliconFlow key from API_KEY environment variable)
python run_all.py --workers 4
python overnight/run_overnight.py --stages 1,2,3,4,5 --workers 2 --pace-sec 0.25
```
