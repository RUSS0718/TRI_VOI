# TRI-VOI: Three-Way Human-AI Cooperative Decision-Making

**Ask the User or Defer to the Expert? Information-Source Disentanglement and Queue-Relief Deferral for Three-Way Human-AI Cooperative Decision-Making**

---

## Unified Research Thesis

> Effective human-AI cooperation requires not only calibrated uncertainty estimation, but also identifying **who holds the information needed to resolve that uncertainty** (the frontline requester vs. an internal domain specialist) and **coupling the Value of Information (EVOI) of asking the user with dynamic specialist queue congestion and reliability drift**.

### Three Linked Motivations
1. **Binary Routing Fallacy**: Existing frameworks split into `{ACT, ASK}` (clarifying uncertainty without specialist escalation) vs. `{ACT, DEFER}` (deferring to experts while assuming complete user inputs). Real-world triage/dispatch flows face orthogonal **Specification Uncertainty** (only the requester knows) and **Policy/Override Uncertainty** (only internal specialists know).
2. **High-Consensus Entropy Collapse**: On natural-language underspecified tickets without explicit placeholder spoilers, modern instruction-tuned LLMs deterministically guess a default assumption ($H=0$, **77.8%–96.1%** across 5 LLMs from 4 families). Even at $K=10$ samples and temperature $T=1.0$, **37.5%–76.7%** of underspecified cases still collapse to $H=0$, disabling entropy/consensus proxy triggers (`Entropy-ASK (Proxy)`, `Consensus-DEFER (Proxy)`).
3. **Three-Regime Piecewise Queue-Relief VoI Theorem**: Under Conditional Escalation (where user clarification resolves routine cases with probability $\hat{p}_{\text{resolve}}$ and confirms confidential escalation with $1-\hat{p}_{\text{resolve}}$), two-stage Bellman optimization exhibits three analytical queue regimes with closed-form boundaries $Q_1^*$ (`ACT`/`DEFER`) and $Q_2^*$ (`OVR`/`DEFER`):
   - **Regime I ($Q < Q_1^*$, `DEFER`-dominant)**: $\partial\,\text{EVOI}(\text{ASK}; Q, \hat{p}_e)/\partial Q = \hat{p}_{\text{resolve}}\gamma > 0$ and $\partial\,\text{EVOI}/\partial \hat{p}_e = -\hat{p}_{\text{resolve}}(R_+ - R_-) < 0$ (asking acts as a **queue-relief valve**).
   - **Regime II ($Q_1^* \le Q < Q_2^*$, turnover)**: once $U_{\text{defer}} < U_{\text{act}}$, the baseline switches to `ACT` and the slope flips negative to $-(1-\hat{p}_{\text{resolve}})\gamma < 0$.
   - **Regime III ($Q \ge Q_2^*$, saturation)**: both baseline and post-ASK revert to `ACT` ($\partial\,\text{EVOI}/\partial Q = 0$).
   Combined with congestion-gated Naor queuing-externality shadow pricing ($Q_t \ge C_m \lor \hat{\rho}_t \ge 0.6$) and conjugate Discounted Beta-Bernoulli tracking, the controller smoothly balances user burden and specialist queue backlog.

---

## Method Architecture (`Tri-VoI-L2D`)
1. **Disentangled Source Diagnosis + Hierarchical Empirical Bayes Held-Out Calibration** (`HeldOutCalibrator`): extracts $(missing\_user\_info, potential\_override, H(x))$ in a single audit pass and calibrates a 3-outcome post-ASK partition $(p_{\text{resolve}}, p_{\text{ovr\_after}}, p_{\text{no\_info}})$ strictly on genuinely clarified held-out splits (zero test or fallback leakage).
2. **Resource-Coupled Two-Stage Bellman EVOI Controller + Queue Externality** (`controller.py`): jointly evaluates `ACT`, `ASK` (with Stage-2 `DEFER` lookahead), and `DEFER` priced with congestion-gated Naor externality $\gamma (Q_t + \hat{\rho}_t (Q_t + 1))$.
3. **Discounted Beta-Bernoulli Specialist Tracker** (`DiscountedBetaTracker`, $\lambda = 0.92$): conjugate closed-form adaptation to sudden, gradual, and shift-based specialist fatigue drift.

---

## Key Experimental Findings (26,039 Cached Live API Calls Across 5 LLMs / 4 Model Families)

### 1. Five-Model Generalization (`N=360` Test Tickets × 5 Seeds per Model, 4 Families)
| Backbone | Missing-Info F1 | Entropy Collapse % | `Naive-Cascade` (`c=1.5`) | `Joint-Greedy` (`c=1.5`) | `Calib-Cascade` (`c=1.5`) | **`Tri-VoI-L2D` (`c=1.5`)** | `Calib-Cascade` (`c=2.5`) | **`Tri-VoI-L2D` (`c=2.5`)** | Severe `Cap=1/3` (`Tri` / `Calib+Naor` / `w/oExt`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `Qwen/Qwen3.5-35B-A3B` | 0.8483 | 77.78 | 3.165 | 5.142 | 4.877 | **5.220** | 4.479 | **5.142** | 3.792 / 3.798 / 2.732 |
| `deepseek-ai/DeepSeek-V3.2` | 0.8946 | 96.11 | 3.305 | 5.391 | 5.550 | **5.523** | 5.075 | **5.259** | 4.062 / 4.080 / 3.470 |
| `zai-org/GLM-4.5-Air` | 0.8883 | 94.44 | 2.597 | 3.750 | 4.054 | **4.317** | 3.459 | **3.910** | 2.348 / 3.284 / 1.966 |
| `Qwen/Qwen3-8B` | 0.2157 | 87.22 | 2.682 | 3.589 | 3.711 | **3.731** | 3.644 | **3.731** | 1.714 / 1.836 / 0.245 |
| `tencent/Hy4-preview` | 0.9831 | 95.00 | 3.441 | 5.390 | 5.833 | **5.842** | 5.350 | **5.468** | 4.477 / 4.548 / 3.846 |

- **Statistical Significance & Fair Naor-Matched Comparison (`Qwen3.5-35B-A3B`, `N=360`, Holm-Bonferroni corrected)**:
  - Under `c_ask=1.5` (`Cap=2`), `Tri-VoI-L2D` achieves **5.220** (83.67% Acc), beating `Direct-ACT`, `Entropy-ASK (Proxy)`, `Consensus-DEFER (Proxy)`, and `Naive-Cascade` (`Boot p_Holm = 0.0018`, **6/6** disjoint episodes), and edging out `Joint-Greedy` (`5.142`), `Joint-Greedy+Naor` (`5.190`), `Calibrated-Cascade` (`4.877`), and `Calibrated-Cascade+Naor` (`4.867`).
  - Under moderate user burden (`c_ask=2.5`), `Tri-VoI-L2D` reaches **5.142**, beating `Calibrated-Cascade` (**4.479**, `Boot p_Holm = 0.0080`) and `Calibrated-Cascade+Naor` (**4.470**, `Boot p_Holm = 0.0080`; **5/6** episodes, `Strat Boot p_Holm = 0.0050`).
  - **Mechanism Disentanglement at `Cap=1/3`**: Adding Naor externality to `Calibrated-Cascade` (`Calibrated-Cascade+Naor`: **3.798**) closes the severe-congestion gap at low `c_ask=1.5`, proving that Naor shadow pricing drives queue-wait compression (`w/o Queue-Externality` drops to **2.732**), while `Tri-VoI-L2D` retains the EVOI edge over `Joint-Greedy+Naor` (**3.792 vs 3.188**) and dominates when both congestion and ask cost are elevated (`c_ask >= 3.0`, $\mu \le 0.33$).

### 2. Multi-Paragraph Noisy Workplace Benchmark + Imperfect LLM User (`N=360`)
- **`Qwen/Qwen3.5-35B-A3B`**: `Tri-VoI-L2D` reaches **3.503** (`c_ask=1.5`, `Boot p_Holm = 0.0096` vs `Naive-Cascade` 2.285) and **3.511** (`c_ask=2.5` vs `Calibrated-Cascade` 2.948); under `Cap=1/3`, `Tri-VoI-L2D` (**1.525**) beats `Joint-Greedy+Naor` (**1.295**) and `Calibrated-Cascade` (**0.396**), while `Calibrated-Cascade+Naor` reaches **1.614**.
- **`tencent/Hy4-preview`**: `Tri-VoI-L2D` reaches **4.380** (`c_ask=1.5`, `Boot p_Holm = 0.0018` vs `Naive-Cascade` 2.666) and **4.082** (`c_ask=2.5` vs `Calibrated-Cascade` 3.803).

### 3. Three-Regime Theory, Monte Carlo Rollout, $K \times T$ Stress & Structural Robustness
- **Three-Regime Theory & Genuine Monte Carlo Verification (`Q=0..10`, `N_MC=200`)**: At `p_e = 0.75` ($Q_1^* = 5.32$), analytical EVOI rises from `+0.779` ($Q=0$) to `+2.912` ($Q=6$) then turns over to `+1.316` ($Q=10$), mirrored by Monte Carlo realized-utility rollout (`-1.176 -> +1.000 -> +0.327`); across `p_e ∈ {0.75, 0.85, 0.92}`, Monte Carlo Defer-regime slopes (`0.4568±0.0137`, `0.4411±0.0065`, `0.4426±0.0036`) match theoretical $\hat{p}_{\text{resolve}}\gamma = 0.4512$ without circular formula fitting.
- **56-Point 2D Operating Phase Map (`7 service rates × 8 ask costs`, 8 methods)**: `Tri-VoI-L2D` is strict overall winner on **27/56** cells (vs runner-up `Calibrated-Cascade+Naor` 13/56, `Joint-Greedy` 8/56) and matches or beats `Calibrated-Cascade` (**53/56**), `Calibrated-Cascade+Naor` (**43/56**), `Joint-Greedy` (**46/56**), `Joint-Greedy+Naor` (**44/56**), and `w/o Queue-Externality` (**49/56**).
- **Decision-Relevant Subset (`328/360` test items excluding invariant Q2) & Realistic Specialist Stress (Stage 5e/5f)**: Excluding 34.3% decision-invariant Q2 tickets preserves `Tri-VoI-L2D`'s performance (`5.253` at `Cap=2, c=1.5`; `5.141` at `c=2.5`); under queuing-delayed feedback and unclarified-ticket rework (`Cap=1/3`), `Tri-VoI-L2D` (`3.561`) strongly outperforms `Joint-Greedy+Naor` (`2.823`) and `Calibrated-Cascade` (`2.765`).
- **Known Boundaries**: (1) `GLM-4.5-Air` underperforms `Calibrated-Cascade` at `Cap=1/3` (`2.348 vs 3.284`) due to small-sample `Q2` calibration variance (`n_cal=60`); (2) `Qwen3-8B` (`Missing F1=0.2157`) reflects small-model diagnostic failure (`Naive-Cascade` 2.434 > `Tri-VoI` 1.714 at `Cap=1/3`); (3) `build_cscw_benchmark_v2` (100% decision-relevant Q2 + zero explicit `"confidential"`/`"override"` strings) and public-dataset transfer remain queued for live API evaluation.

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
