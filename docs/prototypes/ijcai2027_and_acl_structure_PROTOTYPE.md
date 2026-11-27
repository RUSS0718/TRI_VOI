# PROTOTYPE — Throwaway Structural Draft for Review

> **Status:** `PROTOTYPE` (Throwaway structural specification on branch `prototype/ijcai2027-paper-structure`; never merge directly into `main`).  
> **Parent Ticket:** [Draft IJCAI 2027 7-page paper structure and ACL Reframing Contingency Spec](https://github.com/RUSS0718/TRI_VOI/issues/19)  
> **Wayfinder Map:** [Wayfinder Map: Tri-VoI-L2D IJCAI 2027 Submission Freeze (2027-01-05) & ACL Contingency](https://github.com/RUSS0718/TRI_VOI/issues/10)  
> **Canonical Vocabulary:** Follows [GLOSSARY.md](https://github.com/RUSS0718/TRI_VOI/blob/main/GLOSSARY.md) (`Requester`, `Specialist`, `Benchmark v1-Clean`, `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri`, `IJCAI Freeze`, `ACL Reframing Contingency Branch`).

---

## Part 1 — IJCAI 2027 7-Page Double-Column Paper Outline (`ijcai27.sty`)

- **Venue & Deadlines:** IJCAI 2027 Main Track — 7 pages body + unlimited references + technical appendix (Abstract Lock: **2027-01-04**; Internal **IJCAI Freeze**: **2027-01-05**; Buffer Window: **2027-01-06 to 2027-01-10**; Full Paper DDL: **2027-01-11**).
- **Working Title:** *Ask the Requester or Defer to the Specialist? Information-Source Disentanglement and Queue-Relief Deferral for Three-Way Human-AI Decision-Making*

### 1.1 Section-by-Section & Page-Budget Plan (`Total Body = 7.00 Pages`)

| Section | Budget (Pages) | Cumulative Page Range | Core Narrative, Claims & Allocated Visuals |
| :--- | :---: | :---: | :--- |
| **Title + Abstract + Sec. 1 Introduction** | **1.10 p** | `p.1.00 – p.2.10` | Triadic cooperative problem; Binary Routing Fallacy (`{ACT, ASK}` vs. `{ACT, DEFER}`); High-Consensus Entropy Collapse ($H=0$ even at $K=10, T=1.0$); Three-Regime Queue-Relief coupling; `RQ1–RQ3`; **`Fig 1` (Two-column teaser: 3-way topology + Entropy Collapse + Queue-Relief preview, ~0.32 p)**. |
| **Sec. 2 Triadic Problem Formulation & Related Work** | **0.85 p** | `p.2.10 – p.2.95` | Formal triadic MDP / state $(x_t, Q_t, \hat{p}_{e,t}, \hat{\rho}_t)$ separating **Requester** ($U_{\text{spec}}$) vs. **Specialist** ($U_{\text{epist}}$); three-outcome post-`ASK` partition $(p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}})$; contrast against single-channel VoI (`ACL 2026 VoI`), congested-expert L2D (`KL-PBD`), and entropy proxies (`ADEPT`/`DREAM`). |
| **Sec. 3 `Tri-VoI-L2D` Controller** | **1.15 p** | `p.2.95 – p.4.10` | **3-module controller**: (1) Slot-disentangled diagnosis + Hierarchical Empirical Bayes held-out calibration (`k=2.0, k_fine=3.0`); (2) Congestion-gated Naor externality (`q_m >= C_m` or `rho_m >= 0.6`), Two-Stage Bellman EVOI gating, and optional rework-aware $U_{\text{defer}}^{\text{rework}}$; (3) 2-line Discounted Beta-Bernoulli tracker ($\lambda=0.92$). **`Algorithm 1` (Compact 18-line pseudocode box, ~0.35 p)**. |
| **Sec. 4 Theoretical Analysis (Three-Theorem Package)** | **1.00 p** | `p.4.10 – p.5.10` | Compact statements + intuition for **Theorem 1 & Corollary 1** (General 3-outcome three-regime slope $+p_{\text{res}}\gamma / -p_{\text{ovr\_after}}\gamma / 0$ + rework $Q_1^*$ left shift), **Theorem 2 & Corollary 2** (1-Lipschitz miscalibration bound $\Delta R(2\epsilon_p+\epsilon_\pi)$ + $O(\sqrt{\log(1/\delta)/n_b}+k/(n_b+k))$ scaling), and **Theorem 3 (Parts A–B)** (Foster-Lyapunov capacity expansion + $\tau_{\text{ask}}^{\max}$ latency bound). **`Fig 2` (Unified 3-panel theory verification: 200-seed MC rollout + GLM $k/n_{\text{cal}}$ Theorem-2 curve + $\lambda_{\text{tot}}/\tau_{\text{ask}}$ stability, ~0.33 p)**. |
| **Sec. 5 Experimental Setup & Benchmarks** | **0.55 p** | `p.5.10 – p.5.65` | Summary of all four evaluation suites (`Benchmark v1-Clean` `N=420`, `Benchmark v1-Noisy` 15% friction `N=420`, `Benchmark v2 (Implicit-Override)` `N=240`, `Public-Tri` from `CC-BY-4.0` `stellalisy/mediQ` `N=300` with frozen manifest); 5 models across 4 families (`DeepSeek-V3.2`, `Hy4-preview`, `Qwen3.5-35B-A3B`, `GLM-4.5-Air`, `Qwen3-8B`); zero-4-bit Colab BF16/FP8 + SiliconFlow compute parity; paired block bootstrap + McNemar + Holm-Bonferroni protocol. |
| **Sec. 6 Results & Structural Diagnostics** | **1.90 p** | `p.5.65 – p.7.55` -> tighten to `p.6.55` | **`Table 1` (`Benchmark v1-Clean/Noisy` main comparison vs. Naor-matched baselines + `ACL 2026 VoI` + `KL-PBD/KL-PBD+ (Warm)`, ~0.38 p)**; **`Table 2` (Cross-benchmark generalization: `Benchmark v2` 4-model live + `Public-Tri` `MediQ` 3-model live, ~0.34 p)**; **`Fig 3` (2D Phase Map `Cap x c_ask` + `Public-Tri` Pareto frontier, ~0.30 p)**; **`Fig 4` (Ablation, multi-model 5-bar comparison across 4 families, LODO transfer, drift & Stage 5f rework $U_{\text{defer}}^{\text{rework}}$, ~0.35 p)**; explicit transparent boundary reporting (`Qwen3-8B` low F1 floor, `GLM-4.5-Air` small-cal variance). |
| **Sec. 7 Conclusion, Limitations & Ethics** | **0.45 p** | `p.6.55 – p.7.00` | Synthesis of findings, honest scope boundaries (frozen `k=2.0`, simulated Requester/Specialist protocols, clinical triage rubric disclaimer), and anonymized reproducibility statement. |

*(Strict page-budget control: target `Sec 1–6` at `6.55 p` and `Sec 7` at `0.45 p` so total main body lands cleanly at `<= 7.00 p`, zero overflow risk.)*

### 1.2 Appendix Organization (`Appendix A–C`, Unlimited Pages Beyond p.7)

- **Appendix A — Complete Proofs & Derivations**: Full line-by-line proofs of **Theorem 1**, **Corollary 1**, **Theorem 2 & Finite-Sample Corollary**, and **Theorem 3 (Foster-Lyapunov + $\tau_{\text{ask}}$ Jensen/Convexity Bound)**, plus equivalence audit with `controller.py`.
- **Appendix B — Reproducibility, Model Routing & Benchmark Specifications**: Zero-quantization-confound Colab `vLLM` Two-Wave SQLite protocol, SHA-256 cache schema, exact `KL-PBD` & `ACL 2026 VoI` adaptation details, and frozen `Public-Tri` `MediQ` **Clinical Triage Scope Rubric** (`data/public_tri_mediq_manifest.json`).
- **Appendix C — Extended Experimental Tables**: Full 56-point 2D phase sweep, 6-episode per-stream statistics with exact McNemar/Holm-$p$, $K \in \{3,5,10\} \times T \in \{0.7,1.0\}$ entropy-collapse tables, Stage 5e (`328/360`), Stage 5f rework ablation ($U_{\text{defer}}^{\text{rework}}$), Token/Latency Pareto, and Error Taxonomy.

### 1.3 `2027-01-04` Abstract Lock Checklist (6 Hard Gates)

Before submitting the abstract on **2027-01-04**, all 6 conditions must be verified true:
1. **Verified Multi-Benchmark Table Numbers**: `Benchmark v1`, `Benchmark v2 (Implicit-Override)`, and `Public-Tri` (`MediQ`) live runs completed and audited against raw JSON outputs (no placeholder projections).
2. **External Baselines Closed**: Both `ACL 2026 VoI` (`2026.acl-long.1987`) and `KL-PBD / KL-PBD+ (Warm)` (`arXiv:2610.15128`) evaluated across all target benchmarks.
3. **Theory Numbers Matched**: Companion verification numbers (200-seed MC slopes, `GLM` $k/n_{\text{cal}}$ curve, $\lambda_{\text{tot}}$ capacity expansion, and $\tau_{\text{ask}}^{\max}$ inflection) generated offline and cross-checked with exact Theorem 1–3 statements.
4. **Canonical Terminology Lock**: Paper title and abstract strictly obey [GLOSSARY.md](https://github.com/RUSS0718/TRI_VOI/blob/main/GLOSSARY.md) (`Requester`, `Specialist`, 5 models across 4 families, zero legacy acronym matches).
5. **Dual-Submission Cleanliness**: Abstract submitted **exclusively to IJCAI 2027** (zero parallel submission to ACL 2027 ARR `2027-01-04`).
6. **Boundary Honesty**: Abstract claims explicitly respect empirical boundary conditions (non-trivial diagnostic F1 prerequisite and queue-regime scope).

---

## Part 2 — ACL Reframing Contingency Spec (Strictly Non-Dual-Submission)

> **Activation Policy:** Held **strictly in reserve** for a post-IJCAI ACL ARR cycle **only if** IJCAI 2027 is declined or intentionally not pursued. Never submit overlapping content concurrently.

### 2.1 Narrative Pivot & Target Venue Format (`acl_latex.sty`, 8 Pages + Mandatory Limitations)

- **Pivot Angle:** Reframe from *queue-theoretic AI decision control* to **NLP Pragmatic Underspecification, LLM Faithfulness Under Missing Information, and Interactive VoI Communication**:
  - Center **High-Consensus Entropy Collapse** (`H=0` even at $K=10, T=1.0$ across 4 LLM families) as a core *LLM uncertainty-estimation failure mode* rather than just an operations motivation.
  - Frame `Tri-VoI-L2D` as an **information-source-disentangling dialogue policy** connecting computational pragmatics (cooperative clarification) with resource-bounded expert routing.
- **Contingency Working Title:** *When LLMs Guess Unanimously in the Dark: Disentangling Pragmatic Underspecification from Expert-Only Authority in Interactive Decision-Making*
- **8-Page Allocation Plan:**
  - `p.1–2`: Introduction emphasizing LLM pragmatic overconfidence & Entropy Collapse (`Fig 1` teaser + `CLAMBER` / `MediQ` framing).
  - `p.3–4`: Method reframed around LLM diagnostic prompting, Empirical Bayes calibration against uncalibrated sampling variance, and two-stage Bellman communicative VoI (compressing queuing proofs into a compact subsection `0.5 p`, moving full Foster-Lyapunov algebra to Appendix).
  - `p.5–7`: NLP-centric evaluation across `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri` (`MediQ`), **plus CLAMBER auxiliary ambiguity diagnostic** and deep linguistic error analysis (what linguistic cues cause spurious default completion across model families).
  - `p.8`: Human Evaluation / Error Analysis + Conclusion; followed by uncounted mandatory **Limitations** section + Ethics Statement.

### 2.2 Explicit ACL-Only Delta Requirements (Beyond the `2027-01-05` IJCAI Freeze)

If this contingency branch is ever activated after IJCAI, execute these **3 incremental NLP deltas**:
1. **Lightweight Human Evaluation (`n = 60` stratified items)**: 2–3 annotators scoring (a) *clarification question specificity/naturalness*, (b) *gold disentanglement validity* on `Benchmark v2` & `Public-Tri`, and (c) inter-annotator agreement ($\kappa$).
2. **Multi-Turn `MediQ` Extension (`T <= 2` or `T <= 3` turns)**: Allow up to 2 clarification turns on `Public-Tri` (`MediQ`) vs. full multi-turn `ACL 2026 VoI` (`VOI_communication`) to satisfy ACL expectations on conversational depth.
3. **Linguistic Prompt & Default-Completion Error Analysis**: Systematic breakdown of pragmatic implicature triggers and sycophancy/default-filling patterns across Qwen, DeepSeek, Hy4, and GLM families (paired with auxiliary `CLAMBER` ambiguity-detection comparison).

---

## Part 3 — Cross-Ticket Traceability Matrix

| Paper Element | Settled Wayfinder Decision Source | Downstream Execution Issue |
| :--- | :--- | :--- |
| **Core Model Split (`Colab Ultra 5x` + `SiliconFlow`) & `Benchmark v2` 240 Matrix** | [Decide Colab Ultra 5x vs SiliconFlow model split and Benchmark v2 live run matrix](https://github.com/RUSS0718/TRI_VOI/issues/16) | [[P1] External validity roadmap: independent public datasets, benchmark_v2 live run, and IJCAI/ECAI 2027 plan](https://github.com/RUSS0718/TRI_VOI/issues/5) |
| **`Public-Tri` (`MediQ` 60+240, Clinical Triage Rubric, `FactSelect` 15% Friction)** | [Audit candidate public datasets (CLAMBER, MedMCQA, MediQ) for Public-Tri adaptation](https://github.com/RUSS0718/TRI_VOI/issues/11) & [Select independent public dataset and lock Public-Tri Q1-Q4 construction protocol](https://github.com/RUSS0718/TRI_VOI/issues/17) | [[P1] External validity roadmap: independent public datasets, benchmark_v2 live run, and IJCAI/ECAI 2027 plan](https://github.com/RUSS0718/TRI_VOI/issues/5) |
| **`ACL 2026 VoI`, `KL-PBD`, `GLM` Variance Ablation & $U_{\text{defer}}^{\text{rework}}$** | [Verify external baseline papers (ACL 2026 VoI and arXiv:2610.15128 KL-PBD) and extract exact algorithms](https://github.com/RUSS0718/TRI_VOI/issues/12) & [Lock external baseline faithful-reproduction specs, GLM calibration shrinkage, and U_defer rework extension](https://github.com/RUSS0718/TRI_VOI/issues/18) | [[P1] Implement published external baselines (ACL 2026 VoI & arXiv:2610.15128 KL-PBD) and GLM small-calibration variance mitigation](https://github.com/RUSS0718/TRI_VOI/issues/9) |
| **Theorem 1 + Corollary 1, Theorem 2, Theorem 3 (Parts A–B) & Appendix A Proofs** | [Lock the three-theorem mathematical specification for the IJCAI 2027 theory package](https://github.com/RUSS0718/TRI_VOI/issues/15) | [[P1] Theoretical deepening: cross-ticket dynamic queue control, calibration-error sensitivity bounds, and ASK response delay](https://github.com/RUSS0718/TRI_VOI/issues/8) |
| **Canonical Domain Language (`Requester`, `Specialist`, Suite Names)** | [Sync local workspace to remote main commit 1707b98 and initialize canonical GLOSSARY.md](https://github.com/RUSS0718/TRI_VOI/issues/14) | [GLOSSARY.md](https://github.com/RUSS0718/TRI_VOI/blob/main/GLOSSARY.md) |
| **Colab Hardware Provisioning Prerequisite (`gcloud` + `agy-colab-train` + `colab_specs`)** | [Install and authenticate agy-colab-train and probe Colab Ultra 5x GPU specs](https://github.com/RUSS0718/TRI_VOI/issues/13) | Unlocks offline batch inference for `Benchmark v2` & `Public-Tri` |
