# Tri-VoI-MDP: Two-Stage Value-of-Information Control and Bellman Congestion Pricing for Three-Party Collaborative Decision Making

**Unified Primary Method (IJCAI 2027)**: **`Tri-VoI-MDP` (`Deploy-Self`, 100% Zero-Oracle)** — a deployable two-stage sequential control and queueing framework coordinating a **User**, a fast LLM **Generalist**, and a capacity-constrained **Specialist** (`LLM Specialist` or simulated override model).

Compared with its pre-registered 5-call static predecessor (`Tri-VoI-L2D`, preserved intact for lineage comparison), `Tri-VoI-MDP`:
1. Replaces 5-call self-consistency probing with a **2-Call coarse sufficient statistic**, cutting realized LLM token usage by **53.7%–57.8%** across `Benchmark v1-Clean` and `Benchmark v2`.
2. Replaces simulator `oracle_revealed` post-clarification feedback with a strictly local deployable **`post_missing` self-signal** (falling back exclusively to `obs_missing`, never reading simulator ground truth).
3. Replaces myopic Naor static queue cutoffs with **finite-horizon periodic-drain Bellman dynamic programming shadow prices** ($\Delta V(q)$) under server congestion (**Theorem 4 & Proposition 5**).

---

## Method Lineage & Theoretical Guarantees

Full formal statements and machine-checked numerical companions are maintained in [ijcai2027_three_theorem_proofs.md](file:///d:/research/cscwd_tri_voi/docs/theory/ijcai2027_three_theorem_proofs.md) and [GLOSSARY.md](file:///d:/research/cscwd_tri_voi/GLOSSARY.md).

| Component | Role in Paper | Scope & Guarantees |
|:---|:---|:---|
| **`Tri-VoI-L2D`** (`Theorems 1–3`) | Foundational 5-call + Naor predecessor | Two-stage $\mathcal{O}(\log N)$ monotone clarification/deferral policies & submodular value-of-information bounds; frozen unmodified (`[5.220, 5.142, 3.792]` on `v1-Clean / Qwen3.5`). |
| **`Tri-VoI-MDP`** (`Theorem 4 & Proposition 5`) | **Sole Primary Method (`Deploy-Self`, Zero-Oracle)** | Finite-horizon Bellman DP optimal abandonment monotone threshold in queue length $q$ when $\gamma > 0$ (`Theorem 4`) + bounded minimax worst-case regret over 2-call coarse partition & congestion/cost regimes (`Proposition 5`). |
| **`MDP (Oracle-NoInfo Diag)`** | **Diagnostic Ablation Only** | Legacy variant reading `not oracle_revealed` after `ASK`; isolated strictly to explicit diagnostic columns (`Sections 9A, 9B, 10`) and excluded from primary claims. |

---

## Key Empirical Findings (100% Zero-Oracle Single Source of Truth)

All headline figures are automatically compiled into [ijcai2027_canonical_tables.md](file:///d:/research/cscwd_tri_voi/results/ijcai2027_canonical_tables.md) and [ijcai2027_canonical_tables.tex](file:///d:/research/cscwd_tri_voi/results/ijcai2027_canonical_tables.tex):

1. **Universal Severe-Congestion `+MDP` Mechanism Gains (`Section 0A`, `Cap=1/3, c=1.5`, 13 Suites)**:
   Upgrading static Naor thresholds to Bellman DP shadow pricing yields strictly positive gains across **13/13** suites for `CC+Naor -> CC+MDP` (`mean +0.355`) and `ACL+FD+Naor -> ACL+FD+MDP` (`mean +0.443`), and **12/13** suites for `Tri-VoI-L2D -> Tri-VoI-MDP` (`mean +0.429`) and `JG+Naor -> JG+MDP` (`mean +0.429`).
2. **Lowest Cross-Regime Minimax Regret (`Proposition 5`, `Section 0B`)**:
   - Across **26 matched `2C-Self` cells** (`13 suites × {Cap=2, Cap=1/3}`, `c=1.5`, zero Oracle, equal DP pricing): `Tri-VoI-MDP` achieves the highest mean utility (**3.617**), most wins (**10/26**), lowest mean regret (**0.099**), and lowest worst-case regret (**0.464**, vs. `CC+MDP` `0.625`, `ACL+FD+MDP` `0.764`, `JG+MDP` `0.820`).
   - Across **40 `c_ask` cells** (`c_ask ∈ {0.5, 1.5, 3.0, 4.5}`): `Tri-VoI-MDP` achieves the highest mean utility (**3.380**) and lowest worst-case regret (**0.526**, vs. `ACL+FD+MDP` `0.830`, `JG+MDP` `0.927`, `CC+MDP` `1.330`).
   - Across **56-cell 2D sensitivity envelopes (`Table 3`)**: `Tri-VoI-MDP` ranks `#1` in **5/7** suites (`v1-Clean Qwen3.5` **4.623** / worst regret **0.404**, `v2 DeepSeek-V3.2` **4.494** / **0.195**, and all three `Public-Tri` suites).
3. **Token Efficiency (`Section 6`)**:
   Realized LLM token usage drops **53.7%–57.8%** from 5-call `Tri-VoI` (`4952 -> 2111` `-57.4%` on `v1-Clean Qwen3.5`; `4441 -> 2056` `-53.7%` on `v1-Clean DeepSeek-V3.2`; `4967 -> 2097` `-57.8%` on `v2 Qwen3.5`) while preserving or improving net utility.
4. **Pre-Registered Organ-System-Disjoint Blind Clinical Evaluation (`MediQ-Test-Disjoint`, `Section 10`)**:
   Pre-registered at Git tag `prereg-mediq-test-disjoint` (`N_cal=60`, `N_test=180`, `PY_DIFF_LEN=0`, `0` unparsed fallbacks, live API spend `¥3.55`). In cross-group real-LLM Specialist delegation (`Qwen3.5-35B-A3B -> DeepSeek-V3.2`), `Tri-VoI-MDP` achieves **4.266** (`Cap=2, c=1.5`, `81.7%` medical acc / `85.1%` scope-escalation recall, `1823` tokens) and **3.588** (`Cap=1/3, c=1.5`, beating `Tri-VoI-L2D` by `+0.679`, `p=0.0137`; `+0.785` at `c=3.0`, `p=0.0057`), while all four `+MDP` upgrades outperform static Naor under severe congestion (`p < 0.05`).
5. **Honest Empirical Boundaries (`Sections 8, 9D, 9E, 10B`)**:
   - Under matched `2C-Self` equal-budget DP pricing (`Section 9E`), `Tri-VoI-MDP` does **not** achieve single-cell statistical superiority over individual strong `Base+MDP (2C-S)` policies (`p=0.2036–0.9810`), and on `v2 (Qwen3.5)` / `v2 (Hy4-preview)` / `MediQ-Test-Disjoint Setup B`, `CC+MDP (2C-S)`, `ACL+FD+MDP (2C-S)`, or `JG+Naor` lead specific cells.
   - `Benchmark v3` (`Section 8`, `prereg-blind-v3`) has disclosed attribute-template overlap (`50.0%` normalized prose, `35.4%` exact triple overlap across 8 templates/domain); strict zero-overlap slices (`72` and `40` items in `Section 9D`) show parity vs. matched `2C-S` baselines (`p=0.1546–1.0000`).

---

## Repository Structure

```text
.
├── controller.py                        # Core Tri-VoI-MDP (Deploy-Self), Tri-VoI-L2D, and baseline policies
├── external_baselines.py                # Matched 2C-Self & 5C external baselines (KL-PBD+, ACL-2026-VoI, JG, CC)
├── probe_profiles.py                    # Cached/live LLM Generalized Probing & post_missing self-signal extractor
├── llm_client.py                        # SiliconFlow OpenAI-compatible client, cost tracker & SQLite cache
├── benchmark.py / public_tri.py         # Benchmark v1-Clean/v1-Noisy, v2, and Public-Tri (MediQ Scope Triage)
├── benchmark_v3.py                      # Unseen template domains (Benchmark v3) & MediQ-Holdout splits
├── mediq_test_disjoint.py               # Organ-system-disjoint, length-matched MediQ-Test-Disjoint builder
├── generate_ijcai_tables.py             # Generates verified results/ijcai2027_canonical_tables.{md,tex}
├── verify_theory_package.py             # Zero-API verification of Theorems 1–4 & Proposition 5
├── run_deploy_self_canonical_upgrade.py # Offline zero-Oracle upgrade across Sections 0–8
├── run_attribution_and_dedup_audit.py   # Offline Section 9 (matched 2C-S attribution, post_missing audit, v3 slices)
├── run_mediq_test_disjoint_eval.py      # Section 10 runner (MediQ-Test-Disjoint)
├── docs/theory/                         # Proofs & theoretical validation JSON/MD packages
└── results/
    ├── README.md                        # Guide to active vs. archived artifacts
    ├── ijcai2027_canonical_tables.md    # Primary Markdown tables (Sections 0–10)
    ├── ijcai2027_canonical_tables.tex   # LaTeX tables matching Sections 1–8
    ├── attribution_and_dedup_audit.md   # Detailed Section 9 offline attribution & signal audit
    ├── mediq_test_disjoint_eval.md      # Pre-registered MediQ-Test-Disjoint clinical report
    ├── deploy_self_canonical_upgrade.json
    └── archive/                         # Superseded pre-MDP & Oracle-diagnostic stage reports
```

---

## Reproducing Offline Verification & Canonical Tables (`live=0`, `¥0.00`)

All code/table/theory consistency checks run completely offline against the cached SQLite responses and verified JSON outputs:

```bash
python verify_theory_package.py
python run_deploy_self_canonical_upgrade.py
python run_attribution_and_dedup_audit.py --offline-only
python generate_ijcai_tables.py
```

For environment setup and API execution details (when re-running `MediQ-Test-Disjoint` live against SiliconFlow), see [mediq_test_disjoint_eval.md](file:///d:/research/cscwd_tri_voi/results/mediq_test_disjoint_eval.md).
