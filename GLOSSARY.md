# Tri-VoI-MDP Terminology Glossary

> Canonical IJCAI 2027 vocabulary for our three-way human-AI cooperative decision-making and dynamic Value-of-Information (`EVOI`) + periodic-drain Bellman shadow-pricing framework.

## Methods & Variants

- **`Tri-VoI-MDP` (Primary Method — Deploy-Self / 100% Zero-Oracle)**:
  The unified deployable controller combining (i) a **2-Call** coarse sufficient statistic `(obs_missing, obs_pot_ovr)`, (ii) empirical-Bayes held-out calibration over a 3-outcome post-`ASK` partition using strictly self-reported `post_missing` (zero access to simulator `oracle_revealed`), (iii) Discounted Beta-Bernoulli specialist reliability tracking ($\lambda=0.92$), and (iv) finite-horizon periodic-drain 3-action Bellman DP congestion shadow pricing (**Theorem 4**). All main tables (`Sections 0–10` of [`results/ijcai2027_canonical_tables.md`](results/ijcai2027_canonical_tables.md)) report this zero-Oracle controller.
- **`Tri-VoI-L2D` (Predecessor / Ablation Baseline)**:
  The earlier 5-Call myopic threshold controller using three stochastic samples + greedy + source audit and a congestion-gated Naor linear proxy (**Theorems 1–3**). Retained as an explicit ablation/predecessor baseline.
- **`MDP (Oracle-NoInfo Diag)`**:
  Diagnostic ablation variant of `Tri-VoI-MDP` that reads simulator `oracle_revealed` on the post-`ASK` `no_info` branch (`controller.py` `use_self_mdp=False`). Reported exclusively in controlled diagnostic ablations (`Sections 9A/9B/10`), never as a deployable primary metric.

## Cooperative Roles & Mechanisms

- **Requester**: The frontline submitter who holds task-specific context omitted upfront (`ASK` target).
- **Specialist**: The capacity-constrained domain expert pool holding institutional escalation authority or higher diagnostic accuracy (`DEFER` target).
- **Information-Source Disentanglement**: Separating requester specification uncertainty (`missing_user_info`) from specialist policy/capability uncertainty (`potential_override`).
- **Four-Branch Abandonment Threshold ($Q_{\max}^{\text{defer}}$)**: The queue threshold above which `DEFER` is strictly dominated across Stage-1 `ACT` and all three Stage-2 `ASK` branches (`clear`, `ovr_after`, `no_info`), causing the Bellman shadow price $\Delta V$ to saturate to $0$ (**Theorem 4**).

## Evaluation Suites

- **`Benchmark v1-Clean`** & **`Benchmark v1-Noisy`**: `420`-item (`60 cal + 360 test`) 4-domain tri-party benchmarks across 5 LLMs; `v1-Noisy` adds workplace prose noise and a 15%-friction imperfect LLM requester.
- **`Benchmark v2 (Implicit-Override)`**: `240`-item (`60 cal + 180 test`) leakage-hardened suite with implicit escalation cues and 100% decision-relevant `Q2`.
- **`Benchmark v3`**: `192`-item (`48 cal + 144 test`) pre-registered (`prereg-blind-v3`) unseen-domain template suite (`LegalContract`, `BioSafetyLab`, `CloudSecOps`, `AviationDispatch`) with zero protocol code IDs; explicitly disclosed as having `50.0%` normalized prose template reuse between `cal` and `test`, paired with `72`-item and `40`-item strict zero-overlap slices.
- **`Public-Tri`**: `300`-item (`60 cal + 240 test`) clinical scope-of-practice adaptation from `stellalisy/mediQ` (`medqa_dev_convo.jsonl`).
- **`MediQ-Holdout`**: `176`-item (`48 cal + 128 test`) sample-disjoint clinical holdout from unused `medqa_dev_convo.jsonl` entries with natural continuous prose and real-LLM specialists.
- **`MediQ-Test-Disjoint` (Primary Disjoint Blind Benchmark)**: `240`-item (`60 cal + 180 test`) pre-registered (`prereg-mediq-test-disjoint`) blind benchmark built from official `medqa_test_convo.jsonl`, strictly enforcing `0` `dev`-split overlap, `0` clinical organ-system group overlap between `cal` and `test`, `0` upfront `Q4` cues, `4.86%` `Q1/Q2` prose-length match, and `0` JSON truncation fallbacks.
