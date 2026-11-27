# Primary-Source Audit of Candidate Public Datasets (`CLAMBER`, `MedMCQA`, `MediQ`) for `Public-Tri` Adaptation

**Ticket:** [`RUSS0718/TRI_VOI#11`](https://github.com/RUSS0718/TRI_VOI/issues/11)  
**Parent Roadmap:** [`RUSS0718/TRI_VOI#5`](https://github.com/RUSS0718/TRI_VOI/issues/5) (`[P1] External validity roadmap: independent public datasets, benchmark_v2 live run, and IJCAI/ECAI 2027 plan`)  
**Goal:** Evaluate three candidate public datasets against primary sources (official GitHub repositories, HuggingFace dataset APIs/cards, license files, and peer-reviewed papers) for constructing a **240–300 item `Public-Tri` benchmark** that non-circularly separates **Requester-held missing context** (`ASK`) from **latent Specialist-only authority/rules** (`DEFER`), without synthetic keyword spoilers.

---

## Executive Summary

| Dataset | License & Gating | Verified Size & Splits | Native `ASK` Separation (Requester-Held Facts + Gold Answer) | Native `DEFER` Hook (Specialist Authority / Override) | Verdict for `Public-Tri` (240–300 items) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MediQ** (`stellalisy/mediQ`, NeurIPS 2024) | **CC-BY-4.0** (`LICENSE` + HF); ungated | **2,545** MedQA items (`validation`: 1,272; `test`: 1,273) + **140** CRAFT-MD items (`all_craft_md.jsonl`) | **Native & complete.** Splits each case into initial presentation (`context[0]`) vs. withheld `atomic_facts` (`context[1:]`), paired with 4-way `answer_idx` and `FactSelectPatient` simulator. | **Partial hook (strongest available).** `validation` (`1,272` rows) provides empirical `patient.specialties`, `subspecialties`, and `specialty_prob` (`34/100` multi-specialty in sample); clinical escalation triggers sit inside `atomic_facts` (`Q4`). | **Primary Recommendation (Recommended).** Build `Public-Tri` (`60 cal + 240 test = 300`, or `60 + 180 = 240`) from `stellalisy/mediQ` `validation` + a pre-registered clinical subspecialty/escalation scope policy. |
| **MedMCQA** (`medmcqa/medmcqa`, CHIL 2022) | **MIT** (`LICENSE.md`) / **Apache-2.0** (HF card); ungated | **193,155** total (`train`: 182,822; `validation`: 4,183; `test`: 6,150 with `cop=-1` withheld) | **None.** Static single-turn entrance-exam MCQs; creating `ASK` requires artificial stem truncation/rewriting; `39/100` sampled `validation` rows are `choice_type="multi"` trivia; `24/100` are `Dental`. | **None.** Only coarse `subject_name` (`21` subjects) and sparse `topic_name` (`14/100` non-null in sample); no specialist policy or override structure. | **Reject for core `Public-Tri`.** Adapting static exam stems reintroduces synthetic rewriting/spoiler artifacts. |
| **CLAMBER** (`zt991211/CLAMBER`, ACL 2024) | **No `LICENSE` file** on GitHub (`license: null`); ACL paper CC-BY-4.0; ungated | **3,202** items in `clamber_benchmark.jsonl` (`require_clarification`: 1,601 yes / 1,601 no) | **Incomplete (missing gold outcomes).** Has `require_clarification` and `clarifying_question`, but **0/3,202** non-empty `context` and **no ground-truth final answer or Requester resolution**. | **None.** Open-domain general/entity ambiguity (`FD`, `LA`, `MC`); no domain rubric, specialist authority, or escalation ground truth. | **Reject for core `Public-Tri`.** Optional auxiliary sanity check *only* for binary ambiguity detection (`ASK` vs. `NO-ASK`). |

---

## 1. `Tri-VoI-L2D` Structural Requirements for `Public-Tri`

In `benchmark.py` of `RUSS0718/TRI_VOI`, the cooperative triadic routing problem evaluates `ANSWER` vs. `ASK` (to the **Requester**) vs. `DEFER` (to a capacity-constrained **Specialist**) across a 2×2 separation:
- **Sufficient context & AI-solvable (`Q1_clear_easy`)**: No missing Requester facts, no Specialist override → optimal action `ANSWER`.
- **Missing Requester-held context, AI-solvable after clarification (`Q2_underspec_easy`)**: Decisive intake information is held by the Requester; `ASK` resolves ambiguity and enables accurate `ANSWER`.
- **Clear context, unconditional Specialist-only rule/override (`Q3_clear_hard`)**: Initial query already reveals Specialist-scope/policy escalation; `ASK` has zero decision VoI → optimal action `DEFER`.
- **Missing Requester-held context + Conditional Specialist escalation (`Q4_resolvable` / `Q4_escalate`)**: Initial query hides whether a Specialist rule/override applies; the AI must `ASK` the Requester first, and only after observing the clarified fact can it route to `ANSWER` (`Q4_resolvable`) or `DEFER` (`Q4_escalate`).

For `Public-Tri` (`240–300` items) to establish external validity beyond templated tickets (`Benchmark v1` / `v2`), a candidate dataset should satisfy four criteria:
1. **Permissive license & direct programmatic download** (no gated access or manual portals).
2. **Verifiable ground-truth final outcomes** (`answer` / `label`) so post-`ASK` utility $U_{\text{AI}}(\text{post-ASK})$ is measurable, not just whether a question was asked.
3. **Natural initial-vs-withheld information split** (`Requester`-held facts not artificially deleted by ad-hoc keyword masking).
4. **Non-circular `DEFER` separation** (Specialist-only clinical/domain authority or escalation rules grounded in real domain metadata rather than synthetic `"confidential"`/`"override"` strings).

---

## 2. Candidate Dataset Audits (Primary-Source Verified)

### 2.1 CLAMBER (ACL 2024)

- **Primary Sources**:
  - GitHub repository: [`zt991211/CLAMBER`](https://github.com/zt991211/CLAMBER)
  - Direct data file: [`clamber_benchmark.jsonl`](https://raw.githubusercontent.com/zt991211/CLAMBER/main/clamber_benchmark.jsonl)
  - Paper: [ACL Anthology `2024.acl-long.578`](https://aclanthology.org/2024.acl-long.578/) ([arXiv:2405.12063](https://arxiv.org/abs/2405.12063))
- **License & Downloadability**:
  - Directly downloadable as a single raw JSONL file from GitHub with no gating; no official HuggingFace mirror was found via `huggingface.co/api/datasets?search=CLAMBER`.
  - **License gap**: The GitHub repository contains **no `LICENSE` file** (`license: null` via GitHub REST API; repository files are strictly `README.md`, `clamber_benchmark.jsonl`, and `taxonomy.png`). While the ACL Anthology paper text carries ACL's standard CC-BY-4.0 license, the benchmark file itself lacks an explicit dataset license.
- **Exact Schema, Size, and Splits** (verified by parsing all lines of `clamber_benchmark.jsonl`):
  - **Size**: **3,202** instances in a single benchmark split (contrary to secondary summaries claiming ~12,000; each line in `clamber_benchmark.jsonl` is a JSON-encoded string wrapping a JSON object).
  - **Fields**: `question` (`string`), `context` (`string`), `clarifying_question` (`string`), `require_clarification` (`0` or `1`), `category` (`FD`, `LA`, `MC`), `subclass` (`none`, `polysemy`, `co-reference`, `ICL`, `NK`, `what`, `whom`, `when`, `where`), plus baseline prediction columns (`predict_ambiguous`, `predict_is_ambiguous_response`, `predict_clarifying_question`).
  - **Exact Breakdown (`n = 3,202`)**:
    - `require_clarification = 1`: **1,601**; `require_clarification = 0`: **1,601**.
    - `FD` (Epistemic Misalignment): **800** (`400` req=1, `400` req=0; subclasses `NK`: 400, `ICL`: 400).
    - `LA` (Linguistic Ambiguity): **800** (`400` req=1, `400` req=0; subclasses `polysemy`: 400, `co-reference`: 400).
    - `MC` (Aleatoric Output): **1,602** (`801` req=1, `801` req=0; subclasses `none`: 801, `what`: 201, `whom`: 200, `when`: 200, `where`: 200).
    - Non-empty `context` count: **0 / 3,202**.
- **3-Way Cooperative Separation Assessment**:
  - **Structural blocker for `ASK` outcome scoring**: CLAMBER has **no ground-truth final answer** and **no simulated user/Requester profile** holding the answer to `clarifying_question` (e.g., `FD/NK`: *"What is the latitude of the habitat of inyidiidae?"*; `MC/what`: *"Give me a list of good coffee shops?"*). It can evaluate binary ambiguity detection (`require_clarification`) and question quality, but cannot score post-`ASK` decision accuracy or two-stage VoI.
  - **No `DEFER` dimension**: Queries span open-domain lexical/entity ambiguity without any Specialist rubric, capacity-constrained escalation, or latent override rules.

---

### 2.2 MedMCQA (CHIL 2022 / PMLR v174)

- **Primary Sources**:
  - GitHub repository: [`medmcqa/medmcqa`](https://github.com/medmcqa/medmcqa)
  - HuggingFace dataset: [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa)
  - Paper: [PMLR v174 `pal22a`](https://proceedings.mlr.press/v174/pal22a.html) ([arXiv:2203.14371](https://arxiv.org/abs/2203.14371))
- **License & Downloadability**:
  - Permissive open-source: [`medmcqa/medmcqa/LICENSE.md`](https://raw.githubusercontent.com/medmcqa/medmcqa/main/LICENSE.md) is **MIT License**; the [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa/raw/main/README.md) HuggingFace card declares **Apache-2.0**.
  - Freely downloadable without gating via HuggingFace Parquet (~88.3 MB) or the Google Drive archive linked in the official README.
- **Exact Schema, Size, and Splits** (verified via HuggingFace `datasets-server` `/info` and `/rows` against `medmcqa/medmcqa/README.md`):
  - **Total Size**: **193,155** items across 3 splits:
    - `train`: **182,822** (labeled `cop` in `{0, 1, 2, 3}`)
    - `validation`: **4,183** (labeled `cop` in `{0, 1, 2, 3}`)
    - `test`: **6,150** (**ground-truth labels withheld**: `cop = -1`, `exp = ""`; evaluation via Google Form only). *(Note: `medmcqa/medmcqa/README.md` swaps the `Valid` and `Test` row counts in its Markdown table relative to the actual HuggingFace Parquet splits and label availability.)*
  - **Fields**: `id` (`string`), `question` (`string`), `opa`, `opb`, `opc`, `opd` (`string`), `cop` (`ClassLabel`: `0:'a', 1:'b', 2:'c', 3:'d'`, or `-1` on `test`), `choice_type` (`"single"` or `"multi"`), `exp` (`string`, nullable), `subject_name` (`string`, 21 subjects), `topic_name` (`string`, nullable).
  - **Empirical Inspection of First 100 `validation` Rows**:
    - `choice_type`: **61** `single` vs. **39** `multi` (many negative/exception items such as *"Which of the following is not true..."*).
    - `exp` non-empty: **64 / 100**; `topic_name` non-null: **14 / 100**.
    - Top subject in sample: `Dental` (**24 / 100**), followed by `Medicine` (9), `Pediatrics` (8), `Biochemistry` (8).
    - Observed OCR artifacts in exam text/explanations (e.g., `glomerular capillaries')`, `Guyton I LpJ1 4-.;anong`).
- **3-Way Cooperative Separation Assessment**:
  - **No native `ASK` separation**: Questions are static single-turn exam MCQs. Short recall stems cannot be naturally partitioned into initial presentation vs. withheld Requester facts without synthetic LLM rewriting or truncation.
  - **No native `DEFER` separation**: `subject_name` gives broad academic discipline tags (including 24% `Dental`), not clinical escalation authority or conditional overrides. Building `Q2/Q3/Q4` on MedMCQA would require heavy synthetic text generation, undercutting the goal of avoiding synthetic spoilers.

---

### 2.3 MediQ (NeurIPS 2024)

- **Primary Sources**:
  - GitHub repository: [`stellalisy/mediQ`](https://github.com/stellalisy/mediQ) (`src/mediQ_benchmark.py`, `src/patient.py`, `data/`)
  - HuggingFace dataset: [`stellalisy/mediQ`](https://huggingface.co/datasets/stellalisy/mediQ)
  - Paper: [arXiv:2406.00922](https://arxiv.org/abs/2406.00922) (*MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning*, NeurIPS 2024)
- **License & Downloadability**:
  - Explicitly licensed under **CC-BY-4.0** in both [`stellalisy/mediQ/LICENSE`](https://raw.githubusercontent.com/stellalisy/mediQ/main/LICENSE) and the [HuggingFace dataset card](https://huggingface.co/datasets/stellalisy/mediQ/raw/main/README.md).
  - Ungated (`gated: false`), directly downloadable from HuggingFace (~5.57 MB) and GitHub raw (`data/all_dev_good.jsonl`, `data/all_craft_md.jsonl`).
- **Exact Schema, Size, and Splits** (verified via HuggingFace `/info`, `/rows`, and GitHub `data/` + `src/`):
  - **HuggingFace `stellalisy/mediQ`** (**2,545** MedQA-USMLE clinical vignettes):
    - `validation` (`medqa_dev_convo.jsonl`): **1,272** items.
    - `test` (`medqa_test_convo.jsonl`): **1,273** items.
  - **GitHub `stellalisy/mediQ/data/`**:
    - `all_dev_good.jsonl`: **1,272** items (MedQA dev).
    - `all_craft_md.jsonl`: **140** clinical vignettes derived from CRAFT-MD (with `facts` and `patient: {age, gender}`).
  - **Exact Fields (`stellalisy/mediQ`)**:
    - `id` (`int64`), `question` (`string`), `context` (`list[string]` of case sentences), `context_len` (`int64`), `options` (`{A, B, C, D}`), `answer` (`string`), `answer_idx` (`"A"|"B"|"C"|"D"`), `explanation` (`string`).
    - `atomic_facts` (`list[string]`): sentence-decomposed patient facts derived from `context[1:]` (present in **100/100** sampled `validation` rows and **99/100** sampled `test` rows; in `validation` sample: `context_len` min=1, avg=7.21, max=14; `atomic_facts` min=2, avg=9.53, max=23).
    - `patient` (`struct`): `{gender, age, specialties, subspecialties, specialty_prob, subspecialty_prob, gpt_specialty}`. **Crucial split difference**: populated in **100/100** sampled `validation` rows, but **`null` in 100/100 sampled `test` rows** (`facts_old` is also `null` on `test`).
    - Sampled `validation` `patient.specialties` distribution (`n = 100`): `Internal Medicine` (41), `Pediatrics` (19), `Neurology` (12), `Gastroenterology` (6), `Psychiatry` (6), `Cardiology` (5), `Obstetrics and Gynecology` (4), `Pulmonary Disease` (4), `Orthopaedic Surgery` (4), `Emergency Medicine` (4); **34 / 100** cases carry `len(specialties) >= 2`.
- **3-Way Cooperative Separation Assessment**:
  - **Native `ASK` (`Requester`-held context)**: In `src/patient.py` (`Patient.__init__`), the expert system sees only `context[0]` (initial patient chief complaint), `question`, and `options`. All remaining clinical history, vitals, physical exam, and lab findings (`context[1:]`) are stored in `atomic_facts` and revealed only when the expert asks targeted follow-up questions (`FactSelectPatient`). Every item has a verifiable 4-option gold label `answer_idx`.
  - **Non-circular `DEFER` hook (`Specialist` authority & `Q4` conditional escalation)**: No public QA benchmark natively ships with an explicit "confidential hospital override" label, and MediQ's native `answer_idx` still scores USMLE multiple-choice correctness. However, MediQ `validation` uniquely enables a **non-circular, spoiler-free `DEFER` construction** via two primary-source signals:
    1. **Subspecialty & Multi-Specialty Scope Policy (`patient.specialties` / `subspecialties` / `specialty_prob`)**: A transparent, pre-registered **Clinical Triage Scope Rubric** designates generalist primary-care domains (e.g., routine `Internal Medicine` / single-specialty primary care) as AI-manageable (`Q1`/`Q2`), while designated high-acuity subspecialty / multi-specialty cases (e.g., `Pediatrics`, `Neurology`, `Cardiology`, `Obstetrics and Gynecology`, `Orthopaedic Surgery`, or multi-specialty ambiguity `len(specialties) >= 2`) require Specialist `DEFER`—without injecting any `"confidential"` or `"override"` keywords into the patient prompt.
    2. **Natural `Q4` Conditional Escalation inside `atomic_facts`**: In MediQ, `context[0]` is a brief presenting symptom (e.g., *5-year-old girl with 2 hours of nausea and vomiting*, or *67-year-old man with 2-day ear ringing*), whereas escalation-triggering clinical facts (e.g., *bilious vomiting + 2 prior dehydration hospitalizations + BP 82/52 mm Hg*, or *neoadjuvant chemotherapy 1 week ago + 45 dB sensorineural hearing loss*) reside **exclusively in `atomic_facts`**. Thus, `Q4_resolvable` vs. `Q4_escalate` cannot be guessed from `context[0]` and is resolved *only* by executing `ASK` against the Requester.

---

## 3. Concrete Recommendation for `Public-Tri` (240–300 Items)

### Single Strongest Candidate: **MediQ (`stellalisy/mediQ` `validation` split, `n = 1,272`)**

1. **Recommended Split & Sample Size (`N = 300` total: `60` Calibration + `240` Test, or `60 + 180 = 240`)**:
   - Sample exclusively from **`stellalisy/mediQ` `validation` (`medqa_dev_convo.jsonl`, 1,272 items)** because only `validation` carries both `atomic_facts` and populated `patient.specialties` / `subspecialties` / `specialty_prob` metadata (keep `test` `1,273` and `all_craft_md.jsonl` `140` untouched or reserved for auxiliary `ASK`-only transfer checks).
   - Stratify **60 held-out calibration** (`15` per quadrant `Q1..Q4`) and **240 test** (`60` per quadrant; or `180` test at `45` per quadrant to match `Benchmark v1/v2`) across four non-circular operational quadrants:
     - **`Q1_clear_easy` (`ANSWER`)**: Single generalist specialty (`Internal Medicine`), short `context_len`, where `context[0]` + clinical reasoning suffices without high-risk Specialist escalation.
     - **`Q2_underspec_easy` (`ASK -> ANSWER`)**: Generalist specialty where `context[0]` omits decisive lab/exam findings stored in `atomic_facts`, and clarified facts stay within generalist scope.
     - **`Q3_clear_hard` (`DEFER`)**: `context[0]` already indicates a designated Specialist-only domain/acuity trigger under the Clinical Triage Scope Rubric (e.g., acute Neurology/Cardiology/OB-GYN/Surgical presentation or multi-specialty conflict) *without* artificial keyword spoilers.
     - **`Q4_resolvable` / `Q4_escalate` (50/50 split, `ASK -> ANSWER` vs. `ASK -> DEFER`)**: `context[0]` presents an ambiguous generalist-looking symptom; `atomic_facts` either confirm a routine primary-care etiology (`Q4_resolvable`) or reveal a red-flag / subspecialty escalation condition (`Q4_escalate`, e.g., chemotherapy toxicity, hemodynamic instability, surgical abdomen).
2. **Why Single-Dataset MediQ Beats Pairwise Mixing with MedMCQA or CLAMBER**:
   - Mixing MedMCQA adds synthetic truncation/OCR noise without native `atomic_facts` or `FactSelectPatient` compatibility; mixing CLAMBER breaks end-to-end utility scoring because CLAMBER lacks ground-truth final answers and Requester resolutions.
   - If a secondary non-medical check is desired for **RQ1 (`ASK` vs. `NO-ASK` calibration only)**, CLAMBER's `require_clarification` subset can be reported separately as an auxiliary ambiguity-detection diagnostic, keeping `Public-Tri` clean and fully scorable under the 3-way Bellman VoI objective.
3. **Pre-Registration Requirement (Honest Boundary)**:
   - Because MediQ natively labels diagnostic `answer_idx` and `patient.specialties` rather than institutional deferral policies, the **Clinical Triage Scope Rubric** mapping `(context[0], atomic_facts, patient.specialties)` → `{Q1, Q2, Q3, Q4_resolvable, Q4_escalate}` must be **frozen and published verbatim** alongside the item IDs before running live evaluations, and validated by checking that `Q2`/`Q4` items genuinely show low zero-turn accuracy on `context[0]` alone versus high accuracy once `atomic_facts` are revealed.
