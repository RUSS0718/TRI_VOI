# Tri-VoI-L2D

Three-way human-AI cooperative decision-making where an AI agent chooses among acting directly, asking the ticket submitter for missing context, or deferring to a capacity-constrained domain expert.

## Language

### Cooperative Roles & Mechanisms

**Requester**:
The ticket submitter who holds task-specific context missing from the initial prompt and serves as the target of an `ASK` action.
_Avoid_: User, human, customer

**Specialist**:
A capacity-constrained domain expert who holds latent institutional rules or override authority and serves as the target of a `DEFER` action.
_Avoid_: Expert, reviewer, human

**Information-Source Disentanglement**:
The separation of predictive uncertainty caused by missing **Requester** context from uncertainty caused by latent **Specialist** rules or policy overrides.
_Avoid_: Uncertainty decomposition, ambiguity detection

**Queue-Relief Deferral**:
A joint control policy that uses `ASK` to resolve **Requester**-solvable tickets directly, preserving **Specialist** queue capacity for tickets that genuinely require `DEFER`.
_Avoid_: Cascaded deferral, congestion-aware routing

### Evaluation Suites

**Benchmark v1-Clean**:
The original 420-item structured benchmark (`60` calibration + `360` test) evaluating **Information-Source Disentanglement** and **Queue-Relief Deferral** across five models from four families.
_Avoid_: Original benchmark, clean dataset

**Benchmark v1-Noisy**:
The 420-item multi-paragraph workplace-text variant of **Benchmark v1-Clean** paired with an imperfect **Requester** simulator exhibiting 15% clarification friction.
_Avoid_: Realistic benchmark, noisy dataset

**Benchmark v2 (Implicit-Override)**:
The leakage-hardened benchmark where every missing-information ticket changes the optimal decision upon clarification and no explicit override keywords appear in the prompt.
_Avoid_: Strict benchmark, v2 dataset

**Public-Tri**:
The independent 240–300 item three-way cooperative benchmark adapted from a public external dataset to test out-of-distribution generalization beyond synthetic workplace tickets.
_Avoid_: External benchmark, public dataset

### Submission Tracks

**IJCAI Freeze**:
The internal completeness deadline (`2027-01-05`, ahead of the `2027-01-11` IJCAI 2027 full-paper deadline) locking the 7-page manuscript, three-theorem appendix, and anonymized reproduction package.
_Avoid_: Final deadline, soft freeze

**ACL Reframing Contingency Branch**:
A pre-planned, strictly non-dual-submission alternative framing targeting a future ACL ARR cycle (LLM information sufficiency and interactive decision-making) used only if the primary IJCAI submission is not pursued or rejected.
_Avoid_: Parallel ACL submission, dual track
