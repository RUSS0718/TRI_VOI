# Primary-Source Verification & Algorithmic Specification: External Baselines (`ACL 2026 VoI` and `arXiv:2610.15128 KL-PBD`)

- **Ticket:** [`RUSS0718/TRI_VOI#12`](https://github.com/RUSS0718/TRI_VOI/issues/12) (supporting [`#9`](https://github.com/RUSS0718/TRI_VOI/issues/9) and [`#18`](https://github.com/RUSS0718/TRI_VOI/issues/18))
- **Verification Date:** 2026-11-27
- **Summary Verdict:** **Both cited baseline papers exist and resolve directly at their exact cited identifiers** (`2026.acl-long.1987` and `arXiv:2610.15128`), with no year or ID typo. Both provide complete mathematical formulations and algorithms; `ACL 2026 VoI` also has an official public GitHub repository (`dong-river/VOI_communication`), whereas `arXiv:2610.15128` (`KL-PBD`) ships code only in submission supplementary material (no public GitHub URL indexed as of 2026-11-27) and must be reproduced from its primary LaTeX specification.

---

## 1. Verification Summary Table

| Cited Reference | Resolution Check | Canonical Primary Sources | Official Code Repo |
| :--- | :--- | :--- | :--- |
| **ACL 2026 *Value of Information: A Framework for Human–Agent Communication*** (`2026.acl-long.1987`) | **VERIFIED (`HTTP/1.1 200 OK`)** at `https://aclanthology.org/2026.acl-long.1987/` | ACL Anthology: [`2026.acl-long.1987`](https://aclanthology.org/2026.acl-long.1987/) ([PDF](https://aclanthology.org/2026.acl-long.1987.pdf), [BibTeX](https://aclanthology.org/2026.acl-long.1987.bib)); Preprint: [`arXiv:2601.06407`](https://arxiv.org/abs/2601.06407) (submitted 2026-01-10) | [`dong-river/VOI_communication`](https://github.com/dong-river/VOI_communication) (`main` branch contains `mixed_20q/` and `flight_rec/`) |
| **`arXiv:2610.15128` *To Defer or Not to Defer? Online Learning with LLMs and Congested Experts*** (`KL-PBD`) | **VERIFIED (`HTTP/1.1 200 OK`)** at `https://arxiv.org/abs/2610.15128` | arXiv: [`2610.15128v1`](https://arxiv.org/abs/2610.15128) ([PDF](https://arxiv.org/pdf/2610.15128), submitted 2026-10-10; LaTeX source uses `iclr2027_conference.sty`) | **None public** as of 2026-11-27 (paper Reproducibility Statement notes supplementary zip only) |

---

## 2. Baseline A: ACL 2026 VoI (`2026.acl-long.1987` / `arXiv:2601.06407`)

### 2.1 Verified Bibliographic Record

- **Title:** *Value of Information: A Framework for Human–Agent Communication*
- **Authors:** Yijiang River Dong, Tiancheng Hu, Zheng Hui, Caiqi Zhang, Ivan Vulić, Andreea Bobu, and Nigel Collier (University of Cambridge; MIT)
- **Venue:** *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, July 2026, San Diego, California, USA, pp. 42879–42896 (marked **ACL 2026 Oral** in [`VOI_communication/README.md`](https://github.com/dong-river/VOI_communication/blob/main/README.md))
- **DOI:** `10.18653/v1/2026.acl-long.1987` ([ACL Anthology](https://aclanthology.org/2026.acl-long.1987/))
- **arXiv Preprint:** [`arXiv:2601.06407`](https://arxiv.org/abs/2601.06407) (`cs.CL`, DOI `10.48550/arXiv.2601.06407`)

### 2.2 Problem Formulation & Action Space

- **Setting:** Single-user, multi-turn sequential decision-making under an underspecified initial query $S$ and unknown latent user intent/preference $\theta \in \Theta$ (`sections/setup.tex`).
- **Action Space (Binary Clarify-or-Commit):** At each turn $t$ with dialogue history $H_t = (q_1, u_1, \dots, q_{t-1}, u_{t-1})$, the agent chooses:
  1. **`CLARIFY(q_t)`:** Pose a closed-ended clarifying question $q_t \in \mathcal{Q}$ (yes/no or multiple-choice, yielding finite answer space $\mathcal{Y}$), observe user reply $u_t$, update $H_{t+1}$.
  2. **`COMMIT(a)`:** Terminate dialogue and commit to a terminal action $a \in \mathcal{A}$.
- **Structural Difference from Three-Way Cooperative L2D:** There is **no expert deferral action (`DEFER`)**, **no specialist queue/congestion model**, and **no distinction between user-held specification uncertainty and specialist-held policy/epistemic uncertainty** (`sections/setup.tex`, `sections/limitation.tex`).
- **Objective:** Maximize terminal utility minus cumulative communication cost: $U(\theta, a) - c(H)$, instantiated in experiments as linear per-question cost $c(H) = T_{\text{ask}} \cdot c$ (`sections/results.tex`, `sections/limitation.tex`).

### 2.3 Exact Equations & Algorithm (`sections/methods.tex`, Algorithm 1)

1. **Belief & Immediate Expected Utility (`Eq. 1–2`):**
   Given current belief $b(\theta)$ over $\Theta$:
   $$\text{EU}(a \mid b) = \mathbb{E}_{\theta \sim b}[U(\theta, a)] = \sum_{\theta \in \Theta} b(\theta) U(\theta, a), \qquad V(b) = \max_{a \in \mathcal{A}} \text{EU}(a \mid b).$$
2. **One-Step Lookahead Posterior Value (`Eq. 3` & `Sec. 4.2`):**
   For a candidate question $q$ with finite answer space $\mathcal{Y}$:
   $$V_{\text{post}}(b, q) = \sum_{y \in \mathcal{Y}} p(y \mid q, b) \cdot V(b_y), \qquad b_y(\theta) = P(\theta \mid H, q, y),$$
   where response likelihoods are marginalized over the current belief via LLM simulation:
   $$p(y \mid q, b) \approx \sum_{\theta \in \Theta} p(y \mid q, \theta)\, b(\theta).$$
3. **Value of Information & Stopping Rule (`Eq. 4–5`, Algorithm 1):**
   $$\text{VoI}(q) = V_{\text{post}}(b, q) - V(b), \qquad \text{NetVoI}(q) = \text{VoI}(q) - c(q).$$
   At turn $t \le K_{\max}$:
   - Generate candidate questions $Q \gets \mathrm{GenQ}(H)$.
   - Select $q^* = \arg\max_{q \in Q} \text{NetVoI}(q)$.
   - If $\text{NetVoI}(q^*) \le 0$: **break** and return $a^* \in \arg\max_{a \in \mathcal{A}} \mathbb{E}_{\theta \sim b}[U(\theta, a)]$.
   - Else: ask $q^*$, observe $y$, append $(q^*, y)$ to $H$, and re-estimate belief $b \gets \mathrm{Update}(b, q^*, y)$.

### 2.4 Official Implementation Audit (`dong-river/VOI_communication`)

Inspection of the `main` branch of [`dong-river/VOI_communication`](https://github.com/dong-river/VOI_communication) reveals concrete engineering details and subtle differences from the paper text:

- **Available vs. Missing Task Directories:** The repo contains `mixed_20q/` (Animal Guessing, Medical Diagnosis) and `flight_rec/` (Flight Recommendation). The third paper task (*Ambiguous WebShop*) is **not present** on `main` as of 2026-11-27.
- **Model & Execution Pattern:** Paper tables/figures report `GPT-4.1` and `Gemini-2.5-Flash`, whereas released transcripts in `VOI_communication` were run with `openai/gpt-5.4` via OpenRouter (`mixed_20q/README.md`, `flight_rec/README.md`). Both task runners decouple rollout from cost $c$ by logging per-turn `voi` (`utility` / `voi_next`) up to `max_questions` and applying the stopping rule $\text{VoI} < c$ **offline** in evaluation scripts (`mixed_20q/evaluate.py`, `flight_rec/README.md`).
- **Key Implementation Details & Hyperparameters (`mixed_20q/voi_strategy.py`, `run_voi.py`):**
  - **Full-set belief estimation:** Always prompts the LLM (temp `0`, `max_tokens=4000`) for a posterior distribution over the **entire** candidate set `all_subjects` (never hard-eliminated before belief scoring, preventing belief collapse to a fake-certain singleton). Missing candidates receive smoothing `min_prob = 0.01 / n`, then probabilities are renormalized.
  - **Best-of-$N$ question selection:** `top_question_candidates = 3` (proposes up to 3 distinct yes/no questions at temp `0.7`), `lookahead_k = 1`, `batch_size = 25` for `Yes`/`No`/`Maybe` answer simulation (temp `0`).
  - **Note on terminal value formula in code vs. paper:** `mixed_20q/voi_strategy.py` (`calculate_value_function_direct`) computes $V(b) = \max_g [p_g r_g + (1 - p_g)(-r_g)]$ (`r = 1.0` for 20Q, `reward_dict` for medical), whereas `sections/exp_setup.tex` and `mixed_20q/evaluate.py` define terminal reward $U \in \{0, 1\}$ (Animal, 100 candidates, $K_{\max}=20$) and $U \in \{0, 10\}$ (Medical Diagnosis, 15 diseases, $K_{\max}=10$), with per-question costs $c \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$ (`sections/results.tex`, Table 1). Also note `evaluate.py`'s `voi_raw_metrics` enforces `i > 1` (at least 2 questions) before stopping on `u < cost`.
- **Key Implementation Details (`flight_rec/voi_utils.py`):**
  - Factorizes $\Theta$ into 5 independent features: `price`, `stops`, `layover`, `arrival` (3 states each) and `airline` (5 states), with LLM JSON prompts (temp `0.2`) for priors, option-conditioned posteriors, and $P(y \mid q, b)$ / post-answer posteriors; action value is $\max_{a \in \{A,B,C\}} \mathbb{E}[\theta \cdot \text{features}(a)]$.
- **Baselines in Paper:** `No-Question`, `Fixed-Round` ($k \in \{5, 8, 10, 15, 20\}$ in 20Q; $\{1, 2, 3\}$ in Flight Rec), `Adaptive Prompting` ("Auto Stop"), and `Confidence Thresholding` ($\tau \in \{0.5, 0.7, 0.9\}$).

---

## 3. Baseline B: `arXiv:2610.15128` (`KL-PBD`)

### 3.1 Verified Bibliographic Record

- **Title:** *To Defer or Not to Defer? Online Learning with LLMs and Congested Experts*
- **Authors:** Vasileios Saketos\*, William Réveillard\*, Alexandre Proutière, Sebastian Dalleiger, and Ming Xiao (KTH Royal Institute of Technology, Stockholm, Sweden; \*equal contribution)
- **Identifier:** [`arXiv:2610.15128`](https://arxiv.org/abs/2610.15128) [cs.LG], v1 submitted 10 Oct 2026, DOI `10.48550/arXiv.2610.15128` (LaTeX template `iclr2027_conference.sty`)
- **Public Code:** Not hosted on a public GitHub URL as of 2026-11-27; code is referenced only as supplementary material in the paper's Reproducibility Statement.

### 3.2 Problem Formulation, Congestion Model & Action Space (`main_sections/3.problem_formulation.tex`)

- **Horizon & Confidence Discretization:** Over rounds $t = 1, \dots, T$, a query $X_t$ arrives (or no query, $J_t = 0$) and the LLM outputs candidate answer $R_t$ and discrete self-confidence level $J_t \in [K] := \{1, \dots, K\}$ (canonical construction: $G$ self-consistency samples, $K = G + 1$, $J_t = 1$ when no answer is extractable, else $1 +$ size of largest equivalence class). Let $Y_t := \mathbf{1}\{R_t \equiv R_t^\star\}$, arrival probabilities $\omega_j := \mathbb{P}(J_t = j)$, and **unknown** conditional accuracies $h_j^\star := \mathbb{P}(Y_t = 1 \mid J_t = j)$, with $(J_t, Y_t)_{t \ge 1}$ i.i.d.
- **Observation & Action Space (Binary Accept-or-Defer):** The controller observes only $(B_t, J_t)$ (neither $X_t$ nor $R_t$) and chooses $A_t \in \{0, 1\}$ ($1$ = defer to expert queue, $0$ = accept $R_t$; forced $A_t = 0$ if $J_t = 0$ or $B_t = \bar{B}$). **No user clarification action (`ASK`) exists.**
- **Queue & Service Dynamics:**
  - Capacity $\bar{B} \ge 1$, initial backlog $B_1 = 0$.
  - Between rounds $t$ and $t+1$, experts serve $C_t \stackrel{\text{i.i.d.}}{\sim} \rho_C$ queries (independent of $(J_t, Y_t)$), with $\mu := \mathbb{P}(C_t \ge 1) > 0$.
  - Backlog evolution: $B_{t+1} = \max\{0, B_t + A_t - C_t\}$.
  - Experts always return the ground-truth label $R_t^\star$ (so deferred queries incur zero error cost).
- **Selective & Delayed Feedback:**
  - If $A_t = 0$: $Y_t$ is **never observed**.
  - If $A_t = 1$: $Y_t$ is observed **only once the query completes service in the expert queue** (delayed by queue waiting time; completed labels at end of round $t$ are available at round $t+1$).
- **Costs & Terminal Draining Cost:**
  - Per-round cost: $c_t := c_{\mathrm{hold}} B_t + c_{\mathrm{exp}} A_t + c_e (1 - A_t)(1 - Y_t)$, with $c_{\mathrm{hold}} \ge 0$, $c_{\mathrm{exp}} \ge 0$, $c_e > 0$.
  - Terminal clearing cost: $D(B_{T+1})$, where $D(0) = 0$ and $D(b) = c_{\mathrm{hold}} b + \mathbb{E}_{C \sim \rho_C}[D((b - C)^+)]$ for $b \ge 1$ (for $C_t \sim \operatorname{Bernoulli}(\mu)$, $D(b) = c_{\mathrm{hold}} b(b+1)/(2\mu)$).

### 3.3 Optimal Policy & Exact `KL-PBD` Algorithm (`Sec. 3.2`, `Sec. 4.1`, `App. B.5`, `App. F`)

1. **Bellman Marginal Cost of Deferral (known $h \in [0, 1]^K$, `Eq. 1`):**
   Define $h_0 := 1$, expected acceptance error cost $L^h(j) := c_e(1 - h_j)$, and finite-horizon MDP value functions $V_t^h((b, j))$ with terminal value $V_{T+1}^h((b, j)) = D(b)$ and transition kernel $P_a(b, b')$ from $b' = \max\{0, b + a - C\}$. For $b < \bar{B}$:
   $$M_t^h(b) := c_{\mathrm{exp}} + \sum_{b'=0}^{\bar{B}} \bigl(P_1(b, b') - P_0(b, b')\bigr) \sum_{j=0}^K \omega_j V_{t+1}^h((b', j)), \qquad M_t^h(\bar{B}) := +\infty.$$
   **Lemma 1 (Optimal Deferral Policy):** Defer iff $J_t \in [K]$ and $h_{J_t} < \tau_t^h(B_t)$, where $\tau_t^h(b) := 1 - M_t^h(b)/c_e$ (equivalently $L^h(J_t) > M_t^h(B_t)$). By Proposition 1 (`prop:queue-threshold`), $M_t^h(b)$ is nondecreasing in $b$, so optimal deferral has a monotone queue threshold $\tau_{t,j}^{\mathrm{queue}}(h)$.

2. **`KL-PBD` Online Learning Algorithm (`Algorithm 1`):**
   - **Maintain completed feedback statistics per level $j \in [K]$:**
     - $N_{j,t}$ = number of level-$j$ queries whose expert reviews **completed** before round $t$ (note $N_{j,t} \le$ total deferred count due to queue delay).
     - $\widehat h_{j, N_{j,t}} := \frac{1}{N_{j,t}} \sum_{\ell=1}^{N_{j,t}} Y_{j,\ell}$ if $N_{j,t} \ge 1$.
     - $O_{j,t} := \sum_{s=1}^t \mathbf{1}\{J_s = j\}$ (arrival count).
   - **KL Lower Confidence Bound (Pessimism on LLM accuracy):**
     With confidence budget $\phi(T) := \log T + \log \log T$ (for $T \ge 3$),
     $$\underline h_{j,t} := \inf\left\{ q \in [0, \widehat h_{j, N_{j,t}}] : N_{j,t}\, \kl\!\left(\widehat h_{j, N_{j,t}},\, q\right) \le \phi(T) \right\} \quad (N_{j,t} \ge 1), \qquad \underline h_{j,t} := 0 \quad (N_{j,t} = 0),$$
     where $\kl(p, q) = p \log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$ is the Bernoulli KL divergence.
     *(Heuristic/empirical variant `KL-PBD+` in `Sec. 6` & `App. F` replaces $\phi(T)$ with the level-dependent budget $\log(O_{j,t}/N_{j,t})$, achieving up to $4\times$ lower empirical regret while retaining the same structure.)*
   - **Unknown Arrival Distribution $\omega$ (`App. B.5` & `App. F`):**
     Plug in passive empirical arrival frequencies $\widehat\omega_{j,t} = \frac{1}{t}\sum_{s=1}^t \mathbf{1}\{J_s = j\}$ (if $\rho_C$ is also unknown, plug in truncated empirical service $\widehat\rho_{c,t} = \frac{1}{t-1}\sum_{s<t}\mathbf{1}\{\min\{C_s, \bar{B}\} = c\}$).
   - **Decision Rule at Round $t$:**
     Solve the remaining-horizon backward-induction Bellman recursion under $(\underline h_t, \widehat\omega_t)$ to obtain $\widehat\tau_t(B_t) = 1 - M_t^{\underline h_t}(B_t)/c_e$, and choose:
     $$A_t = \mathbf{1}\!\left\{ J_t \in [K],\; \underline h_{J_t, t} < \widehat\tau_t(B_t) \right\}.$$
     *(Why pessimism is required: Greedy plug-in `GBD` with $\widehat h_{j,N_{j,t}}$ suffers selective-feedback lock-in—an early lucky sample raises $\widehat h$ above $\widehat\tau$, halting deferrals and permanently suppressing corrective labels, yielding linear regret with $\rho \in [-0.95, -0.85]$ correlation between deferral rate and regret in `App. H.5`.)*

3. **Regret Guarantees (`Theorems 1–4`):**
   - Gap-dependent: $R_T(\pi^{\texttt{KL-PBD}}) = O(K \log T / \gamma)$ when nonzero threshold gaps satisfy $\gamma_T \ge \gamma > 0$.
   - Worst-case: $R_T(\pi^{\texttt{KL-PBD}}) = \widetilde{O}(\sqrt{KT})$ for $K \le T$, matching minimax and instance-dependent lower bounds up to logarithmic factors.

### 3.4 Experimental Parameters & Baselines (`Sec. 6`, `App. E–H`)

- **Real-World Benchmarks & LLMs:** GSM8K ($8{,}792$), NuminaMath-TIR ($9{,}184$ deduped), MedMCQA ($187{,}005$), evaluated with `Qwen3-4B-Instruct-2507`, `Qwen3-0.6B`, and `Qwen2.5-0.5B-Instruct` ($G=20$ samples at temp $0.6$, top-$p$ $0.95$, $K=21$ levels $p_j \in \{0, 0.05, \dots, 1\}$).
- **Default Queue/Cost Parameters:** $(c_e, c_{\mathrm{exp}}, c_{\mathrm{hold}}) = (1, 0.25, 0.03)$, $\bar{B} = 8$, $C_t \sim \operatorname{Bernoulli}(0.35)$, $T = 10^5$ (sweeps: $T \in \{10^4, 5\cdot 10^4, 10^5, 2\cdot 10^5\}$, $\mu \in \{0.2, 0.35, 0.5, 0.7, 1.0\}$).
- **Baselines Compared in Paper:** `KL-PBD+`, `GBD` ($\widehat h_{j,N_{j,t}}$), `NBD` (frozen calibrated surrogate $h=p$), `FCT` (fixed confidence cutoff tuned under $h=p$), `FQT` (fixed queue threshold per level tuned via coordinate descent under $h=p$), `SALMUT-SF` (adapted from Roy et al., 2021 with forced exploration $\varepsilon_t = \min\{0.5, 2/\sqrt{t}\}$), and queue-unaware `KL-UCB-4P` / `KL-UCB+-4P` (adapted from Achab et al., 2018).

---

## 4. Implications for Reproduction in `RUSS0718/TRI_VOI` (`#9` / `#18`)

| Dimension | `ACL 2026 VoI` (`2026.acl-long.1987`) | `arXiv:2610.15128` (`KL-PBD`) | `Tri-VoI-L2D` (`TRI_VOI`) |
| :--- | :--- | :--- | :--- |
| **Action Space** | `{ACT, ASK}` | `{ACT, DEFER}` | `{ACT, ASK, DEFER}` |
| **Information Source Modeled** | User latent state $\theta \in \Theta$ | Expert oracle only; LLM self-confidence level $J_t \in [K]$ | Disentangled user spec ($U_{\text{spec}}$) vs. specialist policy ($U_{\text{epist}}$) |
| **Queue / Congestion** | None (linear cost $c$ per turn) | Finite capacity $\bar{B}$, i.i.d. service $C_t \sim \rho_C$, holding cost $c_{\mathrm{hold}} B_t$ + Bellman marginal cost $M_t^{\underline h_t}(B_t)$ | Multi-server/single-server queue $Q_t$ with congestion-gated Naor externality |
| **Learning / Calibration** | Zero-shot inference-time LLM belief $b(\theta)$ | Online selective delayed-feedback KL-LCB $\underline h_{j,t}$ | Held-out Empirical Bayes calibration + discounted Beta-Bernoulli expert tracker |
| **Primary Reproduction Source** | `dong-river/VOI_communication` + `arXiv:2601.06407` Sec. 4 & App. B | `arXiv:2610.15128` Sec. 3–4 & App. B.5 / F (custom implementation needed; no public repo) | `controller.py` / `benchmark.py` |

### Practical Adaptation Checklist for Implementing Both Baselines in `TRI_VOI`

1. **ACL 2026 VoI (`{ACT, ASK}` baseline):**
   - Port the exact one-step lookahead VoI rule from `VOI_communication` (`Eq. 1–5`, Algorithm 1): enumerate/propose candidate clarifying questions $q \in Q$ (or use the ticket's elicited clarifying question $q$ when $|Q|=1$), simulate answer probabilities $p(y \mid q, b)$ and post-answer posteriors $b_y$ over the 3 terminal choices `{A, B, C}` using LLM belief prompts (with full-set posterior smoothing `0.01 / n`, no premature collapse), compute $\text{NetVoI}(q) = V_{\text{post}}(b, q) - V(b) - c_{\text{ask}}$, and execute `ASK` iff $\text{NetVoI}(q) > 0$, else `COMMIT` to $\arg\max_{a \in \{A,B,C\}} \text{EU}(a \mid b)$.
   - Because ACL 2026 VoI has no `DEFER` channel, it cleanly isolates the `{ACT, ASK}` ceiling on user-ambiguous items while remaining unable to resolve specialist-override items (`U_epist`) or relieve a congested expert queue via `DEFER`.
2. **`arXiv:2610.15128` `KL-PBD` (`{ACT, DEFER}` online congested-expert baseline):**
   - Implement Algorithm 1 + Appendix B.5 (`\widehat\omega_{j,t}`) from scratch in Python (using e.g. Newton/bisection to invert $N_{j,t}\,\kl(\widehat h_{j,N_{j,t}}, q) = \phi(T)$ for $\underline h_{j,t}$, plus backward dynamic programming over $b \in \{0, \dots, \bar{B}\}$ to compute $M_t^{\underline h_t}(B_t)$).
   - Map `TRI_VOI`'s self-consistency / probe signals (`samples` agreement or `1..K` confidence bins) onto discrete levels $J_t \in [K]$, feed exact queue backlog $B_t$, service completions $C_t$, and delayed labels $Y_t$ (revealed only when a deferred item finishes expert service) into $(N_{j,t}, \widehat h_{j, N_{j,t}}, \underline h_{j,t})$.
   - Include both `KL-PBD` ($\phi(T) = \log T + \log\log T$) and `KL-PBD+` ($\log(O_{j,t}/N_{j,t})$) if comparing online finite-horizon streams, since `arXiv:2610.15128` shows `KL-PBD+` avoids over-conservative exploration on moderate horizons ($T \le 10^4$). Note also that when expert accuracy $\hat{p}_e < 1$, `L^h(j)` or `c_{\mathrm{exp}}` should be aligned with `TRI_VOI`'s expert expected error cost so the cost accounting matches fairly.
