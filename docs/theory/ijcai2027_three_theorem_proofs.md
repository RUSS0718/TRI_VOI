# Mathematical Specification and Full Proofs: IJCAI 2027 Three-Theorem Package

> **Tracking:** [`RUSS0718/TRI_VOI#8`](https://github.com/RUSS0718/TRI_VOI/issues/8) & [`RUSS0718/TRI_VOI#15`](https://github.com/RUSS0718/TRI_VOI/issues/15)  
> **Scope:** Complete statements and line-by-line proofs for Appendix A of the IJCAI 2027 manuscript (*Tri-VoI-L2D*), paired with `¥0` offline numerical verification in [`verify_theory_package.py`](../../verify_theory_package.py).

---

## Notation and Standing Assumptions

At decision step $t$, the **Requester** submits a ticket $x_t$ and the **Specialist** pool (for clarity in single-pool analysis, one server with capacity $C_m$, service rate $\mu = C_m / \Delta_{\text{drain}}$, queue backlog $Q_t \in \mathbb{Z}_{\ge 0}$, marginal linear waiting penalty $\gamma > 0$, deferral fee $c_{\text{defer}} \ge 0$, and clarification cost $c_{\text{ask}} > 0$) has true reliability $p_e \in (0,1)$ and estimate $\hat{p}_e \in (0,1)$. Payoffs are $R_+ > R_-$ with reward span $\Delta R := R_+ - R_- > 0$.

For any observable diagnostic state $s = (\text{obs\_missing}, \text{obs\_pot\_ovr}, \text{unanimous})$:
- $U_{\text{act}} := \hat{p}_{\text{act}} R_+ + (1 - \hat{p}_{\text{act}}) R_-$.
- $U_{\text{clear}} := \hat{p}_{\text{post\_clear}} R_+ + (1 - \hat{p}_{\text{post\_clear}}) R_-$, $U_{\text{ovr}} := \hat{p}_{\text{post\_ovr}} R_+ + (1 - \hat{p}_{\text{post\_ovr}}) R_-$.
- Under linear congestion cost $C_{\text{cong}}(Q) = \gamma Q$, $U_{\text{defer}}(Q) := \hat{p}_e R_+ + (1 - \hat{p}_e) R_- - c_{\text{defer}} - \gamma Q = U_{\text{defer}}(0) - \gamma Q$. *(Remark on Gated Naor Externality: In the controller implementation, queued deferral replaces $Q$ with $Q_{\text{eff}}(Q) = Q + \hat{\rho}(Q+1)\mathbf{1}\{Q \ge C_m \lor \hat{\rho} \ge 0.6\}$, which introduces a single step discontinuity at $Q = C_m$ when $\hat{\rho} < 0.6$ and omits the sub-capacity cross-ticket shadow price when $Q < C_m$. Removing the indicator gate yields a smooth linear-in-$Q$ effective slope $(1 + \hat{\rho})\gamma$ for every $Q \ge 0$.)*
- Following an `ASK` action, outcomes follow the exact three-way probability partition $\pi = (p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}})$ with $p_{\text{res}} + p_{\text{ovr\_after}} + p_{\text{no}} = 1$, $p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}} \ge 0$, corresponding to: (i) missing field revealed and no override applies (`clear`), (ii) missing field revealed and specialist override is confirmed (`ovr_after`), and (iii) no useful clarification obtained (`no_info`).

The two-stage Bellman expected utility of `ASK` and its Expected Value of Information (`EVOI`) over the no-ask baseline are:
$$U_{\text{ASK}}(Q) := p_{\text{res}} \max\{U_{\text{clear}}, U_{\text{defer}}(Q)\} + p_{\text{ovr\_after}} \max\{U_{\text{ovr}}, U_{\text{defer}}(Q)\} + p_{\text{no}} \max\{U_{\text{act}}, U_{\text{defer}}(Q)\} - c_{\text{ask}},$$
$$\text{EVOI}(\text{ASK}; Q) := U_{\text{ASK}}(Q) - \max\{U_{\text{act}}, U_{\text{defer}}(Q)\}.$$

---

## Theorem 1 (General 3-Outcome Three-Regime Piecewise Queue-Relief)

### Statement
Under linear queue cost $C_{\text{cong}}(Q) = \gamma Q$ (or smoothed Naor externality $(1+\hat{\rho})\gamma Q$), suppose the standard ordering $U_{\text{ovr}} < U_{\text{act}} < U_{\text{clear}}$ holds (confirmation of an override degrades AI accuracy below pre-ASK accuracy, whereas resolving a missing user field improves AI accuracy above pre-ASK accuracy). Define the two queue regime boundaries:
$$Q_1^* := \frac{U_{\text{defer}}(0) - U_{\text{act}}}{\gamma}, \qquad Q_2^* := \frac{U_{\text{defer}}(0) - U_{\text{ovr}}}{\gamma},$$
so that $Q_1^* < Q_2^*$. Assume $Q$ is treated on $\mathbb{R}_{\ge 0}$ and further suppose $U_{\text{defer}}(Q) \le U_{\text{clear}}$ on the working region (or more generally, let $Q_0^* := (U_{\text{defer}}(0) - U_{\text{clear}})/\gamma < Q_1^*$; on $Q \in (Q_0^*, \infty)$ the resolved branch chooses `ACT`). Then $\text{EVOI}(\text{ASK}; Q)$ is continuous and piecewise linear in $Q$ across three regimes:
1. **Regime I ($Q_0^* < Q < Q_1^*$, `DEFER`-dominated baseline):**
   $$\frac{\partial\,\text{EVOI}(\text{ASK}; Q)}{\partial Q} = +p_{\text{res}}\,\gamma > 0, \qquad \frac{\partial\,\text{EVOI}(\text{ASK}; Q)}{\partial \hat{p}_e} = -p_{\text{res}}\,\Delta R < 0.$$
   Thus `ASK` acts as a **queue-relief valve** whose value increases strictly with Specialist backlog $Q$ and decreases with Specialist reliability $\hat{p}_e$.
2. **Regime II ($Q_1^* < Q < Q_2^*$, Inversion Regime):**
   $$\frac{\partial\,\text{EVOI}(\text{ASK}; Q)}{\partial Q} = -p_{\text{ovr\_after}}\,\gamma \le 0,$$
   strictly negative whenever $p_{\text{ovr\_after}} > 0$, and recovering $-(1 - p_{\text{res}})\gamma$ in the frictionless special case $p_{\text{no}} = 0$.
3. **Regime III ($Q > Q_2^*$, Saturation Regime):**
   $$\frac{\partial\,\text{EVOI}(\text{ASK}; Q)}{\partial Q} = 0.$$

### Proof
Since $U_{\text{defer}}(Q) = U_{\text{defer}}(0) - \gamma Q$ is strictly decreasing in $Q$ with slope $-\gamma$:
- $U_{\text{defer}}(Q) > U_{\text{clear}} \iff Q < Q_0^*$,
- $U_{\text{defer}}(Q) > U_{\text{act}} \iff Q < Q_1^*$,
- $U_{\text{defer}}(Q) > U_{\text{ovr}} \iff Q < Q_2^*$.
Since $U_{\text{ovr}} < U_{\text{act}} < U_{\text{clear}}$, we have $Q_0^* < Q_1^* < Q_2^*$. Note that the third term $p_{\text{no}} \max\{U_{\text{act}}, U_{\text{defer}}(Q)\}$ matches the baseline $\max\{U_{\text{act}}, U_{\text{defer}}(Q)\}$, so subtracting the baseline yields the identity:
$$\text{EVOI}(\text{ASK}; Q) = p_{\text{res}}\max\{U_{\text{clear}}, U_{\text{defer}}(Q)\} + p_{\text{ovr\_after}}\max\{U_{\text{ovr}}, U_{\text{defer}}(Q)\} - (1 - p_{\text{no}})\max\{U_{\text{act}}, U_{\text{defer}}(Q)\} - c_{\text{ask}},$$
using $1 - p_{\text{no}} = p_{\text{res}} + p_{\text{ovr\_after}}$. Evaluating on each open interval:

1. **Regime I ($Q_0^* < Q < Q_1^*$):** Here $U_{\text{ovr}} < U_{\text{act}} < U_{\text{defer}}(Q) < U_{\text{clear}}$. Substituting the maxima gives:
   $$\text{EVOI}(\text{ASK}; Q) = p_{\text{res}} U_{\text{clear}} + p_{\text{ovr\_after}} U_{\text{defer}}(Q) - (p_{\text{res}} + p_{\text{ovr\_after}}) U_{\text{defer}}(Q) - c_{\text{ask}} = p_{\text{res}}\bigl(U_{\text{clear}} - U_{\text{defer}}(Q)\bigr) - c_{\text{ask}}.$$
   Differentiating with respect to $Q$ and $\hat{p}_e$ (noting $U_{\text{defer}}(Q) = \hat{p}_e R_+ + (1-\hat{p}_e)R_- - c_{\text{defer}} - \gamma Q = R_- + \hat{p}_e\Delta R - c_{\text{defer}} - \gamma Q$) yields:
   $$\frac{\partial\,\text{EVOI}}{\partial Q} = -p_{\text{res}} \frac{\partial U_{\text{defer}}}{\partial Q} = +p_{\text{res}}\gamma > 0, \qquad \frac{\partial\,\text{EVOI}}{\partial \hat{p}_e} = -p_{\text{res}}\Delta R < 0.$$
   (Notice that friction $p_{\text{no}}$ cancels out of the no-info branch and scales the Regime-I slope through $p_{\text{res}}$.)

2. **Regime II ($Q_1^* < Q < Q_2^*$):** Here $U_{\text{ovr}} < U_{\text{defer}}(Q) < U_{\text{act}} < U_{\text{clear}}$. Thus the baseline switches to $U_{\text{act}}$, the `clear` branch stays at $U_{\text{clear}}$, and the `ovr_after` branch still defers at $U_{\text{defer}}(Q)$:
   $$\text{EVOI}(\text{ASK}; Q) = p_{\text{res}} U_{\text{clear}} + p_{\text{ovr\_after}} U_{\text{defer}}(Q) - (p_{\text{res}} + p_{\text{ovr\_after}}) U_{\text{act}} - c_{\text{ask}}.$$
   Differentiating with respect to $Q$ gives:
   $$\frac{\partial\,\text{EVOI}}{\partial Q} = p_{\text{ovr\_after}} \frac{\partial U_{\text{defer}}}{\partial Q} = -p_{\text{ovr\_after}}\gamma.$$
   When $p_{\text{no}} = 0$, $p_{\text{ovr\_after}} = 1 - p_{\text{res}}$, recovering $-(1 - p_{\text{res}})\gamma$.

3. **Regime III ($Q > Q_2^*$):** Here $U_{\text{defer}}(Q) < U_{\text{ovr}} < U_{\text{act}} < U_{\text{clear}}$, so every post-ASK branch and the baseline select direct execution (`ACT`), eliminating dependence on $Q$:
   $$\text{EVOI}(\text{ASK}; Q) = p_{\text{res}} U_{\text{clear}} + p_{\text{ovr\_after}} U_{\text{ovr}} - (p_{\text{res}} + p_{\text{ovr\_after}}) U_{\text{act}} - c_{\text{ask}}, \qquad \frac{\partial\,\text{EVOI}}{\partial Q} = 0.$$
Continuity at $Q_1^*$ and $Q_2^*$ follows immediately from continuity of $\max(\cdot,\cdot)$. $\blacksquare$

---

### Corollary 1 (Unclarified Rework Boundary Shift)
Consider the unclarified-ticket rework setting where deferring an unclarified underspecified ticket incurs an expected rework probability $p_{\text{rework}} = p_{\text{res}} + p_{\text{ovr\_after}} > 0$, extra round-trip cost $c_{\text{ask}} > 0$, and capped specialist reliability $\tilde{p}_e^{\text{rw}} = \min(\hat{p}_e, p_{\text{cap}}) \le \hat{p}_e$ (with $p_{\text{cap}} = 0.55$ in Stage 5f). Pre-ASK deferral utility becomes:
$$U_{\text{defer}}^{\text{rework}}(Q) = (1 - p_{\text{rework}})U_{\text{defer}}(Q) + p_{\text{rework}}\Bigl(R_- + \tilde{p}_e^{\text{rw}}\Delta R - c_{\text{defer}} - c_{\text{ask}} - \gamma Q\Bigr) = U_{\text{defer}}(Q) - \Delta_{\text{rw}},$$
where $\Delta_{\text{rw}} := p_{\text{rework}}\bigl[(\hat{p}_e - \tilde{p}_e^{\text{rw}})\Delta R + c_{\text{ask}}\bigr] > 0$, while **clarified** Stage-2 post-ASK deferral utility on the `clear` and `ovr_after` branches remains $U_{\text{defer}}(Q)$ (and the unclarified `no_info` branch uses $U_{\text{defer}}^{\text{rework}}(Q)$). Then:
1. The Regime I $\leftrightarrow$ II boundary shifts strictly left: $Q_1^{*,\text{rework}} = Q_1^* - \frac{\Delta_{\text{rw}}}{\gamma} < Q_1^*$.
2. The Regime II $\leftrightarrow$ III boundary $Q_2^*$ is invariant on the clarified post-ASK `ovr_after` branch (`DEFER` is chosen iff $Q < Q_2^*$).
3. For every $Q \le Q_1^{*,\text{rework}}$, $\text{EVOI}^{\text{rework}}(\text{ASK}; Q) - \text{EVOI}(\text{ASK}; Q) = (1 - p_{\text{no}})\Delta_{\text{rw}} > 0$, strictly expanding the `ASK` trigger region even under light queue load ($Q = 0$).

#### Proof
Subtracting $U_{\text{defer}}^{\text{rework}}(Q)$ from $U_{\text{defer}}(Q)$ yields $\Delta_{\text{rw}} = p_{\text{rework}}[(\hat{p}_e - \tilde{p}_e^{\text{rw}})\Delta R + c_{\text{ask}}] > 0$. Equating $U_{\text{defer}}^{\text{rework}}(Q) = U_{\text{act}}$ yields $Q_1^{*,\text{rework}} = Q_1^* - \Delta_{\text{rw}}/\gamma < Q_1^*$. On the clarified `clear` and `ovr_after` branches (both implemented cleanly with post-ASK $U_{\text{defer}}(Q)$), $U_{\text{ovr}} = U_{\text{defer}}(Q)$ still occurs at $Q_2^*$. Finally, for $Q \le Q_1^{*,\text{rework}} < Q_1^*$, the pre-ASK `DEFER` baseline drops by $\Delta_{\text{rw}}$, while `clear` is unchanged (`ACT`), `ovr_after` uses $U_{\text{defer}}(Q)$ unchanged, and only the `no_info` branch drops by $p_{\text{no}}\Delta_{\text{rw}}$, giving net EVOI increase $(1 - p_{\text{no}})\Delta_{\text{rw}} > 0$. $\blacksquare$

---

## Theorem 2 (Single-Step Fixed-State Miscalibration Sensitivity Upper Bound & Empirical Bayes Corollary)

### Statement
Fix queue state $Q$, diagnostic bucket $b = (\text{obs\_missing}, \text{obs\_pot\_ovr})$ (for coarse 2-Call `Tri-VoI-MDP`) or $b = (\text{obs\_missing}, \text{obs\_pot\_ovr}, \text{unanimous})$ (for fine `Tri-VoI-L2D`, held fixed; bucket-assignment error from imperfect diagnosis induces an additional bucket-mixing bias noted below), true specialist reliability $p_e$, and tracker estimate $\hat{p}_e$. Let $U_{\text{defer}} := U_{\text{defer}}(Q; p_e)$ and $\widehat{U}_{\text{defer}} := U_{\text{defer}}(Q; \hat{p}_e)$. Suppose true parameters $(p, \pi, p_e)$ and estimates $(\hat{p}, \hat{\pi}, \hat{p}_e)$ satisfy:
$$\max_{a \in \{\text{act}, \text{clear}, \text{ovr}, \text{no\_info}\}} |\hat{p}_a - p_a| \le \epsilon_p, \qquad \|\hat{\pi} - \pi\|_1 = \sum_{k=1}^3 |\hat{\pi}_k - \pi_k| \le \epsilon_\pi, \qquad |\hat{p}_e - p_e| \le \epsilon_e.$$
Let $a^* \in \arg\max_{a \in \{\text{ACT}, \text{ASK}, \text{DEFER}\}} U(a)$ be the true-parameter optimal two-stage policy and $\hat{a} \in \arg\max_{a \in \{\text{ACT}, \text{ASK}, \text{DEFER}\}} \hat{U}(a)$ be the implemented calibrated policy whose Stage-2 choice on outcome $k \in \{\text{clear}, \text{ovr}, \text{no\_info}\}$ (identifying $\text{no\_info}$ with calibrated post-unrevealed execution $\hat{p}_{\text{no\_info}}$ in `Tri-VoI-MDP`) executes `ACT` iff $\hat{U}_k \ge \widehat{U}_{\text{defer}}$, attaining true expected utility $U_{\text{exec}}(\hat{a})$. Define $M_k := \max\{U_k, U_{\text{defer}}\}$, $\widehat{M}_k := \max\{\hat{U}_k, \widehat{U}_{\text{defer}}\}$, $\text{Span}(Q) := \max_k M_k - \min_k M_k$, and $\text{Span}_{\text{exec}}(Q) := \max_k U_{\text{exec},k} - \min_k U_{\text{exec},k}$. Then the single-step suboptimality gap satisfies:
$$U(a^*) - U_{\text{exec}}(\hat{a}) \le 2\,\Delta R\,(\epsilon_p + \epsilon_e) + \frac{\text{Span}(Q) + \text{Span}_{\text{exec}}(Q)}{2}\,\epsilon_\pi \le \Delta R\,(2\,\epsilon_p + 2\,\epsilon_e + \epsilon_\pi),$$
where the last inequality holds whenever $U_{\text{defer}}, U_{\text{exec},k} \in [R_- - c_{\text{defer}}, R_+]$ or $\text{Span}, \text{Span}_{\text{exec}} \le \Delta R$.

### Proof
For $a \in \{\text{act}, \text{clear}, \text{ovr}\}$, $|\hat{U}_a - U_a| \le \Delta R\,\epsilon_p$, and $|\widehat{U}_{\text{defer}} - U_{\text{defer}}| \le \Delta R\,\epsilon_e$. By 1-Lipschitzness of $\max(\cdot,\cdot)$, $|\widehat{M}_k - M_k| \le \max\{|\hat{U}_k - U_k|, |\widehat{U}_{\text{defer}} - U_{\text{defer}}|\} \le \Delta R\max\{\epsilon_p, \epsilon_e\} \le \Delta R(\epsilon_p + \epsilon_e)$ for every $k$.

1. **Exact 4-case one-sided bound on Stage-2 execution gap $\widehat{M}_k - U_{\text{exec},k}$:** Under implemented Stage-2 branching, $U_{\text{exec},k} = U_k$ if $\hat{U}_k \ge \widehat{U}_{\text{defer}}$, and $U_{\text{exec},k} = U_{\text{defer}}$ if $\hat{U}_k < \widehat{U}_{\text{defer}}$. Checking all 4 sign combinations of $(\text{true Stage-2} \times \text{estimated Stage-2})$:
   - Case 1 ($\hat{U}_k \ge \widehat{U}_{\text{defer}},\, U_k \ge U_{\text{defer}}$): $\widehat{M}_k - U_{\text{exec},k} = \hat{U}_k - U_k \le \Delta R\,\epsilon_p \le \Delta R(\epsilon_p + \epsilon_e)$.
   - Case 2 ($\hat{U}_k < \widehat{U}_{\text{defer}},\, U_k < U_{\text{defer}}$): $\widehat{M}_k - U_{\text{exec},k} = \widehat{U}_{\text{defer}} - U_{\text{defer}} \le \Delta R\,\epsilon_e \le \Delta R(\epsilon_p + \epsilon_e)$.
   - Case 3 ($\hat{U}_k \ge \widehat{U}_{\text{defer}}$ so `ACT` is executed, yet $U_k < U_{\text{defer}}$): $\widehat{M}_k - U_{\text{exec},k} = \hat{U}_k - U_k \le \Delta R\,\epsilon_p \le \Delta R(\epsilon_p + \epsilon_e)$ (even though the decision switched!).
   - Case 4 ($\hat{U}_k < \widehat{U}_{\text{defer}}$ so `DEFER` is executed, yet $U_k \ge U_{\text{defer}}$): $\widehat{M}_k - U_{\text{exec},k} = \widehat{U}_{\text{defer}} - U_{\text{defer}} \le \Delta R\,\epsilon_e \le \Delta R(\epsilon_p + \epsilon_e)$.
   In ALL 4 cases, $\widehat{M}_k - U_{\text{exec},k} \le \Delta R(\epsilon_p + \epsilon_e)$ (without incurring a double penalty!).

2. **Upper-bounding $U(a^*) - \hat{U}(a^*)$:** Let $\bar{M} := \frac{1}{2}(\max_k M_k + \min_k M_k)$. Since $\sum_k (\pi_k - \hat{\pi}_k) = 0$:
   - For $a^* = \text{ACT}$: $U(\text{ACT}) - \hat{U}(\text{ACT}) \le \Delta R\,\epsilon_p$.
   - For $a^* = \text{DEFER}$: $U(\text{DEFER}) - \hat{U}(\text{DEFER}) \le \Delta R\,\epsilon_e$.
   - For $a^* = \text{ASK}$:
     $$U(\text{ASK}) - \hat{U}(\text{ASK}) = \sum_k \hat{\pi}_k(M_k - \widehat{M}_k) + \sum_k (\pi_k - \hat{\pi}_k)(M_k - \bar{M}) \le \Delta R(\epsilon_p + \epsilon_e) + \frac{\text{Span}(Q)}{2}\epsilon_\pi.$$

3. **Upper-bounding $\hat{U}(\hat{a}) - U_{\text{exec}}(\hat{a})$:** Let $\bar{M}_{\text{exec}} := \frac{1}{2}(\max_k U_{\text{exec},k} + \min_k U_{\text{exec},k})$.
   - For $\hat{a} = \text{ACT}$: $\hat{U}(\text{ACT}) - U(\text{ACT}) \le \Delta R\,\epsilon_p$.
   - For $\hat{a} = \text{DEFER}$: $\hat{U}(\text{DEFER}) - U(\text{DEFER}) \le \Delta R\,\epsilon_e$.
   - For $\hat{a} = \text{ASK}$:
     $$\hat{U}(\text{ASK}) - U_{\text{exec}}(\text{ASK}) = \sum_k \hat{\pi}_k(\widehat{M}_k - U_{\text{exec},k}) + \sum_k (\hat{\pi}_k - \pi_k)(U_{\text{exec},k} - \bar{M}_{\text{exec}}) \le \Delta R(\epsilon_p + \epsilon_e) + \frac{\text{Span}_{\text{exec}}(Q)}{2}\epsilon_\pi,$$
     using Step 1 on every branch $k$.

4. **Combining via $\hat{U}(a^*) - \hat{U}(\hat{a}) \le 0$:**
   $$U(a^*) - U_{\text{exec}}(\hat{a}) = \bigl(U(a^*) - \hat{U}(a^*)\bigr) + \bigl(\hat{U}(a^*) - \hat{U}(\hat{a})\bigr) + \bigl(\hat{U}(\hat{a}) - U_{\text{exec}}(\hat{a})\bigr) \le 2\,\Delta R\,(\epsilon_p + \epsilon_e) + \frac{\text{Span}(Q) + \text{Span}_{\text{exec}}(Q)}{2}\,\epsilon_\pi.$$
   *(Scope Note: This is a single-step, fixed-queue, fixed-bucket bound. Multi-step cross-ticket queue drift amplification and bucket-assignment misdiagnosis errors—such as `Qwen3-8B` mixing `Q2` into `Q1`—are not absorbed into $(\epsilon_p, \epsilon_\pi)$ and constitute empirical boundary conditions.)* $\blacksquare$

### Corollary 2 (Finite-Sample Empirical Bayes Scaling)
Consider a held-out calibration bucket $b$ containing $n_b \ge 1$ i.i.d. samples, estimated with Dirichlet/Beta shrinkage prior strength $k > 0$ toward prior mean $\mu_0$ (as in `HeldOutCalibrator`). With probability at least $1 - \delta$ ($\delta \in (0,1)$):
$$\epsilon_p(n_b, k),\, \epsilon_\pi(n_b, k) = O\!\left(\sqrt{\frac{\log(1/\delta)}{n_b}} + \frac{k}{n_b + k}\right).$$

#### Proof
For a scalar binomial proportion $\hat{p}_{\text{EB}} = \frac{\sum_{i=1}^{n_b} Z_i + k\mu_0}{n_b + k}$ with $Z_i \in \{0,1\}$ and true mean $p = \mathbb{E}[Z_i]$:
$$|\hat{p}_{\text{EB}} - p| \le \frac{n_b}{n_b + k}\left|\frac{1}{n_b}\sum_{i=1}^{n_b} Z_i - p\right| + \frac{k}{n_b + k}|\mu_0 - p|.$$
By Hoeffding's inequality, $\left|\frac{1}{n_b}\sum_{i=1}^{n_b} Z_i - p\right| \le \sqrt{\frac{\log(2/\delta)}{2n_b}}$ with probability at least $1 - \delta$, while $|\mu_0 - p| \le 1$. Applying a union bound across the 3 multinomial coordinates of $\hat{\pi}$ (each with $|\mu_{0,j} - \pi_j| \le 1$, $\sum |\mu_{0,j} - \pi_j| \le 2$) gives $\|\hat{\pi}_{\text{EB}} - \pi\|_1 \le 3\sqrt{\frac{\log(6/\delta)}{2n_b}} + \frac{2k}{n_b + k}$. Together, both errors scale as $O\bigl(\sqrt{\log(1/\delta)/n_b} + k/(n_b+k)\bigr)$. This explains three key phenomena: (i) `GLM-4.5-Air` variance under small calibration buckets ($n_{\text{cal}}=60 \implies n_b \approx 7\text{--}15$), (ii) why coarse 2-Call bucketing (`b = (obs_missing, obs_pot_ovr)` in `Tri-VoI-MDP`) roughly doubles effective bucket size $n_b$ while cutting LLM probe tokens by $54\%\text{--}58\%$, reducing finite-sample variance when diagnostic flags are informative, and (iii) `Qwen3-8B` diagnostic floor (low Missing-F1 violates the fixed-bucket assumption by mixing `Q2` and `Q1`). $\blacksquare$

---

## Theorem 3 (Accuracy-Constrained Throughput Stability & Non-Zero `ASK` Response Latency)

### Part A — Accuracy-Constrained Capacity-Region Expansion

#### Statement
Consider a discrete-time ticket stream arriving at rate $\lambda_{\text{tot}} > 0$ per slot (bounded-variance arrivals $A_t^{\text{tot}}$ with $\mathbb{E}[A_t^{\text{tot}}] = \lambda_{\text{tot}}$ and $\mathbb{E}[(A_t^{\text{tot}})^2] < \infty$). Tickets belong to:
- AI-solvable clear (`Q1`, share $1 - \alpha_{\text{spec}} - \alpha_{\text{epist}}$),
- Requester-underspecified (`Q2` and `Q4`, share $\alpha_{\text{spec}}$), of which a fraction $\bar{p}_{\text{res}} \in (0,1]$ resolves to `ACT` after `ASK` and $(1 - \bar{p}_{\text{res}})$ still requires Specialist `DEFER`,
- Unconditional Specialist-only (`Q3`, share $\alpha_{\text{epist}}$).

Let the Specialist pool serve $S_t$ tickets per slot with mean service rate $\mu := \mathbb{E}[S_t] > 0$ and $\mathbb{E}[S_t^2] < \infty$, evolving via $Q_{t+1} = \max\{0, Q_t + A_t^{\text{defer}} - S_t\}$. Compare policies inside the **accuracy-preserving class $\Pi_{\text{acc}}$** that route every unresolvable Specialist-only or unclarified-underspecified ticket to Specialist deferral rather than sacrificing accuracy on unclarified/epistemic `ACT` *(Remark: any finite-penalty threshold controller with $\gamma > 0$ remains bounded/positive-recurrent outside $\Pi_{\text{acc}}$ once $Q_t > Q_{\max}$, but does so precisely by dropping tickets onto low-accuracy `ACT`; this happens in practice whenever bursty congestion pushes `Tri-VoI-L2D` into Regimes II–III)*:
1. **Binary `{ACT, DEFER}` Policy in $\Pi_{\text{acc}}$ (no `ASK` channel):** Must defer rate $\lambda_{\text{base}}^{\text{defer}} = \lambda_{\text{tot}}(\alpha_{\text{spec}} + \alpha_{\text{epist}})$, hence requires $\lambda_{\text{tot}} < \lambda_{\text{base}}^* := \frac{\mu}{\alpha_{\text{spec}} + \alpha_{\text{epist}}}$ for queue stability inside $\Pi_{\text{acc}}$.
2. **Ternary `{ACT, ASK, DEFER}` Policy in $\Pi_{\text{acc}}$ (`Tri-VoI-L2D`):** Resolves share $\bar{p}_{\text{res}}\alpha_{\text{spec}}$ via `ASK -> ACT`, lowering Specialist arrival rate to:
   $$\lambda_{\text{Tri}}^{\text{defer}} = \lambda_{\text{tot}}\bigl(\alpha_{\text{epist}} + (1 - \bar{p}_{\text{res}})\alpha_{\text{spec}}\bigr) < \lambda_{\text{base}}^{\text{defer}},$$
   giving stability ceiling $\lambda_{\text{Tri}}^* := \frac{\mu}{\alpha_{\text{epist}} + (1 - \bar{p}_{\text{res}})\alpha_{\text{spec}}} > \lambda_{\text{base}}^*$. Whenever $\lambda_{\text{Tri}}^{\text{defer}} < \mu$, quadratic Lyapunov $V(Q) = \frac{1}{2}Q^2$ establishes positive recurrence with drift gap $\varepsilon := \mu - \lambda_{\text{Tri}}^{\text{defer}} > 0$ and steady-state mean-queue bound:
   $$\limsup_{T \to \infty} \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[Q_t] \le \frac{B}{\varepsilon}, \qquad B := \frac{1}{2}\mathbb{E}\bigl[(A_t^{\text{defer}} - S_t)^2\bigr].$$
   In the throughput window $\lambda_{\text{base}}^* < \lambda_{\text{tot}} < \lambda_{\text{Tri}}^*$, no `{ACT, DEFER}` policy can both maintain $\Pi_{\text{acc}}$ accuracy and keep the queue stable, whereas `{ACT, ASK, DEFER}` admits a stabilizing $\Pi_{\text{acc}}$ policy.

#### Proof
Using $Q_{t+1}^2 \le (Q_t + A_t^{\text{defer}} - S_t)^2$,
$$\mathbb{E}\bigl[V(Q_{t+1}) - V(Q_t) \mid Q_t = q\bigr] \le q(\lambda_{\text{Tri}}^{\text{defer}} - \mu) + \frac{1}{2}\mathbb{E}[(A_t^{\text{defer}} - S_t)^2] = -\varepsilon q + B.$$
By Foster-Lyapunov positive recurrence and telescoping over $t=0\dots T-1$, $\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}[Q_t] \le \frac{\mathbb{E}[V(Q_0)]}{\varepsilon T} + \frac{B}{\varepsilon} \to \frac{B}{\varepsilon}$. Conversely, for any `{ACT, DEFER}` policy in $\Pi_{\text{acc}}$ when $\lambda_{\text{tot}} > \lambda_{\text{base}}^*$, $\mathbb{E}[A_t^{\text{defer}} - S_t] > 0$, so $Q_t \to \infty$ a.s. (or accuracy must be sacrificed). $\blacksquare$

---

### Part B — Non-Zero `ASK` Response Latency ($\tau_{\text{ask}} > 0$)

#### Statement
Suppose an `ASK` reply returns after $\tau_{\text{ask}} \ge 0$ slots, incurring explicit requester waiting penalty $\beta_{\text{wait}} \tau_{\text{ask}}$ ($\beta_{\text{wait}} \ge 0$), while during $[t, t+\tau_{\text{ask}}]$ (incorporating arrival of any in-flight tickets over $\tau_{\text{ask}}$ slots) the Specialist queue evolves randomly to $Q_{t+\tau_{\text{ask}}}$ with conditional expectation $\mathbb{E}[Q_{t+\tau_{\text{ask}}} \mid Q_t] = Q_t + \Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t)$. Let:
$$U_{\text{ASK}}^{(\tau_{\text{ask}})}(Q_t) := \mathbb{E}\Bigl[p_{\text{res}}\max\{U_{\text{clear}}, U_{\text{defer}}(Q_{t+\tau_{\text{ask}}})\} + p_{\text{ovr\_after}}\max\{U_{\text{ovr}}, U_{\text{defer}}(Q_{t+\tau_{\text{ask}}})\} + p_{\text{no}}\max\{U_{\text{act}}, U_{\text{defer}}(Q_{t+\tau_{\text{ask}}})\}\Bigr] - c_{\text{ask}} - \beta_{\text{wait}}\tau_{\text{ask}}.$$
1. **Convexity-Jensen Lower Bound (General Regimes):** Because $q \mapsto \max\{u, U_{\text{defer}}(0) - \gamma q\}$ is convex in $q$, Jensen's inequality gives:
   $$U_{\text{ASK}}^{(\tau_{\text{ask}})}(Q_t) \ge U_{\text{ASK}}\!\bigl(Q_t + \Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t)\bigr) - \beta_{\text{wait}}\tau_{\text{ask}}.$$
2. **Exact Regime-I Shift & Maximum Tolerable Latency:** Whenever $Q_t$ and $Q_{t+\tau_{\text{ask}}}$ remain in Regime I ($Q_0^* < Q < Q_1^*$):
   $$\Delta\text{EVOI}(\tau_{\text{ask}}) := \text{EVOI}^{(\tau_{\text{ask}})}(\text{ASK}; Q_t) - \text{EVOI}^{(0)}(\text{ASK}; Q_t) = -(1 - p_{\text{res}})\,\gamma\,\Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t) - \beta_{\text{wait}}\tau_{\text{ask}},$$
   with closed-form maximum tolerable delay under drift $\bar{\delta}\tau_{\text{ask}}$ ($\bar{\delta} \ge 0$):
   $$\tau_{\text{ask}}^{\max}(Q_t) = \frac{\text{EVOI}^{(0)}(\text{ASK}; Q_t)}{(1 - p_{\text{res}})\,\gamma\,\bar{\delta} + \beta_{\text{wait}}}.$$
   *(Implementation alignment: `Tri-VoI-L2D (Latency-Aware)` approximates this online using EMA-estimated positive backlog growth $\hat{\Delta}_{\text{drift}} = \max(0, \hat{\lambda}_{\text{defer}} - \mu)\tau_{\text{ask}}$.)*

#### Proof
By convexity of $f_u(q) = \max\{u, U_{\text{defer}}(0)-\gamma q\}$ and linearity inside Regime I, both bounds follow immediately as above. $\blacksquare$

---

## Theorem 4 (Finite-Horizon Periodic-Drain 3-Action Bellman Optimality & Shadow-Price Monotonicity)

### Setup
Consider the finite-horizon periodic-drain tri-party queue MDP (`Tri-VoI-MDP` in `solve_3action_mdp_shadow_prices`) with service batch size $C \ge 0$ drained whenever slot index $t \equiv 0 \pmod I$ ($I \ge 1$ drain interval). Let $h = N - 1 - t \ge 0$ denote remaining steps after step $t$, $\phi \in \{0,\dots,I-1\}$ denote the phase $t \bmod I$, $\phi' := (\phi + 1) \bmod I$, and $q_{\text{pre}} \ge 0$ denote pre-drain queue backlog, giving post-drain queue $q = \max\{0, q_{\text{pre}} - C\,\mathbf{1}[\phi = 0]\}$. Given coarse type distribution $\omega \in \Delta^4$ over $b = (\text{obs\_missing}, \text{obs\_pot\_ovr})$, calibrate Stage-1/Stage-2 utilities $(u_{\text{act}, b}, u_{\text{clear}, b}, u_{\text{ovr}, b}, u_{\text{no\_info}, b})$, post-clarification defer base reward $u_{\text{def,post}, 0} := \hat{p}_e R_+ + (1-\hat{p}_e)R_- - c_{\text{defer}}$, and pre-clarification/unclarified defer base reward $u_{\text{def,pre}, b, 0} \le u_{\text{def,post}, 0}$ (accounting for rework penalty $p_{\text{rw}, b}$ when active). Setting $V_0(\phi, q_{\text{pre}}) \equiv 0$, define backward for $h \ge 1$:
$$V_h(\phi, q_{\text{pre}}) = \sum_{b} \omega_b \max\bigl\{u_{\text{act}, b} + V_{h-1}(\phi', q),\; u_{\text{def,pre}, b, 0} - \gamma q + V_{h-1}(\phi', q+1),\; Q_h^{\text{ASK}}(\phi, q; b)\bigr\},$$
where for $\text{obs\_missing}(b) = \text{True}$,
$$Q_h^{\text{ASK}}(\phi, q; b) = p_{\text{res}, b}\max\bigl\{u_{\text{clear}, b} + V_{h-1}(\phi', q),\; u_{\text{def,post}, 0} - \gamma q + V_{h-1}(\phi', q+1)\bigr\} + p_{\text{ovra}, b}\max\bigl\{u_{\text{ovr}, b} + V_{h-1}(\phi', q),\; u_{\text{def,post}, 0} - \gamma q + V_{h-1}(\phi', q+1)\bigr\} + p_{\text{no}, b}\max\bigl\{u_{\text{no\_info}, b} + V_{h-1}(\phi', q),\; u_{\text{def,pre}, b, 0} - \gamma q + V_{h-1}(\phi', q+1)\bigr\} - c_{\text{ask}}$$
(and $\text{ASK}$ is infeasible when $\text{obs\_missing}(b) = \text{False}$). Define the exact one-step marginal shadow price at phase $\phi$ and post-drain queue $q$:
$$\Delta V_{h-1}(\phi', q) := V_{h-1}(\phi', q) - V_{h-1}(\phi', q+1).$$

### Statement
For all horizons $h \ge 1$, phases $\phi \in \{0,\dots,I-1\}$, and queue states $q \ge 0$:
1. **Shadow-Price Threshold Equivalence:** The Bellman-optimal action at $(\phi, q, b)$ coincides with choosing `ACT`, pre-ASK `DEFER`, or 3-outcome `ASK` using effective deferral utilities $\widetilde{U}_{\text{defer,post}}(\phi, q) := u_{\text{def,post}, 0} - \gamma q - \Delta V_{h-1}(\phi', q)$ and $\widetilde{U}_{\text{defer,pre}, b}(\phi, q) := u_{\text{def,pre}, b, 0} - \gamma q - \Delta V_{h-1}(\phi', q)$.
2. **Non-Negativity, Boundedness & High-Queue Abandonment Saturation:** $q_{\text{pre}} \mapsto V_h(\phi, q_{\text{pre}})$ is monotonically non-increasing in $q_{\text{pre}}$, and the marginal shadow price satisfies:
   $$0 \le \Delta V_{h-1}(\phi', q) \le \gamma(h-1), \qquad \lim_{q \to \infty} \Delta V_{h-1}(\phi', q) = 0.$$
   Unlike unbounded linear Naor proxies ($\gamma\rho(q+1) \to \infty$), the Bellman shadow price naturally saturates to $0$ once $q$ crosses the abandonment threshold (where future tickets select `ACT`/`ASK` rather than `DEFER`), while remaining strictly positive at sub-capacity $q < C$ whenever $I > 1$ or future deferrals occur with positive probability.
3. **Finite-Horizon Value Dominance:** Under matched queue dynamics and calibrated parameters $(\omega, p, \pi, \hat{p}_e)$, $V_h(\phi, q_{\text{pre}}) \ge V_h^\pi(\phi, q_{\text{pre}})$ for any myopic single-step policy $\pi$ (including `Direct-ACT`, `Joint-Greedy+Naor`, `Calibrated-Cascade+Naor`, and gated-Naor `Tri-VoI-L2D`).

### Proof
1. **Part 1:** Subtract $V_{h-1}(\phi', q)$ inside every $\max\{\cdot,\cdot\}$ in $V_h$ and $Q_h^{\text{ASK}}$. Since $u_{\text{def}, 0} - \gamma q + V_{h-1}(\phi', q+1) - V_{h-1}(\phi', q) = u_{\text{def}, 0} - \gamma q - \Delta V_{h-1}(\phi', q)$ and all coefficients $(\omega_b, p_{\text{res}, b}, p_{\text{ovra}, b}, p_{\text{no}, b}) \ge 0$, the maximizing Stage-1 and Stage-2 branches remain invariant.
2. **Part 2 (Lower Bound by Induction + Upper Bound by Sample-Path Coupling):**
   - *Monotone non-increasing ($V_h \ge V_h(\cdot+1)$):* At $h=0$, $V_0 \equiv 0$. Assume $q' \mapsto V_{h-1}(\phi', q')$ is non-increasing. Since $q \mapsto -\gamma q + V_{h-1}(\phi', q+1)$ and $q \mapsto V_{h-1}(\phi', q)$ are both non-increasing, and $\max\{\cdot,\cdot\}$, nonnegative linear combinations, and composition with the non-decreasing map $q_{\text{pre}} \mapsto \max\{0, q_{\text{pre}} - C\mathbf{1}[\phi=0]\}$ preserve non-increasingness, $V_h(\phi, q_{\text{pre}})$ is non-increasing, proving $\Delta V_{h-1}(\phi', q) \ge 0$.
   - *Sample-path coupling upper bound:* Fix any $\phi', q$, let $\pi^*$ be an optimal Markov policy starting from $(\phi', q+1)$ over $h-1$ steps, generating stochastic queue sequence $\{Q_s^{(q+1)}\}_{s=0}^{h-2}$ with terminal payoff sum $R^{(q+1)}$. Run the **exact same action choices** (coupled type realizations $b_s$ and Stage-2 outcomes) starting from $(\phi', q)$ to produce queue trajectory $\{Q_s^{(q)}\}_{s=0}^{h-2}$. Because drain $\max\{0, x - C\}$ and unit arrival $x + 1$ are $1$-Lipschitz monotone maps satisfying $0 \le x' - x \le 1 \implies 0 \le f(x') - f(x) \le 1$, by induction $0 \le Q_s^{(q+1)} - Q_s^{(q)} \le 1$ at every step $s$. Hence each coupled `DEFER` step incurs at most $\gamma$ less waiting cost at $q$ than at $q+1$, while non-`DEFER` steps incur identical rewards, yielding $R^{(q+1)} - R^{(q)} \ge -\gamma(h-1)$. By optimality of $V_{h-1}(\phi', q) \ge \mathbb{E}[R^{(q)}]$, $\Delta V_{h-1}(\phi', q) \le \gamma \mathbb{E}_{\pi^*}[\sum_{s=0}^{h-2}\mathbf{1}[\text{defer}_s]] \le \gamma(h-1)$.
   - *Abandonment limit:* Choose $q > Q_{\max}^{\text{defer}} + (h-1)C$, where $Q_{\max}^{\text{defer}} := \gamma^{-1}\max_b(u_{\text{def,post},0} - u_{\text{act},b})$. Even after all $(h-1)$ possible drains, $Q_s^{(q)} > Q_{\max}^{\text{defer}}$, so `ACT` strictly dominates `DEFER` at every remaining step from both $q$ and $q+1$, making $V_{h-1}(\phi', q) = V_{h-1}(\phi', q+1)$ and thus $\Delta V_{h-1}(\phi', q) = 0$.
3. **Part 3:** Immediate from Bellman's Principle of Optimality, as $V_h$ maximizes expected $h$-step cumulative reward over the full Markov policy class containing all stationary myopic threshold policies $\pi$. $\blacksquare$
