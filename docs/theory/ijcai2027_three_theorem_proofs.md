# Mathematical Specification and Full Proofs: IJCAI 2027 Three-Theorem Package

> **Tracking:** [`RUSS0718/TRI_VOI#8`](https://github.com/RUSS0718/TRI_VOI/issues/8) & [`RUSS0718/TRI_VOI#15`](https://github.com/RUSS0718/TRI_VOI/issues/15)  
> **Scope:** Complete statements and line-by-line proofs for Appendix A of the IJCAI 2027 manuscript (*Tri-VoI-L2D*), paired with `¥0` offline numerical verification in [`verify_theory_package.py`](../../verify_theory_package.py).

---

## Notation and Standing Assumptions

At decision step $t$, the **Requester** submits a ticket $x_t$ and the **Specialist** pool (for clarity in single-pool analysis, one server with capacity $C_m$, service rate $\mu = C_m / \Delta_{\text{drain}}$, queue backlog $Q_t \in \mathbb{Z}_{\ge 0}$, marginal linear waiting penalty $\gamma > 0$, deferral fee $c_{\text{defer}} \ge 0$, and clarification cost $c_{\text{ask}} > 0$) has true reliability $p_e \in (0,1)$ and estimate $\hat{p}_e \in (0,1)$. Payoffs are $R_+ > R_-$ with reward span $\Delta R := R_+ - R_- > 0$.

For any observable diagnostic state $s = (\text{obs\_missing}, \text{obs\_pot\_ovr}, \text{unanimous})$:
- $U_{\text{act}} := \hat{p}_{\text{act}} R_+ + (1 - \hat{p}_{\text{act}}) R_-$.
- $U_{\text{clear}} := \hat{p}_{\text{post\_clear}} R_+ + (1 - \hat{p}_{\text{post\_clear}}) R_-$, $U_{\text{ovr}} := \hat{p}_{\text{post\_ovr}} R_+ + (1 - \hat{p}_{\text{post\_ovr}}) R_-$.
- Under linear congestion cost $C_{\text{cong}}(Q) = \gamma Q$, $U_{\text{defer}}(Q) := \hat{p}_e R_+ + (1 - \hat{p}_e) R_- - c_{\text{defer}} - \gamma Q = U_{\text{defer}}(0) - \gamma Q$. (Under gated Naor externality, $Q$ is replaced by the effective congestion term $Q_{\text{eff}}(Q) = Q + \hat{\rho}(Q+1)\mathbf{1}\{Q \ge C_m \lor \hat{\rho} \ge 0.6\}$, which is piecewise linear and strictly increasing in $Q$.)
- Following an `ASK` action, outcomes follow the exact three-way probability partition $\pi = (p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}})$ with $p_{\text{res}} + p_{\text{ovr\_after}} + p_{\text{no}} = 1$, $p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}} \ge 0$, corresponding to: (i) missing field revealed and no override applies (`clear`), (ii) missing field revealed and specialist override is confirmed (`ovr_after`), and (iii) no useful clarification obtained (`no_info`).

The two-stage Bellman expected utility of `ASK` and its Expected Value of Information (`EVOI`) over the no-ask baseline are:
$$U_{\text{ASK}}(Q) := p_{\text{res}} \max\{U_{\text{clear}}, U_{\text{defer}}(Q)\} + p_{\text{ovr\_after}} \max\{U_{\text{ovr}}, U_{\text{defer}}(Q)\} + p_{\text{no}} \max\{U_{\text{act}}, U_{\text{defer}}(Q)\} - c_{\text{ask}},$$
$$\text{EVOI}(\text{ASK}; Q) := U_{\text{ASK}}(Q) - \max\{U_{\text{act}}, U_{\text{defer}}(Q)\}.$$

---

## Theorem 1 (General 3-Outcome Three-Regime Piecewise Queue-Relief)

### Statement
Suppose the standard ordering $U_{\text{ovr}} < U_{\text{act}} < U_{\text{clear}}$ holds (confirmation of an override degrades AI accuracy below pre-ASK accuracy, whereas resolving a missing user field improves AI accuracy above pre-ASK accuracy). Define the two queue regime boundaries:
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
where $\Delta_{\text{rw}} := p_{\text{rework}}\bigl[(\hat{p}_e - \tilde{p}_e^{\text{rw}})\Delta R + c_{\text{ask}}\bigr] > 0$, while Stage-2 post-ASK deferral utility remains $U_{\text{defer}}(Q)$ because clarification has already occurred. Then:
1. The Regime I $\leftrightarrow$ II boundary shifts strictly left: $Q_1^{*,\text{rework}} = Q_1^* - \frac{\Delta_{\text{rw}}}{\gamma} < Q_1^*$.
2. The Regime II $\leftrightarrow$ III boundary $Q_2^*$ is invariant on the post-ASK branches (wait: in the post-ASK `ovr_after` branch, `DEFER` is chosen iff $Q < Q_2^*$).
3. For every $Q \le Q_1^{*,\text{rework}}$, $\text{EVOI}^{\text{rework}}(\text{ASK}; Q) - \text{EVOI}(\text{ASK}; Q) = (1 - p_{\text{no}})\Delta_{\text{rw}} > 0$, strictly expanding the `ASK` trigger region even under light queue load ($Q = 0$).

#### Proof
Subtracting $U_{\text{defer}}^{\text{rework}}(Q)$ from $U_{\text{defer}}(Q)$ directly yields $\Delta_{\text{rw}} = p_{\text{rework}}[(\hat{p}_e - \tilde{p}_e^{\text{rw}})\Delta R + c_{\text{ask}}] > 0$ (since $p_{\text{rework}} > 0$, $c_{\text{ask}} > 0$, and $\hat{p}_e \ge \tilde{p}_e^{\text{rw}}$). Equating $U_{\text{defer}}^{\text{rework}}(Q) = U_{\text{act}}$ yields $Q_1^{*,\text{rework}} = (U_{\text{defer}}(0) - \Delta_{\text{rw}} - U_{\text{act}})/\gamma = Q_1^* - \Delta_{\text{rw}}/\gamma < Q_1^*$. Meanwhile, post-ASK `DEFER` uses standard $U_{\text{defer}}(Q)$, so $U_{\text{ovr}} = U_{\text{defer}}(Q)$ still occurs at $Q_2^*$. Finally, for $Q \le Q_1^{*,\text{rework}} < Q_1^*$, the pre-ASK `DEFER` baseline drops by $\Delta_{\text{rw}}$ while `p_no * U_defer^rework(Q)` in the unclarified fallback branch drops by $p_{\text{no}}\Delta_{\text{rw}}$ (or `max(u_act, u_defer^rework)`), giving a net EVOI increase of $(1 - p_{\text{no}})\Delta_{\text{rw}} > 0$ whenever `ASK` competes with `DEFER`. $\blacksquare$

---

## Theorem 2 (Miscalibration Sensitivity Upper Bound & Finite-Sample Empirical Bayes Corollary)

### Statement
Fix queue state $Q$ and specialist estimate $\hat{p}_e$, and let $U_{\text{defer}} := U_{\text{defer}}(Q)$. Suppose the true outcome probabilities in the current bucket are $p = (p_{\text{act}}, p_{\text{clear}}, p_{\text{ovr}})$ and $\pi = (p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}})$ with $\sum \pi_k = 1$, and the held-out calibrator outputs estimates $\hat{p} = (\hat{p}_{\text{act}}, \hat{p}_{\text{clear}}, \hat{p}_{\text{ovr}})$ and $\hat{\pi} = (\hat{p}_{\text{res}}, \hat{p}_{\text{ovr\_after}}, \hat{p}_{\text{no}})$ satisfying:
$$\max_{a \in \{\text{act}, \text{clear}, \text{ovr}\}} |\hat{p}_a - p_a| \le \epsilon_p, \qquad \|\hat{\pi} - \pi\|_1 = \sum_{k=1}^3 |\hat{\pi}_k - \pi_k| \le \epsilon_\pi.$$
Let $a^* \in \arg\max_{a \in \{\text{ACT}, \text{ASK}, \text{DEFER}\}} U(a)$ be the true-parameter optimal two-stage action and $\hat{a} \in \arg\max_{a \in \{\text{ACT}, \text{ASK}, \text{DEFER}\}} \hat{U}(a)$ be the calibrated action (including Stage-2 post-ASK thresholding between `ACT` and `DEFER`). Define $\text{Span}(Q) := \max_{k \in \{\text{clear}, \text{ovr}, \text{act}\}} \max\{U_k, U_{\text{defer}}\} - \min_{k} \max\{U_k, U_{\text{defer}}\}$. Then the one-step suboptimality gap satisfies:
$$U(a^*) - U(\hat{a}) \le 2\,\Delta R\,\epsilon_p + \text{Span}(Q)\,\epsilon_\pi \le \Delta R\,(2\,\epsilon_p + \epsilon_\pi),$$
provided $U_{\text{defer}} \in [R_- - \text{overhead}, R_+]$ so $\text{Span}(Q) \le \Delta R$ (more generally, $\text{Span}(Q) \le \max\{\Delta R, U_{\text{clear}} - U_{\text{defer}}\}$).

### Proof
For any accuracy scalar $p_a \in [0,1]$, $U_a = R_- + p_a\Delta R$, so $|\hat{U}_a - U_a| = \Delta R\,|\hat{p}_a - p_a| \le \Delta R\,\epsilon_p$ for $a \in \{\text{act}, \text{clear}, \text{ovr}\}$. Let $M_k := \max\{U_k, U_{\text{defer}}\}$ and $\widehat{M}_k := \max\{\hat{U}_k, U_{\text{defer}}\}$ for $k \in \{\text{clear}, \text{ovr}, \text{act}\}$.

1. **Stage-2 suboptimality when `ASK` is taken:** Under true parameters, Stage-2 takes $\max\{U_k, U_{\text{defer}}\}$ for realized branch $k \in \{\text{clear}, \text{ovr}, \text{act}\}$, whereas $\hat{a}$ chooses $\text{ACT}$ iff $\hat{U}_k \ge U_{\text{defer}}$. If the Stage-2 choice differs from optimal, then either $\hat{U}_k \ge U_{\text{defer}} > U_k$ or $U_k > U_{\text{defer}} \ge \hat{U}_k$, which implies $|U_k - U_{\text{defer}}| \le |U_k - \hat{U}_k| \le \Delta R\,\epsilon_p$. Furthermore, the 1-Lipschitzness of $x \mapsto \max\{x, U_{\text{defer}}\}$ gives $|\widehat{M}_k - M_k| \le |\hat{U}_k - U_k| \le \Delta R\,\epsilon_p$ for all $k$.

2. **Bounding $|\hat{U}(a) - U(a)|$ uniformly over $a \in \{\text{ACT}, \text{ASK}, \text{DEFER}\}$:**
   - For $a = \text{ACT}$: $|\hat{U}(\text{ACT}) - U(\text{ACT})| \le \Delta R\,\epsilon_p$.
   - For $a = \text{DEFER}$: $|\hat{U}(\text{DEFER}) - U(\text{DEFER})| = 0$.
   - For $a = \text{ASK}$: let $\bar{M} := \frac{1}{2}(\max_k M_k + \min_k M_k)$ so $\max_k |M_k - \bar{M}| = \frac{1}{2}\text{Span}(Q)$. Using $\sum_k (\hat{\pi}_k - \pi_k) = 0$:
     $$\begin{aligned}
     |\hat{U}(\text{ASK}) - U(\text{ASK})|
     &= \left|\sum_{k=1}^3 \hat{\pi}_k \widehat{M}_k - \sum_{k=1}^3 \pi_k M_k\right| \\
     &\le \sum_{k=1}^3 \hat{\pi}_k |\widehat{M}_k - M_k| + \left|\sum_{k=1}^3 (\hat{\pi}_k - \pi_k)(M_k - \bar{M})\right| \\
     &\le \Delta R\,\epsilon_p + \frac{1}{2}\text{Span}(Q)\,\|\hat{\pi} - \pi\|_1 \\
     &\le \Delta R\,\epsilon_p + \frac{1}{2}\text{Span}(Q)\,\epsilon_\pi.
     \end{aligned}$$
     *(Even without centering by $\bar{M}$, whenever $U_{\text{defer}}$ lies within $[R_-, R_+]$, $0 \le M_k - R_- \le \Delta R$, giving $\text{Span}(Q) \le \Delta R$.)*

3. **Standard optimal-selection perturbation inequality:** Since $\hat{U}(\hat{a}) \ge \hat{U}(a^*)$ and also accounting for possible Stage-2 misrouting loss (bounded by $\Delta R\,\epsilon_p$ inside $U(\hat{a})$ when $\hat{a} = \text{ASK}$):
   $$U(a^*) - U_{\text{ideal}}(\hat{a}) = \bigl(U(a^*) - \hat{U}(a^*)\bigr) + \bigl(\hat{U}(a^*) - \hat{U}(\hat{a})\bigr) + \bigl(\hat{U}(\hat{a}) - U_{\text{ideal}}(\hat{a})\bigr) \le 2\max_a |\hat{U}(a) - U(a)| \le 2\Delta R\,\epsilon_p + \text{Span}(Q)\,\epsilon_\pi.$$
   When $U_{\text{defer}} \in [R_-, R_+]$, $\text{Span}(Q) \le \Delta R$, yielding $U(a^*) - U(\hat{a}) \le \Delta R\,(2\epsilon_p + \epsilon_\pi)$. $\blacksquare$

### Corollary 2 (Finite-Sample Empirical Bayes Scaling)
Consider a held-out calibration bucket $b$ containing $n_b \ge 1$ i.i.d. samples, estimated with Dirichlet/Beta shrinkage prior strength $k > 0$ toward prior mean $\mu_0$ (as in `HeldOutCalibrator`). With probability at least $1 - \delta$ ($\delta \in (0,1)$):
$$\epsilon_p(n_b, k),\, \epsilon_\pi(n_b, k) = O\!\left(\sqrt{\frac{\log(1/\delta)}{n_b}} + \frac{k}{n_b + k}\right).$$

#### Proof
For a scalar binomial proportion $\hat{p}_{\text{EB}} = \frac{\sum_{i=1}^{n_b} Z_i + k\mu_0}{n_b + k}$ with $Z_i \in \{0,1\}$ and true mean $p = \mathbb{E}[Z_i]$:
$$|\hat{p}_{\text{EB}} - p| \le \frac{n_b}{n_b + k}\left|\frac{1}{n_b}\sum_{i=1}^{n_b} Z_i - p\right| + \frac{k}{n_b + k}|\mu_0 - p|.$$
By Hoeffding's inequality, $\left|\frac{1}{n_b}\sum_{i=1}^{n_b} Z_i - p\right| \le \sqrt{\frac{\log(2/\delta)}{2n_b}}$ with probability at least $1 - \delta$, while $|\mu_0 - p| \le 1$. Applying a union bound across the 3 multinomial coordinates of $\hat{\pi}$ (each with $|\mu_{0,j} - \pi_j| \le 1$, $\sum |\mu_{0,j} - \pi_j| \le 2$) gives $\|\hat{\pi}_{\text{EB}} - \pi\|_1 \le 3\sqrt{\frac{\log(6/\delta)}{2n_b}} + \frac{2k}{n_b + k}$. Together, both errors scale as $O\bigl(\sqrt{\log(1/\delta)/n_b} + k/(n_b+k)\bigr)$. This directly explains two empirical boundaries: (i) `GLM-4.5-Air` variance under small calibration buckets ($n_{\text{cal}}=60 \implies n_b \approx 15$ per quadrant, where increasing $k$ trades variance for bounded bias), and (ii) `Qwen3-8B` diagnostic floor (low Missing-F1 mixes `Q2` and `Q1` items into the same bucket, inducing irreducible prior/bucket shift). $\blacksquare$

---

## Theorem 3 (Foster-Lyapunov Capacity Expansion & Non-Zero `ASK` Response Latency)

### Part A — Dynamic Cross-Ticket Queue Stability & Capacity-Region Expansion

#### Statement
Consider a discrete-time ticket stream arriving at rate $\lambda_{\text{tot}} > 0$ per slot (e.g., Bernoulli($\lambda_{\text{tot}}$) or bounded-variance arrivals $A_t^{\text{tot}}$ with $\mathbb{E}[A_t^{\text{tot}}] = \lambda_{\text{tot}}$ and $\mathbb{E}[(A_t^{\text{tot}})^2] < \infty$). Tickets belong to:
- AI-solvable clear (`Q1`, share $1 - \alpha_{\text{spec}} - \alpha_{\text{epist}}$),
- Requester-underspecified (`Q2` and `Q4`, share $\alpha_{\text{spec}}$), of which a fraction $\bar{p}_{\text{res}} \in (0,1]$ resolves to `ACT` after `ASK` and $(1 - \bar{p}_{\text{res}})$ still requires Specialist `DEFER`,
- Unconditional Specialist-only (`Q3`, share $\alpha_{\text{epist}}$).

Let the Specialist pool serve $S_t$ tickets per slot with mean service rate $\mu := \mathbb{E}[S_t] > 0$ and $\mathbb{E}[S_t^2] < \infty$, evolving via $Q_{t+1} = \max\{0, Q_t + A_t^{\text{defer}} - S_t\}$. Assume a thresholded controller with queue-congestion penalty $\gamma Q_t$ defers whenever $U_{\text{defer}}(Q_t) > U_{\text{alt}}$ (which stops deferring all tickets once $Q_t > Q_{\max} := \max_s (U_{\text{defer}}(0) - U_{\text{alt}}(s))/\gamma$), OR compare the *unbounded accuracy-preserving* regime where all unresolvable Specialist tickets are deferred ($A_t^{\text{defer}}$ arrival rate):
1. **Binary `{ACT, DEFER}` Baseline (no `ASK` channel):** To avoid sending unclarified `Q2`/`Q4` tickets to error-prone `ACT`, it must defer rate $\lambda_{\text{base}}^{\text{defer}} = \lambda_{\text{tot}}(\alpha_{\text{spec}} + \alpha_{\text{epist}})$, requiring $\lambda_{\text{tot}}(\alpha_{\text{spec}} + \alpha_{\text{epist}}) < \mu$ for queue stability.
2. **`Tri-VoI-L2D` (`{ACT, ASK, DEFER}`):** Resolves share $\bar{p}_{\text{res}}\alpha_{\text{spec}}$ via `ASK -> ACT`, reducing Specialist arrival rate to:
   $$\lambda_{\text{Tri}}^{\text{defer}} = \lambda_{\text{tot}}\bigl(\alpha_{\text{epist}} + (1 - \bar{p}_{\text{res}})\alpha_{\text{spec}}\bigr) < \lambda_{\text{base}}^{\text{defer}}.$$
   Whenever $\lambda_{\text{Tri}}^{\text{defer}} < \mu$, quadratic Lyapunov $V(Q) = \frac{1}{2}Q^2$ proves positive recurrence with drift gap $\varepsilon := \mu - \lambda_{\text{Tri}}^{\text{defer}} > 0$ and steady-state mean-queue bound:
   $$\limsup_{T \to \infty} \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[Q_t] \le \frac{B}{\varepsilon}, \qquad B := \frac{1}{2}\mathbb{E}\bigl[(A_t^{\text{defer}} - S_t)^2\bigr].$$
   In particular, in the throughput window $\frac{\mu}{\alpha_{\text{spec}} + \alpha_{\text{epist}}} < \lambda_{\text{tot}} < \frac{\mu}{\alpha_{\text{epist}} + (1 - \bar{p}_{\text{res}})\alpha_{\text{spec}}}$, `{ACT, DEFER}` overflows or must drop accuracy into `ACT`, whereas `Tri-VoI-L2D` remains stable **without sacrificing accuracy**.

#### Proof
Using $Q_{t+1}^2 = (\max\{0, Q_t + A_t^{\text{defer}} - S_t\})^2 \le (Q_t + A_t^{\text{defer}} - S_t)^2$:
$$\begin{aligned}
\mathbb{E}\bigl[V(Q_{t+1}) - V(Q_t) \mid Q_t = q\bigr]
&\le \frac{1}{2}\mathbb{E}\bigl[2q(A_t^{\text{defer}} - S_t) + (A_t^{\text{defer}} - S_t)^2 \mid Q_t = q\bigr] \\
&= q\,\bigl(\mathbb{E}[A_t^{\text{defer}} \mid Q_t = q] - \mu\bigr) + \frac{1}{2}\mathbb{E}\bigl[(A_t^{\text{defer}} - S_t)^2 \mid Q_t = q\bigr].
\end{aligned}$$
Under `Tri-VoI-L2D`, whenever $q \le Q_{\max}$, $\mathbb{E}[A_t^{\text{defer}} \mid Q_t = q] \le \lambda_{\text{Tri}}^{\text{defer}}$, so taking $\varepsilon = \mu - \lambda_{\text{Tri}}^{\text{defer}} > 0$ and $B = \sup_q \frac{1}{2}\mathbb{E}[(A_t^{\text{defer}} - S_t)^2 \mid Q_t = q] < \infty$:
$$\mathbb{E}\bigl[V(Q_{t+1}) - V(Q_t) \mid Q_t = q\bigr] \le -\varepsilon q + B.$$
(Note: if the controller additionally enforces congestion backoff for $q > Q_{\max}$, $\mathbb{E}[A_t^{\text{defer}} \mid Q_t = q] = 0$, making the drift $-\mu q + B \le -\varepsilon q + B$ even stronger!)
By the Foster-Lyapunov criterion, the Markov chain $\{Q_t\}$ is positive recurrent. Summing the drift inequality from $t = 0$ to $T - 1$ and telescoping gives:
$$\mathbb{E}[V(Q_T)] - \mathbb{E}[V(Q_0)] \le -\varepsilon \sum_{t=0}^{T-1}\mathbb{E}[Q_t] + B T.$$
Since $V(Q_T) \ge 0$, rearranging and dividing by $\varepsilon T$ yields:
$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}[Q_t] \le \frac{\mathbb{E}[V(Q_0)]}{\varepsilon T} + \frac{B}{\varepsilon} \xrightarrow{T\to\infty} \limsup_{T\to\infty}\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}[Q_t] \le \frac{B}{\varepsilon}.$$
Conversely, when $\lambda_{\text{tot}}(\alpha_{\text{spec}} + \alpha_{\text{epist}}) > \mu$, any `{ACT, DEFER}` policy attempting to defer all Specialist/underspec tickets has $\mathbb{E}[A_t^{\text{defer}} - S_t] > 0$, so by the Strong Law of Large Numbers $Q_t \to \infty$ almost surely (or must reject tickets into inaccurate `ACT`), completing the capacity-region expansion proof. $\blacksquare$

---

### Part B — Non-Zero `ASK` Response Latency ($\tau_{\text{ask}} > 0$)

#### Statement
Suppose an `ASK` reply returns after $\tau_{\text{ask}} \ge 0$ slots, incurring an explicit requester waiting penalty $\beta_{\text{wait}} \tau_{\text{ask}}$ ($\beta_{\text{wait}} \ge 0$), while during the delay window $[t, t+\tau_{\text{ask}}]$ the Specialist queue evolves randomly to $Q_{t+\tau_{\text{ask}}}$ with conditional expectation $\mathbb{E}[Q_{t+\tau_{\text{ask}}} \mid Q_t] = Q_t + \Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t)$. Let:
$$U_{\text{ASK}}^{(\tau_{\text{ask}})}(Q_t) := \mathbb{E}\Bigl[p_{\text{res}}\max\{U_{\text{clear}}, U_{\text{defer}}(Q_{t+\tau_{\text{ask}}})\} + p_{\text{ovr\_after}}\max\{U_{\text{ovr}}, U_{\text{defer}}(Q_{t+\tau_{\text{ask}}})\} + p_{\text{no}}\max\{U_{\text{act}}, U_{\text{defer}}(Q_{t+\tau_{\text{ask}}})\}\Bigr] - c_{\text{ask}} - \beta_{\text{wait}}\tau_{\text{ask}}.$$
1. **Convexity-Jensen Lower Bound (General Regimes):** Because $q \mapsto \max\{u, U_{\text{defer}}(0) - \gamma q\}$ is convex in $q$, Jensen's inequality gives:
   $$U_{\text{ASK}}^{(\tau_{\text{ask}})}(Q_t) \ge U_{\text{ASK}}\!\bigl(Q_t + \Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t)\bigr) - \beta_{\text{wait}}\tau_{\text{ask}}.$$
2. **Exact Regime-I Shift & Maximum Tolerable Latency:** Whenever both $Q_t$ and $Q_{t+\tau_{\text{ask}}}$ remain in Regime I ($Q_0^* < Q < Q_1^*$, with $U_{\text{defer}}(Q_{t+\tau_{\text{ask}}}) > U_{\text{act}}$ almost surely):
   $$\Delta\text{EVOI}(\tau_{\text{ask}}) := \text{EVOI}^{(\tau_{\text{ask}})}(\text{ASK}; Q_t) - \text{EVOI}^{(0)}(\text{ASK}; Q_t) = -(1 - p_{\text{res}})\,\gamma\,\Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t) - \beta_{\text{wait}}\tau_{\text{ask}}.$$
   In particular, under net queue buildup $\Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t) = \bar{\delta}\,\tau_{\text{ask}}$ ($\bar{\delta} \ge 0$), `ASK` remains strictly optimal over $\max\{U_{\text{act}}, U_{\text{defer}}(Q_t)\}$ up to the closed-form maximum tolerable delay:
   $$\tau_{\text{ask}}^{\max}(Q_t) = \frac{\text{EVOI}^{(0)}(\text{ASK}; Q_t)}{(1 - p_{\text{res}})\,\gamma\,\bar{\delta} + \beta_{\text{wait}}}.$$

#### Proof
For any fixed threshold $u \in \mathbb{R}$, the map $f_u(q) := \max\{u, U_{\text{defer}}(0) - \gamma q\}$ is the pointwise maximum of two affine functions of $q$, hence convex on $\mathbb{R}$. Applying Jensen's inequality to each of the three outcome branches in $U_{\text{ASK}}^{(\tau_{\text{ask}})}(Q_t)$ and summing with nonnegative weights $(p_{\text{res}}, p_{\text{ovr\_after}}, p_{\text{no}})$ directly yields:
$$U_{\text{ASK}}^{(\tau_{\text{ask}})}(Q_t) \ge \sum_{k} \pi_k \max\bigl\{U_k, U_{\text{defer}}(0) - \gamma \mathbb{E}[Q_{t+\tau_{\text{ask}}} \mid Q_t]\bigr\} - c_{\text{ask}} - \beta_{\text{wait}}\tau_{\text{ask}} = U_{\text{ASK}}\!\bigl(Q_t + \Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t)\bigr) - \beta_{\text{wait}}\tau_{\text{ask}}.$$
In Regime I ($Q_0^* < Q_t, Q_{t+\tau_{\text{ask}}} < Q_1^*$ a.s.), the `clear` branch chooses $U_{\text{clear}}$ regardless of $\tau_{\text{ask}}$, whereas both the `ovr_after` and `no_info` branches (total weight $p_{\text{ovr\_after}} + p_{\text{no}} = 1 - p_{\text{res}}$) defer at $U_{\text{defer}}(Q_{t+\tau_{\text{ask}}}) = U_{\text{defer}}(Q_t) - \gamma(Q_{t+\tau_{\text{ask}}} - Q_t)$. Taking expectation minus the immediate-reply value $\text{EVOI}^{(0)}(\text{ASK}; Q_t)$ gives:
$$\Delta\text{EVOI}(\tau_{\text{ask}}) = -(1 - p_{\text{res}})\,\gamma\,\mathbb{E}[Q_{t+\tau_{\text{ask}}} - Q_t \mid Q_t] - \beta_{\text{wait}}\tau_{\text{ask}} = -(1 - p_{\text{res}})\,\gamma\,\Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t) - \beta_{\text{wait}}\tau_{\text{ask}}.$$
Setting $\Delta_{\text{drift}}(\tau_{\text{ask}}, Q_t) = \bar{\delta}\,\tau_{\text{ask}}$ and solving $\text{EVOI}^{(0)}(\text{ASK}; Q_t) + \Delta\text{EVOI}(\tau_{\text{ask}}) > 0$ yields $\tau_{\text{ask}} < \tau_{\text{ask}}^{\max}(Q_t)$. Notice that when queue drains during the delay ($\Delta_{\text{drift}} < 0$) and $\beta_{\text{wait}}$ is small, delay can actually *increase* EVOI slightly because post-ASK escalation lands on a shorter queue, whereas queue growth ($\bar{\delta} > 0$) linearly erodes EVOI at rate $(1 - p_{\text{res}})\gamma\bar{\delta} + \beta_{\text{wait}}$. $\blacksquare$
