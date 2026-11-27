import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from controller_utils_compat import parse_json_safe


def bernoulli_kl(p: float, q: float) -> float:
    """Bernoulli KL divergence kl(p, q) with safe boundary handling."""
    if abs(p - q) < 1e-15:
        return 0.0
    if p <= 0.0:
        if q >= 1.0:
            return float("inf")
        return math.log(1.0 / max(1e-15, 1.0 - q))
    if p >= 1.0:
        if q <= 0.0:
            return float("inf")
        return math.log(1.0 / max(1e-15, q))
    if q <= 0.0 or q >= 1.0:
        return float("inf")
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_lcb(p_hat: float, n: int, budget: float) -> float:
    """Compute inf {q in [0, p_hat] : n * kl(p_hat, q) <= budget} via 50-step bisection."""
    if n <= 0 or p_hat <= 0.0:
        return 0.0
    if budget <= 0.0:
        return min(1.0, max(0.0, p_hat))
    lo, hi = 0.0, min(1.0, max(0.0, p_hat))
    if n * bernoulli_kl(p_hat, 0.0) <= budget:
        return 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if n * bernoulli_kl(p_hat, mid) <= budget:
            hi = mid
        else:
            lo = mid
    return hi


class KLPBDController:
    """Faithful implementation of KL-PBD / KL-PBD+ (arXiv:2610.15128, Alg. 1 & App. B.5/F).

    Action space is binary {ACT (0), DEFER (1)} with no ASK channel.
    LLM correctness labels on deferred tickets are revealed only when service finishes.
    """

    NUM_LEVELS = 8

    def __init__(
        self,
        *,
        variant: str = "kl_pbd",  # "kl_pbd" | "kl_pbd_plus" | "kl_pbd_plus_warm"
        total_horizon: int = 360,
        capacity: int = 2,
        drain_interval: int = 2,
        c_hold: float = 0.85,
        c_defer: float = 2.0,
        cal_probed: Optional[List[Dict[str, Any]]] = None,
        sc_only: bool = False,
    ):
        self.variant = variant
        self.sc_only = sc_only
        self.num_levels = 3 if sc_only else self.NUM_LEVELS
        self.total_horizon = max(3, total_horizon)
        self.cap = max(1, int(capacity))
        self.drain_interval = max(1, int(drain_interval))
        self.p_serve = 1.0 / float(self.drain_interval)
        self.c_hold = float(c_hold)
        self.c_defer = float(c_defer)
        self.b_max = max(12, self.cap * 6)

        self.O = [0] * self.num_levels
        self.N = [0] * self.num_levels
        self.S = [0] * self.num_levels
        self.t_arrivals = 0

        if variant == "kl_pbd_plus_warm" and cal_probed:
            for r in cal_probed:
                j = self.level_of(r)
                self.O[j] += 1
                self.N[j] += 1
                if (r["greedy_ans"] == r["item"]["gold"]) and (not r["item"].get("require_defer_for_ok", False)):
                    self.S[j] += 1
                self.t_arrivals += 1

        # Precompute terminal draining cost D(b) for b in 0..b_max:
        # D(0)=0; D(b) = c_hold*b + (1-p_serve)*D(b) + p_serve*D(max(0, b-cap))
        # => D(b) = (c_hold*b)/p_serve + D(max(0, b-cap))
        self.D = [0.0] * (self.b_max + 1)
        for b in range(1, self.b_max + 1):
            b_next = max(0, b - self.cap)
            self.D[b] = (self.c_hold * float(b)) / self.p_serve + self.D[b_next]

    def level_of(self, rec: Dict[str, Any]) -> int:
        if self.sc_only:
            # Self-consistency-only level (maj_count in {1, 2, 3} -> {0, 1, 2}), no disentangled diag
            return max(0, min(2, int(rec["maj_count"]) - 1))
        m_flag = 1 if rec["obs_missing"] else 0
        p_flag = 1 if rec["obs_pot_ovr"] else 0
        u_flag = 1 if rec["maj_count"] == 3 else 0
        return (m_flag << 2) | (p_flag << 1) | u_flag

    def record_arrival(self, j: int) -> None:
        self.O[j] += 1
        self.t_arrivals += 1

    def record_completion(self, j: int, llm_was_correct: bool) -> None:
        self.N[j] += 1
        if llm_was_correct:
            self.S[j] += 1

    def lower_bounds(self) -> List[float]:
        out: List[float] = []
        phi_t = math.log(self.total_horizon) + math.log(max(1.1, math.log(self.total_horizon)))
        for j in range(self.num_levels):
            n_j = self.N[j]
            if n_j <= 0:
                out.append(0.0)
                continue
            p_hat = self.S[j] / float(n_j)
            if self.variant == "kl_pbd":
                budget = phi_t
            else:
                ratio = max(1.0, self.O[j] / float(n_j))
                budget = math.log(ratio)
            out.append(kl_lcb(p_hat, n_j, budget))
        return out

    def arrival_dist(self) -> List[float]:
        if self.t_arrivals <= 0:
            return [1.0 / self.num_levels] * self.num_levels
        inv = 1.0 / float(self.t_arrivals)
        return [self.O[j] * inv for j in range(self.num_levels)]

    def marginal_defer_cost(self, b: int, rem_steps: int, c_e: float, p_e_hat: float) -> float:
        """Compute Bellman marginal cost M_t^{h_lcb}(b) via finite-horizon backward DP."""
        if b >= self.b_max:
            return float("inf")

        h_lcb = self.lower_bounds()
        omega = self.arrival_dist()
        c_exp = self.c_defer + (1.0 - p_e_hat) * c_e
        err_costs = [c_e * (1.0 - h_lcb[j]) for j in range(self.num_levels)]

        horizon = max(1, min(rem_steps, 25))
        # E_V[b'] = sum_j omega[j] * V_{s+1}(b', j)
        e_v = list(self.D)
        p_s = self.p_serve
        q_s = 1.0 - p_s
        b_max = self.b_max
        cap = self.cap
        c_hold = self.c_hold

        m_at_b = c_exp
        for step in range(horizon):
            new_e_v = [0.0] * (b_max + 1)
            for b_cur in range(b_max + 1):
                b0_srv = max(0, b_cur - cap)
                cont_0 = q_s * e_v[b_cur] + p_s * e_v[b0_srv]
                if b_cur < b_max:
                    b1_srv = max(0, (b_cur + 1) - cap)
                    cont_1 = q_s * e_v[b_cur + 1] + p_s * e_v[b1_srv]
                    m_val = c_exp + (cont_1 - cont_0)
                else:
                    m_val = float("inf")

                if step == horizon - 1 and b_cur == b:
                    m_at_b = m_val

                # V_s(b_cur, j) = c_hold*b_cur + cont_0 + min(err_cost_j, m_val)
                base_val = c_hold * float(b_cur) + cont_0
                avg_act_defer = 0.0
                for j in range(self.num_levels):
                    w_j = omega[j]
                    if w_j > 0.0:
                        ec = err_costs[j]
                        avg_act_defer += w_j * (m_val if m_val < ec else ec)
                new_e_v[b_cur] = base_val + avg_act_defer
            e_v = new_e_v

        return m_at_b

    def should_defer(self, rec: Dict[str, Any], b: int, rem_steps: int, r_ok: float, r_err: float, p_e_hat: float) -> bool:
        j = self.level_of(rec)
        self.record_arrival(j)
        if b >= self.b_max:
            return False
        c_e = r_ok - r_err
        m_val = self.marginal_defer_cost(b, rem_steps, c_e, p_e_hat)
        tau = 1.0 - (m_val / max(1e-9, c_e))
        h_lcb = self.lower_bounds()[j]
        return h_lcb < tau


def acl_voi_calib_decision(
    *,
    p_act: float,
    p_res: float,
    p_ovra: float,
    p_no: float,
    p_clear: float,
    p_ovr: float,
    obs_m: bool,
    r_ok: float,
    r_err: float,
    c_ask: float,
) -> Tuple[bool, float]:
    """ACL 2026 VoI one-step lookahead NetVoI rule under {ACT, ASK} (no DEFER/queue)."""
    u_act = p_act * r_ok + (1.0 - p_act) * r_err
    u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
    u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
    v_post = p_res * u_clear + p_ovra * u_ovr + p_no * u_act
    net_voi = v_post - u_act - c_ask
    return bool(obs_m and net_voi > 0.0), float(net_voi)


def build_acl_voi_belief_prompt(
    rubric: str,
    ticket_prose: str,
    clarify_q: str,
    choices: Sequence[str] = ("A", "B", "C"),
) -> str:
    """Build faithful 1-call LLM belief-posterior prompt for ACL 2026 VoI (2026.acl-long.1987)."""
    c_str = ", ".join(choices)
    prior_ex = ", ".join(f'"{c}": 0.33' for c in choices)
    return (
        f"{rubric}\nTicket:\n{ticket_prose}\n\n"
        f"Candidate clarifying question to the requester: \"{clarify_q}\"\n\n"
        f"Estimate your calibrated probability distribution over the optimal decision {{{c_str}}}:\n"
        "1. prior: probability distribution over decisions BEFORE asking.\n"
        "2. answers: 2-3 plausible requester replies (each with probability prob summing to 1.0, and conditional posterior over decisions).\n"
        "Respond in JSON ONLY:\n"
        f'{{"prior": {{{prior_ex}}}, "answers": [{{"summary": "...", "prob": 0.5, "posterior": {{{prior_ex}}}}}]}}'
    )


def _smooth_dist(raw: Any, choices: Sequence[str]) -> Dict[str, float]:
    n = len(choices)
    min_p = 0.01 / float(n)
    vals: Dict[str, float] = {}
    if isinstance(raw, dict):
        for c in choices:
            v = raw.get(c, raw.get(c.lower(), 0.0))
            try:
                fv = max(0.0, float(v))
            except (TypeError, ValueError):
                fv = 0.0
            vals[c] = max(min_p, fv)
    else:
        for c in choices:
            vals[c] = 1.0 / float(n)
    s = sum(vals.values())
    return {c: vals[c] / s for c in choices}


def parse_and_compute_acl_voi(
    raw_content: str,
    choices: Sequence[str] = ("A", "B", "C"),
    r_ok: float = 10.0,
    r_err: float = -10.0,
    c_ask: float = 1.5,
) -> Tuple[float, str, bool]:
    """Compute exact ACL 2026 NetVoI(q) from LLM belief JSON.

    Returns (net_voi, best_pre_choice, parsed_ok).
    """
    obj = parse_json_safe(raw_content)
    if not isinstance(obj, dict) or ("prior" not in obj and "answers" not in obj):
        return -c_ask, choices[0], False

    prior = _smooth_dist(obj.get("prior"), choices)

    def _max_eu(dist: Dict[str, float]) -> Tuple[str, float]:
        best_c = choices[0]
        best_u = -1e18
        for c in choices:
            p_c = dist[c]
            u_c = p_c * r_ok + (1.0 - p_c) * r_err
            if u_c > best_u:
                best_u = u_c
                best_c = c
        return best_c, best_u

    best_pre, v_pre = _max_eu(prior)

    answers = obj.get("answers")
    if not isinstance(answers, list) or len(answers) == 0:
        return -c_ask, best_pre, False

    probs = []
    posts = []
    for item in answers:
        if not isinstance(item, dict):
            continue
        try:
            py = max(1e-4, float(item.get("prob", 0.0)))
        except (TypeError, ValueError):
            py = 1.0 / float(len(answers))
        probs.append(py)
        posts.append(_smooth_dist(item.get("posterior"), choices))

    if not probs:
        return -c_ask, best_pre, False

    p_sum = sum(probs)
    v_post = 0.0
    for py, dist in zip(probs, posts):
        _, u_y = _max_eu(dist)
        v_post += (py / p_sum) * u_y

    net_voi = v_post - v_pre - c_ask
    return float(net_voi), best_pre, True
