import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark import DOMAIN_RUBRIC, user_oracle
from llm_client import call_llm


def parse_json_obj(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    ans_m = re.search(r"\b([A-C])\b", text.upper())
    return {"answer": ans_m.group(1) if ans_m else "A"}


def extract_answer(text: str) -> str:
    obj = parse_json_obj(text)
    ans = str(obj.get("answer", "")).strip().upper()
    if ans in ("A", "B", "C"):
        return ans
    m = re.search(r"\b([A-C])\b", text.upper())
    return m.group(1) if m else "A"


def shannon_entropy(labels: List[str]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def probe_item(item: Dict[str, Any], model_id: str, dry_run: bool) -> Dict[str, Any]:
    """Run 7 fast calls (enable_thinking=False) to extract probes, diagnosis, and post-ASK state."""
    dry_meta = f" [DRY_META:{item['quadrant']}:{item['gold']}]" if dry_run else ""
    base_prompt = (
        f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n\n"
        "Choose the best action A, B, or C. Respond in JSON ONLY: "
        "{\"answer\": \"A\"|\"B\"|\"C\", \"rationale\": \"brief\"}."
    )

    samples = []
    tokens = 0
    for s_idx in range(3):
        r = call_llm(
            [{"role": "user", "content": base_prompt}],
            model_id=model_id,
            temperature=0.7,
            max_tokens=1024,
            tag=f"probe_s{s_idx}",
            seed_idx=s_idx,
            dry_run=dry_run,
        )
        samples.append(extract_answer(r["content"]))
        tokens += r["prompt_tokens"] + r["completion_tokens"]

    r_greedy = call_llm(
        [{"role": "user", "content": base_prompt}],
        model_id=model_id,
        temperature=0.0,
        max_tokens=1024,
        tag="greedy_act",
        seed_idx=0,
        dry_run=dry_run,
    )
    greedy_ans = extract_answer(r_greedy["content"])
    tokens += r_greedy["prompt_tokens"] + r_greedy["completion_tokens"]

    diag_prompt = (
        f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n\n"
        "Audit the ticket against the domain's Required Intake Fields and Internal Advisory Rules:\n"
        "1. missing_user_info: true if ANY Required Intake Field for this domain is omitted/unstated in the Ticket.\n"
        "2. potential_override: true if the Ticket mentions an active confidential specialist protocol, bulletin, runbook, or waiver.\n"
        "3. override_confirmed: true ONLY if an unconditional override applies OR the trigger condition of a conditional protocol is ALREADY confirmed present in the Ticket.\n"
        "4. clarifying_question: if missing_user_info is true, write one short specific question asking for that missing field; else empty string.\n"
        "Respond in JSON ONLY: "
        "{\"missing_user_info\": true|false, \"potential_override\": true|false, \"override_confirmed\": true|false, \"clarifying_question\": \"...\"}."
    )
    r_diag = call_llm(
        [{"role": "user", "content": diag_prompt}],
        model_id=model_id,
        temperature=0.0,
        max_tokens=1024,
        tag="diag_init",
        seed_idx=0,
        dry_run=dry_run,
    )
    diag_obj = parse_json_obj(r_diag["content"])
    tokens += r_diag["prompt_tokens"] + r_diag["completion_tokens"]

    obs_missing = bool(diag_obj.get("missing_user_info", False))
    obs_pot_ovr = bool(diag_obj.get("potential_override", False))
    obs_conf_ovr = bool(diag_obj.get("override_confirmed", False))
    clarify_q = str(diag_obj.get("clarifying_question", "")).strip()

    oracle_reply, oracle_revealed = user_oracle(
        item, clarify_q or f"Could you specify the missing {item['missing_concept'] or 'intake detail'}?"
    )

    post_prompt = (
        f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n"
        f"[Requester Follow-up]: {oracle_reply}\n\n"
        "Re-evaluate after the requester's reply:\n"
        "- answer: best choice A, B, or C.\n"
        "- missing_user_info: true if a required intake field is still missing.\n"
        "- override_confirmed: true if a confidential specialist protocol/override applies to the confirmed ticket.\n"
        "Respond in JSON ONLY: "
        "{\"answer\": \"A\"|\"B\"|\"C\", \"missing_user_info\": true|false, \"override_confirmed\": true|false}."
    )
    r_post = call_llm(
        [{"role": "user", "content": post_prompt}],
        model_id=model_id,
        temperature=0.0,
        max_tokens=1024,
        tag="post_ask_combined",
        seed_idx=0,
        dry_run=dry_run,
    )
    post_obj = parse_json_obj(r_post["content"])
    post_ans = extract_answer(r_post["content"])
    post_ovr_conf = bool(post_obj.get("override_confirmed", obs_conf_ovr))
    tokens += r_post["prompt_tokens"] + r_post["completion_tokens"]

    # Second stochastic post-ASK sample for Naive-Cascade agreement check
    r_post_s1 = call_llm(
        [{"role": "user", "content": f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n[Requester Follow-up]: {oracle_reply}\nRespond in JSON ONLY: {{\"answer\": \"A\"|\"B\"|\"C\"}}."}],
        model_id=model_id,
        temperature=0.7,
        max_tokens=1024,
        tag="post_ask_s1",
        seed_idx=1,
        dry_run=dry_run,
    )
    post_s1 = extract_answer(r_post_s1["content"])
    tokens += r_post_s1["prompt_tokens"] + r_post_s1["completion_tokens"]

    return {
        "item": item,
        "model_id": model_id,
        "samples": samples,
        "entropy": shannon_entropy(samples),
        "maj_count": Counter(samples).most_common(1)[0][1],
        "greedy_ans": greedy_ans,
        "obs_missing": obs_missing,
        "obs_pot_ovr": obs_pot_ovr,
        "obs_conf_ovr": obs_conf_ovr,
        "clarify_q": clarify_q,
        "oracle_reply": oracle_reply,
        "oracle_revealed": oracle_revealed,
        "post_ans": post_ans,
        "post_s1": post_s1,
        "post_ovr_conf": post_ovr_conf,
        "tokens": tokens,
    }


def probe_dataset(items: List[Dict[str, Any]], model_id: str, dry_run: bool, workers: int = 6, label: str = "") -> List[Dict[str, Any]]:
    out_map: Dict[int, Dict[str, Any]] = {}
    if dry_run or workers <= 1:
        for i, it in enumerate(items):
            out_map[i] = probe_item(it, model_id, dry_run)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(probe_item, it, model_id, False): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                idx = futs[fut]
                out_map[idx] = fut.result()
                done += 1
                if done % 30 == 0 or done == len(items):
                    print(f"[{model_id}{' ' + label if label else ''}] probed {done}/{len(items)}", flush=True)
    return [out_map[i] for i in range(len(items))]


class HeldOutCalibrator:
    """Hierarchical Empirical Bayes calibrator on held-out split (zero test leakage)."""

    def __init__(self, cal_probed: List[Dict[str, Any]]):
        k = 2.0  # Coarse Laplace/Dirichlet smoothing strength
        k_fine = 3.0  # Fine shrinkage strength toward coarse (obs_m, obs_p) bucket
        coarse_stats = defaultdict(lambda: [0, 0])
        fine_stats = defaultdict(lambda: [0, 0])
        post_clear = [0, 0]
        post_ovr = [0, 0]
        tri_stats = defaultdict(lambda: [0, 0, 0])  # [n_resolve, n_ovr_after, n_no_info]

        for r in cal_probed:
            gold = r["item"]["gold"]
            m_flag = bool(r["obs_missing"])
            p_flag = bool(r["obs_pot_ovr"])
            u_flag = (r["maj_count"] == 3)
            coarse_stats[(m_flag, p_flag)][1] += 1
            fine_stats[(m_flag, p_flag, u_flag)][1] += 1
            if r["greedy_ans"] == gold:
                coarse_stats[(m_flag, p_flag)][0] += 1
                fine_stats[(m_flag, p_flag, u_flag)][0] += 1

            if r["post_ovr_conf"]:
                post_ovr[1] += 1
                if r["post_ans"] == gold:
                    post_ovr[0] += 1
            elif (not m_flag) or r["oracle_revealed"]:
                post_clear[1] += 1
                if r["post_ans"] == gold:
                    post_clear[0] += 1

            r_key = (m_flag, p_flag)
            if m_flag and r["oracle_revealed"] and (not r["post_ovr_conf"]):
                tri_stats[r_key][0] += 1
            elif m_flag and r["oracle_revealed"] and r["post_ovr_conf"]:
                tri_stats[r_key][1] += 1
            else:
                tri_stats[r_key][2] += 1

        self.coarse_p_act: Dict[Tuple[bool, bool], float] = {}
        for ck, (ok, tot) in coarse_stats.items():
            prior_c = 0.85 if ck == (False, False) else (0.35 if ck[1] else 0.55)
            self.coarse_p_act[ck] = (ok + k * prior_c) / (tot + k)

        self.p_act: Dict[Tuple[bool, bool, bool], float] = {}
        for fk, (ok, tot) in fine_stats.items():
            coarse_prior = self.coarse_p_act.get((fk[0], fk[1]), 0.60)
            self.p_act[fk] = (ok + k_fine * coarse_prior) / (tot + k_fine)

        self.p_post_clear = (post_clear[0] + k * 0.85) / (post_clear[1] + k)
        self.p_post_ovr = (post_ovr[0] + k * 0.35) / (post_ovr[1] + k)

        self.tri_probs: Dict[Tuple[bool, bool], Tuple[float, float, float]] = {}
        for rk, (n_res, n_ovr, n_no) in tri_stats.items():
            if not rk[0]:
                # Information-source gating: no missing user field -> ASK reveals nothing
                self.tri_probs[rk] = (0.0, 0.0, 1.0)
            else:
                prior = (0.45, 0.45, 0.10) if rk[1] else (0.85, 0.05, 0.10)
                denom = n_res + n_ovr + n_no + k
                self.tri_probs[rk] = (
                    (n_res + k * prior[0]) / denom,
                    (n_ovr + k * prior[1]) / denom,
                    (n_no + k * prior[2]) / denom,
                )

    def get_p_act(self, obs_missing: bool, obs_pot_ovr: bool, unanimous: bool) -> float:
        if (obs_missing, obs_pot_ovr, unanimous) in self.p_act:
            return self.p_act[(obs_missing, obs_pot_ovr, unanimous)]
        return self.coarse_p_act.get(
            (obs_missing, obs_pot_ovr),
            0.85 if (not obs_missing and not obs_pot_ovr) else 0.50,
        )

    def get_post_ask_probs(self, obs_missing: bool, obs_pot_ovr: bool) -> Tuple[float, float, float]:
        if not obs_missing:
            return (0.0, 0.0, 1.0)
        default = (0.45, 0.45, 0.10) if obs_pot_ovr else (0.85, 0.05, 0.10)
        return self.tri_probs.get((obs_missing, obs_pot_ovr), default)

    def get_p_resolve(self, obs_missing: bool, obs_pot_ovr: bool) -> float:
        return self.get_post_ask_probs(obs_missing, obs_pot_ovr)[0]


class DiscountedBetaTracker:
    """Conjugate Discounted Beta-Bernoulli specialist reliability tracker."""

    def __init__(self, alpha0: float = 8.5, beta0: float = 1.0, lam: float = 0.92):
        self.alpha = alpha0
        self.beta = beta0
        self.lam = lam

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, correct: bool) -> None:
        y = 1.0 if correct else 0.0
        self.alpha = self.lam * self.alpha + y
        self.beta = self.lam * self.beta + (1.0 - y)


def evaluate_diagnosis_metrics(probed: List[Dict[str, Any]]) -> Dict[str, float]:
    tp_m = fp_m = fn_m = 0
    tp_p = fp_p = fn_p = 0
    collapse_under = 0
    tot_under = 0
    for r in probed:
        it = r["item"]
        true_m = it["is_underspec"]
        true_p = it["is_uncond_ovr"] or it["is_cond_ovr"]
        if r["obs_missing"] and true_m:
            tp_m += 1
        elif r["obs_missing"] and not true_m:
            fp_m += 1
        elif not r["obs_missing"] and true_m:
            fn_m += 1

        if r["obs_pot_ovr"] and true_p:
            tp_p += 1
        elif r["obs_pot_ovr"] and not true_p:
            fp_p += 1
        elif not r["obs_pot_ovr"] and true_p:
            fn_p += 1

        if true_m:
            tot_under += 1
            if r["maj_count"] == 3:
                collapse_under += 1

    f1_m = (2 * tp_m) / max(1, 2 * tp_m + fp_m + fn_m)
    f1_p = (2 * tp_p) / max(1, 2 * tp_p + fp_p + fn_p)
    return {
        "missing_f1": round(f1_m, 4),
        "override_f1": round(f1_p, 4),
        "entropy_collapse_pct": round(100.0 * collapse_under / max(1, tot_under), 2),
        "underspec_acc_greedy_pct": round(
            100.0 * sum(1 for r in probed if r["item"]["is_underspec"] and r["greedy_ans"] == r["item"]["gold"]) / max(1, tot_under),
            2,
        ),
    }


def simulate_stream(
    method: str,
    test_probed: List[Dict[str, Any]],
    calibrator: Optional[HeldOutCalibrator],
    *,
    capacity: int = 2,
    drain_interval: int = 2,
    drift_mode: str = "sudden",  # "none" | "sudden" | "gradual" | "multi_pool"
    asymmetric_risk: bool = False,
    seed: int = 0,
    gamma_queue: float = 0.85,
    c_ask: float = 1.5,
    c_defer_base: float = 2.0,
) -> Dict[str, Any]:
    """Simulate sequential human-AI cooperative stream with queuing & drift."""
    import random
    rng = random.Random(5000 + seed)
    n = len(test_probed)

    # Single specialist or heterogeneous Senior/Junior pool
    if drift_mode == "multi_pool":
        queues = {"Senior": 0, "Junior": 0}
        caps = {"Senior": 1, "Junior": max(1, capacity)}
        costs = {"Senior": 3.5, "Junior": 1.5}
        trackers = {
            "Senior": DiscountedBetaTracker(alpha0=9.0, beta0=0.6, lam=0.92),
            "Junior": DiscountedBetaTracker(alpha0=8.0, beta0=1.2, lam=0.92),
        }
        lam_ema = {"Senior": 0.15, "Junior": 0.30}
    else:
        lam = 1.0 if method == "w/o Discounted-Forgetting" else 0.92
        queues = {"Spec": 0}
        caps = {"Spec": capacity}
        costs = {"Spec": c_defer_base}
        trackers = {"Spec": DiscountedBetaTracker(alpha0=8.5, beta0=1.0, lam=lam)}
        lam_ema = {"Spec": 0.35}

    eff_gamma = 0.0 if method == "w/o Congestion-Term" else gamma_queue
    use_ext_flag = method not in (
        "Direct-ACT",
        "Entropy-Clarify (ADEPT)",
        "Consensus-Defer (DREAM)",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Calibrated-Cascade",
        "Oracle",
        "w/o Queue-Externality",
        "w/o Congestion-Term",
    )

    per_item_utils = np.zeros(n, dtype=float)
    correct_cnt = 0
    ask_cnt = 0
    defer_cnt = 0
    wait_sum = 0
    misroute_cnt = 0
    q4_ask_cnt = 0
    q4_tot_cnt = 0
    q4_res_act_cnt = 0
    q4_res_tot_cnt = 0

    quad_stats = defaultdict(lambda: {"n": 0, "ok": 0, "util": 0.0, "ask": 0, "defer": 0})
    phase2_utils = []
    rolling_acc = []
    queue_trace = []

    for t, rec in enumerate(test_probed):
        it = rec["item"]
        quad = it["quadrant"]
        r_ok = 10.0
        r_err = -20.0 if (asymmetric_risk and it["domain"] in ("ClinicalTriage", "FinCompliance")) else -10.0

        # Queue drainage
        if t % drain_interval == 0:
            for m in queues:
                queues[m] = max(0, queues[m] - caps[m])

        queue_trace.append(sum(queues.values()) / len(queues))

        # True specialist reliability schedule
        frac = t / max(1, n - 1)
        if drift_mode == "none":
            true_acc = {"Spec": 0.92}
        elif drift_mode == "sudden":
            true_acc = {"Spec": 0.60 if t >= (n // 2) else 0.92}
        elif drift_mode == "gradual":
            true_acc = {"Spec": 0.92 - 0.32 * max(0.0, (frac - 0.35) / 0.65)}
        else:  # multi_pool
            true_acc = {
                "Senior": 0.94,
                "Junior": 0.60 if t >= (n // 2) else 0.88,
            }

        def best_specialist_util(use_externality: bool = False) -> Tuple[str, float]:
            best_m = list(queues.keys())[0]
            best_u = -1e9
            for m in queues:
                p_e = trackers[m].mean()
                q_m = queues[m]
                mu_m = caps[m] / float(drain_interval)
                rho_m = min(lam_ema[m] / max(1e-6, mu_m), 1.5)
                if use_externality and eff_gamma > 0.0 and (q_m > 0 or rho_m >= 0.6):
                    cong_cost = eff_gamma * (q_m + rho_m * (q_m + 1.0))
                else:
                    cong_cost = eff_gamma * q_m
                u_m = p_e * r_ok + (1.0 - p_e) * r_err - costs[m] - cong_cost
                if u_m > best_u:
                    best_u = u_m
                    best_m = m
            return best_m, best_u

        ent = rec["entropy"]
        unanimous = (rec["maj_count"] == 3)
        obs_m = rec["obs_missing"]
        obs_p = rec["obs_pot_ovr"]
        obs_c = rec["obs_conf_ovr"]

        did_ask = False
        did_defer = False
        chosen_spec = list(queues.keys())[0]
        final_ans = rec["greedy_ans"]

        if method == "Direct-ACT":
            pass
        elif method == "Entropy-Clarify (ADEPT)":
            if ent > 0.45:
                did_ask = True
                final_ans = rec["post_ans"]
        elif method == "Consensus-Defer (DREAM)":
            if not unanimous:
                did_defer = True
                chosen_spec, _ = best_specialist_util()
        elif method == "Congested-L2D-Only":
            p_act = calibrator.get_p_act(obs_m, obs_p, unanimous) if calibrator else (rec["maj_count"] / 3.0)
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            m_best, u_def = best_specialist_util()
            if u_def > u_act:
                did_defer = True
                chosen_spec = m_best
        elif method == "Naive-Cascade":
            if ent > 0.45:
                did_ask = True
                final_ans = rec["post_ans"]
                if rec["post_ans"] != rec["post_s1"] or rec["post_ovr_conf"]:
                    did_defer = True
                    chosen_spec, _ = best_specialist_util()
            elif obs_c or not unanimous:
                did_defer = True
                chosen_spec, _ = best_specialist_util()
        elif method == "Calibrated-Cascade":
            p_act = calibrator.get_p_act(obs_m, obs_p, unanimous) if calibrator else (rec["maj_count"] / 3.0)
            p_clear = calibrator.p_post_clear if calibrator else 0.88
            p_ovr = calibrator.p_post_ovr if calibrator else 0.40
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            m_best, u_def = best_specialist_util()
            if obs_m:
                did_ask = True
                post_p = p_ovr if rec["post_ovr_conf"] else p_clear
                u_act_post = post_p * r_ok + (1.0 - post_p) * r_err
                if u_def > u_act_post:
                    did_defer = True
                    chosen_spec = m_best
                else:
                    final_ans = rec["post_ans"]
            elif u_def > u_act:
                did_defer = True
                chosen_spec = m_best
        elif method == "Joint-Greedy":
            p_act = calibrator.get_p_act(obs_m, obs_p, unanimous) if calibrator else (rec["maj_count"] / 3.0)
            p_res, p_ovra, p_no = calibrator.get_post_ask_probs(obs_m, obs_p) if calibrator else (
                (0.45, 0.45, 0.10) if (obs_m and obs_p) else ((0.85, 0.05, 0.10) if obs_m else (0.0, 0.0, 1.0))
            )
            p_clear = calibrator.p_post_clear if calibrator else 0.88
            p_ovr = calibrator.p_post_ovr if calibrator else 0.40
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
            u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
            m_best, u_def = best_specialist_util()
            u_ask_myopic = p_res * u_clear + p_ovra * u_ovr + p_no * u_act - c_ask
            if obs_m and u_ask_myopic > max(u_act, u_def):
                did_ask = True
                final_ans = rec["post_ans"]
            elif u_def > u_act:
                did_defer = True
                chosen_spec = m_best
        elif method == "Oracle":
            if quad in ("Q2_underspec_easy", "Q4_resolvable"):
                did_ask = True
                final_ans = rec["post_ans"]
            elif quad in ("Q3_clear_hard", "Q4_escalate"):
                m_best = max(true_acc.keys(), key=lambda m: true_acc[m] * r_ok + (1 - true_acc[m]) * r_err - costs[m] - gamma_queue * queues[m])
                u_def_true = true_acc[m_best] * r_ok + (1 - true_acc[m_best]) * r_err - costs[m_best] - gamma_queue * queues[m_best]
                if u_def_true > (0.40 * r_ok + 0.60 * r_err):
                    did_defer = True
                    chosen_spec = m_best
        else:
            # Tri-VoI-L2D and its ablation variants
            use_diag = (method != "w/o Disentangled-Diag")
            use_calib = (method != "w/o Held-Out-Calib") and (calibrator is not None)

            if use_diag and use_calib:
                p_act = calibrator.get_p_act(obs_m, obs_p, unanimous)
                p_res, p_ovra, p_no = calibrator.get_post_ask_probs(obs_m, obs_p)
                p_clear = calibrator.p_post_clear
                p_ovr = calibrator.p_post_ovr
            elif not use_diag:
                # Ablate disentangled diagnosis: rely purely on entropy
                p_act = 0.55 if ent > 0.45 else 0.86
                p_res, p_ovra, p_no = (0.60, 0.15, 0.25) if ent > 0.45 else (0.05, 0.05, 0.90)
                p_clear = calibrator.p_post_clear if calibrator else 0.88
                p_ovr = calibrator.p_post_ovr if calibrator else 0.40
            else:
                # w/o Held-Out-Calib: raw heuristic priors
                p_act = 0.50 if obs_m else (0.42 if obs_c else (rec["maj_count"] / 3.0))
                if not obs_m:
                    p_res, p_ovra, p_no = (0.0, 0.0, 1.0)
                elif obs_p:
                    p_res, p_ovra, p_no = (0.45, 0.45, 0.10)
                else:
                    p_res, p_ovra, p_no = (0.85, 0.05, 0.10)
                p_clear = 0.88
                p_ovr = 0.40

            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            m_best, u_def = best_specialist_util(use_externality=use_ext_flag)

            # Reference defer utility inside EVOI: live coupled vs static decoupled controls
            if method == "Decoupled-Nominal-EVOI":
                u_def_ref = 0.80 * r_ok + 0.20 * r_err - costs[m_best] - eff_gamma * 1.0
            elif method in ("w/o Coupled-EVOI (Static-Opt)", "w/o Coupled-EVOI"):
                u_def_ref = 0.90 * r_ok + 0.10 * r_err - costs[m_best]
            else:
                u_def_ref = u_def

            u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
            u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err

            # Two-stage Bellman expected utility of ASK over exact 3-outcome partition
            u_ask = (
                p_res * max(u_clear, u_def_ref)
                + p_ovra * max(u_ovr, u_def_ref)
                + p_no * max(u_act, u_def_ref)
                - c_ask
            )

            if u_ask > max(u_act, u_def_ref):
                did_ask = True
                # Stage 2 after requester reply (always uses live u_def)
                post_p = (p_ovr if rec["post_ovr_conf"] else p_clear) if use_diag else (0.55 if rec["post_ans"] != rec["post_s1"] else p_clear)
                u_act_post = post_p * r_ok + (1.0 - post_p) * r_err
                m_best2, u_def2 = best_specialist_util(use_externality=use_ext_flag)
                if u_def2 > u_act_post:
                    did_defer = True
                    chosen_spec = m_best2
                else:
                    final_ans = rec["post_ans"]
            elif u_def > u_act:
                did_defer = True
                chosen_spec = m_best

        # Online arrival-rate EMA update (traffic intensity tracker)
        for m in queues:
            lam_ema[m] = 0.90 * lam_ema[m] + 0.10 * (1.0 if (did_defer and chosen_spec == m) else 0.0)

        # Realize step utility (environment always charges true private wait gamma_queue * Q_t)
        step_u = 0.0
        if did_ask:
            ask_cnt += 1
            step_u -= c_ask

        if did_defer:
            defer_cnt += 1
            wait_sum += queues[chosen_spec]
            step_u -= costs[chosen_spec] + gamma_queue * queues[chosen_spec]
            queues[chosen_spec] += 1
            is_ok = rng.random() < true_acc[chosen_spec]
            trackers[chosen_spec].update(is_ok)
        else:
            is_ok = (final_ans == it["gold"])

        step_u += r_ok if is_ok else r_err
        per_item_utils[t] = step_u
        if is_ok:
            correct_cnt += 1
        rolling_acc.append(1.0 if is_ok else 0.0)
        if t >= n // 2:
            phase2_utils.append(step_u)

        # Misrouting accounting
        if quad in ("Q2_underspec_easy", "Q4_resolvable") and (did_defer and not did_ask):
            misroute_cnt += 1
        elif quad == "Q3_clear_hard" and did_ask:
            misroute_cnt += 1
        elif quad == "Q1_clear_easy" and (did_ask or did_defer):
            misroute_cnt += 1

        if quad in ("Q4_resolvable", "Q4_escalate"):
            q4_tot_cnt += 1
            if did_ask:
                q4_ask_cnt += 1
        if quad == "Q4_resolvable":
            q4_res_tot_cnt += 1
            if not did_defer:
                q4_res_act_cnt += 1

        qs = quad_stats[quad]
        qs["n"] += 1
        qs["ok"] += int(is_ok)
        qs["util"] += step_u
        qs["ask"] += int(did_ask)
        qs["defer"] += int(did_defer)

    return {
        "method": method,
        "acc_pct": 100.0 * correct_cnt / n,
        "net_utility": float(np.sum(per_item_utils)),
        "mean_utility": float(np.mean(per_item_utils)),
        "ask_rate_pct": 100.0 * ask_cnt / n,
        "defer_rate_pct": 100.0 * defer_cnt / n,
        "mean_wait": wait_sum / max(1, defer_cnt),
        "mean_queue_len": float(np.mean(queue_trace)),
        "misroute_pct": 100.0 * misroute_cnt / n,
        "q4_ask_rate_pct": 100.0 * q4_ask_cnt / max(1, q4_tot_cnt),
        "q4_res_self_solve_pct": 100.0 * q4_res_act_cnt / max(1, q4_res_tot_cnt),
        "phase2_utility": float(np.sum(phase2_utils)),
        "per_item_utils": per_item_utils,
        "per_item_correct": np.array(rolling_acc, dtype=float),
        "rolling_acc": rolling_acc,
        "quads": {
            q: {
                "acc_pct": round(100.0 * v["ok"] / max(1, v["n"]), 2),
                "mean_util": round(v["util"] / max(1, v["n"]), 3),
                "ask_pct": round(100.0 * v["ask"] / max(1, v["n"]), 1),
                "defer_pct": round(100.0 * v["defer"] / max(1, v["n"]), 1),
            }
            for q, v in sorted(quad_stats.items())
        },
    }


def run_fixed_state_evoi_sweep(
    test_probed: List[Dict[str, Any]],
    calibrator: HeldOutCalibrator,
    *,
    p_e_fixed: float = 0.85,
    gamma_queue: float = 0.85,
    c_ask: float = 1.5,
    c_def: float = 2.0,
) -> Dict[str, Any]:
    """Fixed-state sweep over injected queue length Q in {0..6} and p_resolve in {0.0, 0.25, 0.50, 0.75, 1.0}."""
    r_ok, r_err = 10.0, -10.0
    q_grid = [0, 1, 2, 3, 4, 5, 6]
    p_res_grid = [0.0, 0.25, 0.50, 0.75, 1.0]

    p_act_q4 = calibrator.get_p_act(True, True, True)
    u_act_q4 = p_act_q4 * r_ok + (1.0 - p_act_q4) * r_err
    u_clear = calibrator.p_post_clear * r_ok + (1.0 - calibrator.p_post_clear) * r_err
    u_ovr = calibrator.p_post_ovr * r_ok + (1.0 - calibrator.p_post_ovr) * r_err
    u_def_nom = 0.80 * r_ok + 0.20 * r_err - c_def - gamma_queue * 1.0

    q4_items = [r for r in test_probed if r["item"]["quadrant"] in ("Q4_resolvable", "Q4_escalate")]
    p_res_cal = calibrator.get_p_resolve(True, True)

    analytical_curves: Dict[str, Any] = {}
    q4_ask_curves: Dict[str, List[float]] = {}

    for p_res in p_res_grid:
        p_ovra = 1.0 - p_res
        evoi_coupled = []
        evoi_decoupled = []
        defer_dom_q = []
        defer_dom_evoi = []
        ask_rates_p = []

        for q in q_grid:
            u_def_q = p_e_fixed * r_ok + (1.0 - p_e_fixed) * r_err - c_def - gamma_queue * q
            u_base_q = max(u_act_q4, u_def_q)
            u_ask_q = p_res * max(u_clear, u_def_q) + p_ovra * max(u_ovr, u_def_q) - c_ask
            ev_q = u_ask_q - u_base_q
            evoi_coupled.append(round(float(ev_q), 4))

            if u_def_q > max(u_act_q4, u_ovr):
                defer_dom_q.append(float(q))
                defer_dom_evoi.append(float(ev_q))

            u_ask_nom = p_res * max(u_clear, u_def_nom) + p_ovra * max(u_ovr, u_def_nom) - c_ask
            evoi_decoupled.append(round(float(u_ask_nom - max(u_act_q4, u_def_nom)), 4))

            # Evaluate actual Q4 test tickets under fixed (Q, p_res override)
            asked = 0
            for rec in q4_items:
                if not rec["obs_missing"]:
                    continue
                unanimous = (rec["maj_count"] == 3)
                p_a = calibrator.get_p_act(rec["obs_missing"], rec["obs_pot_ovr"], unanimous)
                u_a = p_a * r_ok + (1.0 - p_a) * r_err
                u_ask_it = p_res * max(u_clear, u_def_q) + p_ovra * max(u_ovr, u_def_q) - c_ask
                if u_ask_it > max(u_a, u_def_q):
                    asked += 1
            ask_rates_p.append(round(100.0 * asked / max(1, len(q4_items)), 1))

        if len(defer_dom_q) >= 2:
            emp_slope = float(np.polyfit(defer_dom_q, defer_dom_evoi, 1)[0])
        else:
            emp_slope = 0.0
        dec_slope = float(np.polyfit(q_grid, evoi_decoupled, 1)[0])
        analytical_curves[f"{p_res:.2f}"] = {
            "p_resolve": p_res,
            "theoretical_slope": round(p_res * gamma_queue, 4),
            "empirical_slope": round(emp_slope, 4),
            "decoupled_slope": round(dec_slope, 4),
            "evoi_coupled": evoi_coupled,
            "evoi_decoupled": evoi_decoupled,
        }
        q4_ask_curves[f"{p_res:.2f}"] = ask_rates_p

    # 2D Phase Verification across Q in {0..6} x p_e in {0.75, 0.85, 0.92}
    phase_2d_by_pe: Dict[str, Any] = {}
    for pe_val in [0.75, 0.85, 0.92]:
        evoi_pe = []
        cal_q4_ask_pe = []
        nom_q4_ask_pe = []
        for q in q_grid:
            u_def_pe = pe_val * r_ok + (1.0 - pe_val) * r_err - c_def - gamma_queue * q
            u_ask_pe = (
                p_res_cal * max(u_clear, u_def_pe)
                + (1.0 - p_res_cal) * max(u_ovr, u_def_pe)
                - c_ask
            )
            evoi_pe.append(round(float(u_ask_pe - max(u_act_q4, u_def_pe)), 4))

            asked_cal = 0
            asked_nom = 0
            for rec in q4_items:
                unanimous = (rec["maj_count"] == 3)
                p_a = calibrator.get_p_act(rec["obs_missing"], rec["obs_pot_ovr"], unanimous)
                u_a = p_a * r_ok + (1.0 - p_a) * r_err
                pr, po, pn = calibrator.get_post_ask_probs(rec["obs_missing"], rec["obs_pot_ovr"])
                u_ask_cal = pr * max(u_clear, u_def_pe) + po * max(u_ovr, u_def_pe) + pn * max(u_a, u_def_pe) - c_ask
                if u_ask_cal > max(u_a, u_def_pe):
                    asked_cal += 1
                u_ask_nom = pr * max(u_clear, u_def_nom) + po * max(u_ovr, u_def_nom) + pn * max(u_a, u_def_nom) - c_ask
                if u_ask_nom > max(u_a, u_def_nom):
                    asked_nom += 1
            cal_q4_ask_pe.append(round(100.0 * asked_cal / max(1, len(q4_items)), 1))
            nom_q4_ask_pe.append(round(100.0 * asked_nom / max(1, len(q4_items)), 1))
        phase_2d_by_pe[f"{pe_val:.2f}"] = {
            "p_e": pe_val,
            "evoi_by_q": evoi_pe,
            "q4_ask_pct_calibrated": cal_q4_ask_pe,
            "q4_ask_pct_decoupled_nominal": nom_q4_ask_pe,
        }

    cal_q4_ask = phase_2d_by_pe["0.85"]["q4_ask_pct_calibrated"]
    nom_q4_ask = phase_2d_by_pe["0.85"]["q4_ask_pct_decoupled_nominal"]

    return {
        "q_grid": q_grid,
        "p_resolve_grid": p_res_grid,
        "gamma_queue": gamma_queue,
        "p_e_fixed": p_e_fixed,
        "calibrated_p_resolve_q4": round(p_res_cal, 4),
        "n_q4_items": len(q4_items),
        "analytical_curves": analytical_curves,
        "q4_ask_pct_by_p_resolve": q4_ask_curves,
        "q4_ask_pct_calibrated": cal_q4_ask,
        "q4_ask_pct_decoupled_nominal": nom_q4_ask,
        "phase_2d_by_pe": phase_2d_by_pe,
    }
