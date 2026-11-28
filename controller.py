import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark import DOMAIN_RUBRIC, user_oracle
from external_baselines import (
    KLPBDController,
    acl_voi_calib_decision,
    parse_and_compute_acl_voi,
)
from llm_client import call_llm


def parse_json_obj(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    ans_m = re.search(r"\b([A-D])\b", text.upper())
    return {"answer": ans_m.group(1) if ans_m else "A"}


def extract_answer(text: str, valid_choices: Sequence[str] = ("A", "B", "C")) -> str:
    allowed = tuple(c.upper() for c in valid_choices)
    obj = parse_json_obj(text)
    ans = str(obj.get("answer", "")).strip().upper()
    if ans in allowed:
        return ans
    pattern = r"\b([" + "".join(allowed) + r"])\b"
    m = re.search(pattern, text.upper())
    return m.group(1) if m else allowed[0]


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
    tag_tokens: Dict[str, int] = {
        "probe_s0": 0,
        "probe_s1": 0,
        "probe_s2": 0,
        "greedy_act": 0,
        "diag_init": 0,
        "post_ask_combined": 0,
        "post_ask_s1": 0,
    }
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
        t_cur = r["prompt_tokens"] + r["completion_tokens"]
        tokens += t_cur
        tag_tokens[f"probe_s{s_idx}"] = t_cur

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
    t_greedy = r_greedy["prompt_tokens"] + r_greedy["completion_tokens"]
    tokens += t_greedy
    tag_tokens["greedy_act"] = t_greedy

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
    t_diag = r_diag["prompt_tokens"] + r_diag["completion_tokens"]
    tokens += t_diag
    tag_tokens["diag_init"] = t_diag

    obs_missing = bool(diag_obj.get("missing_user_info", False))
    obs_pot_ovr = bool(diag_obj.get("potential_override", False))
    obs_conf_ovr = bool(diag_obj.get("override_confirmed", False))
    clarify_q = str(diag_obj.get("clarifying_question", "")).strip()

    if clarify_q:
        oracle_reply, oracle_revealed = user_oracle(item, clarify_q)

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
        post_missing_parsed = "missing_user_info" in post_obj
        post_missing = bool(post_obj["missing_user_info"]) if post_missing_parsed else bool(obs_missing)
        t_post = r_post["prompt_tokens"] + r_post["completion_tokens"]
        tokens += t_post
        tag_tokens["post_ask_combined"] = t_post

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
        t_ps1 = r_post_s1["prompt_tokens"] + r_post_s1["completion_tokens"]
        tokens += t_ps1
        tag_tokens["post_ask_s1"] = t_ps1
    else:
        oracle_reply = "No clarifying question was asked."
        oracle_revealed = False
        post_ans = greedy_ans
        post_s1 = samples[0]
        post_ovr_conf = obs_conf_ovr
        post_missing_parsed = True
        post_missing = bool(obs_missing)

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
        "post_missing": post_missing,
        "post_missing_parsed": post_missing_parsed,
        "tokens": tokens,
        "tag_tokens": tag_tokens,
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


def _is_self_act_correct(item: Dict[str, Any], ans: str) -> bool:
    """True iff answer matches gold AND item does not require mandatory Specialist escalation."""
    return (ans == item["gold"]) and (not item.get("require_defer_for_ok", False))


class HeldOutCalibrator:
    """Hierarchical Empirical Bayes calibrator on held-out split (zero test leakage)."""

    def __init__(
        self,
        cal_probed: List[Dict[str, Any]],
        *,
        k: float = 2.0,
        k_fine: float = 3.0,
        stratify_post_clear: bool = False,
    ):
        # Defaults k=2.0, k_fine=3.0, stratify_post_clear=False are frozen for Benchmark v1/v2.
        self.stratify_post_clear = stratify_post_clear
        coarse_stats = defaultdict(lambda: [0, 0])
        fine_stats = defaultdict(lambda: [0, 0])
        post_clear = [0, 0]
        post_clear_by_ovr = {False: [0, 0], True: [0, 0]}
        post_ovr = [0, 0]
        post_no_info = [0, 0]
        tri_stats = defaultdict(lambda: [0, 0, 0])  # [n_resolve, n_ovr_after, n_no_info]
        post_clear_self = [0, 0]
        post_clear_by_ovr_self = {False: [0, 0], True: [0, 0]}
        post_no_info_self = [0, 0]
        tri_stats_self = defaultdict(lambda: [0, 0, 0])

        for r in cal_probed:
            it_cal = r["item"]
            m_flag = bool(r["obs_missing"])
            p_flag = bool(r["obs_pot_ovr"])
            u_flag = (r["maj_count"] == 3)
            pm_flag = bool(r["post_missing"]) if "post_missing" in r else bool(m_flag)
            coarse_stats[(m_flag, p_flag)][1] += 1
            fine_stats[(m_flag, p_flag, u_flag)][1] += 1
            if _is_self_act_correct(it_cal, r["greedy_ans"]):
                coarse_stats[(m_flag, p_flag)][0] += 1
                fine_stats[(m_flag, p_flag, u_flag)][0] += 1

            if r["post_ovr_conf"]:
                post_ovr[1] += 1
                if _is_self_act_correct(it_cal, r["post_ans"]):
                    post_ovr[0] += 1
            elif m_flag and r["oracle_revealed"]:
                post_clear[1] += 1
                post_clear_by_ovr[p_flag][1] += 1
                if _is_self_act_correct(it_cal, r["post_ans"]):
                    post_clear[0] += 1
                    post_clear_by_ovr[p_flag][0] += 1
            elif m_flag and (not r["oracle_revealed"]):
                post_no_info[1] += 1
                if _is_self_act_correct(it_cal, r["post_ans"]):
                    post_no_info[0] += 1

            r_key = (m_flag, p_flag)
            if m_flag and r["oracle_revealed"] and (not r["post_ovr_conf"]):
                tri_stats[r_key][0] += 1
            elif m_flag and r["oracle_revealed"] and r["post_ovr_conf"]:
                tri_stats[r_key][1] += 1
            else:
                tri_stats[r_key][2] += 1

            if r["post_ovr_conf"]:
                if m_flag and (not pm_flag):
                    tri_stats_self[r_key][1] += 1
                else:
                    tri_stats_self[r_key][2] += 1
            elif m_flag and (not pm_flag):
                post_clear_self[1] += 1
                post_clear_by_ovr_self[p_flag][1] += 1
                if _is_self_act_correct(it_cal, r["post_ans"]):
                    post_clear_self[0] += 1
                    post_clear_by_ovr_self[p_flag][0] += 1
                tri_stats_self[r_key][0] += 1
            elif m_flag and pm_flag:
                post_no_info_self[1] += 1
                if _is_self_act_correct(it_cal, r["post_ans"]):
                    post_no_info_self[0] += 1
                tri_stats_self[r_key][2] += 1
            else:
                tri_stats_self[r_key][2] += 1

        self.coarse_p_act: Dict[Tuple[bool, bool], float] = {}
        for ck, (ok, tot) in coarse_stats.items():
            prior_c = 0.85 if ck == (False, False) else (0.35 if ck[1] else 0.55)
            self.coarse_p_act[ck] = (ok + k * prior_c) / (tot + k)

        self.p_act: Dict[Tuple[bool, bool, bool], float] = {}
        for fk, (ok, tot) in fine_stats.items():
            coarse_prior = self.coarse_p_act.get((fk[0], fk[1]), 0.60)
            self.p_act[fk] = (ok + k_fine * coarse_prior) / (tot + k_fine)

        self.p_post_clear = (post_clear[0] + k * 0.85) / (post_clear[1] + k) if (post_clear[1] + k) > 0 else 0.85
        self.p_post_clear_by_ovr: Dict[bool, float] = {
            pf: (
                (post_clear_by_ovr[pf][0] + k * (0.85 if not pf else 0.65)) / (post_clear_by_ovr[pf][1] + k)
                if (post_clear_by_ovr[pf][1] + k) > 0
                else (0.85 if not pf else 0.65)
            )
            for pf in (False, True)
        }
        self.p_post_ovr = (post_ovr[0] + k * 0.35) / (post_ovr[1] + k) if (post_ovr[1] + k) > 0 else 0.35
        self.p_post_no_info = (post_no_info[0] + k * 0.65) / (post_no_info[1] + k) if (post_no_info[1] + k) > 0 else 0.65

        self.p_post_clear_self = (post_clear_self[0] + k * 0.85) / (post_clear_self[1] + k) if (post_clear_self[1] + k) > 0 else 0.85
        self.p_post_clear_by_ovr_self: Dict[bool, float] = {
            pf: (
                (post_clear_by_ovr_self[pf][0] + k * (0.85 if not pf else 0.65)) / (post_clear_by_ovr_self[pf][1] + k)
                if (post_clear_by_ovr_self[pf][1] + k) > 0
                else (0.85 if not pf else 0.65)
            )
            for pf in (False, True)
        }
        self.p_post_no_info_self = (post_no_info_self[0] + k * 0.65) / (post_no_info_self[1] + k) if (post_no_info_self[1] + k) > 0 else 0.65

        self.tri_probs: Dict[Tuple[bool, bool], Tuple[float, float, float]] = {}
        self.tri_probs_self: Dict[Tuple[bool, bool], Tuple[float, float, float]] = {}
        for rk in set(tri_stats.keys()) | set(tri_stats_self.keys()):
            n_res, n_ovr, n_no = tri_stats[rk]
            s_res, s_ovr, s_no = tri_stats_self[rk]
            if not rk[0]:
                self.tri_probs[rk] = (0.0, 0.0, 1.0)
                self.tri_probs_self[rk] = (0.0, 0.0, 1.0)
            else:
                prior = (0.45, 0.45, 0.10) if rk[1] else (0.85, 0.05, 0.10)
                denom = n_res + n_ovr + n_no + k
                self.tri_probs[rk] = (
                    ((n_res + k * prior[0]) / denom, (n_ovr + k * prior[1]) / denom, (n_no + k * prior[2]) / denom)
                    if denom > 0
                    else prior
                )
                denom_s = s_res + s_ovr + s_no + k
                self.tri_probs_self[rk] = (
                    ((s_res + k * prior[0]) / denom_s, (s_ovr + k * prior[1]) / denom_s, (s_no + k * prior[2]) / denom_s)
                    if denom_s > 0
                    else prior
                )

    def get_p_act(self, obs_missing: bool, obs_pot_ovr: bool, unanimous: bool) -> float:
        if (obs_missing, obs_pot_ovr, unanimous) in self.p_act:
            return self.p_act[(obs_missing, obs_pot_ovr, unanimous)]
        return self.coarse_p_act.get(
            (obs_missing, obs_pot_ovr),
            0.85 if (not obs_missing and not obs_pot_ovr) else 0.50,
        )

    def get_p_post_clear(self, obs_pot_ovr: bool = False) -> float:
        if self.stratify_post_clear:
            return self.p_post_clear_by_ovr.get(bool(obs_pot_ovr), self.p_post_clear)
        return self.p_post_clear

    def get_p_post_clear_self(self, obs_pot_ovr: bool = False) -> float:
        if self.stratify_post_clear:
            return self.p_post_clear_by_ovr_self.get(bool(obs_pot_ovr), self.p_post_clear_self)
        return self.p_post_clear_self

    def get_post_ask_probs(self, obs_missing: bool, obs_pot_ovr: bool) -> Tuple[float, float, float]:
        if not obs_missing:
            return (0.0, 0.0, 1.0)
        default = (0.45, 0.45, 0.10) if obs_pot_ovr else (0.85, 0.05, 0.10)
        return self.tri_probs.get((obs_missing, obs_pot_ovr), default)

    def get_post_ask_probs_self(self, obs_missing: bool, obs_pot_ovr: bool) -> Tuple[float, float, float]:
        if not obs_missing:
            return (0.0, 0.0, 1.0)
        default = (0.45, 0.45, 0.10) if obs_pot_ovr else (0.85, 0.05, 0.10)
        return self.tri_probs_self.get((obs_missing, obs_pot_ovr), default)

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


def solve_3action_mdp_shadow_prices(
    omega_4: Tuple[float, float, float, float],
    calibrator: Optional[HeldOutCalibrator],
    p_e: float,
    capacity: int,
    drain_interval: int,
    gamma_queue: float,
    c_ask: float,
    c_defer_base: float,
    r_ok: float,
    r_err: float,
    rem_steps: int,
    unclarified_spec_penalty: bool = False,
    q_max: int = 22,
    horizon_cap: int = 25,
    return_all_horizons: bool = False,
    use_self_signal: bool = False,
    policy_mode: str = "tri_voi",
) -> Any:
    """Exact finite-horizon periodic-drain 3-action Bellman DP value table V[phase][q_pre]."""
    di = max(1, int(drain_interval))
    H = min(int(horizon_cap), max(0, int(rem_steps)))
    V_next: List[List[float]] = [[0.0] * (q_max + 1) for _ in range(di)]
    hist: List[List[List[float]]] = [V_next]
    if H <= 0:
        return hist if return_all_horizons else V_next

    s_w = float(sum(omega_4))
    probs = [float(w) / s_w for w in omega_4] if s_w > 0.0 else [0.25, 0.25, 0.25, 0.25]
    buckets = ((False, False), (False, True), (True, False), (True, True))

    cap = max(0, int(capacity))
    u_def_std_0 = p_e * r_ok + (1.0 - p_e) * r_err - c_defer_base
    p_e_rw = min(p_e, 0.55)
    u_def_rw_0 = p_e_rw * r_ok + (1.0 - p_e_rw) * r_err - c_defer_base - c_ask

    b_params = []
    for (om, op), w_b in zip(buckets, probs):
        if calibrator is not None:
            p_act = calibrator.coarse_p_act.get((om, op), 0.85 if (not om and not op) else 0.50)
            if use_self_signal:
                p_res, p_ovra, p_no = calibrator.get_post_ask_probs_self(om, op)
                p_clear = calibrator.get_p_post_clear_self(op)
                p_ni = calibrator.p_post_no_info_self
            else:
                p_res, p_ovra, p_no = calibrator.get_post_ask_probs(om, op)
                p_clear = calibrator.get_p_post_clear(op)
                p_ni = calibrator.p_post_no_info
            p_ovr = calibrator.p_post_ovr
        else:
            p_act = 0.85 if (not om and not op) else (0.35 if op else 0.55)
            p_res, p_ovra, p_no = ((0.45, 0.45, 0.10) if op else (0.85, 0.05, 0.10)) if om else (0.0, 0.0, 1.0)
            p_clear, p_ovr, p_ni = 0.88, 0.40, 0.65
        u_act = p_act * r_ok + (1.0 - p_act) * r_err
        u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
        u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
        u_no_act = (p_ni * r_ok + (1.0 - p_ni) * r_err) if om else u_act
        p_rw = (p_res + p_ovra) if (om and unclarified_spec_penalty) else 0.0
        u_def_pre_base = ((1.0 - p_rw) * u_def_std_0 + p_rw * u_def_rw_0) if p_rw > 0.0 else u_def_std_0
        b_params.append((w_b, om, u_act, u_clear, u_ovr, u_no_act, p_res, p_ovra, p_no, u_def_pre_base))

    for _ in range(H):
        V_cur: List[List[float]] = [[0.0] * (q_max + 1) for _ in range(di)]
        for ph in range(di):
            nxt_ph = (ph + 1) % di
            vn = V_next[nxt_ph]
            vc = V_cur[ph]
            q_start = min(cap, q_max) if (ph == 0 and cap > 0) else 0
            for q_pre in range(q_start, q_max + 1):
                q = (q_pre - cap) if (ph == 0 and cap > 0) else q_pre
                v_stay = vn[q]
                v_push = vn[min(q_max, q + 1)]
                gq = gamma_queue * q
                u_def_post = u_def_std_0 - gq + v_push
                delta_v = -gq + v_push
                total_v = 0.0
                for (w_b, om, u_act, u_clear, u_ovr, u_no_act, p_res, p_ovra, p_no, u_def_pre_base) in b_params:
                    val_act = u_act + v_stay
                    u_def_pre = u_def_pre_base + delta_v
                    if om:
                        c_stay = u_clear + v_stay
                        o_stay = u_ovr + v_stay
                        n_stay = u_no_act + v_stay
                        val_ask = (
                            p_res * (c_stay if c_stay >= u_def_post else u_def_post)
                            + p_ovra * (o_stay if o_stay >= u_def_post else u_def_post)
                            + p_no * (n_stay if n_stay >= u_def_pre else u_def_pre)
                            - c_ask
                        )
                        if policy_mode == "cascade":
                            total_v += w_b * val_ask
                        elif policy_mode == "joint_greedy":
                            val_ask_jg = (p_res * u_clear + p_ovra * u_ovr + p_no * u_no_act - c_ask) + v_stay
                            best_branch = val_act if val_act >= u_def_pre else u_def_pre
                            if val_ask_jg > best_branch:
                                best_branch = val_ask_jg
                            total_v += w_b * best_branch
                        else:
                            best_branch = val_act if val_act >= u_def_pre else u_def_pre
                            if val_ask > best_branch:
                                best_branch = val_ask
                            total_v += w_b * best_branch
                    else:
                        total_v += w_b * (val_act if val_act >= u_def_pre else u_def_pre)
                vc[q_pre] = total_v
            if ph == 0 and cap > 0:
                val0 = vc[q_start]
                for q_z in range(q_start):
                    vc[q_z] = val0
        V_next = V_cur
        if return_all_horizons:
            hist.append(V_next)

    return hist if return_all_horizons else V_next


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
    delayed_feedback: bool = False,
    unclarified_spec_penalty: bool = False,
    cal_probed: Optional[List[Dict[str, Any]]] = None,
    ask_latency: int = 0,
    beta_wait_ask: float = 0.25,
    real_specialist_ok: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Simulate sequential human-AI cooperative stream with queuing & drift."""
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
        "Entropy-ASK (Proxy)",
        "Entropy-Clarify (ADEPT)",
        "Consensus-DEFER (Proxy)",
        "Consensus-Defer (DREAM)",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Calibrated-Cascade",
        "Oracle",
        "w/o Queue-Externality",
        "w/o Congestion-Term",
        "KL-PBD",
        "KL-PBD+",
        "KL-PBD+ (Warm)",
        "KL-PBD+ (SC-Only)",
        "ACL-2026-VoI (Calib)",
        "ACL-2026-VoI+Post-Defer",
        "ACL-2026-VoI (Live)",
        "ACL-2026-VoI (Live)+Post-Defer",
    )

    kl_ctrl: Optional[KLPBDController] = None
    if method in ("KL-PBD", "KL-PBD+", "KL-PBD+ (Warm)", "KL-PBD+ (SC-Only)"):
        kl_var = (
            "kl_pbd"
            if method == "KL-PBD"
            else ("kl_pbd_plus" if method == "KL-PBD+" else "kl_pbd_plus_warm")
        )
        kl_ctrl = KLPBDController(
            variant=kl_var,
            total_horizon=n,
            capacity=capacity,
            drain_interval=drain_interval,
            c_hold=gamma_queue,
            c_defer=c_defer_base,
            cal_probed=cal_probed,
            sc_only=(method == "KL-PBD+ (SC-Only)"),
        )

    per_item_utils = np.zeros(n, dtype=float)
    per_item_tokens = np.zeros(n, dtype=float)
    smooth_naor = (method == "Tri-VoI-L2D (Smooth-Naor)")
    mdp_q_max = 22
    mdp_v_hist: Dict[str, Optional[List[List[List[float]]]]] = {m: None for m in queues}
    mdp_last_pe: Dict[str, float] = {m: -1.0 for m in queues}
    mdp_last_omega: Dict[str, Optional[Tuple[float, float, float, float]]] = {m: None for m in queues}
    omega_counts = [1.0, 1.0, 1.0, 1.0]
    correct_cnt = 0
    raw_correct_cnt = 0
    mandatory_tot = 0
    mandatory_defer = 0
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
    rolling_acc = [0.0] * n
    queue_trace = []
    pending_fb: List[Tuple[int, str, bool]] = []
    pending_kl_fb: List[Tuple[int, int, bool]] = []
    pending_ask: List[Dict[str, Any]] = []

    for t, rec in enumerate(test_probed):
        it = rec["item"]
        quad = it["quadrant"]
        r_ok = 10.0
        r_err = -20.0 if (asymmetric_risk and it["domain"] in ("ClinicalTriage", "FinCompliance")) else -10.0

        # Flush delayed specialist feedback ready by step t
        if delayed_feedback and pending_fb:
            still_pending = []
            for ready_t, spec_m, fb_ok in pending_fb:
                if ready_t <= t:
                    trackers[spec_m].update(fb_ok)
                else:
                    still_pending.append((ready_t, spec_m, fb_ok))
            pending_fb = still_pending

        if kl_ctrl is not None and pending_kl_fb:
            still_kl = []
            for ready_t, lvl_j, llm_ok in pending_kl_fb:
                if ready_t <= t:
                    kl_ctrl.record_completion(lvl_j, llm_ok)
                else:
                    still_kl.append((ready_t, lvl_j, llm_ok))
            pending_kl_fb = still_kl

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

        ent = rec["entropy"]
        unanimous = (rec["maj_count"] == 3)
        obs_m = rec["obs_missing"]
        obs_p = rec["obs_pot_ovr"]
        obs_c = rec["obs_conf_ovr"]

        use_mdp_pricing = ("MDP" in method) or ("OwnDP" in method)
        if use_mdp_pricing:
            b_idx = (2 if obs_m else 0) + (1 if obs_p else 0)
            omega_counts[b_idx] += 1.0
            s_om = omega_counts[0] + omega_counts[1] + omega_counts[2] + omega_counts[3]
            omega_probs = (
                omega_counts[0] / s_om,
                omega_counts[1] / s_om,
                omega_counts[2] / s_om,
                omega_counts[3] / s_om,
            )
            rem_after = n - 1 - t
            h_idx = 25 if rem_after >= 25 else rem_after
            next_ph = (t + 1) % drain_interval

        def mdp_congestion_cost(m_name: str, use_self_sig: bool = False, policy_mode: str = "tri_voi") -> float:
            q_m = queues[m_name]
            if eff_gamma <= 0.0 or rem_after <= 0:
                return eff_gamma * q_m
            p_e_m = trackers[m_name].mean()
            lo = mdp_last_omega[m_name]
            need_solve = (
                mdp_v_hist[m_name] is None
                or abs(p_e_m - mdp_last_pe[m_name]) > 0.015
                or lo is None
                or (abs(omega_probs[0] - lo[0]) + abs(omega_probs[1] - lo[1]) + abs(omega_probs[2] - lo[2]) + abs(omega_probs[3] - lo[3])) > 0.035
            )
            if need_solve:
                mdp_v_hist[m_name] = solve_3action_mdp_shadow_prices(
                    omega_4=omega_probs,
                    calibrator=calibrator,
                    p_e=p_e_m,
                    capacity=caps[m_name],
                    drain_interval=drain_interval,
                    gamma_queue=eff_gamma,
                    c_ask=c_ask,
                    c_defer_base=costs[m_name],
                    r_ok=r_ok,
                    r_err=r_err,
                    rem_steps=25,
                    unclarified_spec_penalty=unclarified_spec_penalty,
                    q_max=mdp_q_max,
                    horizon_cap=25,
                    return_all_horizons=True,
                    use_self_signal=use_self_sig,
                    policy_mode=policy_mode,
                )
                mdp_last_pe[m_name] = p_e_m
                mdp_last_omega[m_name] = omega_probs
            vt = mdp_v_hist[m_name][h_idx][next_ph]
            idx0 = min(mdp_q_max - 1, q_m)
            shadow_m = max(0.0, vt[idx0] - vt[idx0 + 1])
            return eff_gamma * q_m + shadow_m

        def best_specialist_util(
            use_externality: bool = False,
            use_mdp: bool = False,
            use_self_sig: bool = False,
            policy_mode: str = "tri_voi",
        ) -> Tuple[str, float]:
            best_m = list(queues.keys())[0]
            best_u = -1e9
            for m in queues:
                p_e = trackers[m].mean()
                q_m = queues[m]
                if use_mdp:
                    cong_cost = mdp_congestion_cost(m, use_self_sig=use_self_sig, policy_mode=policy_mode)
                else:
                    mu_m = caps[m] / float(drain_interval)
                    rho_m = min(lam_ema[m] / max(1e-6, mu_m), 1.5)
                    if use_externality and eff_gamma > 0.0 and (smooth_naor or q_m >= caps[m] or rho_m >= 0.6):
                        cong_cost = eff_gamma * (q_m + rho_m * (q_m + 1.0))
                    else:
                        cong_cost = eff_gamma * q_m
                u_m = p_e * r_ok + (1.0 - p_e) * r_err - costs[m] - cong_cost
                if u_m > best_u:
                    best_u = u_m
                    best_m = m
            return best_m, best_u

        def best_specialist_util_pre_ask(
            use_externality: bool = False,
            p_rework: float = 0.0,
            use_mdp: bool = False,
            use_self_sig: bool = False,
            policy_mode: str = "tri_voi",
        ) -> Tuple[str, float]:
            if (not unclarified_spec_penalty) or p_rework <= 0.0:
                return best_specialist_util(use_externality=use_externality, use_mdp=use_mdp, use_self_sig=use_self_sig, policy_mode=policy_mode)
            best_m = list(queues.keys())[0]
            best_u = -1e9
            for m in queues:
                p_e = trackers[m].mean()
                q_m = queues[m]
                if use_mdp:
                    cong_cost = mdp_congestion_cost(m, use_self_sig=use_self_sig, policy_mode=policy_mode)
                else:
                    mu_m = caps[m] / float(drain_interval)
                    rho_m = min(lam_ema[m] / max(1e-6, mu_m), 1.5)
                    if use_externality and eff_gamma > 0.0 and (smooth_naor or q_m >= caps[m] or rho_m >= 0.6):
                        cong_cost = eff_gamma * (q_m + rho_m * (q_m + 1.0))
                    else:
                        cong_cost = eff_gamma * q_m
                u_std = p_e * r_ok + (1.0 - p_e) * r_err - costs[m] - cong_cost
                p_e_rw = min(p_e, 0.55)
                u_rw = p_e_rw * r_ok + (1.0 - p_e_rw) * r_err - costs[m] - c_ask - cong_cost
                u_m = (1.0 - p_rework) * u_std + p_rework * u_rw
                if u_m > best_u:
                    best_u = u_m
                    best_m = m
            return best_m, best_u

        did_ask = False
        did_defer = False
        chosen_spec = list(queues.keys())[0]
        final_ans = rec["greedy_ans"]

        if method == "Direct-ACT":
            pass
        elif method in ("Entropy-ASK (Proxy)", "Entropy-Clarify (ADEPT)"):
            if ent > 0.45:
                did_ask = True
                final_ans = rec["post_ans"]
        elif method in ("Consensus-DEFER (Proxy)", "Consensus-Defer (DREAM)"):
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
        elif method in ("KL-PBD", "KL-PBD+", "KL-PBD+ (Warm)", "KL-PBD+ (SC-Only)"):
            assert kl_ctrl is not None
            m_best = max(queues.keys(), key=lambda m: trackers[m].mean() * r_ok + (1.0 - trackers[m].mean()) * r_err - costs[m] - gamma_queue * queues[m])
            if kl_ctrl.should_defer(rec, queues[m_best], n - t, r_ok, r_err, trackers[m_best].mean()):
                did_defer = True
                chosen_spec = m_best
        elif method in (
            "ACL-2026-VoI (Calib)",
            "ACL-2026-VoI+Post-Defer",
            "ACL-2026-VoI+Full-Defer+Naor",
            "ACL-2026-VoI+Full-Defer+MDP",
            "ACL-2026-VoI+Full-Defer+MDP (2C-Self)",
            "ACL-2026-VoI (Live)",
            "ACL-2026-VoI (Live)+Post-Defer",
            "ACL-2026-VoI (Live)+Full-Defer+Naor",
            "ACL-2026-VoI (Live)+Full-Defer+MDP",
            "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)",
        ):
            acl_2c_self = ("(2C-Self)" in method)
            if calibrator is not None:
                if acl_2c_self:
                    p_act = calibrator.coarse_p_act.get((obs_m, obs_p), 0.85 if (not obs_m and not obs_p) else 0.50)
                    p_res, p_ovra, p_no = calibrator.get_post_ask_probs_self(obs_m, obs_p)
                    p_clear = calibrator.get_p_post_clear_self(obs_p)
                    p_ni_acl = calibrator.p_post_no_info_self
                else:
                    p_act = calibrator.get_p_act(obs_m, obs_p, unanimous)
                    p_res, p_ovra, p_no = calibrator.get_post_ask_probs(obs_m, obs_p)
                    p_clear = calibrator.get_p_post_clear(obs_p)
                    p_ni_acl = calibrator.p_post_no_info
                p_ovr = calibrator.p_post_ovr
            else:
                p_act = 0.85 if (acl_2c_self and not obs_m and not obs_p) else (0.50 if acl_2c_self else (rec["maj_count"] / 3.0))
                p_res, p_ovra, p_no = (
                    (0.45, 0.45, 0.10) if (obs_m and obs_p) else ((0.85, 0.05, 0.10) if obs_m else (0.0, 0.0, 1.0))
                )
                p_clear, p_ovr, p_ni_acl = 0.88, 0.40, 0.65

            use_live_acl = "Live" in method and bool(rec.get("acl_belief_raw"))
            if use_live_acl and rec.get("clarify_q"):
                choices_tuple = tuple(rec.get("valid_choices", ("A", "B", "C")))
                net_voi, acl_pre_ans, parsed_ok = parse_and_compute_acl_voi(
                    rec["acl_belief_raw"], choices=choices_tuple, r_ok=r_ok, r_err=r_err, c_ask=c_ask
                )
                if parsed_ok:
                    ask_flag = net_voi > 0.0
                    if not ask_flag:
                        final_ans = acl_pre_ans
                else:
                    ask_flag, _ = acl_voi_calib_decision(
                        p_act=p_act, p_res=p_res, p_ovra=p_ovra, p_no=p_no,
                        p_clear=p_clear, p_ovr=p_ovr, obs_m=obs_m, r_ok=r_ok, r_err=r_err, c_ask=c_ask,
                    )
            else:
                ask_flag, _ = acl_voi_calib_decision(
                    p_act=p_act, p_res=p_res, p_ovra=p_ovra, p_no=p_no,
                    p_clear=p_clear, p_ovr=p_ovr, obs_m=obs_m, r_ok=r_ok, r_err=r_err, c_ask=c_ask,
                )

            acl_use_mdp = ("+MDP" in method)
            if ask_flag:
                did_ask = True
                if "Post-Defer" in method or "Full-Defer" in method:
                    if acl_2c_self:
                        still_missing = bool(rec["post_missing"]) if "post_missing" in rec else bool(obs_m)
                        post_p = p_ni_acl if still_missing else (p_ovr if rec["post_ovr_conf"] else p_clear)
                    else:
                        post_p = p_ovr if rec["post_ovr_conf"] else p_clear
                    u_act_post = post_p * r_ok + (1.0 - post_p) * r_err
                    m_best2, u_def2 = best_specialist_util(
                        use_externality=(use_ext_flag if "Full-Defer" in method else False),
                        use_mdp=acl_use_mdp,
                        use_self_sig=acl_2c_self,
                    )
                    if u_def2 > u_act_post:
                        did_defer = True
                        chosen_spec = m_best2
                    else:
                        final_ans = rec["post_ans"]
                else:
                    final_ans = rec["post_ans"]
            elif "Full-Defer" in method:
                u_act_pre = p_act * r_ok + (1.0 - p_act) * r_err
                m_b0, u_d0 = best_specialist_util(
                    use_externality=use_ext_flag,
                    use_mdp=acl_use_mdp,
                    use_self_sig=acl_2c_self,
                )
                if u_d0 > u_act_pre:
                    did_defer = True
                    chosen_spec = m_b0
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
        elif method in ("Calibrated-Cascade", "Calibrated-Cascade+Naor", "Calibrated-Cascade+MDP"):
            p_act = calibrator.get_p_act(obs_m, obs_p, unanimous) if calibrator else (rec["maj_count"] / 3.0)
            p_clear = calibrator.get_p_post_clear(obs_p) if calibrator else 0.88
            p_ovr = calibrator.p_post_ovr if calibrator else 0.40
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            m_best, u_def = best_specialist_util(use_externality=use_ext_flag, use_mdp=("+MDP" in method))
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
        elif method in ("Calibrated-Cascade+MDP (2C-Self)", "Calibrated-Cascade+OwnDP (2C-Self)"):
            if calibrator is not None:
                p_act = calibrator.coarse_p_act.get((obs_m, obs_p), 0.85 if (not obs_m and not obs_p) else 0.50)
                p_clear = calibrator.get_p_post_clear_self(obs_p)
                p_ovr = calibrator.p_post_ovr
                p_ni = calibrator.p_post_no_info_self
            else:
                p_act, p_clear, p_ovr, p_ni = (0.85 if (not obs_m and not obs_p) else 0.50), 0.88, 0.40, 0.65
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            pol_mode = "cascade" if "OwnDP" in method else "tri_voi"
            m_best, u_def = best_specialist_util(use_externality=use_ext_flag, use_mdp=True, use_self_sig=True, policy_mode=pol_mode)
            if obs_m:
                did_ask = True
                still_missing = bool(rec["post_missing"]) if "post_missing" in rec else bool(obs_m)
                post_p = p_ni if still_missing else (p_ovr if rec["post_ovr_conf"] else p_clear)
                u_act_post = post_p * r_ok + (1.0 - post_p) * r_err
                if u_def > u_act_post:
                    did_defer = True
                    chosen_spec = m_best
                else:
                    final_ans = rec["post_ans"]
            elif u_def > u_act:
                did_defer = True
                chosen_spec = m_best
        elif method in ("Joint-Greedy", "Joint-Greedy+Naor", "Joint-Greedy+MDP"):
            p_act = calibrator.get_p_act(obs_m, obs_p, unanimous) if calibrator else (rec["maj_count"] / 3.0)
            p_res, p_ovra, p_no = calibrator.get_post_ask_probs(obs_m, obs_p) if calibrator else (
                (0.45, 0.45, 0.10) if (obs_m and obs_p) else ((0.85, 0.05, 0.10) if obs_m else (0.0, 0.0, 1.0))
            )
            p_clear = calibrator.get_p_post_clear(obs_p) if calibrator else 0.88
            p_ovr = calibrator.p_post_ovr if calibrator else 0.40
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
            u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
            m_best, u_def = best_specialist_util(use_externality=use_ext_flag, use_mdp=("+MDP" in method))
            u_ask_myopic = p_res * u_clear + p_ovra * u_ovr + p_no * u_act - c_ask
            if obs_m and u_ask_myopic > max(u_act, u_def):
                did_ask = True
                final_ans = rec["post_ans"]
            elif u_def > u_act:
                did_defer = True
                chosen_spec = m_best
        elif method in ("Joint-Greedy+MDP (2C-Self)", "Joint-Greedy+OwnDP (2C-Self)"):
            if calibrator is not None:
                p_act = calibrator.coarse_p_act.get((obs_m, obs_p), 0.85 if (not obs_m and not obs_p) else 0.50)
                p_res, p_ovra, p_no = calibrator.get_post_ask_probs_self(obs_m, obs_p)
                p_clear = calibrator.get_p_post_clear_self(obs_p)
                p_ovr = calibrator.p_post_ovr
                p_ni = calibrator.p_post_no_info_self
            else:
                p_act = 0.85 if (not obs_m and not obs_p) else 0.50
                p_res, p_ovra, p_no = ((0.45, 0.45, 0.10) if obs_p else (0.85, 0.05, 0.10)) if obs_m else (0.0, 0.0, 1.0)
                p_clear, p_ovr, p_ni = 0.88, 0.40, 0.65
            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
            u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
            u_no_act = (p_ni * r_ok + (1.0 - p_ni) * r_err) if obs_m else u_act
            pol_mode = "joint_greedy" if "OwnDP" in method else "tri_voi"
            m_best, u_def = best_specialist_util(use_externality=use_ext_flag, use_mdp=True, use_self_sig=True, policy_mode=pol_mode)
            u_ask_myopic = p_res * u_clear + p_ovra * u_ovr + p_no * u_no_act - c_ask
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
                u_self_base = 0.0 if it.get("require_defer_for_ok", False) else 0.40
                if u_def_true > (u_self_base * r_ok + (1.0 - u_self_base) * r_err):
                    did_defer = True
                    chosen_spec = m_best
        elif method in ("Tri-VoI-MDP", "Tri-VoI-MDP (5-Call)", "Tri-VoI-MDP (Deploy-Self)"):
            use_self_mdp = (method == "Tri-VoI-MDP (Deploy-Self)")
            use_coarse_mdp = (method != "Tri-VoI-MDP (5-Call)")

            if calibrator is not None:
                p_act = (
                    calibrator.coarse_p_act.get((obs_m, obs_p), 0.85 if (not obs_m and not obs_p) else 0.50)
                    if use_coarse_mdp
                    else calibrator.get_p_act(obs_m, obs_p, unanimous)
                )
                if use_self_mdp:
                    p_res, p_ovra, p_no = calibrator.get_post_ask_probs_self(obs_m, obs_p)
                    p_clear = calibrator.get_p_post_clear_self(obs_p)
                    p_ni = calibrator.p_post_no_info_self
                else:
                    p_res, p_ovra, p_no = calibrator.get_post_ask_probs(obs_m, obs_p)
                    p_clear = calibrator.get_p_post_clear(obs_p)
                    p_ni = calibrator.p_post_no_info
                p_ovr = calibrator.p_post_ovr
            else:
                p_act = 0.85 if (not obs_m and not obs_p) else (0.35 if obs_p else 0.55)
                p_res, p_ovra, p_no = ((0.45, 0.45, 0.10) if obs_p else (0.85, 0.05, 0.10)) if obs_m else (0.0, 0.0, 1.0)
                p_clear, p_ovr, p_ni = 0.88, 0.40, 0.65

            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
            u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
            u_no_act = (p_ni * r_ok + (1.0 - p_ni) * r_err) if obs_m else u_act
            p_rework = (p_res + p_ovra) if (obs_m and unclarified_spec_penalty) else 0.0
            m_best, u_def_pre = best_specialist_util_pre_ask(use_externality=use_ext_flag, p_rework=p_rework, use_mdp=True, use_self_sig=use_self_mdp)
            _, u_def_post = best_specialist_util(use_externality=use_ext_flag, use_mdp=True, use_self_sig=use_self_mdp)
            if ask_latency > 0:
                mu_best = caps[m_best] / float(drain_interval)
                delta_drift = max(0.0, lam_ema[m_best] - mu_best) * float(ask_latency)
                lat_pen = beta_wait_ask * float(ask_latency) + (1.0 - p_res) * eff_gamma * delta_drift
            else:
                lat_pen = 0.0

            u_ask = (
                (
                    p_res * max(u_clear, u_def_post)
                    + p_ovra * max(u_ovr, u_def_post)
                    + p_no * max(u_no_act, u_def_pre)
                    - c_ask
                    - lat_pen
                )
                if obs_m
                else -1e9
            )
            if u_ask > max(u_act, u_def_pre):
                did_ask = True
                still_missing = (bool(rec["post_missing"]) if "post_missing" in rec else bool(obs_m)) if use_self_mdp else (not rec["oracle_revealed"])
                if still_missing:
                    u_act_post = p_ni * r_ok + (1.0 - p_ni) * r_err
                    m_best2, u_def2 = best_specialist_util_pre_ask(use_externality=use_ext_flag, p_rework=p_rework, use_mdp=True, use_self_sig=use_self_mdp)
                elif rec["post_ovr_conf"]:
                    u_act_post = p_ovr * r_ok + (1.0 - p_ovr) * r_err
                    m_best2, u_def2 = best_specialist_util(use_externality=use_ext_flag, use_mdp=True, use_self_sig=use_self_mdp)
                else:
                    u_act_post = p_clear * r_ok + (1.0 - p_clear) * r_err
                    m_best2, u_def2 = best_specialist_util(use_externality=use_ext_flag, use_mdp=True, use_self_sig=use_self_mdp)
                if u_def2 > u_act_post:
                    did_defer = True
                    chosen_spec = m_best2
                else:
                    final_ans = rec["post_ans"]
            elif u_def_pre > u_act:
                did_defer = True
                chosen_spec = m_best
        else:
            # Tri-VoI-L2D, Tri-VoI-L2D (2-Call), Tri-VoI-L2D (2-Call+NoInfo), Tri-VoI-L2D (Latency-Aware), and ablation variants
            use_diag = (method != "w/o Disentangled-Diag")
            use_calib = (method != "w/o Held-Out-Calib") and (calibrator is not None)
            use_two_call = method in ("Tri-VoI-L2D (2-Call)", "Tri-VoI-L2D (2-Call+NoInfo)")
            use_no_info_branch = (method == "Tri-VoI-L2D (2-Call+NoInfo)")

            if use_diag and use_calib:
                if use_two_call:
                    p_act = calibrator.coarse_p_act.get(
                        (obs_m, obs_p),
                        0.85 if (not obs_m and not obs_p) else 0.50,
                    )
                else:
                    p_act = calibrator.get_p_act(obs_m, obs_p, unanimous)
                p_res, p_ovra, p_no = calibrator.get_post_ask_probs(obs_m, obs_p)
                p_clear = calibrator.get_p_post_clear(obs_p)
                p_ovr = calibrator.p_post_ovr
                p_ni_l2d = calibrator.p_post_no_info
            elif not use_diag:
                # Ablate disentangled diagnosis: rely purely on entropy
                p_act = 0.55 if ent > 0.45 else 0.86
                p_res, p_ovra, p_no = (0.60, 0.15, 0.25) if ent > 0.45 else (0.05, 0.05, 0.90)
                p_clear = calibrator.get_p_post_clear(obs_p) if calibrator else 0.88
                p_ovr = calibrator.p_post_ovr if calibrator else 0.40
                p_ni_l2d = 0.65
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
                p_ni_l2d = 0.65

            u_act = p_act * r_ok + (1.0 - p_act) * r_err
            p_rework = (p_res + p_ovra) if (obs_m and unclarified_spec_penalty) else 0.0
            m_best, u_def = best_specialist_util_pre_ask(use_externality=use_ext_flag, p_rework=p_rework)
            _, u_def_post = best_specialist_util(use_externality=use_ext_flag)

            # Reference defer utility inside EVOI: live coupled vs static decoupled controls
            if method == "Decoupled-Nominal-EVOI":
                u_def_ref = 0.80 * r_ok + 0.20 * r_err - costs[m_best] - eff_gamma * 1.0
                u_def_post_ref = u_def_ref
            elif method in ("w/o Coupled-EVOI (Static-Opt)", "w/o Coupled-EVOI"):
                u_def_ref = 0.90 * r_ok + 0.10 * r_err - costs[m_best]
                u_def_post_ref = u_def_ref
            else:
                u_def_ref = u_def
                u_def_post_ref = u_def_post

            u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
            u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
            u_no_act_l2d = (p_ni_l2d * r_ok + (1.0 - p_ni_l2d) * r_err) if (use_no_info_branch and obs_m) else u_act
            if method == "Tri-VoI-L2D (Latency-Aware)" and ask_latency > 0:
                mu_best = caps[m_best] / float(drain_interval)
                delta_drift = max(0.0, lam_ema[m_best] - mu_best) * float(ask_latency)
                lat_pen = beta_wait_ask * float(ask_latency) + (1.0 - p_res) * eff_gamma * delta_drift
            else:
                lat_pen = 0.0

            # Two-stage Bellman expected utility of ASK over exact 3-outcome partition (Cor. 1 aligned)
            u_ask = (
                p_res * max(u_clear, u_def_post_ref)
                + p_ovra * max(u_ovr, u_def_post_ref)
                + p_no * max(u_no_act_l2d, u_def_ref)
                - c_ask
                - lat_pen
            )

            if u_ask > max(u_act, u_def_ref):
                did_ask = True
                if use_no_info_branch and (not rec["oracle_revealed"]):
                    u_act_post = p_ni_l2d * r_ok + (1.0 - p_ni_l2d) * r_err
                    m_best2, u_def2 = best_specialist_util_pre_ask(use_externality=use_ext_flag, p_rework=p_rework)
                else:
                    # Stage 2 after requester reply (always uses live standard u_def)
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

        def _realize_item(
            orig_t: int,
            item_rec: Dict[str, Any],
            asked: bool,
            deferred: bool,
            spec_choice: str,
            ans_final: str,
            step_now: int,
            true_acc_now: Dict[str, float],
            extra_ask_wait: float = 0.0,
        ) -> None:
            nonlocal correct_cnt, raw_correct_cnt, mandatory_tot, mandatory_defer
            nonlocal ask_cnt, defer_cnt, wait_sum, misroute_cnt
            nonlocal q4_ask_cnt, q4_tot_cnt, q4_res_act_cnt, q4_res_tot_cnt
            item_it = item_rec["item"]
            item_q = item_it["quadrant"]
            r_ok_i = 10.0
            r_err_i = -20.0 if (asymmetric_risk and item_it["domain"] in ("ClinicalTriage", "FinCompliance")) else -10.0
            s_u = -extra_ask_wait
            if asked:
                ask_cnt += 1
                s_u -= c_ask
            if item_it.get("require_defer_for_ok", False):
                mandatory_tot += 1
                if deferred:
                    mandatory_defer += 1
            if deferred:
                defer_cnt += 1
                wait_sum += queues[spec_choice]
                s_u -= costs[spec_choice] + gamma_queue * queues[spec_choice]
                queues[spec_choice] += 1
                eff_acc = true_acc_now[spec_choice]
                if (
                    unclarified_spec_penalty
                    and item_it["is_underspec"]
                    and item_it.get("missing_changes_gold", True)
                    and not (asked and item_rec["oracle_revealed"])
                ):
                    s_u -= c_ask
                    eff_acc = min(eff_acc, 0.55)
                if real_specialist_ok is not None and item_it["qid"] in real_specialist_ok:
                    is_ok_i = bool(real_specialist_ok[item_it["qid"]])
                else:
                    is_ok_i = rng.random() < eff_acc
                raw_ok_i = is_ok_i
                if kl_ctrl is not None:
                    mu_m = caps[spec_choice] / float(drain_interval)
                    ready_step_kl = step_now + max(1, math.ceil(queues[spec_choice] / max(1e-6, mu_m)))
                    pending_kl_fb.append((ready_step_kl, kl_ctrl.level_of(item_rec), _is_self_act_correct(item_it, item_rec["greedy_ans"])))
                if delayed_feedback:
                    mu_m = caps[spec_choice] / float(drain_interval)
                    ready_step = step_now + max(1, math.ceil(queues[spec_choice] / max(1e-6, mu_m)))
                    pending_fb.append((ready_step, spec_choice, is_ok_i))
                else:
                    trackers[spec_choice].update(is_ok_i)
            else:
                is_ok_i = _is_self_act_correct(item_it, ans_final)
                raw_ok_i = (ans_final == item_it["gold"])

            s_u += r_ok_i if is_ok_i else r_err_i
            per_item_utils[orig_t] = s_u
            tt = item_rec.get("tag_tokens") or {}
            if tt:
                t1_probes = tt.get("probe_s0", 0) + tt.get("probe_s1", 0) + tt.get("probe_s2", 0)
                t1_gd = tt.get("greedy_act", 0) + tt.get("diag_init", 0)
                t_post = tt.get("post_ask_combined", 0) if asked else 0
                if method == "Direct-ACT":
                    tok_i = tt.get("greedy_act", 0)
                elif "ACL-2026-VoI (Live)" in method:
                    tok_i = t1_gd + tt.get("acl_voi_belief", 0) + t_post
                elif method in ("Tri-VoI-L2D (2-Call)", "Tri-VoI-L2D (2-Call+NoInfo)", "Tri-VoI-MDP", "Tri-VoI-MDP (Deploy-Self)") or ("(2C-Self)" in method):
                    tok_i = t1_gd + t_post
                elif method == "KL-PBD+ (SC-Only)":
                    tok_i = t1_probes + tt.get("greedy_act", 0)
                elif method in ("KL-PBD", "KL-PBD+", "KL-PBD+ (Warm)"):
                    tok_i = t1_probes + t1_gd
                elif method == "Naive-Cascade":
                    tok_i = t1_probes + t1_gd + (t_post + tt.get("post_ask_s1", 0) if asked else 0)
                else:
                    tok_i = t1_probes + t1_gd + t_post
            else:
                tok_i = float(item_rec.get("tokens", 0))
            per_item_tokens[orig_t] = float(tok_i)
            if is_ok_i:
                correct_cnt += 1
            if raw_ok_i:
                raw_correct_cnt += 1
            rolling_acc[orig_t] = 1.0 if is_ok_i else 0.0
            if orig_t >= n // 2:
                phase2_utils.append(s_u)

            if item_q in ("Q2_underspec_easy", "Q4_resolvable") and (deferred and not asked):
                misroute_cnt += 1
            elif item_q == "Q3_clear_hard" and asked:
                misroute_cnt += 1
            elif item_q == "Q1_clear_easy" and (asked or deferred):
                misroute_cnt += 1

            if item_q in ("Q4_resolvable", "Q4_escalate"):
                q4_tot_cnt += 1
                if asked:
                    q4_ask_cnt += 1
            if item_q == "Q4_resolvable":
                q4_res_tot_cnt += 1
                if not deferred:
                    q4_res_act_cnt += 1

            qs = quad_stats[item_q]
            qs["n"] += 1
            qs["ok"] += int(is_ok_i)
            qs["util"] += s_u
            qs["ask"] += int(asked)
            qs["defer"] += int(deferred)

        # Flush any delayed ASK items arriving at step t
        if ask_latency > 0 and pending_ask:
            still_ask = []
            for pa in pending_ask:
                if pa["ready_t"] <= t:
                    p_rec = pa["rec"]
                    p_def = False
                    p_spec = chosen_spec
                    p_ans = p_rec["post_ans"]
                    if pa["can_post_defer"]:
                        m_b2, u_d2 = best_specialist_util(
                            use_externality=pa["use_ext"],
                            use_mdp=pa.get("use_mdp", False),
                            use_self_sig=pa.get("use_self_sig", False),
                            policy_mode=pa.get("policy_mode", "tri_voi"),
                        )
                        if u_d2 > pa["u_act_post"]:
                            p_def = True
                            p_spec = m_b2
                    for m in queues:
                        lam_ema[m] = 0.90 * lam_ema[m] + 0.10 * (1.0 if (p_def and p_spec == m) else 0.0)
                    _realize_item(pa["orig_t"], p_rec, True, p_def, p_spec, p_ans, t, true_acc, extra_ask_wait=beta_wait_ask * float(ask_latency))
                else:
                    still_ask.append(pa)
            pending_ask = still_ask

        if ask_latency > 0 and did_ask:
            can_pd = method in (
                "Tri-VoI-MDP",
                "Tri-VoI-MDP (5-Call)",
                "Tri-VoI-MDP (Deploy-Self)",
                "Tri-VoI-L2D",
                "Tri-VoI-L2D (2-Call)",
                "Tri-VoI-L2D (2-Call+NoInfo)",
                "Tri-VoI-L2D (Latency-Aware)",
                "Tri-VoI-L2D (Smooth-Naor)",
                "w/o Disentangled-Diag",
                "w/o Held-Out-Calib",
                "w/o Queue-Externality",
                "w/o Congestion-Term",
                "w/o Discounted-Forgetting",
                "Decoupled-Nominal-EVOI",
                "w/o Coupled-EVOI (Static-Opt)",
                "w/o Coupled-EVOI",
                "Calibrated-Cascade",
                "Calibrated-Cascade+Naor",
                "Calibrated-Cascade+MDP",
                "Calibrated-Cascade+MDP (2C-Self)",
                "Calibrated-Cascade+OwnDP (2C-Self)",
                "Naive-Cascade",
                "ACL-2026-VoI+Post-Defer",
                "ACL-2026-VoI+Full-Defer+Naor",
                "ACL-2026-VoI+Full-Defer+MDP",
                "ACL-2026-VoI+Full-Defer+MDP (2C-Self)",
                "ACL-2026-VoI (Live)+Post-Defer",
                "ACL-2026-VoI (Live)+Full-Defer+Naor",
                "ACL-2026-VoI (Live)+Full-Defer+MDP",
                "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)",
            )
            use_self_l = (method == "Tri-VoI-MDP (Deploy-Self)") or ("(2C-Self)" in method)
            if method == "Naive-Cascade":
                force_pd = (rec["post_ans"] != rec["post_s1"] or rec["post_ovr_conf"])
                u_ap = -1e9 if force_pd else 1e9
            else:
                if calibrator:
                    p_cl = calibrator.get_p_post_clear_self(obs_p) if use_self_l else calibrator.get_p_post_clear(obs_p)
                    p_ov = calibrator.p_post_ovr
                    p_ni_l = calibrator.p_post_no_info_self if use_self_l else calibrator.p_post_no_info
                else:
                    p_cl, p_ov, p_ni_l = 0.88, 0.40, 0.65
                if method in ("Tri-VoI-MDP", "Tri-VoI-MDP (5-Call)", "Tri-VoI-L2D (2-Call+NoInfo)"):
                    pp = p_ni_l if (not rec["oracle_revealed"]) else (p_ov if rec["post_ovr_conf"] else p_cl)
                elif use_self_l:
                    still_m_l = bool(rec["post_missing"]) if "post_missing" in rec else bool(obs_m)
                    pp = p_ni_l if still_m_l else (p_ov if rec["post_ovr_conf"] else p_cl)
                else:
                    pp = p_ov if rec["post_ovr_conf"] else p_cl
                u_ap = pp * r_ok + (1.0 - pp) * r_err
            pol_pa = "cascade" if ("Calibrated-Cascade+OwnDP" in method) else ("joint_greedy" if ("Joint-Greedy+OwnDP" in method) else "tri_voi")
            pending_ask.append({
                "ready_t": t + ask_latency,
                "orig_t": t,
                "rec": rec,
                "can_post_defer": can_pd,
                "use_ext": use_ext_flag if ("Full-Defer" in method or ("ACL" not in method and "Naive" not in method)) else False,
                "use_mdp": ("MDP" in method) or ("OwnDP" in method),
                "use_self_sig": use_self_l,
                "policy_mode": pol_pa,
                "u_act_post": u_ap,
            })
            continue

        # Online arrival-rate EMA update (traffic intensity tracker)
        for m in queues:
            lam_ema[m] = 0.90 * lam_ema[m] + 0.10 * (1.0 if (did_defer and chosen_spec == m) else 0.0)

        _realize_item(t, rec, did_ask, did_defer, chosen_spec, final_ans, t, true_acc)

    # Flush remaining delayed ASK items at end of stream
    if ask_latency > 0 and pending_ask:
        t_end = n
        for pa in pending_ask:
            while t_end < pa["ready_t"]:
                if t_end % drain_interval == 0:
                    for m in queues:
                        queues[m] = max(0, queues[m] - caps[m])
                t_end += 1
            p_rec = pa["rec"]
            p_def = False
            p_spec = list(queues.keys())[0]
            p_ans = p_rec["post_ans"]
            if pa["can_post_defer"]:
                m_b2, u_d2 = best_specialist_util(
                    use_externality=pa["use_ext"],
                    use_mdp=pa.get("use_mdp", False),
                    use_self_sig=pa.get("use_self_sig", False),
                    policy_mode=pa.get("policy_mode", "tri_voi"),
                )
                if u_d2 > pa["u_act_post"]:
                    p_def = True
                    p_spec = m_b2
            _realize_item(pa["orig_t"], p_rec, True, p_def, p_spec, p_ans, t_end, true_acc, extra_ask_wait=beta_wait_ask * float(ask_latency))

    return {
        "method": method,
        "acc_pct": 100.0 * correct_cnt / n,
        "raw_answer_acc_pct": 100.0 * raw_correct_cnt / n,
        "scope_escalation_recall_pct": round(100.0 * mandatory_defer / mandatory_tot, 2) if mandatory_tot > 0 else None,
        "mandatory_escalate_n": mandatory_tot,
        "net_utility": float(np.sum(per_item_utils)),
        "mean_utility": float(np.mean(per_item_utils)),
        "mean_realized_tokens": float(np.mean(per_item_tokens)),
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
    n_mc_seeds: int = 200,
) -> Dict[str, Any]:
    """Three-regime fixed-state sweep over Q in {0..10} with analytical boundaries & Monte Carlo realized rollout."""
    r_ok, r_err = 10.0, -10.0
    q_grid = list(range(0, 11))
    p_res_grid = [0.0, 0.25, 0.50, 0.75, 1.0]

    p_act_q4 = calibrator.get_p_act(True, True, True)
    u_act_q4 = p_act_q4 * r_ok + (1.0 - p_act_q4) * r_err
    u_clear = calibrator.p_post_clear * r_ok + (1.0 - calibrator.p_post_clear) * r_err
    u_ovr = calibrator.p_post_ovr * r_ok + (1.0 - calibrator.p_post_ovr) * r_err
    u_def_nom = 0.80 * r_ok + 0.20 * r_err - c_def - gamma_queue * 1.0

    q4_items = [r for r in test_probed if r["item"]["quadrant"] in ("Q4_resolvable", "Q4_escalate")]
    q4_askable = [r for r in q4_items if r["obs_missing"]]
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

        ana_slope = float(np.polyfit(defer_dom_q, defer_dom_evoi, 1)[0]) if len(defer_dom_q) >= 2 else 0.0
        dec_slope = float(np.polyfit(q_grid, evoi_decoupled, 1)[0])
        analytical_curves[f"{p_res:.2f}"] = {
            "p_resolve": p_res,
            "theoretical_slope": round(p_res * gamma_queue, 4),
            "analytical_defer_slope": round(ana_slope, 4),
            "decoupled_slope": round(dec_slope, 4),
            "evoi_coupled": evoi_coupled,
            "evoi_decoupled": evoi_decoupled,
        }
        q4_ask_curves[f"{p_res:.2f}"] = ask_rates_p

    # 2D Three-Regime Phase Verification + Genuine Monte Carlo Realized-Utility Rollout
    rng_mc = random.Random(2026)
    phase_2d_by_pe: Dict[str, Any] = {}
    for pe_val in [0.75, 0.85, 0.92]:
        u_def_0 = pe_val * r_ok + (1.0 - pe_val) * r_err - c_def
        q1_bound = (u_def_0 - u_act_q4) / gamma_queue
        q2_bound = (u_def_0 - u_ovr) / gamma_queue

        evoi_pe = []
        cal_q4_ask_pe = []
        nom_q4_ask_pe = []
        realized_evoi_mean = []
        realized_evoi_se = []
        real_defer_q = []
        real_defer_seed_means = []

        # Precompute item-level calibrated quantities for Q4 askable items
        item_ctx = []
        for rec in q4_askable:
            unanimous = (rec["maj_count"] == 3)
            p_a = calibrator.get_p_act(rec["obs_missing"], rec["obs_pot_ovr"], unanimous)
            u_a = p_a * r_ok + (1.0 - p_a) * r_err
            post_p = calibrator.p_post_ovr if rec["post_ovr_conf"] else calibrator.p_post_clear
            u_act_post = post_p * r_ok + (1.0 - post_p) * r_err
            u_greedy_real = r_ok if rec["greedy_ans"] == rec["item"]["gold"] else r_err
            u_post_real = r_ok if rec["post_ans"] == rec["item"]["gold"] else r_err
            item_ctx.append((u_a, u_act_post, u_greedy_real, u_post_real))

        for q in q_grid:
            u_def_pe = u_def_0 - gamma_queue * q
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

            # Genuine Monte Carlo realized-utility rollout over Q4 askable tickets
            seed_diffs = []
            for _ in range(n_mc_seeds):
                diff_sum = 0.0
                for u_a, u_act_post, u_greedy_real, u_post_real in item_ctx:
                    spec_ok = rng_mc.random() < pe_val
                    u_def_real = (r_ok if spec_ok else r_err) - c_def - gamma_queue * q
                    u_base_real = u_def_real if u_def_pe > u_a else u_greedy_real
                    u_stage2_real = u_def_real if u_def_pe > u_act_post else u_post_real
                    u_ask_real = u_stage2_real - c_ask
                    diff_sum += (u_ask_real - u_base_real)
                seed_diffs.append(diff_sum / max(1, len(item_ctx)))
            mc_arr = np.asarray(seed_diffs, dtype=float)
            realized_evoi_mean.append(round(float(np.mean(mc_arr)), 4))
            realized_evoi_se.append(round(float(np.std(mc_arr, ddof=1) / math.sqrt(n_mc_seeds)), 4))
            if q <= math.floor(q1_bound):
                real_defer_q.append(float(q))
                real_defer_seed_means.append(mc_arr)

        if len(real_defer_q) >= 2:
            seed_slopes = [
                float(np.polyfit(real_defer_q, [arr[s] for arr in real_defer_seed_means], 1)[0])
                for s in range(n_mc_seeds)
            ]
            real_slope = round(float(np.mean(seed_slopes)), 4)
            real_slope_se = round(float(np.std(seed_slopes, ddof=1) / math.sqrt(n_mc_seeds)), 4)
        else:
            real_slope = 0.0
            real_slope_se = 0.0

        phase_2d_by_pe[f"{pe_val:.2f}"] = {
            "p_e": pe_val,
            "q1_act_defer_boundary": round(float(q1_bound), 3),
            "q2_ovr_defer_boundary": round(float(q2_bound), 3),
            "theoretical_defer_slope": round(p_res_cal * gamma_queue, 4),
            "realized_defer_slope": real_slope,
            "realized_defer_slope_se": real_slope_se,
            "evoi_by_q": evoi_pe,
            "realized_evoi_mean": realized_evoi_mean,
            "realized_evoi_se": realized_evoi_se,
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
        "n_q4_askable": len(q4_askable),
        "analytical_curves": analytical_curves,
        "q4_ask_pct_by_p_resolve": q4_ask_curves,
        "q4_ask_pct_calibrated": cal_q4_ask,
        "q4_ask_pct_decoupled_nominal": nom_q4_ask,
        "phase_2d_by_pe": phase_2d_by_pe,
    }
