import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import build_cscw_benchmark, build_cscw_expansion_batch
from controller import HeldOutCalibrator, probe_dataset, simulate_stream, solve_3action_mdp_shadow_prices


BASE_DIR = Path(__file__).resolve().parent
THEORY_DIR = BASE_DIR / "results" / "theory"
THEORY_DIR.mkdir(parents=True, exist_ok=True)


def verify_theorem_1_and_corollary_1() -> Dict[str, Any]:
    """Verify 3-outcome slopes (+p_res*gamma, -p_ovr_after*gamma, 0), 200-seed MC rollout, and Corollary 1 rework shift."""
    cal_full, test_180 = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    test_ext = build_cscw_expansion_batch(n_per_quad=45, seed=2028, idx_offset=240)
    m_id = "Qwen/Qwen3.5-35B-A3B"
    cal_p = probe_dataset(cal_full, m_id, dry_run=False, workers=1)
    test_p = probe_dataset(test_180 + test_ext, m_id, dry_run=False, workers=1)
    calib = HeldOutCalibrator(cal_p)

    r_ok, r_err = 10.0, -10.0
    gamma = 0.85
    c_ask, c_def = 1.5, 2.0
    p_e = 0.75  # chosen so Q1* and Q2* fall cleanly inside Q in [0..10]

    p_act = calib.get_p_act(True, True, True)
    u_act = p_act * r_ok + (1.0 - p_act) * r_err
    u_clear = calib.p_post_clear * r_ok + (1.0 - calib.p_post_clear) * r_err
    u_ovr = calib.p_post_ovr * r_ok + (1.0 - calib.p_post_ovr) * r_err

    q_grid = np.linspace(0.0, 16.0, 321)
    configs = [
        ("Frictionless (p_no=0.00)", 0.55, 0.45, 0.00),
        ("With Friction (p_no=0.15)", 0.50, 0.35, 0.15),
    ]
    curves: Dict[str, Any] = {}
    u_def_0 = p_e * r_ok + (1.0 - p_e) * r_err - c_def
    q1_star = (u_def_0 - u_act) / gamma
    q2_star = (u_def_0 - u_ovr) / gamma

    for label, p_res, p_ovra, p_no in configs:
        evoi_vals = []
        for q in q_grid:
            u_def_q = u_def_0 - gamma * float(q)
            u_ask_q = (
                p_res * max(u_clear, u_def_q)
                + p_ovra * max(u_ovr, u_def_q)
                + p_no * max(u_act, u_def_q)
                - c_ask
            )
            evoi_vals.append(float(u_ask_q - max(u_act, u_def_q)))
        arr = np.asarray(evoi_vals)
        m1 = (q_grid >= 0.5) & (q_grid <= q1_star - 0.3)
        m2 = (q_grid >= q1_star + 0.3) & (q_grid <= q2_star - 0.3)
        m3 = q_grid >= q2_star + 0.3
        s1 = float(np.polyfit(q_grid[m1], arr[m1], 1)[0])
        s2 = float(np.polyfit(q_grid[m2], arr[m2], 1)[0])
        s3 = float(np.polyfit(q_grid[m3], arr[m3], 1)[0])
        curves[label] = {
            "p_res": p_res,
            "p_ovr_after": p_ovra,
            "p_no": p_no,
            "theo_slope_r1": round(p_res * gamma, 4),
            "emp_slope_r1": round(s1, 4),
            "theo_slope_r2": round(-p_ovra * gamma, 4),
            "emp_slope_r2": round(s2, 4),
            "theo_slope_r3": 0.0,
            "emp_slope_r3": round(s3, 4),
            "evoi_curve": [round(v, 4) for v in evoi_vals],
        }

    # Integer Q in 0..10 with 200-seed MC rollout + Corollary 1 rework shift on calibrated Q4
    p_res_c, p_ovra_c, p_no_c = calib.get_post_ask_probs(True, True)
    p_rw = p_res_c + p_ovra_c
    p_e_rw = min(p_e, 0.55)
    delta_rw = p_rw * ((p_e - p_e_rw) * (r_ok - r_err) + c_ask)
    q1_rw = q1_star - delta_rw / gamma

    q4_askable = [
        r for r in test_p
        if r["item"]["quadrant"] in ("Q4_resolvable", "Q4_escalate") and r["obs_missing"]
    ]
    rng_mc = random.Random(2026)
    q_int = list(range(15))
    evoi_std_int, evoi_rw_int, mc_mean_int, mc_se_int = [], [], [], []
    for q in q_int:
        u_def_q = u_def_0 - gamma * q
        u_def_rw_q = u_def_q - delta_rw
        u_ask_std = (
            p_res_c * max(u_clear, u_def_q)
            + p_ovra_c * max(u_ovr, u_def_q)
            + p_no_c * max(u_act, u_def_q)
            - c_ask
        )
        u_ask_rw = (
            p_res_c * max(u_clear, u_def_q)
            + p_ovra_c * max(u_ovr, u_def_q)
            + p_no_c * max(u_act, u_def_rw_q)
            - c_ask
        )
        evoi_std_int.append(round(float(u_ask_std - max(u_act, u_def_q)), 4))
        evoi_rw_int.append(round(float(u_ask_rw - max(u_act, u_def_rw_q)), 4))

        seed_diffs = []
        for _ in range(200):
            s_diff = 0.0
            for rec in q4_askable:
                ok_spec = rng_mc.random() < p_e
                u_def_real = (r_ok if ok_spec else r_err) - c_def - gamma * q
                u_base = u_def_real if u_def_q > u_act else (r_ok if rec["greedy_ans"] == rec["item"]["gold"] else r_err)
                post_p = calib.p_post_ovr if rec["post_ovr_conf"] else calib.p_post_clear
                u_act_post = post_p * r_ok + (1.0 - post_p) * r_err
                u_s2 = u_def_real if u_def_q > u_act_post else (r_ok if rec["post_ans"] == rec["item"]["gold"] else r_err)
                s_diff += (u_s2 - c_ask) - u_base
            seed_diffs.append(s_diff / max(1, len(q4_askable)))
        arr_mc = np.asarray(seed_diffs)
        mc_mean_int.append(round(float(np.mean(arr_mc)), 4))
        mc_se_int.append(round(float(np.std(arr_mc, ddof=1) / math.sqrt(200)), 4))

    return {
        "q1_star": round(q1_star, 3),
        "q2_star": round(q2_star, 3),
        "q1_rework_star": round(q1_rw, 3),
        "delta_rework": round(delta_rw, 3),
        "q_grid": [round(float(x), 3) for x in q_grid],
        "curves_3outcome": curves,
        "q_int": q_int,
        "evoi_cal_std": evoi_std_int,
        "evoi_cal_rework": evoi_rw_int,
        "mc_mean_int": mc_mean_int,
        "mc_se_int": mc_se_int,
    }


def verify_theorem_2() -> Dict[str, Any]:
    """Verify Theorem 2 miscalibration bound & Finite-Sample EB scaling on cached GLM-4.5-Air and Qwen3-8B."""
    cal_full, test_180 = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    test_ext = build_cscw_expansion_batch(n_per_quad=45, seed=2028, idx_offset=240)
    all_items = cal_full + test_180 + test_ext

    glm_all = probe_dataset(all_items, "zai-org/GLM-4.5-Air", dry_run=False, workers=1)
    qwen8b_all = probe_dataset(all_items, "Qwen/Qwen3-8B", dry_run=False, workers=1)

    # Exact Oracle parameter estimates on full N=420 for GLM to measure empirical eps_p and eps_pi
    by_q = defaultdict(list)
    for r in glm_all:
        by_q[r["item"]["quadrant"]].append(r)

    n_cal_grid = [32, 60, 100, 140, 180, 240]
    k_grid = [2.0, 5.0, 10.0]

    # Constant-size N_test=180 per repetition; truth_calib fitted on disjoint test_s ONLY (zero sample overlap)
    glm_curves: Dict[str, List[Dict[str, Any]]] = {}
    for k_val in k_grid:
        rows = []
        for n_cal in n_cal_grid:
            u_cap2, u_sev, eps_p_list, eps_pi_list, gap_list = [], [], [], [], []
            for rep in range(15):
                rr_split = random.Random(3000 + rep * 17)
                test_s, cal_pool = [], []
                for qk in sorted(by_q.keys()):
                    bkt = list(by_q[qk])
                    rr_split.shuffle(bkt)
                    n_t = 45 if qk in ("Q1_clear_easy", "Q2_underspec_easy", "Q3_clear_hard") else (22 if qk == "Q4_resolvable" else 23)
                    test_s.extend(bkt[:n_t])
                    cal_pool.extend(bkt[n_t:])
                rr_split.shuffle(test_s)

                rr_cal = random.Random(3000 + rep * 17 + n_cal)
                pool_by_q = defaultdict(list)
                for r in cal_pool:
                    pool_by_q[r["item"]["quadrant"]].append(r)
                cal_s = []
                per_q = max(4, n_cal // 4)
                for qk in sorted(pool_by_q.keys()):
                    bkt = list(pool_by_q[qk])
                    rr_cal.shuffle(bkt)
                    take = per_q if qk in ("Q1_clear_easy", "Q2_underspec_easy", "Q3_clear_hard") else (per_q // 2 if qk == "Q4_resolvable" else per_q - per_q // 2)
                    cal_s.extend(bkt[:take])

                eb = HeldOutCalibrator(cal_s, k=k_val, k_fine=k_val)
                truth_calib = HeldOutCalibrator(test_s, k=0.0, k_fine=0.0)

                # Measure calibration error across ALL populated fine buckets + post-ASK outcomes on disjoint test_s
                ep_cands = [
                    abs(eb.p_post_clear - truth_calib.p_post_clear),
                    abs(eb.p_post_ovr - truth_calib.p_post_ovr),
                ]
                for fk in truth_calib.p_act:
                    ep_cands.append(abs(eb.get_p_act(fk[0], fk[1], fk[2]) - truth_calib.p_act[fk]))
                ep_max = max(ep_cands)
                epi = max(
                    sum(abs(a - b) for a, b in zip(eb.get_post_ask_probs(True, pf), truth_calib.get_post_ask_probs(True, pf)))
                    for pf in (False, True)
                )
                eps_p_list.append(ep_max)
                eps_pi_list.append(epi)
                gap_list.append(20.0 * (2.0 * ep_max + epi))

                r_c2 = simulate_stream("Tri-VoI-L2D", test_s, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=rep)
                r_sv = simulate_stream("Tri-VoI-L2D", test_s, eb, capacity=1, drain_interval=3, c_ask=1.5, seed=rep)
                u_cap2.append(r_c2["mean_utility"])
                u_sev.append(r_sv["mean_utility"])

            rows.append({
                "n_cal": len(cal_s),
                "u_cap2_mean": round(float(np.mean(u_cap2)), 3),
                "u_cap2_std": round(float(np.std(u_cap2, ddof=1)), 3),
                "u_sev_mean": round(float(np.mean(u_sev)), 3),
                "u_sev_std": round(float(np.std(u_sev, ddof=1)), 3),
                "mean_eps_p": round(float(np.mean(eps_p_list)), 4),
                "mean_eps_pi": round(float(np.mean(eps_pi_list)), 4),
                "theo_bound_util": round(float(np.mean(gap_list)), 3),
            })
        glm_curves[f"k={int(k_val)}"] = rows

    # Qwen3-8B F1 floor summary
    q8_cal = qwen8b_all[:60]
    q8_test = qwen8b_all[60:]
    q8_eb = HeldOutCalibrator(q8_cal)
    q8_u_cap2 = float(np.mean([
        simulate_stream("Tri-VoI-L2D", q8_test, q8_eb, capacity=2, drain_interval=2, c_ask=1.5, seed=s)["mean_utility"]
        for s in range(5)
    ]))
    q8_u_sev = float(np.mean([
        simulate_stream("Tri-VoI-L2D", q8_test, q8_eb, capacity=1, drain_interval=3, c_ask=1.5, seed=s)["mean_utility"]
        for s in range(5)
    ]))
    return {
        "glm_curves": glm_curves,
        "qwen3_8b_boundary": {
            "u_cap2": round(q8_u_cap2, 3),
            "u_sev": round(q8_u_sev, 3),
        },
    }


def verify_theorem_3() -> Dict[str, Any]:
    """Verify Foster-Lyapunov stability-region expansion and non-zero ASK latency tau_ask."""
    rng = random.Random(2027)
    # Discrete-event queue stability sweep
    # Shares: alpha_spec=0.50 (Q2=0.25, Q4=0.25), alpha_epist=0.25 (Q3), Q1=0.25; p_res_bar = 0.75
    alpha_spec = 0.50
    alpha_epist = 0.25
    p_res_bar = 0.75
    mu = 0.50  # service rate: 1 completion every 2 slots (Bernoulli(0.5) or periodic)

    lam_base_crit = mu / (alpha_spec + alpha_epist)           # 0.667
    lam_tri_crit = mu / (alpha_epist + (1.0 - p_res_bar) * alpha_spec)  # 1.333

    lam_grid = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.45]
    t_steps = 4000
    stability_rows = []
    for lam_tot in lam_grid:
        q_base_sum = 0.0
        q_tri_sum = 0.0
        n_seeds = 10
        for s in range(n_seeds):
            rr = random.Random(700 + s)
            qb, qt = 0, 0
            sum_b, sum_t = 0.0, 0.0
            for _ in range(t_steps):
                # Poisson arrivals per slot (scaled by lam_tot)
                # Knuth exact Poisson draw
                L = math.exp(-lam_tot)
                k_arr, p_acc = 0, 1.0
                while p_acc > L:
                    k_arr += 1
                    p_acc *= rr.random()
                arr_tot = k_arr - 1

                a_base_def = 0
                a_tri_def = 0
                for _i in range(arr_tot):
                    u = rr.random()
                    if u < alpha_epist:
                        a_base_def += 1
                        a_tri_def += 1
                    elif u < alpha_epist + alpha_spec:
                        a_base_def += 1
                        if rr.random() >= p_res_bar:
                            a_tri_def += 1
                srv_b = 1 if rr.random() < mu else 0
                srv_t = 1 if rr.random() < mu else 0
                qb = max(0, qb + a_base_def - srv_b)
                qt = max(0, qt + a_tri_def - srv_t)
                sum_b += qb
                sum_t += qt
            q_base_sum += sum_b / t_steps
            q_tri_sum += sum_t / t_steps

        # Theoretical Foster-Lyapunov bound B / eps when lam_tri_def < mu
        lam_tri_def = lam_tot * (alpha_epist + (1.0 - p_res_bar) * alpha_spec)
        if lam_tri_def < mu:
            eps = mu - lam_tri_def
            # E[(A-S)^2] = Var(A) + Var(S) + (lam_tri_def - mu)^2 = lam_tri_def + mu*(1-mu) + eps^2
            B_val = 0.5 * (lam_tri_def + mu * (1.0 - mu) + eps * eps)
            lyap_bound = round(B_val / eps, 2)
        else:
            lyap_bound = None
        stability_rows.append({
            "lambda_tot": lam_tot,
            "mean_q_base_l2d": round(q_base_sum / n_seeds, 2),
            "mean_q_tri_voi": round(q_tri_sum / n_seeds, 2),
            "lyapunov_bound": lyap_bound,
        })

    # Part B: Non-zero ASK response latency tau_ask in {0, 1, 2, 4, 8}
    r_ok, r_err = 10.0, -10.0
    gamma = 0.85
    c_ask, c_def = 1.5, 2.0
    p_e = 0.88
    p_res, p_ovra, p_no = 0.50, 0.40, 0.10
    p_act, p_clear, p_ovr = 0.45, 0.90, 0.35
    u_act = p_act * r_ok + (1 - p_act) * r_err
    u_clear = p_clear * r_ok + (1 - p_clear) * r_err
    u_ovr = p_ovr * r_ok + (1 - p_ovr) * r_err
    q0 = 2.0
    u_def_q0 = p_e * r_ok + (1 - p_e) * r_err - c_def - gamma * q0
    evoi0 = (
        p_res * max(u_clear, u_def_q0)
        + p_ovra * max(u_ovr, u_def_q0)
        + p_no * max(u_act, u_def_q0)
        - c_ask
        - max(u_act, u_def_q0)
    )
    delta_bar = 0.35
    beta_wait = 0.25
    tau_max_theo = evoi0 / ((1.0 - p_res) * gamma * delta_bar + beta_wait)

    tau_grid = [0, 1, 2, 4, 6, 8]
    latency_rows = []
    for tau in tau_grid:
        # Simulated random queue drift during tau slots: Q_{t+tau} = max(0, q0 + sum_{i=1}^tau (A_i - S_i)) with mean +delta_bar*tau
        evoi_mc_vals = []
        for s in range(2000):
            rr = random.Random(9000 + tau * 100 + s)
            q_fut = q0
            for _ in range(tau):
                arr = 1 if rr.random() < 0.70 else 0
                srv = 1 if rr.random() < 0.35 else 0
                q_fut = max(0.0, q_fut + arr - srv)
            u_def_f = p_e * r_ok + (1 - p_e) * r_err - c_def - gamma * q_fut
            u_ask_f = (
                p_res * max(u_clear, u_def_f)
                + p_ovra * max(u_ovr, u_def_f)
                + p_no * max(u_act, u_def_f)
                - c_ask
                - beta_wait * tau
            )
            evoi_mc_vals.append(u_ask_f - max(u_act, u_def_q0))
        evoi_theo_r1 = evoi0 - ((1.0 - p_res) * gamma * delta_bar + beta_wait) * tau
        latency_rows.append({
            "tau_ask": tau,
            "evoi_mc": round(float(np.mean(evoi_mc_vals)), 3),
            "evoi_theo_r1": round(evoi_theo_r1, 3),
        })

    # Empirical simulate_stream queue-backlog & accuracy sweep across mu in {2.0, 1.0, 0.5, 0.33}
    cal_v1, test_v1a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = test_v1a + build_cscw_expansion_batch(45, 2028, 240)
    q35_cal = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1)
    q35_test = probe_dataset(test_v1, "Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1)
    q35_eb = HeldOutCalibrator(q35_cal)
    stream_mu_rows = []
    for mu_lbl, cap, di in [("2.00", 2, 1), ("1.00", 2, 2), ("0.50", 1, 2), ("0.33", 1, 3)]:
        r_l2d = [simulate_stream("Congested-L2D-Only", q35_test, q35_eb, capacity=cap, drain_interval=di, c_ask=1.5, seed=s) for s in range(5)]
        r_tri = [simulate_stream("Tri-VoI-L2D", q35_test, q35_eb, capacity=cap, drain_interval=di, c_ask=1.5, seed=s) for s in range(5)]
        r_mdp = [simulate_stream("Tri-VoI-MDP", q35_test, q35_eb, capacity=cap, drain_interval=di, c_ask=1.5, seed=s) for s in range(5)]
        stream_mu_rows.append({
            "mu": mu_lbl,
            "l2d_mean_q": round(float(np.mean([r["mean_queue_len"] for r in r_l2d])), 2),
            "l2d_acc_pct": round(float(np.mean([r["acc_pct"] for r in r_l2d])), 2),
            "l2d_util": round(float(np.mean([r["mean_utility"] for r in r_l2d])), 3),
            "tri_mean_q": round(float(np.mean([r["mean_queue_len"] for r in r_tri])), 2),
            "tri_acc_pct": round(float(np.mean([r["acc_pct"] for r in r_tri])), 2),
            "tri_util": round(float(np.mean([r["mean_utility"] for r in r_tri])), 3),
            "mdp_mean_q": round(float(np.mean([r["mean_queue_len"] for r in r_mdp])), 2),
            "mdp_acc_pct": round(float(np.mean([r["acc_pct"] for r in r_mdp])), 2),
            "mdp_util": round(float(np.mean([r["mean_utility"] for r in r_mdp])), 3),
        })

    return {
        "lam_base_crit": round(lam_base_crit, 3),
        "lam_tri_crit": round(lam_tri_crit, 3),
        "stability_rows": stability_rows,
        "stream_mu_rows": stream_mu_rows,
        "tau_max_theo": round(tau_max_theo, 2),
        "latency_rows": latency_rows,
    }


def verify_theorem_4() -> Dict[str, Any]:
    """Verify Theorem 4: shadow-price non-negativity, coupling bound, gamma=0, 4-branch abandonment, and horizon truncation."""
    cal_v1, _ = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    q35_cal = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1)
    calib = HeldOutCalibrator(q35_cal)
    omega = (0.25, 0.25, 0.25, 0.25)
    gamma = 0.85
    p_e = 0.90
    cap, di = 2, 2
    r_ok, r_err, c_ask, c_def = 10.0, -10.0, 1.5, 2.0

    V25 = solve_3action_mdp_shadow_prices(omega, calib, p_e, cap, di, gamma, c_ask, c_def, r_ok, r_err, 25, q_max=70, horizon_cap=25)
    V100 = solve_3action_mdp_shadow_prices(omega, calib, p_e, cap, di, gamma, c_ask, c_def, r_ok, r_err, 100, q_max=70, horizon_cap=100)
    sh_ph1 = [round(V25[1][q] - V25[1][q + 1], 5) for q in range(69)]
    min_sh = float(min(sh_ph1))
    max_sh = float(max(sh_ph1))
    trunc_diff = float(max(abs((V25[1][q] - V25[1][q + 1]) - (V100[1][q] - V100[1][q + 1])) for q in range(8)))

    # Compute exact 4-branch abandonment threshold Q_max^defer + (h-1)*C
    u_def_post0 = p_e * r_ok + (1.0 - p_e) * r_err - c_def
    u_act_vals = [
        (calib.coarse_p_act.get((om, op), 0.85 if (not om and not op) else 0.50) * r_ok
         + (1.0 - calib.coarse_p_act.get((om, op), 0.85 if (not om and not op) else 0.50)) * r_err)
        for om in (False, True) for op in (False, True)
    ]
    u_clear_vals = [calib.get_p_post_clear(op) * r_ok + (1.0 - calib.get_p_post_clear(op)) * r_err for op in (False, True)]
    u_ovr_val = calib.p_post_ovr * r_ok + (1.0 - calib.p_post_ovr) * r_err
    u_ni_val = calib.p_post_no_info * r_ok + (1.0 - calib.p_post_no_info) * r_err
    u_min_self = min(u_act_vals + u_clear_vals + [u_ovr_val, u_ni_val])
    q_act_only = max(0.0, (u_def_post0 - min(u_act_vals)) / gamma)
    q_max_4branch = max(0.0, (u_def_post0 - u_min_self) / gamma)
    q_aband_test = int(math.ceil(q_max_4branch)) + 25 * cap + 2
    sh_high = float(V25[1][q_aband_test] - V25[1][q_aband_test + 1])
    assert abs(sh_high) < 1e-9, f"Expected exact zero shadow price above 4-branch threshold q={q_aband_test}, got {sh_high}"

    # Witness why 4-branch u_min^self is necessary (with cap=0, di=1, q strictly between q_act and q_ovr):
    # Stage-1 DEFER is dominated by ACT, yet Stage-2 ovr_after->DEFER inside ASK still yields Delta V > 0.
    q_witness = int(math.floor(q_act_only)) + 1
    assert q_witness < q_max_4branch, f"Expected gap between u_act ({min(u_act_vals):.2f}) and u_min_self ({u_min_self:.2f})"
    V_wit = solve_3action_mdp_shadow_prices((0.0, 0.0, 0.5, 0.5), calib, p_e, 0, 1, gamma, 0.1, c_def, r_ok, r_err, 3, q_max=40, horizon_cap=10)
    sh_witness_gap = float(V_wit[0][q_witness] - V_wit[0][q_witness + 1])
    assert sh_witness_gap > 1e-4, f"Expected Stage-2 ovr_after shadow price > 0 in gap q={q_witness}, got {sh_witness_gap}"

    # gamma = 0 -> shadow price must vanish identically
    V_g0 = solve_3action_mdp_shadow_prices(omega, calib, p_e, cap, di, 0.0, c_ask, c_def, r_ok, r_err, 10, q_max=10, horizon_cap=10)
    gamma0_max_sh = float(max(abs(V_g0[1][q] - V_g0[1][q + 1]) for q in range(9)))
    assert gamma0_max_sh < 1e-9, f"gamma=0 shadow price non-zero: {gamma0_max_sh}"

    # Check finite horizon bound 0 <= Delta V_h <= gamma * h at h=4
    V4 = solve_3action_mdp_shadow_prices(omega, calib, p_e, cap, di, gamma, c_ask, c_def, r_ok, r_err, 4, q_max=20, horizon_cap=10)
    max_sh_4 = float(max(V4[1][q] - V4[1][q + 1] for q in range(19)))
    assert min_sh >= -1e-9, f"Negative shadow price: {min_sh}"
    assert max_sh_4 <= gamma * 4.0 + 1e-6, f"Coupling upper bound violated: {max_sh_4}"

    # Check degenerate no-defer case even when I = 2 > 1 (c_defer=100 -> DEFER never optimal -> Delta V == 0)
    V_deg = solve_3action_mdp_shadow_prices(omega, calib, p_e, cap, di, gamma, c_ask, 100.0, r_ok, r_err, 10, q_max=10, horizon_cap=10)
    deg_max_sh = float(max(abs(V_deg[1][q] - V_deg[1][q + 1]) for q in range(9)))
    assert deg_max_sh < 1e-9, f"Degenerate zero-defer shadow price non-zero: {deg_max_sh}"

    return {
        "shadow_phase1_q0_to_q9": [round(x, 4) for x in sh_ph1[:10]],
        "subcap_phase1_q1": round(sh_ph1[1], 4),
        "degenerate_cdef100_max_shadow": round(deg_max_sh, 6),
        "gamma0_max_shadow": round(gamma0_max_sh, 6),
        "min_shadow": round(min_sh, 6),
        "max_shadow_h25": round(max_sh, 4),
        "max_shadow_h4": round(max_sh_4, 4),
        "coupling_bound_h4": round(gamma * 4.0, 4),
        "q_act_only_thresh": round(q_act_only, 3),
        "q_max_4branch_thresh": round(q_max_4branch, 3),
        "q_witness_between": q_witness,
        "witness_gap_shadow": round(sh_witness_gap, 4),
        "q_aband_4branch": q_aband_test,
        "high_q_aband_shadow": round(sh_high, 6),
        "horizon_25_vs_100_max_diff_q0_7": round(trunc_diff, 5),
    }


def verify_proposition_5() -> Dict[str, Any]:
    """Verify Proposition 5: exact one-step regret decomposition Regret_CC and Delta_opt across (c_ask, Q)."""
    cal_v1, _ = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    q35_cal = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1)
    calib = HeldOutCalibrator(q35_cal)
    r_ok, r_err, gamma, p_e, c_def = 10.0, -10.0, 0.85, 0.90, 2.0
    u_def0 = p_e * r_ok + (1.0 - p_e) * r_err - c_def
    omega = (0.25, 0.25, 0.25, 0.25)

    p_act = calib.coarse_p_act.get((True, True), 0.35)
    p_res, p_ovra, p_no = calib.get_post_ask_probs(True, True)
    p_clear = calib.get_p_post_clear(True)
    p_ovr = calib.p_post_ovr
    p_ni = calib.p_post_no_info
    u_act = p_act * r_ok + (1.0 - p_act) * r_err
    u_clear = p_clear * r_ok + (1.0 - p_clear) * r_err
    u_ovr = p_ovr * r_ok + (1.0 - p_ovr) * r_err
    u_no = p_ni * r_ok + (1.0 - p_ni) * r_err

    rows = []
    for c_ask in (0.5, 1.5, 3.0, 4.5):
        V25 = solve_3action_mdp_shadow_prices(omega, calib, p_e, 1, 3, gamma, c_ask, c_def, r_ok, r_err, 25, q_max=30, horizon_cap=25)
        for q in (0, 3, 7, 12):
            dv = max(0.0, V25[1][q] - V25[1][q + 1])
            u_def_eff = u_def0 - gamma * q - dv
            u_ask_dp = (
                p_res * max(u_clear, u_def_eff)
                + p_ovra * max(u_ovr, u_def_eff)
                + p_no * max(u_no, u_def_eff)
                - c_ask
            )
            u_ask_jg = p_res * u_clear + p_ovra * u_ovr + p_no * u_no - c_ask
            u_opt = max(u_act, u_def_eff, u_ask_dp)
            evoi_dp = u_ask_dp - max(u_act, u_def_eff)
            regret_cc = u_opt - u_ask_dp
            assert abs(regret_cc - max(0.0, -evoi_dp)) < 1e-9

            delta_opt = (
                p_res * max(0.0, u_def_eff - u_clear)
                + p_ovra * max(0.0, u_def_eff - u_ovr)
                + p_no * max(0.0, u_def_eff - u_no)
            )
            assert abs((u_ask_dp - u_ask_jg) - delta_opt) < 1e-9

            # JG+MDP choice: ASK iff obs_m and u_ask_jg > max(u_act, u_def_eff)
            jg_val = u_ask_jg if u_ask_jg > max(u_act, u_def_eff) else max(u_act, u_def_eff)
            regret_jg = u_opt - jg_val
            assert -1e-9 <= regret_jg <= delta_opt + 1e-9
            rows.append({
                "c_ask": c_ask,
                "q": q,
                "u_def_eff": round(u_def_eff, 3),
                "evoi_dp": round(evoi_dp, 3),
                "regret_cc": round(regret_cc, 3),
                "delta_opt": round(delta_opt, 3),
                "regret_jg": round(regret_jg, 3),
            })
    return {"bucket_true_true_rows": rows}


def plot_fig2(t1: Dict[str, Any], t2: Dict[str, Any], t3: Dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)

    # Panel A: Theorem 1 + Corollary 1
    ax = axes[0]
    q_int = t1["q_int"]
    ax.plot(q_int, t1["evoi_cal_std"], "o-", color="#1f77b4", lw=1.8, ms=4, label="Standard Analytical EVOI")
    ax.errorbar(q_int, t1["mc_mean_int"], yerr=t1["mc_se_int"], fmt="s--", color="#2ca02c", ms=3.5, lw=1.3, capsize=2, label="200-Seed MC Rollout")
    ax.plot(q_int, t1["evoi_cal_rework"], "^-.", color="#d62728", lw=1.6, ms=4, label="Rework-Aware $U_{defer}^{rework}$ (Cor. 1)")
    ax.axvline(t1["q1_star"], color="#1f77b4", ls=":", lw=1.1, label=f"$Q_1^*={t1['q1_star']:.2f}$")
    ax.axvline(t1["q1_rework_star"], color="#d62728", ls=":", lw=1.1, label=f"$Q_1^{{*,rw}}={t1['q1_rework_star']:.2f}$")
    ax.axvline(t1["q2_star"], color="#555555", ls="--", lw=1.0, label=f"$Q_2^*={t1['q2_star']:.2f}$")
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_title("Thm 1 & Cor 1: 3-Regime EVOI & Rework Shift")
    ax.set_xlabel("Specialist Queue Backlog $Q$")
    ax.set_ylabel("EVOI(ASK; $Q$)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)

    # Panel B: Theorem 2 GLM-4.5-Air n_cal & k sweep
    ax = axes[1]
    colors = {"k=2": "#d62728", "k=5": "#ff7f0e", "k=10": "#1f77b4"}
    for klbl, rows in t2["glm_curves"].items():
        xs = [r["n_cal"] for r in rows]
        ys = [r["u_sev_mean"] for r in rows]
        es = [r["u_sev_std"] for r in rows]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=colors[klbl], lw=1.6, capsize=2.5, label=f"GLM-4.5-Air ({klbl}) Cap=1/3")
    ax.axhline(t2["qwen3_8b_boundary"]["u_sev"], color="#7f7f7f", ls="--", lw=1.1, label=f"Qwen3-8B Floor ({t2['qwen3_8b_boundary']['u_sev']:.2f})")
    ax.set_title("Thm 2 & Cor 2: GLM Shrinkage ($k, n_{cal}$)")
    ax.set_xlabel("Calibration Set Size $n_{cal}$")
    ax.set_ylabel("Mean Utility (Severe Cap=1/3)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    # Panel C: Theorem 3 Capacity Expansion + Inset/Twin Latency
    ax = axes[2]
    lams = [r["lambda_tot"] for r in t3["stability_rows"]]
    qb = [min(r["mean_q_base_l2d"], 400.0) for r in t3["stability_rows"]]
    qt = [r["mean_q_tri_voi"] for r in t3["stability_rows"]]
    ax.plot(lams, qb, "s--", color="#d62728", lw=1.6, ms=4, label="{ACT, DEFER} Backlog (capped 400)")
    ax.plot(lams, qt, "o-", color="#1f77b4", lw=1.8, ms=4.5, label="Tri-VoI-L2D Backlog")
    lyap_x = [r["lambda_tot"] for r in t3["stability_rows"] if r["lyapunov_bound"] is not None]
    lyap_y = [r["lyapunov_bound"] for r in t3["stability_rows"] if r["lyapunov_bound"] is not None]
    ax.plot(lyap_x, lyap_y, "^:", color="#2ca02c", lw=1.4, label="Thm 3 Lyapunov Bound $B/\\varepsilon$")
    ax.axvline(t3["lam_base_crit"], color="#d62728", ls=":", lw=1.0, label=f"$\\lambda_{{base}}^*={t3['lam_base_crit']:.2f}$")
    ax.axvline(t3["lam_tri_crit"], color="#1f77b4", ls=":", lw=1.0, label=f"$\\lambda_{{Tri}}^*={t3['lam_tri_crit']:.2f}$")
    ax.set_yscale("log")
    ax.set_title("Thm 3: Capacity Expansion & Latency $\\tau_{ask}$")
    ax.set_xlabel("Total Arrival Rate $\\lambda_{tot}$")
    ax.set_ylabel("Steady-State Mean Queue $\\mathbb{E}[Q]$ (log)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True)

    out_pdf = THEORY_DIR / "fig2_theory_verification.pdf"
    out_png = THEORY_DIR / "fig2_theory_verification.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t1 = verify_theorem_1_and_corollary_1()
    t2 = verify_theorem_2()
    t3 = verify_theorem_3()
    t4 = verify_theorem_4()
    p5 = verify_proposition_5()
    plot_fig2(t1, t2, t3)

    payload = {"theorem_1": t1, "theorem_2": t2, "theorem_3": t3, "theorem_4": t4, "proposition_5": p5}
    (THEORY_DIR / "theory_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Offline Verification Report: IJCAI 2027 Three-Theorem Package",
        "",
        "## Theorem 1 & Corollary 1 (3-Outcome Three-Regime EVOI + Rework Shift)",
        f"- Boundaries ($p_e=0.75$): $Q_1^* = {t1['q1_star']}$, $Q_2^* = {t1['q2_star']}$, Rework $Q_1^{{*,rw}} = {t1['q1_rework_star']}$ ($\\Delta_{{rw}} = {t1['delta_rework']}$).",
    ]
    for lbl, c in t1["curves_3outcome"].items():
        lines.append(
            f"- **{lbl}**: R1 slope `{c['emp_slope_r1']}` (theo `{c['theo_slope_r1']}`), "
            f"R2 slope `{c['emp_slope_r2']}` (theo `{c['theo_slope_r2']}`), "
            f"R3 slope `{c['emp_slope_r3']}` (theo `{c['theo_slope_r3']}`)"
        )
    lines.extend([
        "",
        "## Theorem 2 & Corollary 2 (GLM-4.5-Air Shrinkage & Miscalibration Bound)",
    ])
    for klbl, rows in t2["glm_curves"].items():
        r60 = rows[1]
        r240 = rows[-1]
        lines.append(
            f"- **{klbl}** (fixed $N_{{test}}=180$): `n_cal={r60['n_cal']}` Cap=1/3 `{r60['u_sev_mean']}±{r60['u_sev_std']}` ($\\epsilon_p={r60['mean_eps_p']}, \\epsilon_\\pi={r60['mean_eps_pi']}$) "
            f"-> `n_cal={r240['n_cal']}` Cap=1/3 `{r240['u_sev_mean']}±{r240['u_sev_std']}` ($\\epsilon_p={r240['mean_eps_p']}, \\epsilon_\\pi={r240['mean_eps_pi']}$)"
        )
    lines.append(f"- **Qwen3-8B F1 Floor Boundary**: Cap=2 `{t2['qwen3_8b_boundary']['u_cap2']}`, Cap=1/3 `{t2['qwen3_8b_boundary']['u_sev']}`.")
    lines.extend([
        "",
        "## Theorem 3 (Foster-Lyapunov Stability Expansion & Non-Zero ASK Latency)",
        f"- Stability Thresholds ($\\mu=0.50$): `{{ACT, DEFER}}` $\\lambda_{{base}}^* = {t3['lam_base_crit']}$ vs `Tri-VoI-L2D` $\\lambda_{{Tri}}^* = {t3['lam_tri_crit']}$ (2.0x capacity expansion).",
        f"- Latency Tolerance ($q_0=2$, $\\bar{{\\delta}}=0.35$, $\\beta_{{wait}}=0.25$): Closed-form $\\tau_{{ask}}^{{\\max}} = {t3['tau_max_theo']}$ slots.",
        "| $\\lambda_{tot}$ | `{ACT, DEFER}` Mean Q | `Tri-VoI-L2D` Mean Q | Lyapunov Bound $B/\\varepsilon$ |",
        "|---:|---:|---:|---:|",
    ])
    for r in t3["stability_rows"]:
        lb = str(r["lyapunov_bound"]) if r["lyapunov_bound"] is not None else "Unstable"
        lines.append(f"| {r['lambda_tot']:.2f} | {r['mean_q_base_l2d']:.2f} | {r['mean_q_tri_voi']:.2f} | {lb} |")
    lines.extend([
        "",
        "### Latency Sweep ($\\tau_{ask}$)",
        "| $\\tau_{ask}$ | MC EVOI | Regime-I Linear Prediction |",
        "|---:|---:|---:|",
    ])
    for r in t3["latency_rows"]:
        lines.append(f"| {r['tau_ask']} | {r['evoi_mc']:.3f} | {r['evoi_theo_r1']:.3f} |")

    lines.extend([
        "",
        "### Real `simulate_stream` Empirical Queue vs. Accuracy (`v1-Clean`, `Qwen3.5-35B-A3B`)",
        "| Service Rate $\\mu$ | `Congested-L2D-Only` $\\bar{Q}$ (Acc%, Util) | `Tri-VoI-L2D` $\\bar{Q}$ (Acc%, Util) | `Tri-VoI-MDP` $\\bar{Q}$ (Acc%, Util) |",
        "|---:|---:|---:|---:|",
    ])
    for r in t3["stream_mu_rows"]:
        lines.append(
            f"| {r['mu']} | {r['l2d_mean_q']:.2f} ({r['l2d_acc_pct']:.2f}%, {r['l2d_util']:.3f}) | "
            f"{r['tri_mean_q']:.2f} ({r['tri_acc_pct']:.2f}%, {r['tri_util']:.3f}) | "
            f"{r['mdp_mean_q']:.2f} ({r['mdp_acc_pct']:.2f}%, {r['mdp_util']:.3f}) |"
        )

    lines.extend([
        "",
        "## Theorem 4 (Exact 3-Action Periodic-Drain Bellman Optimality & Shadow-Price Monotonicity)",
        f"- Phase-1 shadow prices $\\Delta V_{{25}}(1, q)$ for $q=0..9$: `{t4['shadow_phase1_q0_to_q9']}` (sub-capacity $q=1 < C$: `{t4['subcap_phase1_q1']}` > 0).",
        f"- Non-negativity min $\\Delta V$: `{t4['min_shadow']}`, $\\gamma=0$ max $|\\Delta V|$: `{t4['gamma0_max_shadow']}`, Coupling bound ($h=4$): `{t4['max_shadow_h4']} <= {t4['coupling_bound_h4']}`.",
        f"- 4-branch abandonment threshold: $Q_{{act}}={t4['q_act_only_thresh']}$, $Q_{{\\max}}^{{\\text{{defer}}}}={t4['q_max_4branch_thresh']}$ (witness $q={t4['q_witness_between']}$ gap $\\Delta V={t4['witness_gap_shadow']}>0$; above 4-branch bound $q={t4['q_aband_4branch']}$: `{t4['high_q_aband_shadow']}`), Operating-queue diff ($q\\le 7$): `{t4['horizon_25_vs_100_max_diff_q0_7']}`.",
        "",
        "## Proposition 5 (Tri-Party EVOI Regret Decomposition under Matched Dynamic Shadow Pricing)",
        "| $c_{\\text{ask}}$ | $Q$ | $\\widetilde{U}_{\\text{def,eff}}$ | $\\text{EVOI}_{\\text{DP}}$ | $\\text{Regret}_{\\text{CC}}$ | $\\Delta_{\\text{opt}}$ | $\\text{Regret}_{\\text{JG}}$ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in p5["bucket_true_true_rows"]:
        lines.append(
            f"| {row['c_ask']} | {row['q']} | {row['u_def_eff']:.3f} | {row['evoi_dp']:+.3f} | "
            f"{row['regret_cc']:.3f} | {row['delta_opt']:.3f} | {row['regret_jg']:.3f} |"
        )

    (THEORY_DIR / "theory_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved theory verification JSON, MD, and fig2_theory_verification.{pdf,png}")


if __name__ == "__main__":
    main()
