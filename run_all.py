import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, ttest_rel

from benchmark import (
    build_cscw_benchmark,
    build_cscw_expansion_batch,
    split_stratified_episodes,
)
from controller import (
    HeldOutCalibrator,
    evaluate_diagnosis_metrics,
    probe_dataset,
    run_fixed_state_evoi_sweep,
    simulate_stream,
)
from llm_client import RESULTS_DIR

FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.5-Air",
    "Qwen/Qwen3-8B",
]

MAIN_METHODS = [
    "Direct-ACT",
    "Entropy-ASK (Proxy)",
    "Consensus-DEFER (Proxy)",
    "Congested-L2D-Only",
    "Naive-Cascade",
    "Joint-Greedy",
    "Joint-Greedy+Naor",
    "Calibrated-Cascade",
    "Calibrated-Cascade+Naor",
    "Tri-VoI-L2D",
    "Oracle",
]

ABLATIONS = [
    "Tri-VoI-L2D",
    "w/o Disentangled-Diag",
    "w/o Held-Out-Calib",
    "w/o Queue-Externality",
    "Decoupled-Nominal-EVOI",
    "w/o Coupled-EVOI (Static-Opt)",
    "w/o Congestion-Term",
    "w/o Discounted-Forgetting",
]


def block_paired_bootstrap(
    ours_vec: np.ndarray,
    base_vec: np.ndarray,
    *,
    block_size: int = 10,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Circular moving-block paired bootstrap for serially dependent ticket streams."""
    diff = np.asarray(ours_vec, dtype=float) - np.asarray(base_vec, dtype=float)
    n = len(diff)
    mean_d = float(np.mean(diff))
    if np.allclose(diff, 0.0):
        return {
            "delta_mean": 0.0,
            "ci95": [0.0, 0.0],
            "boot_p": 1.0,
        }

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = float(np.mean(diff[idx]))

    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    centered = boots - mean_d
    boot_p = (1.0 + float(np.sum(np.abs(centered) >= abs(mean_d)))) / (n_boot + 1.0)
    return {
        "delta_mean": round(mean_d, 3),
        "ci95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
        "boot_p": float(f"{boot_p:.4e}"),
    }


def stratified_block_paired_bootstrap(
    ours_eps: List[np.ndarray],
    base_eps: List[np.ndarray],
    *,
    block_size: int = 10,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Stratified circular moving-block bootstrap resampling within each independent episode."""
    diffs = [np.asarray(o, dtype=float) - np.asarray(b, dtype=float) for o, b in zip(ours_eps, base_eps)]
    full_diff = np.concatenate(diffs)
    mean_d = float(np.mean(full_diff))
    if np.allclose(full_diff, 0.0):
        return {"delta_mean": 0.0, "ci95": [0.0, 0.0], "boot_p": 1.0}

    rng = np.random.default_rng(seed)
    offsets = np.arange(block_size)
    n_eps = len(diffs)
    ep_ns = [len(d) for d in diffs]
    ep_nblocks = [int(np.ceil(n / block_size)) for n in ep_ns]
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        tot_sum = 0.0
        tot_cnt = 0
        for e in range(n_eps):
            ne = ep_ns[e]
            starts = rng.integers(0, ne, size=ep_nblocks[e])
            idx = ((starts[:, None] + offsets[None, :]) % ne).ravel()[:ne]
            tot_sum += float(np.sum(diffs[e][idx]))
            tot_cnt += ne
        boots[b] = tot_sum / tot_cnt

    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    centered = boots - mean_d
    boot_p = (1.0 + float(np.sum(np.abs(centered) >= abs(mean_d)))) / (n_boot + 1.0)
    return {
        "delta_mean": round(mean_d, 3),
        "ci95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
        "boot_p": float(f"{boot_p:.4e}"),
    }


def paired_mcnemar_exact(ours_ok: np.ndarray, base_ok: np.ndarray) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired binary correctness indicators."""
    ours_b = np.asarray(ours_ok) > 0.5
    base_b = np.asarray(base_ok) > 0.5
    b = int(np.sum(ours_b & (~base_b)))
    c = int(np.sum((~ours_b) & base_b))
    if (b + c) == 0:
        pval = 1.0
    else:
        pval = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return {
        "b_ours_win": b,
        "c_base_win": c,
        "mcnemar_p": float(f"{pval:.4e}"),
    }


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Step-down Holm-Bonferroni family-wise error correction."""
    if not pvals:
        return {}
    m = len(pvals)
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(ordered):
        raw_adj = min(1.0, (m - i) * float(p))
        running_max = max(running_max, raw_adj)
        adjusted[k] = float(f"{running_max:.4e}")
    return adjusted


def stream_ordering_stability_check(
    test_probed: List[Dict[str, Any]],
    calibrator: HeldOutCalibrator,
    *,
    n_perms: int = 30,
) -> List[Dict[str, Any]]:
    """Secondary check: sensitivity to stream arrival order & specialist RNG across 30 permutations."""
    baselines = [
        "Direct-ACT",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
    ]
    counts: Dict[str, Dict[str, int]] = {b: {"wins": 0, "losses": 0, "ties": 0} for b in baselines}
    deltas: Dict[str, List[float]] = {b: [] for b in baselines}

    for p_idx in range(n_perms):
        perm_items = random.Random(100 + p_idx).sample(test_probed, len(test_probed))
        ours_r = simulate_stream(
            "Tri-VoI-L2D",
            perm_items,
            calibrator,
            capacity=2,
            drain_interval=2,
            drift_mode="sudden",
            seed=p_idx,
        )
        for b in baselines:
            base_r = simulate_stream(
                b,
                perm_items,
                calibrator,
                capacity=2,
                drain_interval=2,
                drift_mode="sudden",
                seed=p_idx,
            )
            du = ours_r["mean_utility"] - base_r["mean_utility"]
            deltas[b].append(du)
            if du > 1e-9:
                counts[b]["wins"] += 1
            elif du < -1e-9:
                counts[b]["losses"] += 1
            else:
                counts[b]["ties"] += 1

    out = []
    for b in baselines:
        w = counts[b]["wins"]
        l = counts[b]["losses"]
        sign_p = 1.0 if (w + l) == 0 else float(binomtest(w, w + l, 0.5, alternative="two-sided").pvalue)
        out.append({
            "baseline": b,
            "n_streams": n_perms,
            "wins": w,
            "losses": l,
            "ties": counts[b]["ties"],
            "mean_delta_util": round(float(np.mean(deltas[b])), 3),
            "sign_test_p": float(f"{sign_p:.4e}"),
        })
    return out


def eval_methods_multi_seed(
    methods: List[str],
    test_probed: List[Dict[str, Any]],
    calibrator: HeldOutCalibrator,
    *,
    n_seeds: int = 5,
    capacity: int = 2,
    drain_interval: int = 2,
    drift_mode: str = "sudden",
    asymmetric_risk: bool = False,
    c_ask: float = 1.5,
    compute_sig: bool = False,
) -> Dict[str, Any]:
    raw_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    for s in range(n_seeds):
        for m in methods:
            raw_runs[m].append(
                simulate_stream(
                    m,
                    test_probed,
                    calibrator,
                    capacity=capacity,
                    drain_interval=drain_interval,
                    drift_mode=drift_mode,
                    asymmetric_risk=asymmetric_risk,
                    seed=s,
                    c_ask=c_ask,
                )
            )

    has_ours = "Tri-VoI-L2D" in raw_runs
    ours_item_vec = np.mean([r["per_item_utils"] for r in raw_runs["Tri-VoI-L2D"]], axis=0) if has_ours else None
    ours_acc_vec = np.mean([r["per_item_correct"] for r in raw_runs["Tri-VoI-L2D"]], axis=0) if has_ours else None
    ours_ok_seed0 = raw_runs["Tri-VoI-L2D"][0]["per_item_correct"] if has_ours else None

    rows = []
    boot_p_map: Dict[str, float] = {}
    mcnemar_p_map: Dict[str, float] = {}

    for m in methods:
        rs = raw_runs[m]
        row: Dict[str, Any] = {
            "method": m,
            "acc_pct": round(float(np.mean([r["acc_pct"] for r in rs])), 2),
            "acc_std": round(float(np.std([r["acc_pct"] for r in rs])), 2),
            "net_utility": round(float(np.mean([r["net_utility"] for r in rs])), 2),
            "net_util_std": round(float(np.std([r["net_utility"] for r in rs])), 2),
            "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
            "ask_rate_pct": round(float(np.mean([r["ask_rate_pct"] for r in rs])), 1),
            "defer_rate_pct": round(float(np.mean([r["defer_rate_pct"] for r in rs])), 1),
            "mean_wait": round(float(np.mean([r["mean_wait"] for r in rs])), 2),
            "mean_queue_len": round(float(np.mean([r["mean_queue_len"] for r in rs])), 2),
            "misroute_pct": round(float(np.mean([r["misroute_pct"] for r in rs])), 2),
            "q4_ask_rate_pct": round(float(np.mean([r["q4_ask_rate_pct"] for r in rs])), 1),
            "q4_res_self_solve_pct": round(float(np.mean([r["q4_res_self_solve_pct"] for r in rs])), 1),
            "phase2_utility": round(float(np.mean([r["phase2_utility"] for r in rs])), 2),
            "quads": rs[0]["quads"],
        }
        if compute_sig and has_ours and m != "Tri-VoI-L2D":
            base_item_vec = np.mean([r["per_item_utils"] for r in rs], axis=0)
            base_acc_vec = np.mean([r["per_item_correct"] for r in rs], axis=0)
            base_ok_seed0 = rs[0]["per_item_correct"]

            ub = block_paired_bootstrap(ours_item_vec, base_item_vec, block_size=10, n_boot=5000, seed=42)
            ab = block_paired_bootstrap(ours_acc_vec * 100.0, base_acc_vec * 100.0, block_size=10, n_boot=5000, seed=42)
            mc = paired_mcnemar_exact(ours_ok_seed0, base_ok_seed0)
            row["sig_vs_ours"] = {
                "delta_util": ub["delta_mean"],
                "util_ci95": ub["ci95"],
                "boot_p": ub["boot_p"],
                "delta_acc_pp": round(ab["delta_mean"], 2),
                "acc_ci95": ab["ci95"],
                "mcnemar_b": mc["b_ours_win"],
                "mcnemar_c": mc["c_base_win"],
                "mcnemar_p": mc["mcnemar_p"],
            }
            if m != "Oracle":
                boot_p_map[m] = float(ub["boot_p"])
                mcnemar_p_map[m] = float(mc["mcnemar_p"])
        rows.append(row)

    if compute_sig and boot_p_map:
        boot_holm = holm_bonferroni(boot_p_map)
        mc_holm = holm_bonferroni(mcnemar_p_map)
        for row in rows:
            m = row["method"]
            if m in boot_holm:
                row["sig_vs_ours"]["boot_p_holm"] = boot_holm[m]
                row["sig_vs_ours"]["mcnemar_p_holm"] = mc_holm[m]

    rolling = {m: np.mean([r["rolling_acc"] for r in raw_runs[m]], axis=0).tolist() for m in methods}
    return {"rows": rows, "rolling_acc": rolling}


def eval_independent_episodes(
    methods: List[str],
    episodes: List[List[Dict[str, Any]]],
    calibrator: HeldOutCalibrator,
    *,
    n_seeds: int = 5,
    capacity: int = 2,
    drain_interval: int = 2,
    drift_mode: str = "sudden",
    c_ask: float = 1.5,
) -> Dict[str, Any]:
    """Evaluate across N disjoint 60-ticket episodes (fresh queue & tracker per episode)."""
    n_eps = len(episodes)
    ep_means_util: Dict[str, List[float]] = {m: [] for m in methods}
    ep_means_acc: Dict[str, List[float]] = {m: [] for m in methods}
    ep_asks: Dict[str, List[float]] = {m: [] for m in methods}
    ep_q4_asks: Dict[str, List[float]] = {m: [] for m in methods}
    ep_waits: Dict[str, List[float]] = {m: [] for m in methods}
    ep_item_utils: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    ep_seed0_ok: Dict[str, List[np.ndarray]] = {m: [] for m in methods}

    for ep_idx, ep_items in enumerate(episodes):
        for m in methods:
            runs = [
                simulate_stream(
                    m,
                    ep_items,
                    calibrator,
                    capacity=capacity,
                    drain_interval=drain_interval,
                    drift_mode=drift_mode,
                    seed=ep_idx * 100 + s,
                    c_ask=c_ask,
                )
                for s in range(n_seeds)
            ]
            ep_means_util[m].append(float(np.mean([r["mean_utility"] for r in runs])))
            ep_means_acc[m].append(float(np.mean([r["acc_pct"] for r in runs])))
            ep_asks[m].append(float(np.mean([r["ask_rate_pct"] for r in runs])))
            ep_q4_asks[m].append(float(np.mean([r["q4_ask_rate_pct"] for r in runs])))
            ep_waits[m].append(float(np.mean([r["mean_wait"] for r in runs])))
            ep_item_utils[m].append(np.mean([r["per_item_utils"] for r in runs], axis=0))
            ep_seed0_ok[m].append(runs[0]["per_item_correct"])

    ours_ep_u = np.array(ep_means_util["Tri-VoI-L2D"], dtype=float)
    ours_pooled_ok = np.concatenate(ep_seed0_ok["Tri-VoI-L2D"])

    rows = []
    boot_p_map: Dict[str, float] = {}
    mc_p_map: Dict[str, float] = {}
    ep_t_map: Dict[str, float] = {}

    for m in methods:
        u_arr = np.array(ep_means_util[m], dtype=float)
        a_arr = np.array(ep_means_acc[m], dtype=float)
        row: Dict[str, Any] = {
            "method": m,
            "ep_acc_mean": round(float(np.mean(a_arr)), 2),
            "ep_acc_std": round(float(np.std(a_arr, ddof=1)), 2),
            "ep_util_mean": round(float(np.mean(u_arr)), 3),
            "ep_util_std": round(float(np.std(u_arr, ddof=1)), 3),
            "ask_rate_pct": round(float(np.mean(ep_asks[m])), 1),
            "q4_ask_rate_pct": round(float(np.mean(ep_q4_asks[m])), 1),
            "mean_wait": round(float(np.mean(ep_waits[m])), 2),
        }
        if m != "Tri-VoI-L2D":
            wins = int(np.sum(ours_ep_u > u_arr + 1e-9))
            diff_ep = ours_ep_u - u_arr
            if np.allclose(diff_ep, 0.0):
                ep_tp = 1.0
            else:
                _, ep_tp_val = ttest_rel(ours_ep_u, u_arr)
                ep_tp = float(ep_tp_val)
            sb = stratified_block_paired_bootstrap(
                ep_item_utils["Tri-VoI-L2D"], ep_item_utils[m], block_size=10, n_boot=5000, seed=42
            )
            base_pooled_ok = np.concatenate(ep_seed0_ok[m])
            mc = paired_mcnemar_exact(ours_pooled_ok, base_pooled_ok)
            row["sig_vs_ours"] = {
                "ep_wins": wins,
                "n_episodes": n_eps,
                "ep_t_p": float(f"{ep_tp:.4e}"),
                "delta_util": sb["delta_mean"],
                "util_ci95": sb["ci95"],
                "boot_p": sb["boot_p"],
                "mcnemar_p": mc["mcnemar_p"],
            }
            if m != "Oracle":
                boot_p_map[m] = float(sb["boot_p"])
                mc_p_map[m] = float(mc["mcnemar_p"])
                ep_t_map[m] = ep_tp
        rows.append(row)

    boot_holm = holm_bonferroni(boot_p_map)
    mc_holm = holm_bonferroni(mc_p_map)
    ept_holm = holm_bonferroni(ep_t_map)
    for row in rows:
        m = row["method"]
        if m in boot_holm:
            row["sig_vs_ours"]["boot_p_holm"] = boot_holm[m]
            row["sig_vs_ours"]["mcnemar_p_holm"] = mc_holm[m]
            row["sig_vs_ours"]["ep_t_p_holm"] = ept_holm[m]

    return {"n_episodes": n_eps, "c_ask": c_ask, "rows": rows}


def generate_figures(all_out: Dict[str, Any]) -> List[str]:
    saved = []

    # Fig 1: Pre-Exp Entropy Collapse vs Greedy Accuracy on Underspecified Tickets
    pre = all_out["pre_exp_entropy_collapse"]
    models = [x["short_name"] for x in pre]
    collapse = [x["entropy_collapse_pct"] for x in pre]
    under_acc = [x["underspec_acc_greedy_pct"] for x in pre]
    miss_f1 = [x["missing_f1"] * 100.0 for x in pre]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(models))
    w = 0.26
    ax.bar(x - w, collapse, width=w, label="Entropy Collapse (H=0) %", color="#d95f02")
    ax.bar(x, under_acc, width=w, label="Underspec Greedy Acc %", color="#7570b3")
    ax.bar(x + w, miss_f1, width=w, label="Disentangled Missing-Info F1 %", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Finding 2: High-Consensus Entropy Collapse Across 4 LLM Backbones")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = str(FIGS_DIR / f"fig1_entropy_collapse.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/figs/fig1_entropy_collapse.{ext}")
    plt.close(fig)

    # Fig 2: 3-Panel Queue-Relief Coupling, Phase Transition, Naor Externality, and c_ask Sensitivity
    fs = all_out["exp2_fixed_state_theory"]
    q_grid = fs["q_grid"]
    curves = fs["analytical_curves"]
    p2d = fs["phase_2d_by_pe"]
    sweep = all_out["exp2_congestion_sweep"]
    cask_rows = all_out["exp2_cask_sweep"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.1))

    # Panel (a): Fixed-state EVOI(Q) across p_resolve + phase transition Q4 ASK curve (p_e=0.92) on twin axis
    palette = {
        "0.00": ("#444444", "--", "p_res=0.00 (Neg Ctrl, slope=0)"),
        "0.25": ("#a6bddb", "-", "p_res=0.25"),
        "0.50": ("#3690c0", "-", "p_res=0.50"),
        "0.75": ("#02818a", "-", "p_res=0.75"),
        "1.00": ("#1b9e77", "-", "p_res=1.00"),
    }
    for k_p in ["0.00", "0.25", "0.50", "0.75", "1.00"]:
        col, ls, lbl = palette[k_p]
        ax1.plot(q_grid, curves[k_p]["evoi_coupled"], marker="o", markersize=3.5, linewidth=1.8, linestyle=ls, color=col, label=lbl)
    ax1.plot(q_grid, curves["0.50"]["evoi_decoupled"], linestyle=":", linewidth=1.6, color="#7570b3", label="Decoupled-Nominal")
    ax1.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax1.set_xlabel("Injected Specialist Queue Q")
    ax1.set_ylabel("Analytical EVOI(ASK; Q)")
    ax1.set_title("(a) Fixed-State Slope & Threshold Flip")
    ax1.grid(alpha=0.25)

    ax1_r = ax1.twinx()
    l_flip, = ax1_r.plot(
        q_grid,
        p2d["0.92"]["q4_ask_pct_calibrated"],
        marker="D",
        markersize=4.5,
        linewidth=2.0,
        linestyle="-.",
        color="#e7298a",
        label="Calib Q4 ASK% (p_e=0.92)",
    )
    ax1_r.set_ylabel("Calibrated Q4 ASK % (p_e=0.92)", color="#e7298a", fontsize=8.5)
    ax1_r.tick_params(axis="y", labelcolor="#e7298a", labelsize=8)
    ax1_r.set_ylim(-5, 108)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + [l_flip], l1 + ["Q4 ASK% Flip (p_e=0.92)"], fontsize=6.6, loc="upper left")

    # Panel (b): Dynamic Load Sweep (Q4 ASK % on left, Severe/High Mean Wait on right)
    xs = np.arange(len(sweep))
    x_labels = [f"{s['regime_label'].split('(')[1].rstrip(')')}\n(Q={s['tri_mean_queue_len']:.2f})" for s in sweep]
    ax2.plot(xs, [s["tri_q4_ask_pct"] for s in sweep], marker="o", linewidth=2.2, color="#1b9e77", label="Tri-VoI Q4 ASK%")
    ax2.plot(xs, [s["decoupled_nominal_q4_ask_pct"] for s in sweep], marker="^", linestyle="--", linewidth=1.7, color="#7570b3", label="Decoupled-Nom Q4 ASK%")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(x_labels, fontsize=7.8)
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel("Q4 Pre-Clarification ASK (%)")
    ax2.set_title("(b) Dynamic Load & Externality Relief")
    ax2.grid(alpha=0.25)

    ax2_r = ax2.twinx()
    bw = 0.19
    b1 = ax2_r.bar(xs - bw, [s["tri_mean_wait"] for s in sweep], width=bw, alpha=0.45, color="#1b9e77", label="Tri-VoI Wait (Naor Ext)")
    b2 = ax2_r.bar(xs, [s["wo_ext_mean_wait"] for s in sweep], width=bw, alpha=0.45, color="#d95f02", label="w/o Externality Wait")
    b3 = ax2_r.bar(xs + bw, [s["cascade_mean_wait"] for s in sweep], width=bw, alpha=0.45, color="#386cb0", label="Calib-Cascade Wait")
    ax2_r.set_ylabel("Mean Specialist Wait (Queue)")
    h2, l2 = ax2.get_legend_handles_labels()
    h2r, l2r = ax2_r.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, fontsize=6.6, loc="upper left")

    # Panel (c): c_ask sensitivity sweep at Cap=2 (N=360)
    c_vals = [r["c_ask"] for r in cask_rows]
    for meth, col, mk, ls in [
        ("Tri-VoI-L2D", "#1b9e77", "o", "-"),
        ("Calibrated-Cascade+Naor", "#005a32", "^", "--"),
        ("Calibrated-Cascade", "#386cb0", "^", "-"),
        ("Joint-Greedy+Naor", "#b34700", "s", "--"),
        ("Joint-Greedy", "#d95f02", "s", "-"),
        ("Naive-Cascade", "#e7298a", "d", "-"),
    ]:
        ys = [r["methods"][meth]["mean_utility"] for r in cask_rows]
        ax3.plot(c_vals, ys, marker=mk, linestyle=ls, linewidth=1.8, color=col, label=meth)
    ax3.set_xlabel("User Clarification Burden c_ask")
    ax3.set_ylabel("Mean Net Utility / Ticket")
    ax3.set_title("(c) Cost Sensitivity c_ask (N=360, Cap=2)")
    ax3.legend(fontsize=6.6, loc="lower left")
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = str(FIGS_DIR / f"fig2_queue_relief_coupling.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/figs/fig2_queue_relief_coupling.{ext}")
    plt.close(fig)

    # Fig 3: Drift Adaptation + Stress Scenarios (N=360)
    roll = all_out["exp1_main"]["rolling_acc"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    win = 25
    for m, col in [
        ("Tri-VoI-L2D", "#1b9e77"),
        ("Calibrated-Cascade", "#386cb0"),
        ("Joint-Greedy", "#d95f02"),
        ("Naive-Cascade", "#e7298a"),
        ("Direct-ACT", "#666666"),
    ]:
        arr = np.array(roll[m]) * 100.0
        sm = np.convolve(arr, np.ones(win) / win, mode="valid")
        ax1.plot(range(win - 1, len(arr)), sm, label=m, color=col, linewidth=1.9)
    ax1.axvline(len(roll["Tri-VoI-L2D"]) // 2, color="red", linestyle=":", label="Drift Onset (0.92->0.60)")
    ax1.set_title("Rolling Accuracy Under Fatigue Drift (N=360)")
    ax1.set_xlabel("Ticket Index t")
    ax1.set_ylabel("Rolling Accuracy (%)")
    ax1.legend(fontsize=7.2, loc="lower left")
    ax1.grid(alpha=0.25)

    stress = all_out["exp2_stress_summary"]
    scenarios = ["Sudden Drift", "Gradual Drift", "Senior+Junior Pool", "Asymmetric Risk"]
    x3 = np.arange(len(scenarios))
    w3 = 0.18
    for idx, (m, col) in enumerate([
        ("Joint-Greedy", "#d95f02"),
        ("Naive-Cascade", "#e7298a"),
        ("Calibrated-Cascade", "#386cb0"),
        ("Tri-VoI-L2D", "#1b9e77"),
    ]):
        vals = [stress[s][m] for s in scenarios]
        ax2.bar(x3 + (idx - 1.5) * w3, vals, width=w3, label=m, color=col)
    ax2.set_xticks(x3)
    ax2.set_xticklabels(scenarios, fontsize=8, rotation=10)
    ax2.set_ylabel("Mean Net Utility per Ticket")
    ax2.set_title("CSCW Stress Scenarios (N=360, c_ask=1.5)")
    ax2.legend(fontsize=7.2)
    ax2.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = str(FIGS_DIR / f"fig3_drift_and_multiexpert.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/figs/fig3_drift_and_multiexpert.{ext}")
    plt.close(fig)

    # Fig 4: Cross-Model Generalization & Ablation (N=360 primary + c_ask=2.5 ablation contrast)
    cross = all_out["exp3_cross_model"]
    cm_names = [c["short_name"] for c in cross]
    x4 = np.arange(len(cm_names))
    w4 = 0.18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for idx, (key, lbl, col) in enumerate([
        ("naive_mean_util", "Naive-Cascade", "#e7298a"),
        ("greedy_mean_util", "Joint-Greedy", "#d95f02"),
        ("cascade_mean_util", "Calibrated-Cascade", "#386cb0"),
        ("tri_mean_util", "Tri-VoI-L2D", "#1b9e77"),
    ]):
        ax1.bar(x4 + (idx - 1.5) * w4, [c[key] for c in cross], width=w4, label=lbl, color=col)
    ax1.set_xticks(x4)
    ax1.set_xticklabels(cm_names, fontsize=8.5)
    ax1.set_ylabel("Mean Net Utility / Ticket")
    ax1.set_title("Cross-Model Generalization (4 Heterogeneous Backbones)")
    ax1.legend(fontsize=7.2)
    ax1.grid(axis="y", alpha=0.25)

    abls_15 = {a["method"]: a["mean_utility"] for a in all_out["exp4_ablation"]["rows"]}
    abls_25 = {a["method"]: a["mean_utility"] for a in all_out["exp4_ablation_c25"]["rows"]}
    anames = [a["method"] for a in all_out["exp4_ablation"]["rows"]]
    y_pos = np.arange(len(anames))
    bh = 0.36
    ax2.barh(y_pos - bh / 2, [abls_15[nm] for nm in anames], height=bh, color="#1b9e77", label="c_ask = 1.5")
    ax2.barh(y_pos + bh / 2, [abls_25[nm] for nm in anames], height=bh, color="#386cb0", label="c_ask = 2.5")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(anames, fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Net Utility / Ticket")
    ax2.set_title("Component Ablation Study (N=360)")
    ax2.legend(fontsize=7.5)
    ax2.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = str(FIGS_DIR / f"fig4_ablation_and_crossmodel.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/figs/fig4_ablation_and_crossmodel.{ext}")
    plt.close(fig)

    return saved


def _format_main_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Method | Acc (%) | Net Util | Mean Util | ASK (%) | DEFER (%) | Misroute (%) | Mean Wait | Q4 ASK (%) | ΔUtil [95% Block CI] | Boot p (Holm) | McNemar p (Holm) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    for r in rows:
        sig = r.get("sig_vs_ours")
        if sig is None:
            du_str = "-"
            bp_str = "-"
            mp_str = "-"
        else:
            du_str = f"{sig['delta_util']:+.3f} [{sig['util_ci95'][0]:+.2f}, {sig['util_ci95'][1]:+.2f}]"
            bp_str = f"{sig['boot_p']} ({sig['boot_p_holm']})" if "boot_p_holm" in sig else str(sig["boot_p"])
            mp_str = f"{sig['mcnemar_p']} ({sig['mcnemar_p_holm']})" if "mcnemar_p_holm" in sig else str(sig["mcnemar_p"])
        lines.append(
            f"| {r['method']} | {r['acc_pct']}±{r['acc_std']} | {r['net_utility']} | {r['mean_utility']} | "
            f"{r['ask_rate_pct']} | {r['defer_rate_pct']} | {r['misroute_pct']} | {r['mean_wait']} | "
            f"{r['q4_ask_rate_pct']} | {du_str} | {bp_str} | {mp_str} |"
        )
    return lines


def _format_episode_table(ep_res: Dict[str, Any]) -> List[str]:
    lines = [
        f"| Method | Ep Acc (%) Mean±Std | Ep Util Mean±Std | ASK (%) | Q4 ASK (%) | Wait | Wins/{ep_res['n_episodes']} | Ep Paired-t p (Holm) | Pooled ΔUtil [95% Strat CI] | Strat Boot p (Holm) | McNemar p (Holm) |",
        "| :--- | :---: | :---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |",
    ]
    for r in ep_res["rows"]:
        sig = r.get("sig_vs_ours")
        if sig is None:
            w_str = "-"
            tp_str = "-"
            du_str = "-"
            bp_str = "-"
            mp_str = "-"
        else:
            w_str = f"{sig['ep_wins']}/{sig['n_episodes']}"
            tp_str = f"{sig['ep_t_p']} ({sig['ep_t_p_holm']})" if "ep_t_p_holm" in sig else str(sig["ep_t_p"])
            du_str = f"{sig['delta_util']:+.3f} [{sig['util_ci95'][0]:+.2f}, {sig['util_ci95'][1]:+.2f}]"
            bp_str = f"{sig['boot_p']} ({sig['boot_p_holm']})" if "boot_p_holm" in sig else str(sig["boot_p"])
            mp_str = f"{sig['mcnemar_p']} ({sig['mcnemar_p_holm']})" if "mcnemar_p_holm" in sig else str(sig["mcnemar_p"])
        lines.append(
            f"| {r['method']} | {r['ep_acc_mean']}±{r['ep_acc_std']} | {r['ep_util_mean']}±{r['ep_util_std']} | "
            f"{r['ask_rate_pct']} | {r['q4_ask_rate_pct']} | {r['mean_wait']} | {w_str} | {tp_str} | {du_str} | {bp_str} | {mp_str} |"
        )
    return lines


def write_outputs(all_out: Dict[str, Any], dry_run: bool) -> None:
    prefix = "dry_run_" if dry_run else ""
    json_path = RESULTS_DIR / f"{prefix}all_metrics.json"
    md_path = RESULTS_DIR / f"{prefix}report.md"
    tex_path = RESULTS_DIR / f"{prefix}latex_tables.tex"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)

    lines = [
        f"# Tri-VoI-L2D Phase 2 Experiment Report (N=360 + 6 Episodes) — [{all_out['mode']}]",
        "",
        "## Pre-Exp: High-Consensus Entropy Collapse Across 4 Backbones",
        "",
        "| Backbone | Test N | Missing-Info F1 | Override F1 | Entropy Collapse (H=0 on Underspec) % | Underspec Greedy Acc % |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in all_out["pre_exp_entropy_collapse"]:
        lines.append(f"| {p['model_id']} | {p.get('n_test', '-')} | {p['missing_f1']} | {p['override_f1']} | {p['entropy_collapse_pct']} | {p['underspec_acc_greedy_pct']} |")

    lines += [
        "",
        "## Exp 1a: Continuous Stream on Qwen/Qwen3.5-35B-A3B (`N=360` Test Tickets × 5 Seeds, `c_ask=1.5`)",
        "",
    ] + _format_main_table(all_out["exp1_main"]["rows"])

    lines += [
        "",
        "## Exp 1b: Continuous Stream on Qwen/Qwen3.5-35B-A3B (`N=360` Test Tickets × 5 Seeds, Moderate Burden `c_ask=2.5`)",
        "",
    ] + _format_main_table(all_out["exp1_main_c25"]["rows"])

    lines += [
        "",
        "## Exp 1c: 6 Disjoint 60-Ticket Independent Episodes (`N=360`, Fresh Queue/Tracker per Episode, `c_ask=1.5`)",
        "",
    ] + _format_episode_table(all_out["exp1_episodes_c15"])

    lines += [
        "",
        "## Exp 1d: 6 Disjoint 60-Ticket Independent Episodes (`N=360`, Moderate User Burden `c_ask=2.5`)",
        "",
    ] + _format_episode_table(all_out["exp1_episodes_c25"])

    lines += [
        "",
        "### Exp 1 Secondary Check: 30-Permutation Stream-Order & Specialist-Stochasticity Stability (`N=360`)",
        "> Note: Permutes the 360 test tickets across 30 arrival orders (queue/RNG robustness check only).",
        "",
        "| Baseline | Wins / 30 Streams | Mean ΔUtil | Sign-Test p |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for st in all_out["exp1_stream_stability"]:
        lines.append(f"| {st['baseline']} | {st['wins']}/{st['n_streams']} (Ties: {st['ties']}) | {st['mean_delta_util']:+.3f} | {st['sign_test_p']} |")

    lines += [
        "",
        "## Exp 2a: Queue-Relief VoI & Naor Externality Load Sweep (`N=360`, `c_ask=1.5`)",
        "",
        "| Regime | Tri Qbar | Tri Q4 ASK% | Dec-Nom Q4 ASK% | Tri Util / Wait | w/o Ext Util / Wait | Calib-Casc+Naor Util / Wait | Calib-Casc Util / Wait | Joint-Greedy+Naor Util / Wait | Joint-Greedy Util / Wait |",
        "| :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in all_out["exp2_congestion_sweep"]:
        lines.append(
            f"| {s['regime_label']} | {s['tri_mean_queue_len']:.2f} | {s['tri_q4_ask_pct']} | {s['decoupled_nominal_q4_ask_pct']} | "
            f"**{s['tri_mean_util']:.3f} / {s['tri_mean_wait']:.2f}** | "
            f"{s['wo_ext_mean_util']:.3f} / {s['wo_ext_mean_wait']:.2f} | "
            f"{s['cascade_naor_mean_util']:.3f} / {s['cascade_naor_mean_wait']:.2f} | "
            f"{s['cascade_mean_util']:.3f} / {s['cascade_mean_wait']:.2f} | "
            f"{s['greedy_naor_mean_util']:.3f} / {s['greedy_naor_mean_wait']:.2f} | "
            f"{s['greedy_mean_util']:.3f} / {s['greedy_mean_wait']:.2f} |"
        )

    fs = all_out["exp2_fixed_state_theory"]
    lines += [
        "",
        f"## Exp 2a-Theory: Three-Regime Fixed-State Queue Sweep (Q∈{{0..10}}, fixed p_e={fs['p_e_fixed']}, γ={fs['gamma_queue']}) & Negative Control (`N_Q4={fs['n_q4_items']}`)",
        "",
        "| p_resolve | Theoretical Defer Slope (p_res·γ) | Analytical Defer Slope | Decoupled-Nominal Slope | Q=0 EVOI | Q=6 EVOI | Q=10 EVOI | Q4 ASK% (Q=0 -> Q=10) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for kp, cv in fs["analytical_curves"].items():
        q4c = fs["q4_ask_pct_by_p_resolve"][kp]
        lines.append(
            f"| {kp} | {cv['theoretical_slope']:.4f} | {cv['analytical_defer_slope']:.4f} | {cv['decoupled_slope']:.4f} | "
            f"{cv['evoi_coupled'][0]:+.3f} | {cv['evoi_coupled'][6]:+.3f} | {cv['evoi_coupled'][-1]:+.3f} | {q4c[0]:.1f}% -> {q4c[-1]:.1f}% |"
        )

    lines += [
        "",
        "### Exp 2a-2D Three-Regime Phase Transition & Monte Carlo Realized-Utility Rollout (`p_e ∈ {0.75, 0.85, 0.92}`, `N_MC=200`)",
        "",
        "| p_e | Q1* (ACT/DEFER) | Q2* (OVR/DEFER) | Theor Defer Slope | MC Realized Slope ± SE | Analytical EVOI (Q=0 -> Q=6 -> Q=10) | MC Realized EVOI (Q=0 -> Q=6 -> Q=10) | Tri-VoI Q4 ASK% (Q=0 -> Q=10) |",
        "| ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |",
    ]
    for kpe, pv in fs["phase_2d_by_pe"].items():
        lines.append(
            f"| {kpe} | {pv['q1_act_defer_boundary']:.2f} | {pv['q2_ovr_defer_boundary']:.2f} | "
            f"{pv['theoretical_defer_slope']:.4f} | {pv['realized_defer_slope']:.4f}±{pv['realized_defer_slope_se']:.4f} | "
            f"{pv['evoi_by_q'][0]:+.3f} -> {pv['evoi_by_q'][6]:+.3f} -> {pv['evoi_by_q'][-1]:+.3f} | "
            f"{pv['realized_evoi_mean'][0]:+.3f} -> {pv['realized_evoi_mean'][6]:+.3f} -> {pv['realized_evoi_mean'][-1]:+.3f} | "
            f"{pv['q4_ask_pct_calibrated'][0]:.1f}% -> {pv['q4_ask_pct_calibrated'][-1]:.1f}% |"
        )

    lines += [
        "",
        "## Exp 2a-Cost: User Clarification Burden Sweep (`c_ask ∈ {0.5, 1.5, 2.5, 3.0, 5.0}`, `N=360`, Cap=2)",
        "",
        "| c_ask | Naive-Cascade | Joint-Greedy | Joint-Greedy+Naor | Calib-Cascade | Calib-Cascade+Naor | Tri-VoI-L2D |",
        "| ---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cr in all_out["exp2_cask_sweep"]:
        ms = cr["methods"]
        lines.append(
            f"| {cr['c_ask']:.1f} | {ms['Naive-Cascade']['mean_utility']:.3f} ({ms['Naive-Cascade']['ask_rate_pct']}%) | "
            f"{ms['Joint-Greedy']['mean_utility']:.3f} ({ms['Joint-Greedy']['ask_rate_pct']}%) | "
            f"{ms['Joint-Greedy+Naor']['mean_utility']:.3f} ({ms['Joint-Greedy+Naor']['ask_rate_pct']}%) | "
            f"{ms['Calibrated-Cascade']['mean_utility']:.3f} ({ms['Calibrated-Cascade']['ask_rate_pct']}%) | "
            f"{ms['Calibrated-Cascade+Naor']['mean_utility']:.3f} ({ms['Calibrated-Cascade+Naor']['ask_rate_pct']}%) | "
            f"**{ms['Tri-VoI-L2D']['mean_utility']:.3f} ({ms['Tri-VoI-L2D']['ask_rate_pct']}%)** |"
        )

    lines += [
        "",
        "## Exp 2b/c: CSCW Drift, Heterogeneous Pool & Asymmetric Risk (`N=360`, Mean Utility)",
        "",
        "| Scenario | Direct-ACT | Congested-L2D-Only | Naive-Cascade | Joint-Greedy | Joint-Greedy+Naor | Calibrated-Cascade | Calib-Casc+Naor | Tri-VoI-L2D |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sc_name, vals in all_out["exp2_stress_summary"].items():
        lines.append(
            f"| {sc_name} | {vals['Direct-ACT']} | {vals['Congested-L2D-Only']} | {vals['Naive-Cascade']} | "
            f"{vals['Joint-Greedy']} | {vals['Joint-Greedy+Naor']} | {vals['Calibrated-Cascade']} | {vals['Calibrated-Cascade+Naor']} | **{vals['Tri-VoI-L2D']}** |"
        )

    lines += [
        "",
        "## Exp 3: Cross-Model Generalization (Mean Utility / Acc %, `c_ask=1.5`)",
        "",
        "| Backbone | Test N | Direct-ACT | Congested-L2D | Naive-Cascade | Joint-Greedy | Joint-Greedy+Naor | Calibrated-Cascade | Calib-Casc+Naor | Tri-VoI-L2D |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in all_out["exp3_cross_model"]:
        lines.append(
            f"| {c['model_id']} | {c['n_test']} | {c['direct_mean_util']} ({c['direct_acc']}%) | "
            f"{c['l2d_mean_util']} ({c['l2d_acc']}%) | {c['naive_mean_util']} ({c['naive_acc']}%) | "
            f"{c['greedy_mean_util']} ({c['greedy_acc']}%) | {c['greedy_naor_mean_util']} ({c['greedy_naor_acc']}%) | "
            f"{c['cascade_mean_util']} ({c['cascade_acc']}%) | {c['cascade_naor_mean_util']} ({c['cascade_naor_acc']}%) | "
            f"**{c['tri_mean_util']} ({c['tri_acc']}%)** |"
        )

    lines += [
        "",
        "## Exp 4: Component Ablation Study (`N=360` Test Tickets × 5 Seeds, `c_ask=1.5` vs `c_ask=2.5`)",
        "",
        "| Variant | Acc% (1.5) | Util (1.5) | ASK% (1.5) | Q4 ASK% (1.5) | Acc% (2.5) | Util (2.5) | ASK% (2.5) | Phase2 Util (1.5) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    c25_map = {r["method"]: r for r in all_out["exp4_ablation_c25"]["rows"]}
    for r in all_out["exp4_ablation"]["rows"]:
        r2 = c25_map[r["method"]]
        lines.append(
            f"| {r['method']} | {r['acc_pct']} | {r['mean_utility']} | {r['ask_rate_pct']} | {r['q4_ask_rate_pct']} | "
            f"{r2['acc_pct']} | {r2['mean_utility']} | {r2['ask_rate_pct']} | {r['phase2_utility']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact IEEE LaTeX Table 1 export (N=360, c_ask=1.5)
    tex_lines = [
        "% Auto-generated Tri-VoI-L2D Table 1 (Phase 2: N=360 test tickets x 5 seeds)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main comparison on natural-language CSCW cooperative tasks (360 independent test tickets $\\times$ 5 seeds, $c_{\\text{ask}}=1.5$).}",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "Method & Acc(\\%) & MeanUtil & ASK(\\%) & Defer(\\%) & Misrt(\\%) & $\\Delta$Util [95\\% Block CI] & McNemar $p_{\\text{Holm}}$ \\\\",
        "\\hline",
    ]
    for r in all_out["exp1_main"]["rows"]:
        sig = r.get("sig_vs_ours")
        if sig is None:
            du_tex = "--"
            mc_tex = "--"
        else:
            du_tex = f"${sig['delta_util']:+.2f}$ [{sig['util_ci95'][0]:+.2f}, {sig['util_ci95'][1]:+.2f}]"
            mc_tex = str(sig.get("mcnemar_p_holm", sig["mcnemar_p"]))
        tex_lines.append(
            f"{r['method']} & {r['acc_pct']:.2f} & {r['mean_utility']:.3f} & {r['ask_rate_pct']:.1f} & "
            f"{r['defer_rate_pct']:.1f} & {r['misroute_pct']:.2f} & {du_tex} & {mc_tex} \\\\"
        )
    tex_lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved {json_path}, {md_path}, {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Tri-VoI-L2D Phase 2 Experiments (A+B) (N=360)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cal-check-only", action="store_true", help="Probe primary 60 cal items first to verify F1 >= 0.65")
    args = parser.parse_args()

    cal_full, test_180 = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    test_ext_180 = build_cscw_expansion_batch(n_per_quad=45, seed=2028, idx_offset=240)
    primary_model = ALL_MODELS[0]

    # Step A: Probe primary calibration split first & check F1 gate (100% cached)
    cal_prim = probe_dataset(cal_full, primary_model, dry_run=args.dry_run, workers=args.workers, label="cal60")
    cal_diag = evaluate_diagnosis_metrics(cal_prim)
    print(f"[Gate Check] {primary_model} 60-item cal diagnosis: {cal_diag}", flush=True)
    if args.cal_check_only:
        return

    # Existing 180 test tickets (100% cached) + 180 expansion tickets (probed once & cached)
    test_prim_base = probe_dataset(test_180, primary_model, dry_run=args.dry_run, workers=args.workers, label="test180_cached")
    test_prim_ext = probe_dataset(test_ext_180, primary_model, dry_run=args.dry_run, workers=args.workers, label="test180_new")
    test_prim = test_prim_base + test_prim_ext
    prim_calib = HeldOutCalibrator(cal_prim)

    # Probe secondary 3 models (40 cal + 80 test stratified subset — 100% cached)
    cal_sec = cal_full[:40]
    test_sec = test_180[:80]
    model_data = {
        primary_model: {"cal": cal_prim, "test": test_prim, "calib": prim_calib},
    }
    for m in ALL_MODELS[1:]:
        mc = probe_dataset(cal_sec, m, dry_run=args.dry_run, workers=args.workers, label="cal40")
        mt = probe_dataset(test_sec, m, dry_run=args.dry_run, workers=args.workers, label="test80")
        model_data[m] = {"cal": mc, "test": mt, "calib": HeldOutCalibrator(mc)}

    # Pre-Exp: 4-model Entropy Collapse diagnostic on test split
    pre_exp = []
    for m in ALL_MODELS:
        dm = evaluate_diagnosis_metrics(model_data[m]["test"])
        dm["model_id"] = m
        dm["short_name"] = m.split("/")[-1]
        dm["n_test"] = len(model_data[m]["test"])
        pre_exp.append(dm)

    # Exp 1a & 1b: Continuous stream on N=360 (c_ask=1.5 and c_ask=2.5)
    exp1_main = eval_methods_multi_seed(
        MAIN_METHODS, test_prim, prim_calib, n_seeds=5, capacity=2, drain_interval=2, drift_mode="sudden", c_ask=1.5, compute_sig=True
    )
    exp1_main_c25 = eval_methods_multi_seed(
        MAIN_METHODS, test_prim, prim_calib, n_seeds=5, capacity=2, drain_interval=2, drift_mode="sudden", c_ask=2.5, compute_sig=True
    )

    # Exp 1c & 1d: 6 disjoint 60-ticket independent episodes
    episodes = split_stratified_episodes(test_prim, n_episodes=6, seed=2029)
    exp1_episodes_c15 = eval_independent_episodes(
        MAIN_METHODS, episodes, prim_calib, n_seeds=5, capacity=2, drain_interval=2, drift_mode="sudden", c_ask=1.5
    )
    exp1_episodes_c25 = eval_independent_episodes(
        MAIN_METHODS, episodes, prim_calib, n_seeds=5, capacity=2, drain_interval=2, drift_mode="sudden", c_ask=2.5
    )

    exp1_stream_stability = stream_ordering_stability_check(test_prim, prim_calib, n_perms=30)

    # Exp 2a: Queue-Relief Coupling Load Sweep (N=360, includes w/o Queue-Externality)
    regimes = [
        ("Low Load (Cap=4)", 4, 1),
        ("Medium Load (Cap=2)", 2, 2),
        ("High Congestion (Cap=1)", 1, 2),
        ("Severe Bottleneck (Cap=1/3)", 1, 3),
    ]
    exp2_sweep = []
    sweep_methods = [
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Decoupled-Nominal-EVOI",
        "w/o Coupled-EVOI (Static-Opt)",
        "w/o Queue-Externality",
        "Tri-VoI-L2D",
    ]
    for label, cap, d_int in regimes:
        sw = eval_methods_multi_seed(
            sweep_methods,
            test_prim,
            prim_calib,
            n_seeds=5,
            capacity=cap,
            drain_interval=d_int,
            drift_mode="sudden",
            c_ask=1.5,
        )
        rmap = {r["method"]: r for r in sw["rows"]}
        exp2_sweep.append(
            {
                "regime_label": label,
                "tri_q4_ask_pct": rmap["Tri-VoI-L2D"]["q4_ask_rate_pct"],
                "tri_ask_pct": rmap["Tri-VoI-L2D"]["ask_rate_pct"],
                "tri_mean_queue_len": rmap["Tri-VoI-L2D"]["mean_queue_len"],
                "tri_mean_wait": rmap["Tri-VoI-L2D"]["mean_wait"],
                "tri_mean_util": rmap["Tri-VoI-L2D"]["mean_utility"],
                "wo_ext_mean_util": rmap["w/o Queue-Externality"]["mean_utility"],
                "wo_ext_mean_wait": rmap["w/o Queue-Externality"]["mean_wait"],
                "wo_ext_q4_ask_pct": rmap["w/o Queue-Externality"]["q4_ask_rate_pct"],
                "decoupled_nominal_q4_ask_pct": rmap["Decoupled-Nominal-EVOI"]["q4_ask_rate_pct"],
                "static_opt_q4_ask_pct": rmap["w/o Coupled-EVOI (Static-Opt)"]["q4_ask_rate_pct"],
                "cascade_mean_util": rmap["Calibrated-Cascade"]["mean_utility"],
                "cascade_mean_wait": rmap["Calibrated-Cascade"]["mean_wait"],
                "cascade_q4_ask_pct": rmap["Calibrated-Cascade"]["q4_ask_rate_pct"],
                "cascade_naor_mean_util": rmap["Calibrated-Cascade+Naor"]["mean_utility"],
                "cascade_naor_mean_wait": rmap["Calibrated-Cascade+Naor"]["mean_wait"],
                "cascade_naor_q4_ask_pct": rmap["Calibrated-Cascade+Naor"]["q4_ask_rate_pct"],
                "greedy_mean_util": rmap["Joint-Greedy"]["mean_utility"],
                "greedy_mean_wait": rmap["Joint-Greedy"]["mean_wait"],
                "greedy_q4_ask_pct": rmap["Joint-Greedy"]["q4_ask_rate_pct"],
                "greedy_naor_mean_util": rmap["Joint-Greedy+Naor"]["mean_utility"],
                "greedy_naor_mean_wait": rmap["Joint-Greedy+Naor"]["mean_wait"],
                "greedy_naor_q4_ask_pct": rmap["Joint-Greedy+Naor"]["q4_ask_rate_pct"],
                "naive_mean_util": rmap["Naive-Cascade"]["mean_utility"],
                "naive_mean_wait": rmap["Naive-Cascade"]["mean_wait"],
                "naive_mean_queue_len": rmap["Naive-Cascade"]["mean_queue_len"],
            }
        )

    # Exp 2a-Theory: Three-regime Fixed-State Q x p_e x p_resolve sweep on N=360 (90 Q4 items)
    exp2_fixed_state_theory = run_fixed_state_evoi_sweep(test_prim, prim_calib)

    # Exp 2a-Cost: c_ask sensitivity sweep at Cap=2 on N=360
    cask_methods = [
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
    ]
    exp2_cask_sweep = []
    for ca in [0.5, 1.5, 2.5, 3.0, 5.0]:
        cr = eval_methods_multi_seed(
            cask_methods,
            test_prim,
            prim_calib,
            n_seeds=5,
            capacity=2,
            drain_interval=2,
            drift_mode="sudden",
            c_ask=ca,
        )
        rm = {r["method"]: r for r in cr["rows"]}
        exp2_cask_sweep.append({
            "c_ask": ca,
            "methods": {
                m: {
                    "mean_utility": rm[m]["mean_utility"],
                    "acc_pct": rm[m]["acc_pct"],
                    "ask_rate_pct": rm[m]["ask_rate_pct"],
                    "q4_ask_rate_pct": rm[m]["q4_ask_rate_pct"],
                }
                for m in cask_methods
            },
        })

    # Exp 2b/c: Drift modes + Multi-specialist Senior/Junior Pool + Asymmetric Risk (N=360)
    stress_cfgs = [
        ("Sudden Drift", dict(drift_mode="sudden", asymmetric_risk=False)),
        ("Gradual Drift", dict(drift_mode="gradual", asymmetric_risk=False)),
        ("Senior+Junior Pool", dict(drift_mode="multi_pool", asymmetric_risk=False)),
        ("Asymmetric Risk", dict(drift_mode="sudden", asymmetric_risk=True)),
    ]
    stress_methods = [
        "Direct-ACT",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
    ]
    exp2_stress = {}
    for sname, kw in stress_cfgs:
        sr = eval_methods_multi_seed(
            stress_methods,
            test_prim,
            prim_calib,
            n_seeds=5,
            capacity=2,
            drain_interval=2,
            **kw,
        )
        rm = {r["method"]: r["mean_utility"] for r in sr["rows"]}
        exp2_stress[sname] = rm

    # Exp 3: Cross-model Generalization
    exp3_cross = []
    cross_methods = [
        "Direct-ACT",
        "Entropy-ASK (Proxy)",
        "Consensus-DEFER (Proxy)",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
    ]
    for m in ALL_MODELS:
        md = model_data[m]
        cr = eval_methods_multi_seed(
            cross_methods,
            md["test"],
            md["calib"],
            n_seeds=5,
            capacity=2,
            drain_interval=2,
            drift_mode="sudden",
        )
        rm = {r["method"]: r for r in cr["rows"]}
        exp3_cross.append(
            {
                "model_id": m,
                "short_name": m.split("/")[-1],
                "n_test": len(md["test"]),
                "direct_mean_util": rm["Direct-ACT"]["mean_utility"],
                "direct_acc": rm["Direct-ACT"]["acc_pct"],
                "entropy_mean_util": rm["Entropy-ASK (Proxy)"]["mean_utility"],
                "entropy_acc": rm["Entropy-ASK (Proxy)"]["acc_pct"],
                "consensus_mean_util": rm["Consensus-DEFER (Proxy)"]["mean_utility"],
                "consensus_acc": rm["Consensus-DEFER (Proxy)"]["acc_pct"],
                "l2d_mean_util": rm["Congested-L2D-Only"]["mean_utility"],
                "l2d_acc": rm["Congested-L2D-Only"]["acc_pct"],
                "naive_mean_util": rm["Naive-Cascade"]["mean_utility"],
                "naive_acc": rm["Naive-Cascade"]["acc_pct"],
                "greedy_mean_util": rm["Joint-Greedy"]["mean_utility"],
                "greedy_acc": rm["Joint-Greedy"]["acc_pct"],
                "greedy_naor_mean_util": rm["Joint-Greedy+Naor"]["mean_utility"],
                "greedy_naor_acc": rm["Joint-Greedy+Naor"]["acc_pct"],
                "cascade_mean_util": rm["Calibrated-Cascade"]["mean_utility"],
                "cascade_acc": rm["Calibrated-Cascade"]["acc_pct"],
                "cascade_naor_mean_util": rm["Calibrated-Cascade+Naor"]["mean_utility"],
                "cascade_naor_acc": rm["Calibrated-Cascade+Naor"]["acc_pct"],
                "tri_mean_util": rm["Tri-VoI-L2D"]["mean_utility"],
                "tri_acc": rm["Tri-VoI-L2D"]["acc_pct"],
            }
        )

    # Exp 4: Component Ablation on N=360 (both c_ask=1.5 and c_ask=2.5)
    exp4_ablation = eval_methods_multi_seed(
        ABLATIONS, test_prim, prim_calib, n_seeds=5, capacity=2, drain_interval=2, drift_mode="sudden", c_ask=1.5
    )
    exp4_ablation_c25 = eval_methods_multi_seed(
        ABLATIONS, test_prim, prim_calib, n_seeds=5, capacity=2, drain_interval=2, drift_mode="sudden", c_ask=2.5
    )

    all_out = {
        "mode": "DRY_RUN" if args.dry_run else "LIVE_SILICONFLOW_4MODELS_PHASE2_N360",
        "cal_gate_primary": cal_diag,
        "pre_exp_entropy_collapse": pre_exp,
        "exp1_main": exp1_main,
        "exp1_main_c25": exp1_main_c25,
        "exp1_episodes_c15": exp1_episodes_c15,
        "exp1_episodes_c25": exp1_episodes_c25,
        "exp1_stream_stability": exp1_stream_stability,
        "exp2_congestion_sweep": exp2_sweep,
        "exp2_fixed_state_theory": exp2_fixed_state_theory,
        "exp2_cask_sweep": exp2_cask_sweep,
        "exp2_stress_summary": exp2_stress,
        "exp3_cross_model": exp3_cross,
        "exp4_ablation": exp4_ablation,
        "exp4_ablation_c25": exp4_ablation_c25,
    }
    figs = generate_figures(all_out)
    all_out["figures"] = figs
    write_outputs(all_out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
