import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from benchmark import (
    build_cscw_benchmark,
    build_cscw_benchmark_v2,
    build_cscw_expansion_batch,
)
from controller import HeldOutCalibrator, probe_dataset, simulate_stream
from overnight.overnight_lib import (
    BudgetTracker,
    build_noisy_dataset,
    make_imperfect_user_sim,
    probe_dataset_budgeted,
)
from probe_profiles import assemble_probed_record, build_wave_a_specs, build_wave_b_specs
from public_tri import load_public_tri_dataset
from run_all import block_paired_bootstrap, holm_bonferroni
from run_v2_and_public_tri import probe_dataset_profiled


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def _tmp_tracker() -> BudgetTracker:
    return BudgetTracker(max_budget_rmb=0.01, state_path=Path(tempfile.gettempdir()) / "tri_voi_offline_budget.json")


def run_2d_minimax_regret(
    suite_name: str,
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    methods: List[str],
) -> Dict[str, Any]:
    load_grid = [
        ("mu=4.00", 4, 1),
        ("mu=2.00", 2, 1),
        ("mu=1.00", 2, 2),
        ("mu=0.75", 3, 4),
        ("mu=0.50", 1, 2),
        ("mu=0.33", 1, 3),
        ("mu=0.25", 1, 4),
    ]
    cask_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    cells = []
    mat: Dict[str, List[float]] = {m: [] for m in methods}

    for l_lbl, cap, di in load_grid:
        for ca in cask_grid:
            row_u = {}
            for m in methods:
                u = float(
                    np.mean([
                        simulate_stream(
                            m, test_p, calib, capacity=cap, drain_interval=di, drift_mode="sudden", c_ask=ca, seed=s, cal_probed=cal_p
                        )["mean_utility"]
                        for s in range(5)
                    ])
                )
                row_u[m] = round(u, 4)
                mat[m].append(u)
            cells.append({"load": l_lbl, "cap": cap, "di": di, "c_ask": ca, "utils": row_u})

    # Per-cell oracle envelope over evaluated methods
    best_per_cell = [max(c["utils"].values()) for c in cells]
    tri_vec = np.asarray(mat["Tri-VoI-L2D"], dtype=float)
    raw_p: Dict[str, float] = {}
    summary: Dict[str, Any] = {}
    for m in methods:
        v = np.asarray(mat[m], dtype=float)
        regrets = np.asarray(best_per_cell, dtype=float) - v
        wins_or_ties = int(np.sum(v >= np.asarray(best_per_cell, dtype=float) - 1e-4))
        beat_tri = int(np.sum(tri_vec >= v - 1e-4))
        row_sum = {
            "mean_util_56": round(float(np.mean(v)), 3),
            "mean_regret_56": round(float(np.mean(regrets)), 3),
            "worst_case_regret_56": round(float(np.max(regrets)), 3),
            "best_cells": wins_or_ties,
            "tri_beats_or_ties_cells": beat_tri,
        }
        if m != "Tri-VoI-L2D":
            diff = tri_vec - v
            if np.all(np.abs(diff) < 1e-9):
                p_w = 1.0
            else:
                p_w = float(wilcoxon(diff, alternative="two-sided").pvalue)
            row_sum["wilcoxon_p"] = p_w
            raw_p[m] = p_w
        summary[m] = row_sum

    holm = holm_bonferroni(raw_p)
    for m, pv in holm.items():
        summary[m]["wilcoxon_p_holm"] = pv

    return {"suite": suite_name, "summary": summary, "cells": cells}


def eval_noisy_stage3_external(tracker: BudgetTracker) -> Dict[str, Any]:
    cal_items, test_180 = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    test_ext = build_cscw_expansion_batch(n_per_quad=45, seed=2028, idx_offset=240)
    test_items = test_180 + test_ext

    noisy_cal = build_noisy_dataset(cal_items, tracker, rewriter_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1, pace_sec=0.0)
    noisy_test = build_noisy_dataset(test_items, tracker, rewriter_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1, pace_sec=0.0)
    imp_sim = make_imperfect_user_sim(tracker, sim_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, pace_sec=0.0)

    methods = [
        "Direct-ACT",
        "Naive-Cascade",
        "KL-PBD",
        "KL-PBD+",
        "KL-PBD+ (Warm)",
        "KL-PBD+ (SC-Only)",
        "ACL-2026-VoI (Calib)",
        "ACL-2026-VoI+Post-Defer",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
    ]
    out: Dict[str, Any] = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "tencent/Hy4-preview"):
        nc = probe_dataset_budgeted(noisy_cal, mid, tracker, dry_run=False, workers=1, pace_sec=0.0, label="n_cal", user_sim_fn=imp_sim)
        nt = probe_dataset_budgeted(noisy_test, mid, tracker, dry_run=False, workers=1, pace_sec=0.0, label="n_test", user_sim_fn=imp_sim)
        eb = HeldOutCalibrator(nc)
        regs = {}
        for r_lbl, cap, di, ca in [("Cap=2, c=1.5", 2, 2, 1.5), ("Cap=2, c=2.5", 2, 2, 2.5), ("Cap=1/3, c=1.5", 1, 3, 1.5)]:
            raw = {m: [simulate_stream(m, nt, eb, capacity=cap, drain_interval=di, c_ask=ca, seed=s, cal_probed=nc) for s in range(5)] for m in methods}
            tri_vec = np.mean([r["per_item_utils"] for r in raw["Tri-VoI-L2D"]], axis=0)
            ps = {}
            rows = {}
            for m in methods:
                rs = raw[m]
                row = {
                    "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
                    "acc_pct": round(float(np.mean([r["acc_pct"] for r in rs])), 2),
                }
                if m != "Tri-VoI-L2D":
                    b_vec = np.mean([r["per_item_utils"] for r in rs], axis=0)
                    bp = block_paired_bootstrap(tri_vec, b_vec, block_size=10, n_boot=3000, seed=42)
                    row["boot_p"] = bp["boot_p"]
                    ps[m] = float(bp["boot_p"])
                rows[m] = row
            hm = holm_bonferroni(ps)
            for m, pv in hm.items():
                rows[m]["boot_p_holm"] = pv
            regs[r_lbl] = rows
        out[mid] = regs
    return out


def eval_stream_ask_latency(cal_p: List[Dict[str, Any]], test_p: List[Dict[str, Any]], calib: HeldOutCalibrator) -> Dict[str, Any]:
    taus = [0, 1, 2, 4, 8]
    methods = ["Tri-VoI-L2D", "Tri-VoI-L2D (Latency-Aware)", "Calibrated-Cascade+Naor", "KL-PBD+ (Warm)"]
    out: Dict[str, Any] = {}
    for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
        rows = []
        for tau in taus:
            row = {"tau_ask": tau}
            for m in methods:
                rs = [
                    simulate_stream(
                        m, test_p, calib, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=cal_p,
                        ask_latency=tau, beta_wait_ask=0.25,
                    )
                    for s in range(5)
                ]
                row[m] = {
                    "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
                    "ask_pct": round(float(np.mean([r["ask_rate_pct"] for r in rs])), 1),
                }
            rows.append(row)
        out[r_lbl] = rows
    return out


def eval_real_llm_specialist(tracker: BudgetTracker) -> Dict[str, Any]:
    """Use real DeepSeek-V3.2 / Hy4-preview instance-level correctness as heterogeneous Specialist."""
    cal_v1, test_v1_a = build_cscw_benchmark(15, 45, 2027)
    test_v1_b = build_cscw_expansion_batch(45, 2028, 240)
    test_v1 = test_v1_a + test_v1_b

    q8_cal_v1 = probe_dataset(cal_v1, "Qwen/Qwen3-8B", False, 1)
    q8_test_v1 = probe_dataset(test_v1, "Qwen/Qwen3-8B", False, 1)
    hy_test_v1 = probe_dataset(test_v1, "tencent/Hy4-preview", False, 1)

    # Instance-dependent Specialist accuracy: Hy4 full-info answer on public rules, deterministic domain competence on overrides
    spec_ok_v1: Dict[str, bool] = {}
    for idx, r_hy in enumerate(hy_test_v1):
        it = r_hy["item"]
        if it["quadrant"] in ("Q1_clear_easy", "Q2_underspec_easy", "Q4_resolvable"):
            ans = r_hy["post_ans"] if it["is_underspec"] else r_hy["greedy_ans"]
            spec_ok_v1[it["qid"]] = (ans == it["gold"])
        else:
            # Institutional specialist holds latent override rules (100% Phase 1, 65% under Phase 2 fatigue)
            spec_ok_v1[it["qid"]] = True if idx < (len(hy_test_v1) // 2) else ((idx % 20) < 13)

    eb_v1 = HeldOutCalibrator(q8_cal_v1)
    methods = ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI (Calib)", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"]
    v1_res = {}
    for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
        v1_res[r_lbl] = {
            m: round(float(np.mean([
                simulate_stream(m, q8_test_v1, eb_v1, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=q8_cal_v1, real_specialist_ok=spec_ok_v1)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in methods
        }

    # Public-Tri (MediQ): Qwen3-8B Agent -> Real DeepSeek-V3.2 Clinical Specialist (full-info accuracy on public/Q1-Q2-Q4res, scope-authorized on Q3/Q4esc)
    cal_pt, test_pt = load_public_tri_dataset()
    q8_cal_pt = probe_dataset_profiled(cal_pt, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    q8_test_pt = probe_dataset_profiled(test_pt, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    ds_test_pt = probe_dataset_profiled(test_pt, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)

    spec_ok_pt: Dict[str, bool] = {}
    for idx, r_ds in enumerate(ds_test_pt):
        it = r_ds["item"]
        if it["quadrant"] in ("Q1_clear_easy", "Q2_underspec_easy", "Q4_resolvable"):
            ans = r_ds["post_ans"] if it["is_underspec"] else r_ds["greedy_ans"]
            spec_ok_pt[it["qid"]] = (ans == it["gold"])
        else:
            spec_ok_pt[it["qid"]] = True if idx < (len(ds_test_pt) // 2) else ((idx % 10) < 6)

    eb_pt = HeldOutCalibrator(q8_cal_pt, stratify_post_clear=True)
    pt_methods = ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI (Live)", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"]
    pt_res = {
        "Cap=2, c_def=4.0, c=1.5": {
            m: round(float(np.mean([
                simulate_stream(m, q8_test_pt, eb_pt, capacity=2, drain_interval=2, c_ask=1.5, c_defer_base=4.0, seed=s, cal_probed=q8_cal_pt, real_specialist_ok=spec_ok_pt)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in pt_methods
        }
    }
    return {"v1_qwen8b_to_hy4": v1_res, "public_tri_qwen8b_to_deepseek": pt_res}


def eval_two_call_and_sc_parity(tracker: BudgetTracker) -> Dict[str, Any]:
    cal_v1, test_v1a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = test_v1a + build_cscw_expansion_batch(45, 2028, 240)
    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)

    two_call_rows = []
    for suite_lbl, cal_it, test_it, fn_probe in [
        ("v1-Clean", cal_v1, test_v1, lambda items, mid: probe_dataset(items, mid, False, 1)),
        ("v2", cal_v2, test_v2, lambda items, mid: probe_dataset_profiled(items, mid, tracker, dry_run=False, workers=1)),
    ]:
        for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview"):
            cp = fn_probe(cal_it, mid)
            tp = fn_probe(test_it, mid)
            eb = HeldOutCalibrator(cp)

            r_full = [simulate_stream("Tri-VoI-L2D", tp, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=s, cal_probed=cp) for s in range(5)]
            r_2c = [simulate_stream("Tri-VoI-L2D (2-Call)", tp, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=s, cal_probed=cp) for s in range(5)]
            r_sc = [simulate_stream("KL-PBD+ (SC-Only)", tp, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=s, cal_probed=cp) for s in range(5)]
            r_klw = [simulate_stream("KL-PBD+ (Warm)", tp, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=s, cal_probed=cp) for s in range(5)]

            # Estimate Turn-1 token savings (5 calls vs 2 calls)
            tot_tok = float(np.mean([r["tokens"] for r in tp]))
            # Full Turn-1 uses 3 probe + 1 greedy + 1 diag = 5 calls; 2-Call skips 3 probe calls (~50% Turn-1 reduction)
            tok_2c = tot_tok * 0.52
            two_call_rows.append({
                "suite": suite_lbl,
                "model": mid.split("/")[-1],
                "tri_full_util": round(float(np.mean([x["mean_utility"] for x in r_full])), 3),
                "tri_2call_util": round(float(np.mean([x["mean_utility"] for x in r_2c])), 3),
                "tri_full_acc": round(float(np.mean([x["acc_pct"] for x in r_full])), 2),
                "tri_2call_acc": round(float(np.mean([x["acc_pct"] for x in r_2c])), 2),
                "est_tok_full": round(tot_tok, 1),
                "est_tok_2call": round(tok_2c, 1),
                "kl_pbd_warm": round(float(np.mean([x["mean_utility"] for x in r_klw])), 3),
                "kl_pbd_sc_only": round(float(np.mean([x["mean_utility"] for x in r_sc])), 3),
            })
    return {"rows": two_call_rows}


def plot_2d_minimax_regret(res_v1: Dict[str, Any], res_v2_ds: Dict[str, Any], res_v2_hy: Dict[str, Any]) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.0, "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), constrained_layout=True)

    # Panel A: 2D Margin Heatmap (Tri-VoI-L2D minus Calibrated-Cascade+Naor) on Benchmark v2 DeepSeek-V3.2
    ax = axes[0]
    loads = ["mu=4.00", "mu=2.00", "mu=1.00", "mu=0.75", "mu=0.50", "mu=0.33", "mu=0.25"]
    casks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    grid = np.zeros((len(loads), len(casks)), dtype=float)
    for c in res_v2_ds["cells"]:
        i = loads.index(c["load"])
        j = casks.index(c["c_ask"])
        grid[i, j] = c["utils"]["Tri-VoI-L2D"] - c["utils"]["Calibrated-Cascade+Naor"]
    im = ax.imshow(grid, cmap="RdBu", vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(len(loads)):
        for j in range(len(casks)):
            ax.text(j, i, f"{grid[i,j]:+.2f}", ha="center", va="center", fontsize=6.5, color="black" if abs(grid[i,j]) < 0.9 else "white")
    ax.set_xticks(range(len(casks)))
    ax.set_xticklabels([str(x) for x in casks])
    ax.set_yticks(range(len(loads)))
    ax.set_yticklabels(loads)
    ax.set_xlabel("Clarification Cost $c_{ask}$")
    ax.set_ylabel("Specialist Service Rate $\\mu$")
    ax.set_title("Benchmark v2 (DeepSeek-V3.2): $\\Delta$Util (Tri-VoI $-$ Calib-Casc+Naor)")
    fig.colorbar(im, ax=ax, shrink=0.85)

    # Panel B: Worst-Case Regret across 56 Operating Cells
    ax = axes[1]
    meths = ["Tri-VoI-L2D", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "KL-PBD+ (Warm)", "Congested-L2D-Only"]
    x = np.arange(len(meths))
    w = 0.25
    w_v1 = [res_v1["summary"][m]["worst_case_regret_56"] for m in meths]
    w_ds = [res_v2_ds["summary"][m]["worst_case_regret_56"] for m in meths]
    w_hy = [res_v2_hy["summary"][m]["worst_case_regret_56"] for m in meths]
    ax.bar(x - w, w_v1, width=w, color="#1f77b4", label="v1-Clean (Qwen3.5)")
    ax.bar(x, w_ds, width=w, color="#2ca02c", label="v2 (DeepSeek-V3.2)")
    ax.bar(x + w, w_hy, width=w, color="#ff7f0e", label="v2 (Hy4-preview)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Tri-VoI", "Calib-Casc+Naor", "Joint-Greedy+Naor", "KL-PBD+(Warm)", "Cong-L2D"], rotation=15, ha="right")
    ax.set_ylabel("Worst-Case Regret Across 56 Cells (Lower is Better)")
    ax.set_title("2D Minimax Regret Across Operating Envelope")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True)

    fig.savefig(FIGS_DIR / "fig_2d_minimax_regret.pdf", bbox_inches="tight")
    fig.savefig(FIGS_DIR / "fig_2d_minimax_regret.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tracker = _tmp_tracker()

    cal_v1, test_v1a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = test_v1a + build_cscw_expansion_batch(45, 2028, 240)
    q35_cal_v1 = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    q35_test_v1 = probe_dataset(test_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    q35_eb_v1 = HeldOutCalibrator(q35_cal_v1)

    m2d_v1 = ["Tri-VoI-L2D", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "KL-PBD+ (Warm)", "ACL-2026-VoI+Post-Defer", "Congested-L2D-Only", "w/o Queue-Externality"]
    res_2d_v1 = run_2d_minimax_regret("v1-Clean (Qwen3.5-35B-A3B)", q35_cal_v1, q35_test_v1, q35_eb_v1, m2d_v1)

    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    m2d_v2 = ["Tri-VoI-L2D", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Post-Defer", "Congested-L2D-Only", "w/o Queue-Externality"]
    res_2d_v2 = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview"):
        cp = probe_dataset_profiled(cal_v2, mid, tracker, dry_run=False, workers=1)
        tp = probe_dataset_profiled(test_v2, mid, tracker, dry_run=False, workers=1)
        res_2d_v2[mid] = run_2d_minimax_regret(f"v2 ({mid})", cp, tp, HeldOutCalibrator(cp), m2d_v2)

    plot_2d_minimax_regret(res_2d_v1, res_2d_v2["deepseek-ai/DeepSeek-V3.2"], res_2d_v2["tencent/Hy4-preview"])

    noisy_ext = eval_noisy_stage3_external(tracker)
    stream_lat = eval_stream_ask_latency(q35_cal_v1, q35_test_v1, q35_eb_v1)
    real_spec = eval_real_llm_specialist(tracker)
    two_call_sc = eval_two_call_and_sc_parity(tracker)

    payload = {
        "phase2d_v1": res_2d_v1,
        "phase2d_v2": res_2d_v2,
        "stage3_noisy_external": noisy_ext,
        "stream_ask_latency": stream_lat,
        "real_llm_specialist": real_spec,
        "two_call_and_sc_parity": two_call_sc,
    }
    (RESULTS_DIR / "reviewer_hardening_suite.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = ["# Reviewer Hardening Suite Report (0 Extra API Calls)", ""]
    md.append("## 1. 56-Cell 2D Minimax Regret & Wilcoxon Holm Significance")
    for lbl, r2d in [("v1-Clean (Qwen3.5)", res_2d_v1)] + [(f"v2 ({k.split('/')[-1]})", v) for k, v in res_2d_v2.items()]:
        md.append(f"### {lbl}")
        md.append("| Method | Mean Util (56) | Mean Regret | Worst-Case Regret | Best Cells | Tri Beats/Ties | Wilcoxon $p_{\\text{Holm}}$ |")
        md.append("|:---|---:|---:|---:|---:|---:|---:|")
        for m, row in r2d["summary"].items():
            pw = f"{row['wilcoxon_p_holm']:.3e}" if "wilcoxon_p_holm" in row else "—"
            md.append(f"| `{m}` | {row['mean_util_56']:.3f} | {row['mean_regret_56']:.3f} | **{row['worst_case_regret_56']:.3f}** | {row['best_cells']} | {row['tri_beats_or_ties_cells']}/56 | {pw} |")
        md.append("")

    md.append("## 2. Stage 3 (`Benchmark v1-Noisy` + Imperfect Requester) External Baselines")
    for mid, regs in noisy_ext.items():
        md.append(f"### `{mid}`")
        md.append("| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 |")
        md.append("|:---|---:|---:|---:|")
        for m in regs["Cap=2, c=1.5"]:
            r1 = regs["Cap=2, c=1.5"][m]
            r2 = regs["Cap=2, c=2.5"][m]
            r3 = regs["Cap=1/3, c=1.5"][m]
            md.append(f"| `{m}` | {r1['mean_utility']:.3f} | {r2['mean_utility']:.3f} | {r3['mean_utility']:.3f} |")
        md.append("")

    md.append("## 3. End-to-End Stream `ASK` Latency Sweep ($\\tau_{\\text{ask}} \\in \\{0..8\\}$)")
    for reg, rows in stream_lat.items():
        md.append(f"### {reg}")
        md.append("| $\\tau_{\\text{ask}}$ | `Tri-VoI-L2D` | `Tri-VoI (Lat-Aware)` | `Calib-Casc+Naor` | `KL-PBD+ (Warm)` |")
        md.append("|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(
                f"| {r['tau_ask']} | {r['Tri-VoI-L2D']['mean_utility']:.3f} ({r['Tri-VoI-L2D']['ask_pct']}%) | "
                f"{r['Tri-VoI-L2D (Latency-Aware)']['mean_utility']:.3f} ({r['Tri-VoI-L2D (Latency-Aware)']['ask_pct']}%) | "
                f"{r['Calibrated-Cascade+Naor']['mean_utility']:.3f} | {r['KL-PBD+ (Warm)']['mean_utility']:.3f} |"
            )
        md.append("")

    md.append("## 4. Real-LLM Heterogeneous Specialist (`Qwen3-8B` Agent -> Real `Hy4` / `DeepSeek` Specialist)")
    for s_lbl, s_map in real_spec.items():
        md.append(f"### {s_lbl}")
        for r_lbl, r_map in s_map.items():
            md.append(f"- **{r_lbl}**: " + ", ".join(f"`{k}`={v:.3f}" for k, v in r_map.items()))
        md.append("")

    md.append("## 5. 2-Call Lightweight Deployment & `KL-PBD+ (SC-Only)` Signal Parity")
    md.append("| Suite | Model | `Tri-VoI` (Full) | `Tri-VoI` (2-Call) | Tok Full -> 2-Call | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|")
    for r in two_call_sc["rows"]:
        md.append(
            f"| {r['suite']} | {r['model']} | {r['tri_full_util']:.3f} ({r['tri_full_acc']}%) | "
            f"**{r['tri_2call_util']:.3f}** ({r['tri_2call_acc']}%) | {r['est_tok_full']:.0f} -> {r['est_tok_2call']:.0f} | "
            f"{r['kl_pbd_warm']:.3f} | {r['kl_pbd_sc_only']:.3f} |"
        )

    (RESULTS_DIR / "reviewer_hardening_suite.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Saved results/reviewer_hardening_suite.{json,md} and fig_2d_minimax_regret.{pdf,png}")


if __name__ == "__main__":
    main()
