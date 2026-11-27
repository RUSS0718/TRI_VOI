import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import (
    build_cscw_benchmark,
    build_cscw_benchmark_v2,
    build_cscw_expansion_batch,
    split_stratified_episodes,
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
from run_all import block_paired_bootstrap, eval_independent_episodes, holm_bonferroni
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
    c_defer_base: float = 2.0,
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
                            m, test_p, calib, capacity=cap, drain_interval=di, drift_mode="sudden", c_ask=ca, c_defer_base=c_defer_base, seed=s, cal_probed=cal_p
                        )["mean_utility"]
                        for s in range(5)
                    ])
                )
                row_u[m] = round(u, 4)
                mat[m].append(u)
            cells.append({"load": l_lbl, "cap": cap, "di": di, "c_ask": ca, "utils": row_u})

    # Descriptive 56-cell summary ONLY (no i.i.d. Wilcoxon test across hyperparameter cells sharing the same items)
    best_per_cell = [max(c["utils"].values()) for c in cells]
    tri_vec = np.asarray(mat["Tri-VoI-L2D"], dtype=float)
    summary: Dict[str, Any] = {}
    for m in methods:
        v = np.asarray(mat[m], dtype=float)
        regrets = np.asarray(best_per_cell, dtype=float) - v
        wins_or_ties = int(np.sum(v >= np.asarray(best_per_cell, dtype=float) - 1e-4))
        tri_wins = int(np.sum(tri_vec > v + 1e-4))
        tri_ties = int(np.sum(np.abs(tri_vec - v) <= 1e-4))
        tri_losses = int(np.sum(v > tri_vec + 1e-4))
        delta_vs_tri = float(np.mean(v - tri_vec))  # >0 means baseline beats Tri-VoI-L2D
        summary[m] = {
            "mean_util_56": round(float(np.mean(v)), 3),
            "mean_delta_vs_tri": round(delta_vs_tri, 3),
            "mean_regret_56": round(float(np.mean(regrets)), 3),
            "worst_case_regret_56": round(float(np.max(regrets)), 3),
            "best_cells": wins_or_ties,
            "tri_wins": tri_wins,
            "tri_ties": tri_ties,
            "tri_losses": tri_losses,
        }

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
    """Compare genuine full-quadrant Real-LLM Clinical Specialists (Public-Tri) vs. Semi-Synthetic Specialists (v1/v2)."""
    cal_pt, test_pt = load_public_tri_dataset()
    q35_cal_pt = probe_dataset_profiled(cal_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    q35_test_pt = probe_dataset_profiled(test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    q8_cal_pt = probe_dataset_profiled(cal_pt, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    q8_test_pt = probe_dataset_profiled(test_pt, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    ds_test_pt = probe_dataset_profiled(test_pt, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)

    # Genuine Real-LLM correctness across ALL 5 quadrants (ZERO hardcoded percentages)
    ds_by_qid = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in ds_test_pt
    }
    q8_by_qid = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in q8_test_pt
    }
    spec_ok_pt_ds = {it["item"]["qid"]: ds_by_qid[it["item"]["qid"]] for it in ds_test_pt}
    # Genuine Real-LLM handoff drift: DeepSeek-V3.2 (first half) -> Qwen3-8B (second half) across all quadrants
    spec_ok_pt_shift = {
        r["item"]["qid"]: (ds_by_qid[r["item"]["qid"]] if idx < (len(ds_test_pt) // 2) else q8_by_qid[r["item"]["qid"]])
        for idx, r in enumerate(ds_test_pt)
    }

    pt_methods = [
        "Direct-ACT",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "Calibrated-Cascade+Naor",
        "Joint-Greedy+Naor",
        "Tri-VoI-L2D",
    ]
    eb_q35_pt = HeldOutCalibrator(q35_cal_pt, stratify_post_clear=True)
    eb_q8_pt = HeldOutCalibrator(q8_cal_pt, stratify_post_clear=True)

    pt_q35_ds = {}
    pt_q35_shift = {}
    for r_lbl, cap, di, c_def in [
        ("Cap=2, c_def=2.0, c=1.5", 2, 2, 2.0),
        ("Cap=2, c_def=4.0, c=1.5", 2, 2, 4.0),
        ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 2.0),
    ]:
        pt_q35_ds[r_lbl] = {
            m: round(float(np.mean([
                simulate_stream(m, q35_test_pt, eb_q35_pt, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=s, cal_probed=q35_cal_pt, real_specialist_ok=spec_ok_pt_ds)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in pt_methods
        }
        pt_q35_shift[r_lbl] = {
            m: round(float(np.mean([
                simulate_stream(m, q35_test_pt, eb_q35_pt, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=s, cal_probed=q35_cal_pt, real_specialist_ok=spec_ok_pt_shift)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in pt_methods
        }

    pt_q8_ds = {
        r_lbl: {
            m: round(float(np.mean([
                simulate_stream(m, q8_test_pt, eb_q8_pt, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=s, cal_probed=q8_cal_pt, real_specialist_ok=spec_ok_pt_ds)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in pt_methods
        }
        for r_lbl, cap, di, c_def in [("Cap=2, c_def=4.0, c=1.5", 2, 2, 4.0), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 2.0)]
    }

    # Semi-Synthetic v1/v2 Specialists: Real Hy4 full-info on public items, simulated 100%/65% on confidential overrides
    cal_v1, test_v1_a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = test_v1_a + build_cscw_expansion_batch(45, 2028, 240)
    hy_test_v1 = probe_dataset(test_v1, "tencent/Hy4-preview", False, 1)
    spec_ok_v1: Dict[str, bool] = {}
    for idx, r_hy in enumerate(hy_test_v1):
        it = r_hy["item"]
        if it["quadrant"] in ("Q1_clear_easy", "Q2_underspec_easy", "Q4_resolvable"):
            ans = r_hy["post_ans"] if it["is_underspec"] else r_hy["greedy_ans"]
            spec_ok_v1[it["qid"]] = (ans == it["gold"])
        else:
            spec_ok_v1[it["qid"]] = True if idx < (len(hy_test_v1) // 2) else ((idx % 20) < 13)

    methods_v1 = ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI+Full-Defer+Naor", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"]
    q8_cal_v1 = probe_dataset(cal_v1, "Qwen/Qwen3-8B", False, 1)
    q8_test_v1 = probe_dataset(test_v1, "Qwen/Qwen3-8B", False, 1)
    eb_q8_v1 = HeldOutCalibrator(q8_cal_v1)
    q35_cal_v1 = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    q35_test_v1 = probe_dataset(test_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    eb_q35_v1 = HeldOutCalibrator(q35_cal_v1)

    v1_q35_res = {
        r_lbl: {
            m: round(float(np.mean([
                simulate_stream(m, q35_test_v1, eb_q35_v1, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=q35_cal_v1, real_specialist_ok=spec_ok_v1)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in methods_v1
        }
        for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]
    }
    v1_q8_res = {
        r_lbl: {
            m: round(float(np.mean([
                simulate_stream(m, q8_test_v1, eb_q8_v1, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=q8_cal_v1, real_specialist_ok=spec_ok_v1)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in methods_v1
        }
        for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]
    }

    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    q35_cal_v2 = probe_dataset_profiled(cal_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    q35_test_v2 = probe_dataset_profiled(test_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    hy_test_v2 = probe_dataset_profiled(test_v2, "tencent/Hy4-preview", tracker, dry_run=False, workers=1)
    spec_ok_v2: Dict[str, bool] = {}
    for idx, r_hy in enumerate(hy_test_v2):
        it = r_hy["item"]
        if it["quadrant"] in ("Q1_clear_easy", "Q2_underspec_easy", "Q4_resolvable"):
            ans = r_hy["post_ans"] if it["is_underspec"] else r_hy["greedy_ans"]
            spec_ok_v2[it["qid"]] = (ans == it["gold"])
        else:
            spec_ok_v2[it["qid"]] = True if idx < (len(hy_test_v2) // 2) else ((idx % 20) < 13)

    eb_q35_v2 = HeldOutCalibrator(q35_cal_v2)
    methods_v2 = ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Full-Defer+Naor", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"]
    v2_q35_res = {
        r_lbl: {
            m: round(float(np.mean([
                simulate_stream(m, q35_test_v2, eb_q35_v2, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=q35_cal_v2, real_specialist_ok=spec_ok_v2)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in methods_v2
        }
        for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]
    }

    return {
        "real_public_tri_qwen35_to_deepseek_stationary": pt_q35_ds,
        "real_public_tri_qwen35_to_ds_to_qwen8b_shift": pt_q35_shift,
        "real_public_tri_qwen8b_to_deepseek_stationary": pt_q8_ds,
        "semi_synthetic_v1_qwen35_to_hy4": v1_q35_res,
        "semi_synthetic_v2_qwen35_to_hy4": v2_q35_res,
        "semi_synthetic_v1_qwen8b_to_hy4": v1_q8_res,
    }


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

            # Exact SQLite measured token accounting (turn-1 and end-to-end realized)
            t1_full = float(np.mean([
                sum(r["tag_tokens"][t] for t in ("probe_s0", "probe_s1", "probe_s2", "greedy_act", "diag_init"))
                for r in tp
            ]))
            t1_2c = float(np.mean([r["tag_tokens"]["greedy_act"] + r["tag_tokens"]["diag_init"] for r in tp]))
            real_tok_full = float(np.mean([x["mean_realized_tokens"] for x in r_full]))
            real_tok_2c = float(np.mean([x["mean_realized_tokens"] for x in r_2c]))
            real_tok_klw = float(np.mean([x["mean_realized_tokens"] for x in r_klw]))
            real_tok_sc = float(np.mean([x["mean_realized_tokens"] for x in r_sc]))

            two_call_rows.append({
                "suite": suite_lbl,
                "model": mid.split("/")[-1],
                "tri_full_util": round(float(np.mean([x["mean_utility"] for x in r_full])), 3),
                "tri_2call_util": round(float(np.mean([x["mean_utility"] for x in r_2c])), 3),
                "tri_full_acc": round(float(np.mean([x["acc_pct"] for x in r_full])), 2),
                "tri_2call_acc": round(float(np.mean([x["acc_pct"] for x in r_2c])), 2),
                "t1_tok_full": round(t1_full, 1),
                "t1_tok_2call": round(t1_2c, 1),
                "realized_tok_full": round(real_tok_full, 1),
                "realized_tok_2call": round(real_tok_2c, 1),
                "tok_reduction_pct": round(100.0 * (real_tok_full - real_tok_2c) / max(1.0, real_tok_full), 1),
                "realized_tok_klw": round(real_tok_klw, 1),
                "realized_tok_sc": round(real_tok_sc, 1),
                "kl_pbd_warm": round(float(np.mean([x["mean_utility"] for x in r_klw])), 3),
                "kl_pbd_sc_only": round(float(np.mean([x["mean_utility"] for x in r_sc])), 3),
            })
    return {"rows": two_call_rows}


def eval_2x2_evoi_naor_decomposition(tracker: BudgetTracker) -> Dict[str, Any]:
    """Explicit 2x2 factorial separation (+-EVOI x +-Naor) and fair ACL-2026-VoI+Full-Defer+Naor."""
    cal_v1, test_v1a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = test_v1a + build_cscw_expansion_batch(45, 2028, 240)
    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    cal_pt, test_pt = load_public_tri_dataset()

    configs = [
        ("v1-Clean / Qwen3.5", "Qwen/Qwen3.5-35B-A3B", probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", False, 1), probe_dataset(test_v1, "Qwen/Qwen3.5-35B-A3B", False, 1), False, 2.0, False),
        ("v2 / Qwen3.5", "Qwen/Qwen3.5-35B-A3B", probe_dataset_profiled(cal_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, 2.0, False),
        ("v2 / DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2", probe_dataset_profiled(cal_v2, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v2, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), True, 2.0, False),
        ("v2 / Hy4-preview", "tencent/Hy4-preview", probe_dataset_profiled(cal_v2, "tencent/Hy4-preview", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v2, "tencent/Hy4-preview", tracker, dry_run=False, workers=1), True, 2.0, False),
        ("Public-Tri / Qwen3.5 (c_def=2)", "Qwen/Qwen3.5-35B-A3B", probe_dataset_profiled(cal_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, 2.0, True),
        ("Public-Tri / Qwen3.5 (c_def=4)", "Qwen/Qwen3.5-35B-A3B", probe_dataset_profiled(cal_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, 4.0, True),
        ("Public-Tri / DeepSeek (c_def=2)", "deepseek-ai/DeepSeek-V3.2", probe_dataset_profiled(cal_pt, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_pt, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), True, 2.0, True),
    ]
    rows = []
    for lbl, mid, cp, tp, use_live_acl, c_def, strat in configs:
        eb = HeldOutCalibrator(cp, stratify_post_clear=strat)
        acl_fd = "ACL-2026-VoI (Live)+Full-Defer+Naor" if use_live_acl else "ACL-2026-VoI+Full-Defer+Naor"
        meths = [
            "Joint-Greedy",
            "Joint-Greedy+Naor",
            "Calibrated-Cascade",
            "Calibrated-Cascade+Naor",
            acl_fd,
            "w/o Queue-Externality",
            "Tri-VoI-L2D",
            "Tri-VoI-L2D (Smooth-Naor)",
        ]
        for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
            vals = {
                m: round(float(np.mean([
                    simulate_stream(m, tp, eb, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=s, cal_probed=cp)["mean_utility"]
                    for s in range(5)
                ])), 3)
                for m in meths
            }
            rows.append({
                "suite_model": lbl,
                "regime": r_lbl,
                "c_defer": c_def,
                "vals": vals,
                "acl_full_defer_key": acl_fd,
                "naor_gain_tri": round(vals["Tri-VoI-L2D"] - vals["w/o Queue-Externality"], 3),
                "evoi_gain_over_cc_naor": round(vals["Tri-VoI-L2D"] - vals["Calibrated-Cascade+Naor"], 3),
                "evoi_gain_over_jg_naor": round(vals["Tri-VoI-L2D"] - vals["Joint-Greedy+Naor"], 3),
                "evoi_gain_over_acl_fd": round(vals["Tri-VoI-L2D"] - vals[acl_fd], 3),
            })
    return {"rows": rows}


def eval_repeated_resplit_and_episodes(tracker: BudgetTracker) -> Dict[str, Any]:
    """20 stratified random (cal, test) resplits capturing BOTH calibration & stream variance + 6-episode warm-fixed check."""
    cal_v1, test_v1a = build_cscw_benchmark(15, 45, 2027)
    pool_v1 = probe_dataset(cal_v1 + test_v1a + build_cscw_expansion_batch(45, 2028, 240), "Qwen/Qwen3.5-35B-A3B", False, 1)
    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    pool_v2_q35 = probe_dataset_profiled(cal_v2 + test_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    pool_v2_ds = probe_dataset_profiled(cal_v2 + test_v2, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)
    cal_pt, test_pt = load_public_tri_dataset()
    pool_pt_q35 = probe_dataset_profiled(cal_pt + test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    pool_pt_ds = probe_dataset_profiled(cal_pt + test_pt, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)

    suites = [
        ("v1-Clean (Qwen3.5)", pool_v1, 15, False, False, 2.0),
        ("v2 (Qwen3.5)", pool_v2_q35, 15, True, False, 2.0),
        ("v2 (DeepSeek-V3.2)", pool_v2_ds, 15, True, False, 2.0),
        ("Public-Tri (Qwen3.5)", pool_pt_q35, 15, True, True, 2.0),
        ("Public-Tri (DeepSeek-V3.2)", pool_pt_ds, 15, True, True, 2.0),
    ]
    resplit_out = []
    for s_lbl, full_pool, n_cal_pq, live_acl, strat, c_def in suites:
        by_q = defaultdict(list)
        for r in full_pool:
            by_q[r["item"]["quadrant"]].append(r)
        acl_fd = "ACL-2026-VoI (Live)+Full-Defer+Naor" if live_acl else "ACL-2026-VoI+Full-Defer+Naor"
        comp_meths = ["KL-PBD+ (Warm)", acl_fd, "Joint-Greedy+Naor", "Calibrated-Cascade+Naor", "Tri-VoI-L2D"]

        for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
            per_m: Dict[str, List[float]] = {m: [] for m in comp_meths}
            for rep in range(20):
                rr = random.Random(8100 + rep * 31)
                cal_s, test_s = [], []
                for qk in sorted(by_q.keys()):
                    bkt = list(by_q[qk])
                    rr.shuffle(bkt)
                    n_c = n_cal_pq if qk in ("Q1_clear_easy", "Q2_underspec_easy", "Q3_clear_hard") else (n_cal_pq // 2 if qk == "Q4_resolvable" else n_cal_pq - n_cal_pq // 2)
                    cal_s.extend(bkt[:n_c])
                    test_s.extend(bkt[n_c:])
                rr.shuffle(cal_s)
                rr.shuffle(test_s)
                eb = HeldOutCalibrator(cal_s, stratify_post_clear=strat)
                for m in comp_meths:
                    res = simulate_stream(m, test_s, eb, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=rep, cal_probed=cal_s)
                    per_m[m].append(res["mean_utility"])

            tri_arr = np.asarray(per_m["Tri-VoI-L2D"], dtype=float)
            m_stats = {}
            for m in comp_meths:
                arr = np.asarray(per_m[m], dtype=float)
                diff = tri_arr - arr
                m_stats[m] = {
                    "mean": round(float(np.mean(arr)), 3),
                    "std": round(float(np.std(arr, ddof=1)), 3),
                    "delta_tri_minus_base": round(float(np.mean(diff)), 3),
                    "delta_p5_p95": [round(float(np.percentile(diff, 5)), 3), round(float(np.percentile(diff, 95)), 3)],
                    "tri_wins_of_20": int(np.sum(diff > 1e-4)),
                }
            resplit_out.append({"suite": s_lbl, "regime": r_lbl, "stats": m_stats})

    # Fixed 6-episode evaluation (with cal_probed warm-start properly passed)
    q35_cal_v1 = pool_v1[:60]
    q35_test_v1 = pool_v1[60:]
    # Re-fetch canonical fixed splits so episodes match standard calibration
    c1, t1a = build_cscw_benchmark(15, 45, 2027)
    t1 = t1a + build_cscw_expansion_batch(45, 2028, 240)
    c1_p = probe_dataset(c1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    t1_p = probe_dataset(t1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    ep_v1 = eval_independent_episodes(
        ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI+Full-Defer+Naor", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"],
        split_stratified_episodes(t1_p, n_episodes=6, seed=2029),
        HeldOutCalibrator(c1_p),
        n_seeds=5, capacity=2, drain_interval=2, c_ask=1.5, cal_probed=c1_p,
    )
    return {"resplits_20": resplit_out, "episodes_v1_warm_fixed": ep_v1}


def plot_2d_minimax_regret(res_v1: Dict[str, Any], res_v2_ds: Dict[str, Any], res_pt_q35_cd4: Dict[str, Any]) -> None:
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
    w_pt = [res_pt_q35_cd4["summary"][m]["worst_case_regret_56"] for m in meths]
    ax.bar(x - w, w_v1, width=w, color="#1f77b4", label="v1-Clean (Qwen3.5)")
    ax.bar(x, w_ds, width=w, color="#2ca02c", label="v2 (DeepSeek-V3.2)")
    ax.bar(x + w, w_pt, width=w, color="#ff7f0e", label="Public-Tri (Qwen3.5, c_def=4)")
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

    m2d_v1 = [
        "Tri-VoI-L2D",
        "Calibrated-Cascade+Naor",
        "Joint-Greedy+Naor",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI+Post-Defer",
        "ACL-2026-VoI+Full-Defer+Naor",
        "Congested-L2D-Only",
        "w/o Queue-Externality",
    ]
    res_2d_v1 = run_2d_minimax_regret("v1-Clean (Qwen3.5-35B-A3B)", q35_cal_v1, q35_test_v1, q35_eb_v1, m2d_v1)

    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    m2d_v2 = [
        "Tri-VoI-L2D",
        "Calibrated-Cascade+Naor",
        "Joint-Greedy+Naor",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)+Post-Defer",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "Congested-L2D-Only",
        "w/o Queue-Externality",
    ]
    res_2d_v2 = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview"):
        cp = probe_dataset_profiled(cal_v2, mid, tracker, dry_run=False, workers=1)
        tp = probe_dataset_profiled(test_v2, mid, tracker, dry_run=False, workers=1)
        res_2d_v2[mid] = run_2d_minimax_regret(f"v2 ({mid})", cp, tp, HeldOutCalibrator(cp), m2d_v2)

    cal_pt, test_pt = load_public_tri_dataset()
    res_2d_pt = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        cp = probe_dataset_profiled(cal_pt, mid, tracker, dry_run=False, workers=1)
        tp = probe_dataset_profiled(test_pt, mid, tracker, dry_run=False, workers=1)
        eb_pt = HeldOutCalibrator(cp, stratify_post_clear=True)
        for c_def in (2.0, 4.0):
            key = f"{mid} (c_def={c_def:.1f})"
            res_2d_pt[key] = run_2d_minimax_regret(f"Public-Tri ({mid}, c_def={c_def:.1f})", cp, tp, eb_pt, m2d_v2, c_defer_base=c_def)

    plot_2d_minimax_regret(res_2d_v1, res_2d_v2["deepseek-ai/DeepSeek-V3.2"], res_2d_pt["Qwen/Qwen3.5-35B-A3B (c_def=4.0)"])

    noisy_ext = eval_noisy_stage3_external(tracker)
    stream_lat = eval_stream_ask_latency(q35_cal_v1, q35_test_v1, q35_eb_v1)
    real_spec = eval_real_llm_specialist(tracker)
    two_call_sc = eval_two_call_and_sc_parity(tracker)
    decomp_2x2 = eval_2x2_evoi_naor_decomposition(tracker)
    resplits_eps = eval_repeated_resplit_and_episodes(tracker)

    payload = {
        "phase2d_v1": res_2d_v1,
        "phase2d_v2": res_2d_v2,
        "phase2d_public_tri": res_2d_pt,
        "stage3_noisy_external": noisy_ext,
        "stream_ask_latency": stream_lat,
        "real_llm_specialist": real_spec,
        "two_call_and_sc_parity": two_call_sc,
        "decomposition_2x2": decomp_2x2,
        "resplits_and_episodes": resplits_eps,
    }
    (RESULTS_DIR / "reviewer_hardening_suite.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = ["# Reviewer Hardening Suite Report (0 Extra API Calls)", ""]
    md.append("## 1. 56-Cell 2D Minimax Regret (Descriptive Envelope — No Pseudoreplication p-values)")
    phase2d_items = (
        [("v1-Clean (Qwen3.5)", res_2d_v1)]
        + [(f"v2 ({k.split('/')[-1]})", v) for k, v in res_2d_v2.items()]
        + [(f"Public-Tri ({k.split('/')[-1]})", v) for k, v in res_2d_pt.items()]
    )
    for lbl, r2d in phase2d_items:
        lbl_clean = lbl.replace("))", ")")
        md.append(f"### {lbl_clean}")
        md.append("| Method | Mean Util (56) | $\\Delta$ vs Tri | Mean Regret | Worst-Case Regret | Best Cells | Tri W / T / L |")
        md.append("|:---|---:|---:|---:|---:|---:|---:|")
        for m, row in r2d["summary"].items():
            d_str = "—" if m == "Tri-VoI-L2D" else f"{row['mean_delta_vs_tri']:+.3f}"
            wtl = "—" if m == "Tri-VoI-L2D" else f"{row['tri_wins']} / {row['tri_ties']} / {row['tri_losses']}"
            md.append(
                f"| `{m}` | {row['mean_util_56']:.3f} | {d_str} | {row['mean_regret_56']:.3f} | "
                f"**{row['worst_case_regret_56']:.3f}** | {row['best_cells']} | {wtl} |"
            )
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

    md.append("## 4. Real-LLM Clinical Specialist (`Public-Tri`, Full-Quadrant) vs. Semi-Synthetic Specialist (`v1/v2`)")
    for s_lbl, s_map in real_spec.items():
        md.append(f"### {s_lbl}")
        for r_lbl, r_map in s_map.items():
            md.append(f"- **{r_lbl}**: " + ", ".join(f"`{k}`={v:.3f}" for k, v in r_map.items()))
        md.append("")

    md.append("## 5. 2-Call Measured SQLite Token Cost & `KL-PBD+ (SC-Only)` Signal Parity")
    md.append("| Suite | Model | `Tri-VoI` (Full) | `Tri-VoI` (2-Call) | T1 Tok (Full->2C) | Realized Tok (Full->2C, Δ%) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|---:|")
    for r in two_call_sc["rows"]:
        md.append(
            f"| {r['suite']} | {r['model']} | {r['tri_full_util']:.3f} ({r['tri_full_acc']}%) | "
            f"**{r['tri_2call_util']:.3f}** ({r['tri_2call_acc']}%) | {r['t1_tok_full']:.0f} -> {r['t1_tok_2call']:.0f} | "
            f"{r['realized_tok_full']:.0f} -> {r['realized_tok_2call']:.0f} (-{r['tok_reduction_pct']:.1f}%) | "
            f"{r['kl_pbd_warm']:.3f} | {r['kl_pbd_sc_only']:.3f} |"
        )
    md.append("")

    md.append("## 6. Explicit 2x2 Decomposition (`±EVOI × ±Naor`) & Fair `ACL-2026-VoI+Full-Defer+Naor`")
    md.append("| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI` | `Smooth-Naor` | Naor Δ | EVOI vs CC+N |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in decomp_2x2["rows"]:
        v = r["vals"]
        acl_k = r["acl_full_defer_key"]
        md.append(
            f"| {r['suite_model']} | {r['regime']} | {v['Joint-Greedy']:.3f} | {v['Joint-Greedy+Naor']:.3f} | "
            f"{v['Calibrated-Cascade']:.3f} | {v['Calibrated-Cascade+Naor']:.3f} | {v[acl_k]:.3f} | "
            f"{v['w/o Queue-Externality']:.3f} | {v['Tri-VoI-L2D']:.3f} | {v['Tri-VoI-L2D (Smooth-Naor)']:.3f} | "
            f"{r['naor_gain_tri']:+.3f} | {r['evoi_gain_over_cc_naor']:+.3f} |"
        )
    md.append("")

    md.append("## 7. 20 Stratified Random `(cal, test)` Resplits & 6 Disjoint-Episode Check")
    md.append("| Suite | Regime | `Tri-VoI` (mean±sd) | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | Tri Wins/20 vs (KL, ACL, JG+N, CC+N) |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|:---:|")
    for row in resplits_eps["resplits_20"]:
        st = row["stats"]
        acl_k = [k for k in st if "ACL-2026-VoI" in k][0]
        w_str = f"{st['KL-PBD+ (Warm)']['tri_wins_of_20']}/{st[acl_k]['tri_wins_of_20']}/{st['Joint-Greedy+Naor']['tri_wins_of_20']}/{st['Calibrated-Cascade+Naor']['tri_wins_of_20']}"
        md.append(
            f"| {row['suite']} | {row['regime']} | {st['Tri-VoI-L2D']['mean']:.3f}±{st['Tri-VoI-L2D']['std']:.2f} | "
            f"{st['KL-PBD+ (Warm)']['mean']:.3f} | {st[acl_k]['mean']:.3f} | {st['Joint-Greedy+Naor']['mean']:.3f} | "
            f"{st['Calibrated-Cascade+Naor']['mean']:.3f} | {w_str} |"
        )

    (RESULTS_DIR / "reviewer_hardening_suite.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Saved results/reviewer_hardening_suite.{json,md} and fig_2d_minimax_regret.{pdf,png}")


if __name__ == "__main__":
    main()
