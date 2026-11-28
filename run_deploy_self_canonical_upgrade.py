import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark import build_cscw_benchmark, build_cscw_benchmark_v2, build_cscw_expansion_batch
from benchmark_v3 import load_benchmark_v3_dataset
from controller import HeldOutCalibrator, probe_dataset, simulate_stream
from llm_client import RESULTS_DIR
from overnight.overnight_lib import (
    BudgetTracker,
    build_noisy_dataset,
    make_imperfect_user_sim,
    probe_dataset_budgeted,
)
from public_tri import load_public_tri_dataset, load_public_tri_holdout_dataset
from run_v2_and_public_tri import probe_dataset_profiled


def _tmp_tracker() -> BudgetTracker:
    return BudgetTracker(
        max_budget_rmb=0.01,
        state_path=Path(tempfile.gettempdir()) / "tri_voi_deploy_self_upgrade_budget.json",
    )


def _mean_5(
    method: str,
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    cap: int,
    di: int,
    c_ask: float,
    c_def: float = 2.0,
    real_ok: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    rs = [
        simulate_stream(
            method,
            test_p,
            calib,
            capacity=cap,
            drain_interval=di,
            drift_mode="sudden",
            c_ask=c_ask,
            c_defer_base=c_def,
            seed=s,
            cal_probed=cal_p,
            real_specialist_ok=real_ok,
        )
        for s in range(5)
    ]
    scope_vals = [r["scope_escalation_recall_pct"] for r in rs if r["scope_escalation_recall_pct"] is not None]
    return {
        "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
        "acc_pct": round(float(np.mean([r["acc_pct"] for r in rs])), 2),
        "raw_answer_acc_pct": round(float(np.mean([r["raw_answer_acc_pct"] for r in rs])), 2),
        "scope_escalation_recall_pct": round(float(np.mean(scope_vals)), 2) if scope_vals else None,
        "mean_realized_tokens": round(float(np.mean([r["mean_realized_tokens"] for r in rs])), 1),
    }


def _upgrade_56_minimax(
    base_r2d: Dict[str, Any],
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    c_def: float = 2.0,
) -> Dict[str, Any]:
    new_cells = []
    methods = list(base_r2d["summary"].keys())
    mat: Dict[str, List[float]] = {m: [] for m in methods}
    for cell in base_r2d["cells"]:
        u_ds = float(
            np.mean([
                simulate_stream(
                    "Tri-VoI-MDP (Deploy-Self)",
                    test_p,
                    calib,
                    capacity=int(cell["cap"]),
                    drain_interval=int(cell["di"]),
                    drift_mode="sudden",
                    c_ask=float(cell["c_ask"]),
                    c_defer_base=c_def,
                    seed=s,
                    cal_probed=cal_p,
                )["mean_utility"]
                for s in range(5)
            ])
        )
        nu = dict(cell["utils"])
        nu["Tri-VoI-MDP"] = round(u_ds, 4)
        new_cells.append({**cell, "utils": nu})
        for m in methods:
            mat[m].append(float(nu[m]))

    best_per_cell = [max(c["utils"].values()) for c in new_cells]
    mdp_vec = np.asarray(mat["Tri-VoI-MDP"], dtype=float)
    summary: Dict[str, Any] = {}
    for m in methods:
        v = np.asarray(mat[m], dtype=float)
        regrets = np.asarray(best_per_cell, dtype=float) - v
        summary[m] = {
            "mean_util_56": round(float(np.mean(v)), 3),
            "mean_delta_vs_mdp": round(float(np.mean(v - mdp_vec)), 3),
            "mean_regret_56": round(float(np.mean(regrets)), 3),
            "worst_case_regret_56": round(float(np.max(regrets)), 3),
            "best_cells": int(np.sum(v >= np.asarray(best_per_cell, dtype=float) - 1e-4)),
            "mdp_wins": int(np.sum(mdp_vec > v + 1e-4)),
            "mdp_ties": int(np.sum(np.abs(mdp_vec - v) <= 1e-4)),
            "mdp_losses": int(np.sum(v > mdp_vec + 1e-4)),
        }
    return {"suite": base_r2d["suite"], "summary": summary, "cells": new_cells}


def main() -> None:
    tracker = _tmp_tracker()
    hard = json.loads((RESULTS_DIR / "reviewer_hardening_suite.json").read_text(encoding="utf-8"))

    # Load all cached datasets
    cal_v1, t1a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = t1a + build_cscw_expansion_batch(45, 2028, 240)
    q35_cal_v1 = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    q35_test_v1 = probe_dataset(test_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    eb_q35_v1 = HeldOutCalibrator(q35_cal_v1, stratify_post_clear=False)

    ds_cal_v1 = probe_dataset(cal_v1, "deepseek-ai/DeepSeek-V3.2", False, 1)
    ds_test_v1 = probe_dataset(test_v1, "deepseek-ai/DeepSeek-V3.2", False, 1)
    eb_ds_v1 = HeldOutCalibrator(ds_cal_v1, stratify_post_clear=False)

    hy_cal_v1 = probe_dataset(cal_v1, "tencent/Hy4-preview", False, 1)
    hy_test_v1 = probe_dataset(test_v1, "tencent/Hy4-preview", False, 1)
    eb_hy_v1 = HeldOutCalibrator(hy_cal_v1, stratify_post_clear=False)

    noisy_cal = build_noisy_dataset(cal_v1, tracker, rewriter_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1, pace_sec=0.0)
    noisy_test = build_noisy_dataset(test_v1, tracker, rewriter_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1, pace_sec=0.0)
    imp_sim = make_imperfect_user_sim(tracker, sim_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, pace_sec=0.0)
    q35_cal_vn = probe_dataset_budgeted(noisy_cal, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1, pace_sec=0.0, user_sim_fn=imp_sim)
    q35_test_vn = probe_dataset_budgeted(noisy_test, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1, pace_sec=0.0, user_sim_fn=imp_sim)
    eb_q35_vn = HeldOutCalibrator(q35_cal_vn, stratify_post_clear=False)

    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    v2_cp, v2_tp, v2_eb = {}, {}, {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview"):
        v2_cp[mid] = probe_dataset_profiled(cal_v2, mid, tracker, dry_run=False, workers=1)
        v2_tp[mid] = probe_dataset_profiled(test_v2, mid, tracker, dry_run=False, workers=1)
        v2_eb[mid] = HeldOutCalibrator(v2_cp[mid], stratify_post_clear=False)

    cal_pt, test_pt = load_public_tri_dataset()
    pt_cp, pt_tp, pt_eb_unif = {}, {}, {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        pt_cp[mid] = probe_dataset_profiled(cal_pt, mid, tracker, dry_run=False, workers=1)
        pt_tp[mid] = probe_dataset_profiled(test_pt, mid, tracker, dry_run=False, workers=1)
        pt_eb_unif[mid] = HeldOutCalibrator(pt_cp[mid], stratify_post_clear=False)
    q8_cal_pt = probe_dataset_profiled(cal_pt, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    q8_test_pt = probe_dataset_profiled(test_pt, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    eb_q8_pt_unif = HeldOutCalibrator(q8_cal_pt, stratify_post_clear=False)

    # Table 1: v1-Clean & v1-Noisy (Deploy-Self)
    table1 = {
        "v1_clean": {
            "Cap=2, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_v1, q35_test_v1, eb_q35_v1, 2, 2, 1.5)["mean_utility"],
            "Cap=2, c=2.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_v1, q35_test_v1, eb_q35_v1, 2, 2, 2.5)["mean_utility"],
            "Cap=1/3, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_v1, q35_test_v1, eb_q35_v1, 1, 3, 1.5)["mean_utility"],
        },
        "v1_noisy": {
            "Cap=2, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_vn, q35_test_vn, eb_q35_vn, 2, 2, 1.5)["mean_utility"],
            "Cap=2, c=2.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_vn, q35_test_vn, eb_q35_vn, 2, 2, 2.5)["mean_utility"],
            "Cap=1/3, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_vn, q35_test_vn, eb_q35_vn, 1, 3, 1.5)["mean_utility"],
        },
    }

    # Table 2: v2 (3 models) & Public-Tri (2 models)
    table2 = {"v2": {}, "public_tri": {}}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview"):
        table2["v2"][mid] = {
            "Cap=2, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", v2_cp[mid], v2_tp[mid], v2_eb[mid], 2, 2, 1.5)["mean_utility"],
            "Cap=1/3, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", v2_cp[mid], v2_tp[mid], v2_eb[mid], 1, 3, 1.5)["mean_utility"],
        }
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        table2["public_tri"][mid] = {
            "Cap=2, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", pt_cp[mid], pt_tp[mid], pt_eb_unif[mid], 2, 2, 1.5)["mean_utility"],
            "Cap=1/3, c=1.5": _mean_5("Tri-VoI-MDP (Deploy-Self)", pt_cp[mid], pt_tp[mid], pt_eb_unif[mid], 1, 3, 1.5)["mean_utility"],
        }

    # Table 3: 56-Cell 2D Minimax Regret across all 7 suites
    phase2d_v1 = _upgrade_56_minimax(hard["phase2d_v1"], q35_cal_v1, q35_test_v1, eb_q35_v1, c_def=2.0)
    phase2d_v2 = {
        mid: _upgrade_56_minimax(hard["phase2d_v2"][mid], v2_cp[mid], v2_tp[mid], v2_eb[mid], c_def=2.0)
        for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview")
    }
    phase2d_pt = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        for c_def in (2.0, 4.0):
            k = f"{mid} (c_def={c_def:.1f})"
            if k in hard["phase2d_public_tri"]:
                phase2d_pt[k] = _upgrade_56_minimax(hard["phase2d_public_tri"][k], pt_cp[mid], pt_tp[mid], pt_eb_unif[mid], c_def=c_def)

    # Section 4: 2x2 Decomposition (14 rows)
    decomp_cfg = {
        "v1-Clean / Qwen3.5": (q35_cal_v1, q35_test_v1, eb_q35_v1),
        "v2 / Qwen3.5": (v2_cp["Qwen/Qwen3.5-35B-A3B"], v2_tp["Qwen/Qwen3.5-35B-A3B"], v2_eb["Qwen/Qwen3.5-35B-A3B"]),
        "v2 / DeepSeek-V3.2": (v2_cp["deepseek-ai/DeepSeek-V3.2"], v2_tp["deepseek-ai/DeepSeek-V3.2"], v2_eb["deepseek-ai/DeepSeek-V3.2"]),
        "v2 / Hy4-preview": (v2_cp["tencent/Hy4-preview"], v2_tp["tencent/Hy4-preview"], v2_eb["tencent/Hy4-preview"]),
        "Public-Tri / Qwen3.5 (c_def=2)": (pt_cp["Qwen/Qwen3.5-35B-A3B"], pt_tp["Qwen/Qwen3.5-35B-A3B"], pt_eb_unif["Qwen/Qwen3.5-35B-A3B"]),
        "Public-Tri / Qwen3.5 (c_def=4)": (pt_cp["Qwen/Qwen3.5-35B-A3B"], pt_tp["Qwen/Qwen3.5-35B-A3B"], pt_eb_unif["Qwen/Qwen3.5-35B-A3B"]),
        "Public-Tri / DeepSeek (c_def=2)": (pt_cp["deepseek-ai/DeepSeek-V3.2"], pt_tp["deepseek-ai/DeepSeek-V3.2"], pt_eb_unif["deepseek-ai/DeepSeek-V3.2"]),
    }
    decomp_rows = []
    for r in hard["decomposition_2x2"]["rows"]:
        cp_d, tp_d, eb_d = decomp_cfg[r["suite_model"]]
        cap_d, di_d = (2, 2) if "Cap=2" in r["regime"] else (1, 3)
        u_ds = _mean_5("Tri-VoI-MDP (Deploy-Self)", cp_d, tp_d, eb_d, cap_d, di_d, 1.5, c_def=float(r["c_defer"]))["mean_utility"]
        nv = dict(r["vals"])
        nv["Tri-VoI-MDP"] = u_ds
        decomp_rows.append({
            **r,
            "vals": nv,
            "mdp_gain_over_tri": round(u_ds - nv["Tri-VoI-L2D"], 3),
            "mdp_gain_over_cc_naor": round(u_ds - nv["Calibrated-Cascade+Naor"], 3),
        })

    # Section 5: 20 Stratified Random Resplits (exact same random seeds 8100 + rep*31)
    resplit_cfg = [
        ("v1-Clean (Qwen3.5)", q35_cal_v1 + q35_test_v1, 15, False, False, 2.0),
        ("v2 (Qwen3.5)", v2_cp["Qwen/Qwen3.5-35B-A3B"] + v2_tp["Qwen/Qwen3.5-35B-A3B"], 15, True, False, 2.0),
        ("v2 (DeepSeek-V3.2)", v2_cp["deepseek-ai/DeepSeek-V3.2"] + v2_tp["deepseek-ai/DeepSeek-V3.2"], 15, True, False, 2.0),
        ("Public-Tri (Qwen3.5)", pt_cp["Qwen/Qwen3.5-35B-A3B"] + pt_tp["Qwen/Qwen3.5-35B-A3B"], 15, True, True, 2.0),
        ("Public-Tri (DeepSeek-V3.2)", pt_cp["deepseek-ai/DeepSeek-V3.2"] + pt_tp["deepseek-ai/DeepSeek-V3.2"], 15, True, True, 2.0),
    ]
    resplits_20 = []
    for s_lbl, full_pool, n_cal_pq, live_acl, strat, c_def in resplit_cfg:
        by_q = defaultdict(list)
        for r in full_pool:
            by_q[r["item"]["quadrant"]].append(r)
        acl_fd = "ACL-2026-VoI (Live)+Full-Defer+Naor" if live_acl else "ACL-2026-VoI+Full-Defer+Naor"
        comp_meths = ["KL-PBD+ (Warm)", acl_fd, "Joint-Greedy+Naor", "Calibrated-Cascade+Naor", "Tri-VoI-L2D", "Tri-VoI-MDP (Deploy-Self)"]
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
                eb_unif = HeldOutCalibrator(cal_s, stratify_post_clear=False)
                for m in comp_meths:
                    res = simulate_stream(m, test_s, eb_unif if m == "Tri-VoI-MDP (Deploy-Self)" else eb, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=rep, cal_probed=cal_s)
                    per_m[m].append(res["mean_utility"])
            ds_arr = np.asarray(per_m["Tri-VoI-MDP (Deploy-Self)"], dtype=float)
            m_stats = {}
            for m in comp_meths:
                out_k = "Tri-VoI-MDP" if m == "Tri-VoI-MDP (Deploy-Self)" else m
                arr = np.asarray(per_m[m], dtype=float)
                m_stats[out_k] = {
                    "mean": round(float(np.mean(arr)), 3),
                    "std": round(float(np.std(arr, ddof=1)), 3),
                    "mdp_wins_of_20": int(np.sum(ds_arr > arr + 1e-4)),
                }
            resplits_20.append({"suite": s_lbl, "regime": r_lbl, "stats": m_stats})

    # Section 6: Measured SQLite Token Cost (6 rows)
    tok_rows = []
    for r in hard["two_call_and_sc_parity"]["rows"]:
        mid_full = "Qwen/Qwen3.5-35B-A3B" if r["model"] == "Qwen3.5-35B-A3B" else ("deepseek-ai/DeepSeek-V3.2" if r["model"] == "DeepSeek-V3.2" else "tencent/Hy4-preview")
        if r["suite"] == "v1-Clean":
            cp_t = probe_dataset(cal_v1, mid_full, False, 1)
            tp_t = probe_dataset(test_v1, mid_full, False, 1)
        else:
            cp_t, tp_t = v2_cp[mid_full], v2_tp[mid_full]
        eb_t = HeldOutCalibrator(cp_t, stratify_post_clear=False)
        m5 = _mean_5("Tri-VoI-MDP (Deploy-Self)", cp_t, tp_t, eb_t, 2, 2, 1.5)
        tok_rows.append({
            **r,
            "tri_mdp_util": m5["mean_utility"],
            "realized_tok_mdp": m5["mean_realized_tokens"],
            "mdp_tok_reduction_pct": round(100.0 * (float(r["realized_tok_full"]) - m5["mean_realized_tokens"]) / max(1.0, float(r["realized_tok_full"])), 1),
        })

    # Section 7: Real-LLM Specialist (Public-Tri) & Semi-Synthetic
    ds_by_qid_pt = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in pt_tp["deepseek-ai/DeepSeek-V3.2"]
    }
    q8_by_qid_pt = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in q8_test_pt
    }
    q35_by_qid_pt = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in pt_tp["Qwen/Qwen3.5-35B-A3B"]
    }
    spec_ok_pt_shift = {
        r["item"]["qid"]: (ds_by_qid_pt[r["item"]["qid"]] if idx < (len(pt_tp["deepseek-ai/DeepSeek-V3.2"]) // 2) else q8_by_qid_pt[r["item"]["qid"]])
        for idx, r in enumerate(pt_tp["deepseek-ai/DeepSeek-V3.2"])
    }
    q8_cal_cap = [{**r, "item": {**r["item"], "require_defer_for_ok": False}} for r in q8_cal_pt]
    q8_test_cap = [{**r, "item": {**r["item"], "require_defer_for_ok": False}} for r in q8_test_pt]
    eb_q8_cap = HeldOutCalibrator(q8_cal_cap, stratify_post_clear=False)

    spec_ok_v1: Dict[str, bool] = {}
    for idx, r_hy in enumerate(hy_test_v1):
        it = r_hy["item"]
        if it["quadrant"] in ("Q1_clear_easy", "Q2_underspec_easy", "Q4_resolvable"):
            spec_ok_v1[it["qid"]] = ((r_hy["post_ans"] if it["is_underspec"] else r_hy["greedy_ans"]) == it["gold"])
        else:
            spec_ok_v1[it["qid"]] = True if idx < (len(hy_test_v1) // 2) else ((idx % 20) < 13)

    hy_test_v2 = v2_tp["tencent/Hy4-preview"]
    spec_ok_v2: Dict[str, bool] = {}
    for idx, r_hy in enumerate(hy_test_v2):
        it = r_hy["item"]
        if it["quadrant"] in ("Q1_clear_easy", "Q2_underspec_easy", "Q4_resolvable"):
            spec_ok_v2[it["qid"]] = ((r_hy["post_ans"] if it["is_underspec"] else r_hy["greedy_ans"]) == it["gold"])
        else:
            spec_ok_v2[it["qid"]] = True if idx < (len(hy_test_v2) // 2) else ((idx % 20) < 13)

    sec7 = {
        "real_public_tri_qwen35_to_deepseek_stationary": {},
        "real_public_tri_qwen35_split_metrics": {},
        "real_public_tri_qwen35_to_ds_to_qwen8b_shift": {},
        "real_public_tri_qwen8b_to_deepseek_stationary": {},
        "pure_cap_gap_qwen8b_to_qwen35": {},
        "pure_cap_gap_qwen8b_to_deepseek": {},
        "semi_synthetic_v1_qwen35_to_hy4": {},
        "semi_synthetic_v2_qwen35_to_hy4": {},
    }
    q35_cp_pt, q35_tp_pt = pt_cp["Qwen/Qwen3.5-35B-A3B"], pt_tp["Qwen/Qwen3.5-35B-A3B"]
    q35_eb_pt = pt_eb_unif["Qwen/Qwen3.5-35B-A3B"]
    for r_lbl, cap, di, c_def in [("Cap=2, c_def=2.0, c=1.5", 2, 2, 2.0), ("Cap=2, c_def=4.0, c=1.5", 2, 2, 4.0), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 2.0)]:
        m_st = _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cp_pt, q35_tp_pt, q35_eb_pt, cap, di, 1.5, c_def=c_def, real_ok=ds_by_qid_pt)
        m_sh = _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cp_pt, q35_tp_pt, q35_eb_pt, cap, di, 1.5, c_def=c_def, real_ok=spec_ok_pt_shift)
        sec7["real_public_tri_qwen35_to_deepseek_stationary"][r_lbl] = m_st["mean_utility"]
        sec7["real_public_tri_qwen35_to_ds_to_qwen8b_shift"][r_lbl] = m_sh["mean_utility"]
        sec7["real_public_tri_qwen35_split_metrics"][r_lbl] = {
            "mean_utility": m_st["mean_utility"],
            "raw_answer_acc_pct": m_st["raw_answer_acc_pct"],
            "scope_escalation_recall_pct": m_st["scope_escalation_recall_pct"],
        }
    for r_lbl, cap, di, c_def in [("Cap=2, c_def=4.0, c=1.5", 2, 2, 4.0), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 2.0)]:
        sec7["real_public_tri_qwen8b_to_deepseek_stationary"][r_lbl] = _mean_5("Tri-VoI-MDP (Deploy-Self)", q8_cal_pt, q8_test_pt, eb_q8_pt_unif, cap, di, 1.5, c_def=c_def, real_ok=ds_by_qid_pt)["mean_utility"]
    for r_lbl, cap, di in [("Cap=2, c_def=2.0, c=1.5", 2, 2), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3)]:
        sec7["pure_cap_gap_qwen8b_to_qwen35"][r_lbl] = _mean_5("Tri-VoI-MDP (Deploy-Self)", q8_cal_cap, q8_test_cap, eb_q8_cap, cap, di, 1.5, c_def=2.0, real_ok=q35_by_qid_pt)["mean_utility"]
        sec7["pure_cap_gap_qwen8b_to_deepseek"][r_lbl] = _mean_5("Tri-VoI-MDP (Deploy-Self)", q8_cal_cap, q8_test_cap, eb_q8_cap, cap, di, 1.5, c_def=2.0, real_ok=ds_by_qid_pt)["mean_utility"]
    for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
        sec7["semi_synthetic_v1_qwen35_to_hy4"][r_lbl] = _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_cal_v1, q35_test_v1, eb_q35_v1, cap, di, 1.5, c_def=2.0, real_ok=spec_ok_v1)["mean_utility"]
        sec7["semi_synthetic_v2_qwen35_to_hy4"][r_lbl] = _mean_5("Tri-VoI-MDP (Deploy-Self)", v2_cp["Qwen/Qwen3.5-35B-A3B"], v2_tp["Qwen/Qwen3.5-35B-A3B"], v2_eb["Qwen/Qwen3.5-35B-A3B"], cap, di, 1.5, c_def=2.0, real_ok=spec_ok_v2)["mean_utility"]

    # Section 8: Suite A (Benchmark v3) & Suite B (MediQ-Holdout)
    cal_v3, test_v3 = load_benchmark_v3_dataset()
    sec8_v3 = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        cp3 = probe_dataset_profiled(cal_v3, mid, tracker, dry_run=False, workers=1)
        tp3 = probe_dataset_profiled(test_v3, mid, tracker, dry_run=False, workers=1)
        eb3 = HeldOutCalibrator(cp3, stratify_post_clear=False)
        sec8_v3[mid] = {
            reg: _mean_5("Tri-VoI-MDP (Deploy-Self)", cp3, tp3, eb3, cap, di, ca, c_def=2.0)
            for reg, cap, di, ca in [("Cap=2, c=1.5", 2, 2, 1.5), ("Cap=2, c=2.5", 2, 2, 2.5), ("Cap=1/3, c=1.5", 1, 3, 1.5)]
        }

    cal_h, test_h = load_public_tri_holdout_dataset()
    q8_ch = probe_dataset_profiled(cal_h, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    q8_th = probe_dataset_profiled(test_h, "Qwen/Qwen3-8B", tracker, dry_run=False, workers=1)
    q35_ch = probe_dataset_profiled(cal_h, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    q35_th = probe_dataset_profiled(test_h, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1)
    ds_th = probe_dataset_profiled(test_h, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)
    q35_ok_h = {r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"]) for r in q35_th}
    ds_ok_h = {r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"]) for r in ds_th}
    eb_q35_h = HeldOutCalibrator(q35_ch, stratify_post_clear=False)
    q8_ch_cap = [{**r, "item": {**r["item"], "require_defer_for_ok": False}} for r in q8_ch]
    q8_th_cap = [{**r, "item": {**r["item"], "require_defer_for_ok": False}} for r in q8_th]
    eb_q8_h_cap = HeldOutCalibrator(q8_ch_cap, stratify_post_clear=False)

    sec8_holdout = {
        "scope_compliance_qwen35_to_deepseek": {
            reg: _mean_5("Tri-VoI-MDP (Deploy-Self)", q35_ch, q35_th, eb_q35_h, cap, di, 1.5, c_def=c_def, real_ok=ds_ok_h)
            for reg, cap, di, c_def in [("Cap=2, c_def=2.0, c=1.5", 2, 2, 2.0), ("Cap=2, c_def=4.0, c=1.5", 2, 2, 4.0), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 2.0)]
        },
        "pure_cap_gap_qwen8b_to_qwen35": {
            reg: _mean_5("Tri-VoI-MDP (Deploy-Self)", q8_ch_cap, q8_th_cap, eb_q8_h_cap, cap, di, 1.5, c_def=2.0, real_ok=q35_ok_h)
            for reg, cap, di in [("Cap=2, c_def=2.0, c=1.5", 2, 2), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3)]
        },
        "pure_cap_gap_qwen8b_to_deepseek": {
            reg: _mean_5("Tri-VoI-MDP (Deploy-Self)", q8_ch_cap, q8_th_cap, eb_q8_h_cap, cap, di, 1.5, c_def=2.0, real_ok=ds_ok_h)
            for reg, cap, di in [("Cap=2, c_def=2.0, c=1.5", 2, 2), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3)]
        },
    }

    snap = tracker.snapshot()
    assert snap["live_calls"] == 0, f"Unexpected live API call! {snap}"
    payload = {
        "budget": snap,
        "table1": table1,
        "table2": table2,
        "phase2d_v1": phase2d_v1,
        "phase2d_v2": phase2d_v2,
        "phase2d_public_tri": phase2d_pt,
        "decomposition_2x2_rows": decomp_rows,
        "resplits_20": resplits_20,
        "two_call_tok_rows": tok_rows,
        "section7_real_spec": sec7,
        "section8_v3": sec8_v3,
        "section8_holdout": sec8_holdout,
    }
    (RESULTS_DIR / "deploy_self_canonical_upgrade.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results/deploy_self_canonical_upgrade.json | live={snap['live_calls']} rmb=RMB {snap['estimated_rmb']:.2f}")


if __name__ == "__main__":
    main()
