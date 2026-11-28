import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark import build_cscw_benchmark, build_cscw_benchmark_v2, build_cscw_expansion_batch
from benchmark_v3 import load_benchmark_v3_dataset
from controller import HeldOutCalibrator, parse_json_obj, probe_dataset, simulate_stream
from llm_client import RESULTS_DIR
from overnight.overnight_lib import (
    BudgetTracker,
    build_noisy_dataset,
    make_imperfect_user_sim,
    probe_dataset_budgeted,
)
from public_tri import load_public_tri_dataset, load_public_tri_holdout_dataset
from run_all import block_paired_bootstrap
from run_v2_and_public_tri import probe_dataset_profiled


def _norm_v3(s: str) -> str:
    return re.sub(r"#\d+", "#ID", s)


def _mean_util(
    method: str,
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    cal_p: List[Dict[str, Any]],
    *,
    cap: int,
    di: int,
    c_ask: float = 1.5,
    c_def: float = 2.0,
    real_spec_ok: Optional[Dict[str, bool]] = None,
    n_seeds: int = 5,
) -> float:
    return round(
        float(
            np.mean([
                simulate_stream(
                    method,
                    test_p,
                    calib,
                    capacity=cap,
                    drain_interval=di,
                    c_ask=c_ask,
                    c_defer_base=c_def,
                    seed=s,
                    cal_probed=cal_p,
                    real_specialist_ok=real_spec_ok,
                )["mean_utility"]
                for s in range(n_seeds)
            ])
        ),
        3,
    )


def _mean_metrics(
    method: str,
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    cal_p: List[Dict[str, Any]],
    *,
    cap: int,
    di: int,
    c_ask: float = 1.5,
    c_def: float = 2.0,
    real_spec_ok: Optional[Dict[str, bool]] = None,
    n_seeds: int = 5,
) -> Dict[str, Any]:
    runs = [
        simulate_stream(
            method,
            test_p,
            calib,
            capacity=cap,
            drain_interval=di,
            c_ask=c_ask,
            c_defer_base=c_def,
            seed=s,
            cal_probed=cal_p,
            real_specialist_ok=real_spec_ok,
        )
        for s in range(n_seeds)
    ]
    return {
        "mean_utility": round(float(np.mean([r["mean_utility"] for r in runs])), 3),
        "mean_tokens": round(float(np.mean([r["mean_realized_tokens"] for r in runs])), 1),
        "per_item_utils_mean": np.mean([r["per_item_utils"] for r in runs], axis=0),
    }


def _post_missing_metrics(probed: List[Dict[str, Any]]) -> Dict[str, Any]:
    sub = [r for r in probed if r["obs_missing"]]
    tp = fp = fn = tn = 0
    unparsed_cnt = 0
    for r in sub:
        if not bool(r.get("post_missing_parsed", True)):
            unparsed_cnt += 1
        true_ni = not bool(r["oracle_revealed"])
        pred_ni = bool(r.get("post_missing", False))
        if pred_ni and true_ni:
            tp += 1
        elif pred_ni and not true_ni:
            fp += 1
        elif not pred_ni and true_ni:
            fn += 1
        else:
            tn += 1
    tot = max(1, len(sub))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-9, prec + rec)
    return {
        "n_obs_missing": len(sub),
        "true_no_info": tp + fn,
        "pred_post_missing": tp + fp,
        "unparsed_fallback_to_obs": unparsed_cnt,
        "agreement_pct": round(100.0 * (tp + tn) / tot, 2),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
    }


def _audit_sqlite_post_missing() -> Dict[str, Any]:
    """Scan all SQLite caches to count exact historical post_ask_combined rows missing `missing_user_info`."""
    per_db = []
    tot_rows = 0
    tot_missing_key = 0
    for db in sorted(RESULTS_DIR.glob("cache_*.sqlite")):
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT tag, content, finish_reason FROM cache_v1 WHERE tag LIKE '%post_ask_combined%'").fetchall()
        conn.close()
        if not rows:
            continue
        db_missing = 0
        by_tag: Dict[str, Dict[str, int]] = {}
        for tag, content, _ in rows:
            obj = parse_json_obj(content or "")
            st = by_tag.setdefault(tag, {"total": 0, "missing_key": 0})
            st["total"] += 1
            if "missing_user_info" not in obj:
                db_missing += 1
                st["missing_key"] += 1
        tot_rows += len(rows)
        tot_missing_key += db_missing
        per_db.append({
            "db": db.name,
            "total": len(rows),
            "missing_key": db_missing,
            "by_tag": by_tag,
        })
    return {
        "total_post_ask_combined_rows": tot_rows,
        "total_missing_key_fallbacks": tot_missing_key,
        "per_db": per_db,
    }


def main() -> None:
    tracker = BudgetTracker(max_budget_rmb=0.01, state_path=Path(tempfile.gettempdir()) / "tri_voi_attr_audit_budget.json")

    cal_v1, t1a = build_cscw_benchmark(15, 45, 2027)
    test_v1 = t1a + build_cscw_expansion_batch(45, 2028, 240)
    v1_cal_p = probe_dataset(cal_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)
    v1_test_p = probe_dataset(test_v1, "Qwen/Qwen3.5-35B-A3B", False, 1)

    noisy_cal = build_noisy_dataset(cal_v1, tracker, rewriter_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1, pace_sec=0.0)
    noisy_test = build_noisy_dataset(test_v1, tracker, rewriter_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, workers=1, pace_sec=0.0)
    imp_sim = make_imperfect_user_sim(tracker, sim_model="Qwen/Qwen3.5-35B-A3B", dry_run=False, pace_sec=0.0)
    vn_cal_p = probe_dataset_budgeted(noisy_cal, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1, pace_sec=0.0, user_sim_fn=imp_sim)
    vn_test_p = probe_dataset_budgeted(noisy_test, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1, pace_sec=0.0, user_sim_fn=imp_sim)

    cal_v2, test_v2 = build_cscw_benchmark_v2(15, 45, 2027)
    cal_pt, test_pt = load_public_tri_dataset()
    cal_v3, test_v3 = load_benchmark_v3_dataset()
    cal_h, test_h = load_public_tri_holdout_dataset()

    ds_test_h = probe_dataset_profiled(test_h, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)
    ds_ok_h = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in ds_test_h
    }
    ds_test_pt = probe_dataset_profiled(test_pt, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1)
    ds_ok_pt = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in ds_test_pt
    }

    suites: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], bool, Optional[Dict[str, bool]], bool]] = [
        ("v1-Clean / Qwen3.5", v1_cal_p, v1_test_p, False, None, False),
        ("v1-Noisy / Qwen3.5", vn_cal_p, vn_test_p, False, None, False),
        ("v2 / Qwen3.5", probe_dataset_profiled(cal_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v2, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, None, False),
        ("v2 / DeepSeek-V3.2", probe_dataset_profiled(cal_v2, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v2, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), True, None, False),
        ("v2 / Hy4-preview", probe_dataset_profiled(cal_v2, "tencent/Hy4-preview", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v2, "tencent/Hy4-preview", tracker, dry_run=False, workers=1), True, None, False),
        ("Public-Tri / Qwen3.5 (strat=False)", probe_dataset_profiled(cal_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, None, False),
        ("Public-Tri / Qwen3.5 (strat=True)", probe_dataset_profiled(cal_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, None, True),
        ("Public-Tri / DS (RealSpec, strat=False)", probe_dataset_profiled(cal_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_pt, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, ds_ok_pt, False),
        ("v3 / Qwen3.5", probe_dataset_profiled(cal_v3, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v3, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, None, False),
        ("v3 / DeepSeek-V3.2", probe_dataset_profiled(cal_v3, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_v3, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=False, workers=1), True, None, False),
        ("MediQ-Holdout / Q3.5->RealDS", probe_dataset_profiled(cal_h, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), probe_dataset_profiled(test_h, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=False, workers=1), True, ds_ok_h, False),
    ]

    ablation_methods = [
        "Tri-VoI-L2D",
        "Tri-VoI-L2D (2-Call)",
        "Tri-VoI-L2D (2-Call+NoInfo)",
        "Tri-VoI-MDP (5-Call)",
        "Tri-VoI-MDP",
        "Tri-VoI-MDP (Deploy-Self)",
    ]

    part_a_rows = []
    part_b_rows = []
    part_c_signal = {}
    part_e_rows = []
    part_f_rows = []

    regime_map_suites = {
        "v1-Clean / Qwen3.5",
        "v1-Noisy / Qwen3.5",
        "v2 / DeepSeek-V3.2",
        "v3 / DeepSeek-V3.2",
        "MediQ-Holdout / Q3.5->RealDS",
    }

    for lbl, cp, tp, live_acl, real_ok, strat in suites:
        cb = HeldOutCalibrator(cp, stratify_post_clear=strat)
        acl_naor = "ACL-2026-VoI (Live)+Full-Defer+Naor" if live_acl else "ACL-2026-VoI+Full-Defer+Naor"
        acl_mdp = "ACL-2026-VoI (Live)+Full-Defer+MDP" if live_acl else "ACL-2026-VoI+Full-Defer+MDP"
        acl_2c = "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)" if live_acl else "ACL-2026-VoI+Full-Defer+MDP (2C-Self)"
        fair_methods = [
            "Calibrated-Cascade+Naor",
            "Calibrated-Cascade+MDP",
            "Joint-Greedy+Naor",
            "Joint-Greedy+MDP",
            acl_naor,
            acl_mdp,
            "Tri-VoI-MDP",
            "Tri-VoI-MDP (Deploy-Self)",
        ]
        matched_2c_methods = [
            "Calibrated-Cascade+MDP (2C-Self)",
            "Calibrated-Cascade+OwnDP (2C-Self)",
            "Joint-Greedy+MDP (2C-Self)",
            "Joint-Greedy+OwnDP (2C-Self)",
            acl_2c,
            "Tri-VoI-MDP (Deploy-Self)",
        ]
        if lbl not in part_c_signal and "strat=True" not in lbl and "RealSpec" not in lbl:
            part_c_signal[lbl] = _post_missing_metrics(tp)

        for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
            vals_a = {m: _mean_util(m, tp, cb, cp, cap=cap, di=di, real_spec_ok=real_ok) for m in ablation_methods}
            part_a_rows.append({"suite": lbl, "regime": r_lbl, "vals": vals_a})

            vals_b = {m: _mean_util(m, tp, cb, cp, cap=cap, di=di, real_spec_ok=real_ok) for m in fair_methods}
            best_base_mdp = max(vals_b["Calibrated-Cascade+MDP"], vals_b["Joint-Greedy+MDP"], vals_b[acl_mdp])
            part_b_rows.append({
                "suite": lbl,
                "regime": r_lbl,
                "acl_naor_key": acl_naor,
                "acl_mdp_key": acl_mdp,
                "vals": vals_b,
                "mdp_gain_cc": round(vals_b["Calibrated-Cascade+MDP"] - vals_b["Calibrated-Cascade+Naor"], 3),
                "mdp_gain_jg": round(vals_b["Joint-Greedy+MDP"] - vals_b["Joint-Greedy+Naor"], 3),
                "mdp_gain_acl": round(vals_b[acl_mdp] - vals_b[acl_naor], 3),
                "evoi_net_vs_best_base_mdp": round(vals_b["Tri-VoI-MDP"] - best_base_mdp, 3),
                "deploy_self_net_vs_best_base_mdp": round(vals_b["Tri-VoI-MDP (Deploy-Self)"] - best_base_mdp, 3),
            })

            met_e = {m: _mean_metrics(m, tp, cb, cp, cap=cap, di=di, real_spec_ok=real_ok) for m in matched_2c_methods}
            vals_e = {m: met_e[m]["mean_utility"] for m in matched_2c_methods}
            toks_e = {m: met_e[m]["mean_tokens"] for m in matched_2c_methods}
            shared_base_keys = ["Calibrated-Cascade+MDP (2C-Self)", "Joint-Greedy+MDP (2C-Self)", acl_2c]
            own_base_keys = ["Calibrated-Cascade+OwnDP (2C-Self)", "Joint-Greedy+OwnDP (2C-Self)", acl_2c]
            best_shared_k = max(shared_base_keys, key=lambda mk: vals_e[mk])
            best_own_k = max(own_base_keys, key=lambda mk: vals_e[mk])
            bp_shared = block_paired_bootstrap(
                met_e["Tri-VoI-MDP (Deploy-Self)"]["per_item_utils_mean"],
                met_e[best_shared_k]["per_item_utils_mean"],
                block_size=6,
                n_boot=3000,
                seed=2029,
            )
            bp_own = block_paired_bootstrap(
                met_e["Tri-VoI-MDP (Deploy-Self)"]["per_item_utils_mean"],
                met_e[best_own_k]["per_item_utils_mean"],
                block_size=6,
                n_boot=3000,
                seed=2029,
            )
            part_e_rows.append({
                "suite": lbl,
                "regime": r_lbl,
                "acl_2c_key": acl_2c,
                "vals": vals_e,
                "tokens": toks_e,
                "best_shared_2c_key": best_shared_k,
                "best_own_2c_key": best_own_k,
                "evoi_net_vs_best_shared_2c": round(vals_e["Tri-VoI-MDP (Deploy-Self)"] - vals_e[best_shared_k], 3),
                "evoi_net_vs_best_own_2c": round(vals_e["Tri-VoI-MDP (Deploy-Self)"] - vals_e[best_own_k], 3),
                "boot_p_vs_best_shared_2c": bp_shared["boot_p"],
                "boot_p_vs_best_own_2c": bp_own["boot_p"],
            })

        if lbl in regime_map_suites:
            rm_meths = [
                "Calibrated-Cascade+MDP (2C-Self)",
                "Joint-Greedy+MDP (2C-Self)",
                acl_2c,
                "Tri-VoI-MDP (Deploy-Self)",
            ]
            for cap_lbl, cap, di in [("Cap=2", 2, 2), ("Cap=1/3", 1, 3)]:
                for ca in (0.5, 1.5, 3.0, 4.5):
                    vals_f = {
                        m: _mean_util(m, tp, cb, cp, cap=cap, di=di, c_ask=ca, real_spec_ok=real_ok)
                        for m in rm_meths
                    }
                    best_f = max(vals_f["Calibrated-Cascade+MDP (2C-Self)"], vals_f["Joint-Greedy+MDP (2C-Self)"], vals_f[acl_2c])
                    part_f_rows.append({
                        "suite": lbl,
                        "cap_regime": cap_lbl,
                        "c_ask": ca,
                        "acl_2c_key": acl_2c,
                        "vals": vals_f,
                        "evoi_net_vs_best_2c": round(vals_f["Tri-VoI-MDP (Deploy-Self)"] - best_f, 3),
                    })

    # Part D: Benchmark v3 Content Overlap Audit & Strict Zero-Overlap Slices
    cal_v3_proses = {_norm_v3(it["prose"]) for it in cal_v3}
    cal_v3_triples = {(_norm_v3(it["prose"]), it["missing_val"], it["gold"]) for it in cal_v3}
    overlap_audit = {
        "n_cal": len(cal_v3),
        "n_test": len(test_v3),
        "uniq_cal_prose": len(cal_v3_proses),
        "uniq_test_prose": len({_norm_v3(it["prose"]) for it in test_v3}),
        "test_prose_in_cal": sum(1 for it in test_v3 if _norm_v3(it["prose"]) in cal_v3_proses),
        "exact_triple_in_cal": sum(1 for it in test_v3 if (_norm_v3(it["prose"]), it["missing_val"], it["gold"]) in cal_v3_triples),
    }

    zero_overlap_qids = {it["qid"] for it in test_v3 if _norm_v3(it["prose"]) not in cal_v3_proses}
    seen_zero_proses = set()
    zero_uniq_qids = set()
    for it in test_v3:
        np_s = _norm_v3(it["prose"])
        if np_s not in cal_v3_proses and np_s not in seen_zero_proses:
            seen_zero_proses.add(np_s)
            zero_uniq_qids.add(it["qid"])

    v3_eval_meths = [
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "ACL-2026-VoI (Live)+Full-Defer+MDP",
        "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)",
        "Joint-Greedy+Naor",
        "Joint-Greedy+MDP",
        "Joint-Greedy+MDP (2C-Self)",
        "Calibrated-Cascade+Naor",
        "Calibrated-Cascade+MDP",
        "Calibrated-Cascade+MDP (2C-Self)",
        "Tri-VoI-L2D",
        "Tri-VoI-MDP",
        "Tri-VoI-MDP (Deploy-Self)",
    ]
    part_d_rows = []
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        cp = probe_dataset_profiled(cal_v3, mid, tracker, dry_run=False, workers=1)
        tp = probe_dataset_profiled(test_v3, mid, tracker, dry_run=False, workers=1)
        cb = HeldOutCalibrator(cp, stratify_post_clear=False)
        tp_zero = [r for r in tp if r["item"]["qid"] in zero_overlap_qids]
        tp_zuniq = [r for r in tp if r["item"]["qid"] in zero_uniq_qids]
        for slice_lbl, slice_tp in [
            (f"Full ({len(tp)})", tp),
            (f"Zero-Cal-Prose-Overlap ({len(tp_zero)})", tp_zero),
            (f"Zero-Overlap-Unique-Prose ({len(tp_zuniq)})", tp_zuniq),
        ]:
            for r_lbl, cap, di in [("Cap=2, c=1.5", 2, 2), ("Cap=1/3, c=1.5", 1, 3)]:
                vals_d = {m: _mean_util(m, slice_tp, cb, cp, cap=cap, di=di) for m in v3_eval_meths}
                raw_self = [simulate_stream("Tri-VoI-MDP (Deploy-Self)", slice_tp, cb, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=cp)["per_item_utils"] for s in range(5)]
                best_bm = max(["Calibrated-Cascade+MDP", "Joint-Greedy+MDP", "ACL-2026-VoI (Live)+Full-Defer+MDP"], key=lambda km: vals_d[km])
                raw_bm = [simulate_stream(best_bm, slice_tp, cb, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=cp)["per_item_utils"] for s in range(5)]
                bp = block_paired_bootstrap(np.mean(raw_self, axis=0), np.mean(raw_bm, axis=0), block_size=6, n_boot=3000, seed=2029)
                best_2cbm = max(["Calibrated-Cascade+MDP (2C-Self)", "Joint-Greedy+MDP (2C-Self)", "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)"], key=lambda km: vals_d[km])
                raw_2cbm = [simulate_stream(best_2cbm, slice_tp, cb, capacity=cap, drain_interval=di, c_ask=1.5, seed=s, cal_probed=cp)["per_item_utils"] for s in range(5)]
                bp_2c = block_paired_bootstrap(np.mean(raw_self, axis=0), np.mean(raw_2cbm, axis=0), block_size=6, n_boot=3000, seed=2029)
                part_d_rows.append({
                    "model": mid.split("/")[-1],
                    "slice": slice_lbl,
                    "regime": r_lbl,
                    "vals": vals_d,
                    "best_base_mdp_name": best_bm,
                    "boot_p_deploy_vs_best_base_mdp": bp["boot_p"],
                    "best_2c_self_base_mdp_name": best_2cbm,
                    "boot_p_deploy_vs_best_2c_self_base_mdp": bp_2c["boot_p"],
                })

    sqlite_audit = _audit_sqlite_post_missing()
    snap = tracker.snapshot()
    assert snap["live_calls"] == 0, f"Unexpected live API call! {snap}"

    payload = {
        "budget": snap,
        "sqlite_post_missing_fallback_audit": sqlite_audit,
        "part_a_stepwise_ablation": part_a_rows,
        "part_b_fair_baseline_mdp": part_b_rows,
        "part_c_post_missing_signal_audit": part_c_signal,
        "part_d_v3_overlap_audit": overlap_audit,
        "part_d_v3_zero_overlap_slices": part_d_rows,
        "part_e_matched_2c_self_attribution": part_e_rows,
        "part_f_c_ask_regime_map": part_f_rows,
    }
    (RESULTS_DIR / "attribution_and_dedup_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Offline Attribution, Fair 2-Call Zero-Oracle `Baseline+MDP`, `c_ask` Regime Map & `Benchmark v3` Audit (`live=0`, `¥0.00`)",
        "",
        f"- **Global SQLite `post_ask_combined*` Fallback Audit**: `{sqlite_audit['total_missing_key_fallbacks']}/{sqlite_audit['total_post_ask_combined_rows']}` rows historically lacked `missing_user_info` (all truncated `length` JSON responses; `Deploy-Self` now strictly falls back to `obs_missing` and never accesses `oracle_revealed`).",
        "",
        "## Part A: Step-by-Step Controlled Ablation (`L2D` -> `2-Call` -> `2-Call+NoInfo` -> `MDP (5-Call)` -> `Tri-VoI-MDP` -> `Deploy-Self`)",
        "| Suite | Regime | `L2D (5C)` | `+2-Call` | `+2C+NoInfo` | `MDP (5C)` | `Tri-VoI-MDP` | `Deploy-Self` |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in part_a_rows:
        v = r["vals"]
        md.append(
            f"| {r['suite']} | {r['regime']} | {v['Tri-VoI-L2D']:.3f} | {v['Tri-VoI-L2D (2-Call)']:.3f} | "
            f"{v['Tri-VoI-L2D (2-Call+NoInfo)']:.3f} | {v['Tri-VoI-MDP (5-Call)']:.3f} | "
            f"{v['Tri-VoI-MDP']:.3f} | {v['Tri-VoI-MDP (Deploy-Self)']:.3f} |"
        )

    md.extend([
        "",
        "## Part B: 5-Call `Baseline+MDP` vs. `Tri-VoI-MDP` (Historical Cross-Call Comparison)",
        "| Suite | Regime | `CC+N` | `CC+MDP` | `JG+N` | `JG+MDP` | `ACL+FD+N` | `ACL+FD+MDP` | `Tri-VoI-MDP` | `Deploy-Self` | EVOI Net (Self vs Best Base+MDP) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in part_b_rows:
        v = r["vals"]
        an, am = r["acl_naor_key"], r["acl_mdp_key"]
        md.append(
            f"| {r['suite']} | {r['regime']} | {v['Calibrated-Cascade+Naor']:.3f} | {v['Calibrated-Cascade+MDP']:.3f} | "
            f"{v['Joint-Greedy+Naor']:.3f} | {v['Joint-Greedy+MDP']:.3f} | {v[an]:.3f} | {v[am]:.3f} | "
            f"{v['Tri-VoI-MDP']:.3f} | {v['Tri-VoI-MDP (Deploy-Self)']:.3f} | {r['deploy_self_net_vs_best_base_mdp']:+.3f} |"
        )

    md.extend([
        "",
        "## Part C: Deployable `post_missing` Signal vs. Simulator `not oracle_revealed` (on `obs_missing=True` Test Items)",
        "| Suite | `obs_missing` N | True NoInfo | Pred `post_missing` | Unparsed (`obs_missing` Fallback) | Agreement% | Precision | Recall | F1 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for s_lbl, st in part_c_signal.items():
        md.append(
            f"| {s_lbl} | {st['n_obs_missing']} | {st['true_no_info']} | {st['pred_post_missing']} | "
            f"{st['unparsed_fallback_to_obs']} | {st['agreement_pct']:.2f}% | {st['precision']:.3f} | {st['recall']:.3f} | {st['f1']:.3f} |"
        )

    md.extend([
        "",
        "## Part D: `Benchmark v3` Content Overlap Disclosure & Strict Zero-Overlap Test Slices",
        f"- **Overlap Audit**: `n_cal={overlap_audit['n_cal']}`, `n_test={overlap_audit['n_test']}`, `uniq_cal_prose={overlap_audit['uniq_cal_prose']}`, `uniq_test_prose={overlap_audit['uniq_test_prose']}`, `test_prose_in_cal={overlap_audit['test_prose_in_cal']}/{overlap_audit['n_test']}` (`{100*overlap_audit['test_prose_in_cal']/overlap_audit['n_test']:.1f}%`), `exact_(prose,missing_val,gold)_in_cal={overlap_audit['exact_triple_in_cal']}/{overlap_audit['n_test']}` (`{100*overlap_audit['exact_triple_in_cal']/overlap_audit['n_test']:.1f}%`).",
        "",
        "| Model | Slice | Regime | `ACL+FD+MDP (2C-S)` | `JG+MDP (2C-S)` | `CC+MDP (2C-S)` | `CC+MDP (5C)` | `JG+MDP (5C)` | `Deploy-Self` | p (Self vs Best 2C-S) |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in part_d_rows:
        v = r["vals"]
        md.append(
            f"| {r['model']} | {r['slice']} | {r['regime']} | "
            f"{v['ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)']:.3f} | {v['Joint-Greedy+MDP (2C-Self)']:.3f} | "
            f"{v['Calibrated-Cascade+MDP (2C-Self)']:.3f} | {v['Calibrated-Cascade+MDP']:.3f} | "
            f"{v['Joint-Greedy+MDP']:.3f} | {v['Tri-VoI-MDP (Deploy-Self)']:.3f} | "
            f"{r['boot_p_deploy_vs_best_2c_self_base_mdp']:.4f} |"
        )

    md.extend([
        "",
        "## Part E: Strictly Matched 2-Call Zero-Oracle Fair Comparison (Equal Probe Cost, Zero Oracle & Same DP Pricing)",
        "| Suite | Regime | `CC+MDP (2C-S)` | `CC+OwnDP` | `JG+MDP (2C-S)` | `JG+OwnDP` | `ACL+FD+MDP (2C-S)` | `Deploy-Self` | Net vs Shared (`p`) | Net vs OwnDP (`p`) | Tok (`CC/JG/Self`) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ])
    for r in part_e_rows:
        v = r["vals"]
        tk = r["tokens"]
        a2c = r["acl_2c_key"]
        md.append(
            f"| {r['suite']} | {r['regime']} | {v['Calibrated-Cascade+MDP (2C-Self)']:.3f} | "
            f"{v['Calibrated-Cascade+OwnDP (2C-Self)']:.3f} | {v['Joint-Greedy+MDP (2C-Self)']:.3f} | "
            f"{v['Joint-Greedy+OwnDP (2C-Self)']:.3f} | {v[a2c]:.3f} | {v['Tri-VoI-MDP (Deploy-Self)']:.3f} | "
            f"{r['evoi_net_vs_best_shared_2c']:+.3f} (`{r['boot_p_vs_best_shared_2c']:.4f}`) | "
            f"{r['evoi_net_vs_best_own_2c']:+.3f} (`{r['boot_p_vs_best_own_2c']:.4f}`) | "
            f"{tk['Calibrated-Cascade+MDP (2C-Self)']:.0f}/{tk['Joint-Greedy+MDP (2C-Self)']:.0f}/{tk['Tri-VoI-MDP (Deploy-Self)']:.0f} |"
        )

    md.extend([
        "",
        "## Part F: Clarification-Cost `c_ask` Regime Map Under Matched 2-Call Zero-Oracle Pricing (Proposition 5)",
        "| Suite | Cap | `c_ask` | `CC+MDP (2C-S)` | `JG+MDP (2C-S)` | `ACL+FD+MDP (2C-S)` | `Deploy-Self` | EVOI Net vs Best 2C-S |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in part_f_rows:
        v = r["vals"]
        a2c = r["acl_2c_key"]
        md.append(
            f"| {r['suite']} | {r['cap_regime']} | {r['c_ask']:.1f} | {v['Calibrated-Cascade+MDP (2C-Self)']:.3f} | "
            f"{v['Joint-Greedy+MDP (2C-Self)']:.3f} | {v[a2c]:.3f} | {v['Tri-VoI-MDP (Deploy-Self)']:.3f} | "
            f"{r['evoi_net_vs_best_2c']:+.3f} |"
        )

    (RESULTS_DIR / "attribution_and_dedup_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved results/attribution_and_dedup_audit.{{json,md}} | live={snap['live_calls']} rmb=¥{snap['estimated_rmb']:.2f}")


if __name__ == "__main__":
    main()
