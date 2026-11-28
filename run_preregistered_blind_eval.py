import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark_v3 import load_benchmark_v3_dataset
from controller import HeldOutCalibrator, evaluate_diagnosis_metrics, simulate_stream
from llm_client import RESULTS_DIR
from overnight.overnight_lib import BudgetTracker
from public_tri import load_public_tri_holdout_dataset
from run_all import block_paired_bootstrap, holm_bonferroni
from run_v2_and_public_tri import probe_dataset_profiled


BLIND_METHODS = [
    "Direct-ACT",
    "Naive-Cascade",
    "KL-PBD+ (Warm)",
    "ACL-2026-VoI (Live)+Post-Defer",
    "ACL-2026-VoI (Live)+Full-Defer+Naor",
    "Joint-Greedy+Naor",
    "Calibrated-Cascade+Naor",
    "Tri-VoI-L2D",
    "Tri-VoI-MDP",
    "Oracle",
]


def eval_blind_regimes(
    methods: List[str],
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    regimes: List[tuple],
    *,
    real_specialist_ok: Dict[str, bool] | None = None,
    n_seeds: int = 5,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r_lbl, cap, di, c_ask, c_def in regimes:
        raw: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
        for s in range(n_seeds):
            for m in methods:
                raw[m].append(
                    simulate_stream(
                        m,
                        test_p,
                        calib,
                        capacity=cap,
                        drain_interval=di,
                        drift_mode="sudden",
                        c_ask=c_ask,
                        c_defer_base=c_def,
                        seed=s,
                        cal_probed=cal_p,
                        real_specialist_ok=real_specialist_ok,
                    )
                )
        mdp_vec = np.mean([r["per_item_utils"] for r in raw["Tri-VoI-MDP"]], axis=0)
        tri_vec = np.mean([r["per_item_utils"] for r in raw["Tri-VoI-L2D"]], axis=0)
        ps_mdp: Dict[str, float] = {}
        rows: Dict[str, Any] = {}
        for m in methods:
            rs = raw[m]
            scope_vals = [r["scope_escalation_recall_pct"] for r in rs if r["scope_escalation_recall_pct"] is not None]
            row = {
                "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
                "acc_pct": round(float(np.mean([r["acc_pct"] for r in rs])), 2),
                "raw_answer_acc_pct": round(float(np.mean([r["raw_answer_acc_pct"] for r in rs])), 2),
                "scope_escalation_recall_pct": round(float(np.mean(scope_vals)), 2) if scope_vals else None,
                "ask_rate_pct": round(float(np.mean([r["ask_rate_pct"] for r in rs])), 1),
                "defer_rate_pct": round(float(np.mean([r["defer_rate_pct"] for r in rs])), 1),
                "mean_wait": round(float(np.mean([r["mean_wait"] for r in rs])), 2),
                "mean_realized_tokens": round(float(np.mean([r["mean_realized_tokens"] for r in rs])), 1),
            }
            b_vec = np.mean([r["per_item_utils"] for r in rs], axis=0)
            if m not in ("Tri-VoI-MDP", "Oracle"):
                bp = block_paired_bootstrap(mdp_vec, b_vec, block_size=8, n_boot=3000, seed=2029)
                row["boot_p_vs_mdp"] = bp["boot_p"]
                ps_mdp[m] = float(bp["boot_p"])
            if m not in ("Tri-VoI-L2D", "Oracle"):
                bp_l2d = block_paired_bootstrap(tri_vec, b_vec, block_size=8, n_boot=3000, seed=2029)
                row["boot_p_vs_l2d"] = bp_l2d["boot_p"]
            rows[m] = row
        hm = holm_bonferroni(ps_mdp)
        for m, pv in hm.items():
            rows[m]["boot_p_mdp_holm"] = pv
        out[r_lbl] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-Registered Zero-Touch Blind Evaluation (Benchmark v3 + MediQ-Holdout)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-budget-rmb", type=float, default=10.0)
    args = parser.parse_args()

    budget_path = RESULTS_DIR / ("dry_run_blind_budget.json" if args.dry_run else "blind_v3_budget_state.json")
    if args.dry_run and budget_path.exists():
        budget_path.unlink()
    tracker = BudgetTracker(max_budget_rmb=args.max_budget_rmb, state_path=budget_path)

    # Suite A: Benchmark v3 (4 Unseen Length-Matched Domains, Zero Protocol Codes)
    cal_v3, test_v3 = load_benchmark_v3_dataset()
    v3_regimes = [
        ("Cap=2, c=1.5", 2, 2, 1.5, 2.0),
        ("Cap=2, c=2.5", 2, 2, 2.5, 2.0),
        ("Cap=1/3, c=1.5", 1, 3, 1.5, 2.0),
    ]
    v3_results: Dict[str, Any] = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        cp = probe_dataset_profiled(cal_v3, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="v3_cal")
        tp = probe_dataset_profiled(test_v3, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="v3_test")
        eb = HeldOutCalibrator(cp, stratify_post_clear=False)  # FROZEN hyperparameters
        v3_results[mid] = {
            "diagnosis_test": evaluate_diagnosis_metrics(tp),
            "regimes": eval_blind_regimes(BLIND_METHODS, cp, tp, eb, v3_regimes, n_seeds=5),
        }

    # Suite B: Disjoint MediQ-Holdout (Pure Capability-Gap Real-LLM + Scope Compliance Split)
    cal_h, test_h = load_public_tri_holdout_dataset()
    q8_cal_h = probe_dataset_profiled(cal_h, "Qwen/Qwen3-8B", tracker, dry_run=args.dry_run, workers=args.workers, label="h_q8_cal")
    q8_test_h = probe_dataset_profiled(test_h, "Qwen/Qwen3-8B", tracker, dry_run=args.dry_run, workers=args.workers, label="h_q8_test")
    q35_cal_h = probe_dataset_profiled(cal_h, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=args.dry_run, workers=args.workers, label="h_q35_cal")
    q35_test_h = probe_dataset_profiled(test_h, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=args.dry_run, workers=args.workers, label="h_q35_test")
    ds_test_h = probe_dataset_profiled(test_h, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=args.dry_run, workers=args.workers, label="h_ds_test")

    q35_ok_h = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in q35_test_h
    }
    ds_ok_h = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in ds_test_h
    }

    # B1: Scope-Compliance Real-LLM (Qwen3.5 -> DeepSeek-V3.2, with require_defer_for_ok preserved, split reporting)
    eb_q35_h = HeldOutCalibrator(q35_cal_h, stratify_post_clear=False)
    h_scope_regimes = [
        ("Cap=2, c_def=2.0, c=1.5", 2, 2, 1.5, 2.0),
        ("Cap=2, c_def=4.0, c=1.5", 2, 2, 1.5, 4.0),
        ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 1.5, 2.0),
    ]
    h_scope_q35_ds = eval_blind_regimes(
        BLIND_METHODS, q35_cal_h, q35_test_h, eb_q35_h, h_scope_regimes, real_specialist_ok=ds_ok_h, n_seeds=5
    )

    # B2: Pure Capability-Gap Real-LLM Specialist (require_defer_for_ok=False everywhere; zero forced DEFER penalty)
    q8_cal_cap = [{**r, "item": {**r["item"], "require_defer_for_ok": False}} for r in q8_cal_h]
    q8_test_cap = [{**r, "item": {**r["item"], "require_defer_for_ok": False}} for r in q8_test_h]
    eb_q8_cap = HeldOutCalibrator(q8_cal_cap, stratify_post_clear=False)
    h_cap_regimes = [
        ("Cap=2, c_def=2.0, c=1.5", 2, 2, 1.5, 2.0),
        ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 1.5, 2.0),
    ]
    h_cap_q8_q35 = eval_blind_regimes(
        BLIND_METHODS, q8_cal_cap, q8_test_cap, eb_q8_cap, h_cap_regimes, real_specialist_ok=q35_ok_h, n_seeds=5
    )
    h_cap_q8_ds = eval_blind_regimes(
        BLIND_METHODS, q8_cal_cap, q8_test_cap, eb_q8_cap, h_cap_regimes, real_specialist_ok=ds_ok_h, n_seeds=5
    )

    snap = tracker.snapshot()
    payload = {
        "schema_version": "preregistered_blind_eval_v3",
        "budget": snap,
        "suite_a_benchmark_v3": v3_results,
        "suite_b_mediq_holdout": {
            "q8_diag": evaluate_diagnosis_metrics(q8_test_h),
            "q35_diag": evaluate_diagnosis_metrics(q35_test_h),
            "scope_compliance_qwen35_to_deepseek": h_scope_q35_ds,
            "pure_cap_gap_qwen8b_to_qwen35": h_cap_q8_q35,
            "pure_cap_gap_qwen8b_to_deepseek": h_cap_q8_ds,
        },
    }

    prefix = "dry_run_" if args.dry_run else ""
    out_json = RESULTS_DIR / f"{prefix}preregistered_blind_eval.json"
    out_md = RESULTS_DIR / f"{prefix}preregistered_blind_eval.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Pre-Registered Zero-Touch Blind Evaluation (`Benchmark v3` + Pure Capability-Gap `MediQ-Holdout`)",
        f"- Live Calls: `{snap['live_calls']}`, Cached Hits: `{snap['cached_hits']}`, Spent: `¥{snap['estimated_rmb']:.2f}`",
        "",
        "## Suite A: `Benchmark v3` (4 Unseen Length-Matched Domains, Zero Protocol Codes)",
    ]
    for mid, mdata in v3_results.items():
        md.append(f"### `{mid}` (Missing-F1=`{mdata['diagnosis_test']['missing_f1']}`, Override-F1=`{mdata['diagnosis_test']['override_f1']}`)")
        md.append("| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 | Realized Tok (Cap=2) |")
        md.append("|:---|---:|---:|---:|---:|")
        for m in BLIND_METHODS:
            r1 = mdata["regimes"]["Cap=2, c=1.5"][m]
            r2 = mdata["regimes"]["Cap=2, c=2.5"][m]
            r3 = mdata["regimes"]["Cap=1/3, c=1.5"][m]
            md.append(
                f"| `{m}` | {r1['mean_utility']:.3f} ({r1['acc_pct']:.1f}%) | "
                f"{r2['mean_utility']:.3f} ({r2['acc_pct']:.1f}%) | "
                f"{r3['mean_utility']:.3f} ({r3['acc_pct']:.1f}%) | {r1['mean_realized_tokens']:.0f} |"
            )
        md.append("")

    md.append("## Suite B1: `MediQ-Holdout` Scope-Compliance Real-LLM Specialist (`Qwen3.5` -> `DeepSeek-V3.2`)")
    md.append("| Method | Cap=2 (Util / RawMedAcc% / EscRec%) | Cap=2 HighCost (Util / RawMedAcc% / EscRec%) | Cap=1/3 (Util / RawMedAcc% / EscRec%) |")
    md.append("|:---|---:|---:|---:|")
    for m in BLIND_METHODS:
        r1 = h_scope_q35_ds["Cap=2, c_def=2.0, c=1.5"][m]
        r2 = h_scope_q35_ds["Cap=2, c_def=4.0, c=1.5"][m]
        r3 = h_scope_q35_ds["Cap=1/3, c_def=2.0, c=1.5"][m]
        md.append(
            f"| `{m}` | {r1['mean_utility']:.3f} / {r1['raw_answer_acc_pct']:.1f}% / {r1['scope_escalation_recall_pct']:.1f}% | "
            f"{r2['mean_utility']:.3f} / {r2['raw_answer_acc_pct']:.1f}% / {r2['scope_escalation_recall_pct']:.1f}% | "
            f"{r3['mean_utility']:.3f} / {r3['raw_answer_acc_pct']:.1f}% / {r3['scope_escalation_recall_pct']:.1f}% |"
        )
    md.append("")

    md.append("## Suite B2: `MediQ-Holdout` Pure Capability-Gap Real-LLM Specialist (No Forced Escalation Penalty)")
    for lbl, bmap in [
        ("`Qwen3-8B` -> Real `Qwen3.5-35B-A3B`", h_cap_q8_q35),
        ("`Qwen3-8B` -> Real `DeepSeek-V3.2`", h_cap_q8_ds),
    ]:
        md.append(f"### {lbl}")
        md.append("| Method | Cap=2 (Util / Acc%) | Cap=1/3 (Util / Acc%) | Realized Tok (Cap=2) |")
        md.append("|:---|---:|---:|---:|")
        for m in BLIND_METHODS:
            r1 = bmap["Cap=2, c_def=2.0, c=1.5"][m]
            r2 = bmap["Cap=1/3, c_def=2.0, c=1.5"][m]
            md.append(f"| `{m}` | {r1['mean_utility']:.3f} ({r1['raw_answer_acc_pct']:.1f}%) | {r2['mean_utility']:.3f} ({r2['raw_answer_acc_pct']:.1f}%) | {r1['mean_realized_tokens']:.0f} |")
        md.append("")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {out_json} and {out_md} | Spent ¥{snap['estimated_rmb']:.2f}")


if __name__ == "__main__":
    main()
