import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from controller import HeldOutCalibrator, evaluate_diagnosis_metrics, simulate_stream
from llm_client import RESULTS_DIR
from mediq_test_disjoint import load_mediq_test_disjoint_dataset
from overnight.overnight_lib import BudgetTracker
from run_all import block_paired_bootstrap
from run_attribution_and_dedup_audit import _post_missing_metrics
from run_v2_and_public_tri import probe_dataset_profiled


EVAL_METHODS = [
    "Direct-ACT",
    "KL-PBD+ (Warm)",
    "ACL-2026-VoI (Live)+Full-Defer+Naor",
    "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)",
    "Joint-Greedy+Naor",
    "Joint-Greedy+MDP (2C-Self)",
    "Joint-Greedy+OwnDP (2C-Self)",
    "Calibrated-Cascade+Naor",
    "Calibrated-Cascade+MDP (2C-Self)",
    "Calibrated-Cascade+OwnDP (2C-Self)",
    "Tri-VoI-L2D",
    "Tri-VoI-MDP",
    "Tri-VoI-MDP (Deploy-Self)",
    "Oracle",
]


def _eval_setup(
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    regimes: List[tuple],
    *,
    real_specialist_ok: Optional[Dict[str, bool]] = None,
    n_seeds: int = 5,
) -> Dict[str, Any]:
    calib = HeldOutCalibrator(cal_p, stratify_post_clear=False)
    out: Dict[str, Any] = {}
    for r_lbl, cap, di, c_ask, c_def in regimes:
        raw: Dict[str, List[Dict[str, Any]]] = {m: [] for m in EVAL_METHODS}
        for s in range(n_seeds):
            for m in EVAL_METHODS:
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
        ds_vec = np.mean([r["per_item_utils"] for r in raw["Tri-VoI-MDP (Deploy-Self)"]], axis=0)
        rows: Dict[str, Any] = {}
        for m in EVAL_METHODS:
            rs = raw[m]
            scope_vals = [r["scope_escalation_recall_pct"] for r in rs if r["scope_escalation_recall_pct"] is not None]
            rows[m] = {
                "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
                "acc_pct": round(float(np.mean([r["acc_pct"] for r in rs])), 2),
                "raw_answer_acc_pct": round(float(np.mean([r["raw_answer_acc_pct"] for r in rs])), 2),
                "scope_escalation_recall_pct": round(float(np.mean(scope_vals)), 2) if scope_vals else None,
                "ask_rate_pct": round(float(np.mean([r["ask_rate_pct"] for r in rs])), 1),
                "defer_rate_pct": round(float(np.mean([r["defer_rate_pct"] for r in rs])), 1),
                "mean_realized_tokens": round(float(np.mean([r["mean_realized_tokens"] for r in rs])), 1),
            }
        best_2c_shared = max(
            ["Calibrated-Cascade+MDP (2C-Self)", "Joint-Greedy+MDP (2C-Self)", "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)"],
            key=lambda mk: rows[mk]["mean_utility"],
        )
        best_2c_own = max(
            ["Calibrated-Cascade+OwnDP (2C-Self)", "Joint-Greedy+OwnDP (2C-Self)", "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)"],
            key=lambda mk: rows[mk]["mean_utility"],
        )
        bp_shared = block_paired_bootstrap(
            ds_vec,
            np.mean([r["per_item_utils"] for r in raw[best_2c_shared]], axis=0),
            block_size=6,
            n_boot=3000,
            seed=2029,
        )
        bp_own = block_paired_bootstrap(
            ds_vec,
            np.mean([r["per_item_utils"] for r in raw[best_2c_own]], axis=0),
            block_size=6,
            n_boot=3000,
            seed=2029,
        )
        bp_l2d = block_paired_bootstrap(
            ds_vec,
            np.mean([r["per_item_utils"] for r in raw["Tri-VoI-L2D"]], axis=0),
            block_size=6,
            n_boot=3000,
            seed=2029,
        )
        out[r_lbl] = {
            "methods": rows,
            "summary": {
                "best_2c_shared_name": best_2c_shared,
                "deploy_self_vs_best_2c_shared": round(rows["Tri-VoI-MDP (Deploy-Self)"]["mean_utility"] - rows[best_2c_shared]["mean_utility"], 3),
                "boot_p_vs_best_2c_shared": bp_shared["boot_p"],
                "best_2c_own_name": best_2c_own,
                "deploy_self_vs_best_2c_own": round(rows["Tri-VoI-MDP (Deploy-Self)"]["mean_utility"] - rows[best_2c_own]["mean_utility"], 3),
                "boot_p_vs_best_2c_own": bp_own["boot_p"],
                "deploy_self_vs_l2d": round(rows["Tri-VoI-MDP (Deploy-Self)"]["mean_utility"] - rows["Tri-VoI-L2D"]["mean_utility"], 3),
                "boot_p_vs_l2d": bp_l2d["boot_p"],
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-Registered Organ-System-Disjoint MediQ-Test Blind Evaluation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-budget-rmb", type=float, default=7.0)
    args = parser.parse_args()

    budget_path = RESULTS_DIR / ("dry_run_mediq_test_disjoint_budget.json" if args.dry_run else "mediq_test_disjoint_budget_state.json")
    if args.dry_run and budget_path.exists():
        budget_path.unlink()
    tracker = BudgetTracker(max_budget_rmb=args.max_budget_rmb, state_path=budget_path)

    cal_items, test_items = load_mediq_test_disjoint_dataset()

    q35_cal = probe_dataset_profiled(cal_items, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=args.dry_run, workers=args.workers, label="mtd_q35_cal")
    q35_test = probe_dataset_profiled(test_items, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=args.dry_run, workers=args.workers, label="mtd_q35_test")

    ds_cal = probe_dataset_profiled(cal_items, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=args.dry_run, workers=args.workers, label="mtd_ds_cal")
    ds_test = probe_dataset_profiled(test_items, "deepseek-ai/DeepSeek-V3.2", tracker, dry_run=args.dry_run, workers=args.workers, label="mtd_ds_test")

    ds_ok = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in ds_test
    }

    regimes = [
        ("Cap=2, c=1.5", 2, 2, 1.5, 2.0),
        ("Cap=2, c=3.0", 2, 2, 3.0, 2.0),
        ("Cap=1/3, c=1.5", 1, 3, 1.5, 2.0),
        ("Cap=1/3, c=3.0", 1, 3, 3.0, 2.0),
    ]

    res_q35_realds = _eval_setup(q35_cal, q35_test, regimes, real_specialist_ok=ds_ok, n_seeds=5)
    res_ds_gen = _eval_setup(ds_cal, ds_test, regimes, real_specialist_ok=None, n_seeds=5)

    snap = tracker.snapshot()
    payload = {
        "schema_version": "mediq_test_disjoint_eval_v1",
        "budget": snap,
        "diagnosis": {
            "Qwen/Qwen3.5-35B-A3B": evaluate_diagnosis_metrics(q35_test),
            "deepseek-ai/DeepSeek-V3.2": evaluate_diagnosis_metrics(ds_test),
        },
        "post_missing_audit": {
            "Qwen/Qwen3.5-35B-A3B": _post_missing_metrics(q35_test),
            "deepseek-ai/DeepSeek-V3.2": _post_missing_metrics(ds_test),
        },
        "setup_qwen35_to_real_ds": res_q35_realds,
        "setup_deepseek_generalist": res_ds_gen,
    }

    prefix = "dry_run_" if args.dry_run else ""
    out_json = RESULTS_DIR / f"{prefix}mediq_test_disjoint_eval.json"
    out_md = RESULTS_DIR / f"{prefix}mediq_test_disjoint_eval.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Pre-Registered Organ-System-Disjoint `MediQ-Test-Disjoint` Blind Evaluation (`N_cal=60, N_test=180`)",
        f"- Live Calls: `{snap['live_calls']}`, Cached Hits: `{snap['cached_hits']}`, Spent: `¥{snap['estimated_rmb']:.2f}`",
        "",
        "## Diagnostic & `post_missing` Signal Audit (Zero Oracle Fallback)",
    ]
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        dg = payload["diagnosis"][mid]
        pm = payload["post_missing_audit"][mid]
        md.append(
            f"- **`{mid}`**: Miss-F1=`{dg['missing_f1']:.3f}`, Override-F1=`{dg['override_f1']:.3f}`, "
            f"`post_missing` Agree=`{pm['agreement_pct']:.2f}%`, F1=`{pm['f1']:.3f}`, Unparsed Fallback=`{pm['unparsed_fallback_to_obs']}`."
        )

    for title, sdata in [
        ("## Setup A: Cross-Group Real-LLM Specialist (`Qwen3.5-35B-A3B` -> `DeepSeek-V3.2`)", res_q35_realds),
        ("## Setup B: Cross-Group Generalist (`DeepSeek-V3.2`)", res_ds_gen),
    ]:
        md.extend(["", title, "| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 | Tok (Cap=2, c=1.5) |", "|:---|---:|---:|---:|---:|---:|"])
        for m in EVAL_METHODS:
            c1 = sdata["Cap=2, c=1.5"]["methods"][m]
            c2 = sdata["Cap=2, c=3.0"]["methods"][m]
            c3 = sdata["Cap=1/3, c=1.5"]["methods"][m]
            c4 = sdata["Cap=1/3, c=3.0"]["methods"][m]
            md.append(
                f"| `{m}` | {c1['mean_utility']:.3f} ({c1['raw_answer_acc_pct']:.1f}%/{c1['scope_escalation_recall_pct']:.1f}%) | "
                f"{c2['mean_utility']:.3f} | {c3['mean_utility']:.3f} ({c3['raw_answer_acc_pct']:.1f}%/{c3['scope_escalation_recall_pct']:.1f}%) | "
                f"{c4['mean_utility']:.3f} | {c1['mean_realized_tokens']:.0f} |"
            )
        md.append("")
        for r_lbl, _, _, _, _ in regimes:
            sm = sdata[r_lbl]["summary"]
            md.append(
                f"- **{r_lbl}**: `Deploy-Self` vs `{sm['best_2c_shared_name']}` = `{sm['deploy_self_vs_best_2c_shared']:+.3f}` (`p={sm['boot_p_vs_best_2c_shared']:.4f}`), "
                f"vs `{sm['best_2c_own_name']}` = `{sm['deploy_self_vs_best_2c_own']:+.3f}` (`p={sm['boot_p_vs_best_2c_own']:.4f}`), "
                f"vs `Tri-VoI-L2D` = `{sm['deploy_self_vs_l2d']:+.3f}` (`p={sm['boot_p_vs_l2d']:.4f}`)."
            )

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {out_json} and {out_md} | Spent RMB {snap['estimated_rmb']:.2f}")


if __name__ == "__main__":
    main()
