import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark import DOMAIN_RUBRIC, build_cscw_benchmark_v2
from controller import HeldOutCalibrator, evaluate_diagnosis_metrics, simulate_stream
from llm_client import RESULTS_DIR
from overnight.overnight_lib import BudgetTracker
from public_tri import load_public_tri_holdout_dataset
from run_v2_and_public_tri import compute_public_tri_noncircular_stats, eval_regime_with_sig, probe_dataset_profiled


DOMAIN_RUBRIC_UNSEEN = re.sub(r"\(e\.g\.,[^)]+\)\s*", "", DOMAIN_RUBRIC)

_UNSEEN_MAP = [
    ("Onco-B7", "Track Delta-44"),
    ("Cardiac-Watch CW-7", "Clinical Pathway K-91"),
    ("Compliance Matrix M-14", "Compliance Matrix R-82"),
    ("Treasury Bulletin CB-19", "Treasury Bulletin F-33"),
    ("SRE Escalation Clause VIP-9", "SRE Escalation Clause Z-17"),
    ("SRE Runbook LPM-4", "SRE Runbook T-60"),
    ("Customs Waiver PX-8", "Customs Waiver N-25"),
    ("Carrier Notice PA-12", "Carrier Notice V-49"),
]


def to_unseen_v2_item(it: Dict[str, Any]) -> Dict[str, Any]:
    new_it = dict(it)
    prose = it["prose"]
    for old_tok, new_tok in _UNSEEN_MAP:
        prose = prose.replace(old_tok, new_tok)
    new_it["qid"] = f"UNSEEN_{it['qid']}"
    new_it["prose"] = prose
    new_it["custom_rubric"] = DOMAIN_RUBRIC_UNSEEN
    return new_it


def main() -> None:
    p = argparse.ArgumentParser(description="Single-Shot Zero-Touch Holdout & Unseen-ID Verification")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-budget-rmb", type=float, default=10.0)
    args = p.parse_args()

    prefix = "dry_run_" if args.dry_run else ""
    tracker = BudgetTracker(max_budget_rmb=args.max_budget_rmb, state_path=RESULTS_DIR / f"{prefix}holdout_budget_state.json")

    # Part A: Strictly Zero-Touch Disjoint MediQ Holdout (format-debiased natural prose, zero bullet headers)
    cal_h, test_h = load_public_tri_holdout_dataset()
    methods_pt = [
        "Direct-ACT",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "w/o Queue-Externality",
        "Tri-VoI-L2D",
        "Tri-VoI-L2D (2-Call)",
        "Tri-VoI-L2D (Smooth-Naor)",
        "Oracle",
    ]

    holdout_models: Dict[str, Any] = {}
    probed_cache: Dict[str, Any] = {}
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        cal_p = probe_dataset_profiled(cal_h, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="holdout_cal48")
        test_p = probe_dataset_profiled(test_h, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="holdout_test128")
        probed_cache[mid] = (cal_p, test_p)
        diag = evaluate_diagnosis_metrics(test_p)
        raw_acc = compute_public_tri_noncircular_stats(test_p)

        # Primary unified default calibrator (stratify_post_clear=False — identical to v1/v2) vs stratified comparison
        by_cal_mode: Dict[str, Any] = {}
        for strat_flag, mode_lbl in [(False, "unified_default"), (True, "stratify_post_clear")]:
            eb = HeldOutCalibrator(cal_p, stratify_post_clear=strat_flag)
            by_cal_mode[mode_lbl] = {
                "Cap=2, c=1.5": eval_regime_with_sig(methods_pt, cal_p, test_p, eb, cap=2, di=2, c_ask=1.5),
                "Cap=2, c_def=4.0, c=1.5": eval_regime_with_sig(methods_pt, cal_p, test_p, eb, cap=2, di=2, c_ask=1.5, c_defer_base=4.0),
                "Cap=1/3, c=1.5": eval_regime_with_sig(methods_pt, cal_p, test_p, eb, cap=1, di=3, c_ask=1.5),
            }
        holdout_models[mid] = {
            "diagnosis": diag,
            "raw_usmle_acc": raw_acc,
            "regimes_by_calibrator": by_cal_mode,
        }

    # Genuine Full-Quadrant Real-LLM Clinical Specialist on Zero-Touch Holdout (Qwen3.5 Agent -> Real DeepSeek-V3.2 Specialist)
    q35_cal_h, q35_test_h = probed_cache["Qwen/Qwen3.5-35B-A3B"]
    _, ds_test_h = probed_cache["deepseek-ai/DeepSeek-V3.2"]
    spec_ok_holdout = {
        r["item"]["qid"]: ((r["post_ans"] if r["item"]["is_underspec"] else r["greedy_ans"]) == r["item"]["gold"])
        for r in ds_test_h
    }
    eb_q35_h = HeldOutCalibrator(q35_cal_h, stratify_post_clear=False)
    spec_methods = [
        "Direct-ACT",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "Calibrated-Cascade+Naor",
        "Joint-Greedy+Naor",
        "Tri-VoI-L2D",
        "Tri-VoI-L2D (Smooth-Naor)",
    ]
    holdout_real_spec = {}
    for r_lbl, cap, di, c_def in [("Cap=2, c_def=2.0, c=1.5", 2, 2, 2.0), ("Cap=2, c_def=4.0, c=1.5", 2, 2, 4.0), ("Cap=1/3, c_def=2.0, c=1.5", 1, 3, 2.0)]:
        holdout_real_spec[r_lbl] = {
            m: round(float(np.mean([
                simulate_stream(m, q35_test_h, eb_q35_h, capacity=cap, drain_interval=di, c_ask=1.5, c_defer_base=c_def, seed=s, cal_probed=q35_cal_h, real_specialist_ok=spec_ok_holdout)["mean_utility"]
                for s in range(5)
            ])), 3)
            for m in spec_methods
        }

    # Part B: Benchmark v2 Unseen-Protocol-ID Sanity Check (rubric code examples stripped; unseen protocol names)
    cal_u_raw, test_u_raw = build_cscw_benchmark_v2(n_cal_per_quad=10, n_test_per_quad=30, seed=2031)
    cal_u = [to_unseen_v2_item(it) for it in cal_u_raw]
    test_u = [to_unseen_v2_item(it) for it in test_u_raw]
    for it in cal_u + test_u:
        for old_tok, _ in _UNSEEN_MAP:
            assert old_tok not in it["prose"], f"Old protocol token leaked: {old_tok}"
        assert "Onco-B7" not in DOMAIN_RUBRIC_UNSEEN and "M-14" not in DOMAIN_RUBRIC_UNSEEN

    u_cal_p = probe_dataset_profiled(cal_u, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=args.dry_run, workers=args.workers, label="v2_unseen_cal40")
    u_test_p = probe_dataset_profiled(test_u, "Qwen/Qwen3.5-35B-A3B", tracker, dry_run=args.dry_run, workers=args.workers, label="v2_unseen_test120")
    u_diag = evaluate_diagnosis_metrics(u_test_p)
    u_eb = HeldOutCalibrator(u_cal_p, stratify_post_clear=False)
    u_methods = [
        "Direct-ACT",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
        "Tri-VoI-L2D (2-Call)",
        "Tri-VoI-L2D (Smooth-Naor)",
    ]
    unseen_regimes = {
        "Cap=2, c=1.5": eval_regime_with_sig(u_methods, u_cal_p, u_test_p, u_eb, cap=2, di=2, c_ask=1.5),
        "Cap=1/3, c=1.5": eval_regime_with_sig(u_methods, u_cal_p, u_test_p, u_eb, cap=1, di=3, c_ask=1.5),
    }

    tracker.save()
    snap = tracker.snapshot()
    payload = {
        "holdout_mediq_disjoint": {
            "n_cal": len(cal_h),
            "n_test": len(test_h),
            "models": holdout_models,
            "real_llm_specialist_q35_to_ds": holdout_real_spec,
        },
        "v2_unseen_protocol_ids": {
            "model": "Qwen/Qwen3.5-35B-A3B",
            "n_cal": len(cal_u),
            "n_test": len(test_u),
            "diagnosis": u_diag,
            "regimes": unseen_regimes,
        },
        "budget": snap,
    }
    (RESULTS_DIR / f"{prefix}zero_touch_holdout.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        f"# Single-Shot Zero-Touch Holdout & Unseen-ID Report (Spend: ¥{snap['estimated_rmb']:.4f})",
        "",
        f"## Part A: Disjoint MediQ Holdout (`N_cal={len(cal_h)}`, `N_test={len(test_h)}`, Natural Prose — Zero Bullet Shortcut)",
    ]
    for mid, mdata in holdout_models.items():
        d = mdata["diagnosis"]
        nc = mdata["raw_usmle_acc"]
        md.append(
            f"### `{mid}` (Missing-F1={d['missing_f1']}, Scope-F1={d['override_f1']}, Collapse={d['entropy_collapse_pct']}%, "
            f"Raw USMLE Q1={nc['Q1_clear_easy']['zero_turn_acc_pct']}%, Q3={nc['Q3_clear_hard']['zero_turn_acc_pct']}%, "
            f"Q4res={nc['Q4_resolvable']['zero_turn_acc_pct']}%->{nc['Q4_resolvable']['post_ask_acc_pct']}%, "
            f"Q4esc={nc['Q4_escalate']['zero_turn_acc_pct']}%->{nc['Q4_escalate']['post_ask_acc_pct']}%)"
        )
        for m_lbl, regs in mdata["regimes_by_calibrator"].items():
            md.append(f"#### Calibrator: `{m_lbl}`")
            md.append("| Method | Cap=2, c=1.5 | Cap=2, c_def=4.0, c=1.5 | Cap=1/3, c=1.5 |")
            md.append("|:---|---:|---:|---:|")
            for meth in methods_pt:
                r1 = regs["Cap=2, c=1.5"][meth]
                r2 = regs["Cap=2, c_def=4.0, c=1.5"][meth]
                r3 = regs["Cap=1/3, c=1.5"][meth]
                md.append(f"| `{meth}` | {r1['mean_utility']:.3f} ({r1['acc_pct']:.1f}%) | {r2['mean_utility']:.3f} ({r2['acc_pct']:.1f}%) | {r3['mean_utility']:.3f} ({r3['acc_pct']:.1f}%) |")
            md.append("")

    md.append("### Holdout Real-LLM Clinical Specialist (`Qwen3.5-35B` Agent -> Real `DeepSeek-V3.2` Full-Quadrant)")
    for r_lbl, r_map in holdout_real_spec.items():
        md.append(f"- **{r_lbl}**: " + ", ".join(f"`{k}`={v:.3f}" for k, v in r_map.items()))
    md.append("")

    md.append(f"## Part B: `Benchmark v2` Unseen-Protocol-ID Sanity Check (`Qwen3.5-35B`, `N_cal={len(cal_u)}`, `N_test={len(test_u)}`)")
    md.append(f"- Missing-F1=`{u_diag['missing_f1']}`, Override-F1=`{u_diag['override_f1']}`, Collapse=`{u_diag['entropy_collapse_pct']}%`")
    md.append("| Method | Cap=2, c=1.5 | Cap=1/3, c=1.5 |")
    md.append("|:---|---:|---:|")
    for meth in u_methods:
        r1 = unseen_regimes["Cap=2, c=1.5"][meth]
        r2 = unseen_regimes["Cap=1/3, c=1.5"][meth]
        md.append(f"| `{meth}` | {r1['mean_utility']:.3f} ({r1['acc_pct']:.1f}%) | {r2['mean_utility']:.3f} ({r2['acc_pct']:.1f}%) |")

    (RESULTS_DIR / f"{prefix}zero_touch_holdout.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Completed single-shot zero-touch holdout (dry_run={args.dry_run}). Spend: ¥{snap['estimated_rmb']:.4f}")


if __name__ == "__main__":
    main()
