import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import build_cscw_benchmark_v2, split_stratified_episodes
from controller import HeldOutCalibrator, evaluate_diagnosis_metrics, simulate_stream
from llm_client import RESULTS_DIR, call_llm
from overnight.overnight_lib import BudgetTracker
from probe_profiles import assemble_probed_record, build_wave_a_specs, build_wave_b_specs
from public_tri import load_public_tri_dataset
from run_all import block_paired_bootstrap, eval_independent_episodes, holm_bonferroni


FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

V2_MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-8B",
    "tencent/Hy4-preview",
]

PUBLIC_TRI_MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-8B",
]

METHODS_V2 = [
    "Direct-ACT",
    "Entropy-ASK (Proxy)",
    "Consensus-DEFER (Proxy)",
    "Congested-L2D-Only",
    "Naive-Cascade",
    "KL-PBD",
    "KL-PBD+",
    "KL-PBD+ (Warm)",
    "ACL-2026-VoI (Calib)",
    "ACL-2026-VoI+Post-Defer",
    "ACL-2026-VoI (Live)",
    "ACL-2026-VoI (Live)+Post-Defer",
    "Joint-Greedy+Naor",
    "Calibrated-Cascade+Naor",
    "Tri-VoI-L2D",
    "Oracle",
]

METHODS_PT = [
    "Direct-ACT",
    "Congested-L2D-Only",
    "Naive-Cascade",
    "KL-PBD",
    "KL-PBD+",
    "KL-PBD+ (Warm)",
    "ACL-2026-VoI (Calib)",
    "ACL-2026-VoI+Post-Defer",
    "ACL-2026-VoI (Live)",
    "ACL-2026-VoI (Live)+Post-Defer",
    "Joint-Greedy+Naor",
    "Calibrated-Cascade+Naor",
    "Tri-VoI-L2D",
    "Oracle",
]


def _call_with_451_retry(spec: Dict[str, Any], model_id: str, dry_run: bool) -> Dict[str, Any]:
    for attempt in range(2):
        try:
            return call_llm(
                spec["messages"],
                model_id=model_id,
                temperature=spec["temperature"],
                max_tokens=spec["max_tokens"],
                tag=spec["tag"],
                seed_idx=spec["seed_idx"],
                dry_run=dry_run,
            )
        except RuntimeError as exc:
            if attempt == 0 and "451" in str(exc):
                time.sleep(1.0)
                continue
            raise
    raise RuntimeError("Unreachable")


def probe_item_with_profile(
    item: Dict[str, Any],
    model_id: str,
    tracker: BudgetTracker,
    *,
    dry_run: bool = False,
    pace_sec: float = 0.05,
) -> Dict[str, Any]:
    responses: Dict[str, Dict[str, Any]] = {}
    for spec in build_wave_a_specs(item, dry_run=dry_run):
        tracker.check_budget()
        resp = _call_with_451_retry(spec, model_id, dry_run)
        tracker.record(model_id, resp["prompt_tokens"], resp["completion_tokens"], bool(resp.get("cached")), dry_run)
        responses[spec["tag"]] = resp
        if not resp.get("cached") and not dry_run and pace_sec > 0:
            time.sleep(pace_sec)

    from controller import parse_json_obj
    diag_obj = parse_json_obj(responses["diag_init"]["content"])
    clarify_q = str(diag_obj.get("clarifying_question", "")).strip()

    oracle_reply, oracle_revealed, specs_b = build_wave_b_specs(
        item, clarify_q, dry_run=dry_run, include_acl_voi=True
    )
    for spec in specs_b:
        tracker.check_budget()
        resp = _call_with_451_retry(spec, model_id, dry_run)
        tracker.record(model_id, resp["prompt_tokens"], resp["completion_tokens"], bool(resp.get("cached")), dry_run)
        responses[spec["tag"]] = resp
        if not resp.get("cached") and not dry_run and pace_sec > 0:
            time.sleep(pace_sec)

    return assemble_probed_record(
        item,
        model_id,
        responses,
        oracle_reply=oracle_reply,
        oracle_revealed=oracle_revealed,
    )


def probe_dataset_profiled(
    items: List[Dict[str, Any]],
    model_id: str,
    tracker: BudgetTracker,
    *,
    dry_run: bool = False,
    workers: int = 6,
    label: str = "",
) -> List[Dict[str, Any]]:
    out_map: Dict[int, Dict[str, Any]] = {}
    if dry_run or workers <= 1:
        for i, it in enumerate(items):
            out_map[i] = probe_item_with_profile(it, model_id, tracker, dry_run=dry_run)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(probe_item_with_profile, it, model_id, tracker, dry_run=False): i
                for i, it in enumerate(items)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                out_map[idx] = fut.result()
                done += 1
                if done % 40 == 0 or done == len(items):
                    snap = tracker.snapshot()
                    print(
                        f"[{model_id} {label}] {done}/{len(items)} | live={snap['live_calls']} "
                        f"cached={snap['cached_hits']} rmb=¥{snap['estimated_rmb']:.2f}",
                        flush=True,
                    )
    return [out_map[i] for i in range(len(items))]


def eval_regime_with_sig(
    methods: List[str],
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    *,
    cap: int,
    di: int,
    c_ask: float,
    n_seeds: int = 5,
    unclarified_spec_penalty: bool = False,
) -> Dict[str, Any]:
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
                    seed=s,
                    unclarified_spec_penalty=unclarified_spec_penalty,
                    cal_probed=cal_p,
                )
            )
    ours_vec = np.mean([r["per_item_utils"] for r in raw["Tri-VoI-L2D"]], axis=0)
    boot_ps: Dict[str, float] = {}
    rows: Dict[str, Any] = {}
    for m in methods:
        rs = raw[m]
        row = {
            "mean_utility": round(float(np.mean([r["mean_utility"] for r in rs])), 3),
            "acc_pct": round(float(np.mean([r["acc_pct"] for r in rs])), 2),
            "ask_pct": round(float(np.mean([r["ask_rate_pct"] for r in rs])), 1),
            "defer_pct": round(float(np.mean([r["defer_rate_pct"] for r in rs])), 1),
            "mean_wait": round(float(np.mean([r["mean_wait"] for r in rs])), 2),
        }
        if m != "Tri-VoI-L2D":
            base_vec = np.mean([r["per_item_utils"] for r in rs], axis=0)
            bp = block_paired_bootstrap(ours_vec, base_vec, block_size=10, n_boot=3000, seed=42)
            row["delta_vs_tri"] = bp["delta_mean"]
            row["ci95"] = bp["ci95"]
            row["boot_p"] = bp["boot_p"]
            if m != "Oracle":
                boot_ps[m] = float(bp["boot_p"])
        rows[m] = row
    holm = holm_bonferroni(boot_ps)
    for m, pv in holm.items():
        rows[m]["boot_p_holm"] = pv
    return rows


def compute_public_tri_noncircular_stats(test_p: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify zero-turn vs post-ASK accuracy improvement on withheld-fact MediQ quadrants."""
    out: Dict[str, Any] = {}
    for qk in ("Q1_clear_easy", "Q2_underspec_easy", "Q3_clear_hard", "Q4_resolvable", "Q4_escalate"):
        sub = [r for r in test_p if r["item"]["quadrant"] == qk]
        g_acc = 100.0 * sum(1 for r in sub if r["greedy_ans"] == r["item"]["gold"]) / max(1, len(sub))
        p_acc = 100.0 * sum(1 for r in sub if r["post_ans"] == r["item"]["gold"]) / max(1, len(sub))
        rev = 100.0 * sum(1 for r in sub if r["oracle_revealed"]) / max(1, len(sub))
        out[qk] = {
            "n": len(sub),
            "zero_turn_acc_pct": round(g_acc, 2),
            "post_ask_acc_pct": round(p_acc, 2),
            "oracle_revealed_pct": round(rev, 1),
        }
    return out


def plot_v2_and_public_tri(v2_res: Dict[str, Any], pt_res: Dict[str, Any], prefix: str = "") -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.0,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)

    # Panel A: Benchmark v2 (4 models) across key methods at Cap=2, c=2.5 & Cap=1/3, c=1.5
    ax = axes[0]
    m_list = list(v2_res["models"].keys())
    short_names = [m.split("/")[-1] for m in m_list]
    compare_meths = ["KL-PBD+ (Warm)", "ACL-2026-VoI (Live)", "Calibrated-Cascade+Naor", "Tri-VoI-L2D"]
    colors = ["#8c564b", "#9467bd", "#ff7f0e", "#1f77b4"]
    x = np.arange(len(m_list))
    w = 0.19
    for idx, (meth, col) in enumerate(zip(compare_meths, colors)):
        vals = [v2_res["models"][m]["regimes"]["Cap=2, c=2.5"][meth]["mean_utility"] for m in m_list]
        ax.bar(x + (idx - 1.5) * w, vals, width=w, color=col, label=meth)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.set_ylabel("Mean Utility (Cap=2, c_ask=2.5)")
    ax.set_title("Benchmark v2 (Implicit-Override, 4 Models)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=True)

    # Panel B: Public-Tri (MediQ) 3 models at Cap=1/3, c=1.5 & Cap=2, c=3.0
    ax = axes[1]
    pt_models = list(pt_res["models"].keys())
    pt_short = [m.split("/")[-1] for m in pt_models]
    x2 = np.arange(len(pt_models))
    for idx, (meth, col) in enumerate(zip(compare_meths, colors)):
        vals = [pt_res["models"][m]["regimes"]["Cap=1/3, c=1.5"][meth]["mean_utility"] for m in pt_models]
        ax.bar(x2 + (idx - 1.5) * w, vals, width=w, color=col, label=meth)
    ax.set_xticks(x2)
    ax.set_xticklabels(pt_short, rotation=15, ha="right")
    ax.set_ylabel("Mean Utility (Severe Cap=1/3, c_ask=1.5)")
    ax.set_title("Public-Tri (MediQ Clinical Triage, 3 Models)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=True)

    out_pdf = FIGS_DIR / f"{prefix}fig_v2_and_public_tri.pdf"
    out_png = FIGS_DIR / f"{prefix}fig_v2_and_public_tri.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Live Evaluation on Benchmark v2 & Public-Tri (MediQ)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-budget-rmb", type=float, default=35.0)
    args = p.parse_args()

    prefix = "dry_run_" if args.dry_run else ""
    budget_file = RESULTS_DIR / f"{prefix}v2_pt_budget_state.json"
    tracker = BudgetTracker(max_budget_rmb=args.max_budget_rmb, state_path=budget_file)

    # 1. Benchmark v2 (Implicit-Override)
    cal_v2, test_v2 = build_cscw_benchmark_v2(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    spoiler_hits = sum(
        1 for it in cal_v2 + test_v2
        if ("confidential" in it["prose"].lower() or "override" in it["prose"].lower())
    )
    assert spoiler_hits == 0, f"Benchmark v2 spoiler leak detected: {spoiler_hits}"

    v2_models_out: Dict[str, Any] = {}
    for mid in V2_MODELS:
        cal_p = probe_dataset_profiled(cal_v2, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="v2_cal60")
        test_p = probe_dataset_profiled(test_v2, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="v2_test180")
        calib = HeldOutCalibrator(cal_p)
        diag = evaluate_diagnosis_metrics(test_p)
        regimes = {
            "Cap=2, c=1.5": eval_regime_with_sig(METHODS_V2, cal_p, test_p, calib, cap=2, di=2, c_ask=1.5),
            "Cap=2, c=2.5": eval_regime_with_sig(METHODS_V2, cal_p, test_p, calib, cap=2, di=2, c_ask=2.5),
            "Cap=1/3, c=1.5": eval_regime_with_sig(METHODS_V2, cal_p, test_p, calib, cap=1, di=3, c_ask=1.5),
        }
        eps = split_stratified_episodes(test_p, n_episodes=6, seed=2029)
        ep_res = eval_independent_episodes(
            ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI (Live)", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"],
            eps, calib, n_seeds=5, capacity=2, drain_interval=2, c_ask=2.5
        )
        extra = {}
        if mid == V2_MODELS[0]:
            extra["rework_cap2"] = eval_regime_with_sig(
                METHODS_V2, cal_p, test_p, calib, cap=2, di=2, c_ask=1.5, unclarified_spec_penalty=True
            )
            extra["rework_sev"] = eval_regime_with_sig(
                METHODS_V2, cal_p, test_p, calib, cap=1, di=3, c_ask=1.5, unclarified_spec_penalty=True
            )
        v2_models_out[mid] = {
            "diagnosis": diag,
            "regimes": regimes,
            "episodes_c25": ep_res,
            **extra,
        }

    v2_payload = {
        "suite": "Benchmark v2 (Implicit-Override)",
        "spoiler_hits": spoiler_hits,
        "n_cal": len(cal_v2),
        "n_test": len(test_v2),
        "models": v2_models_out,
    }

    # 2. Public-Tri (MediQ)
    cal_pt, test_pt = load_public_tri_dataset()
    pt_models_out: Dict[str, Any] = {}
    for mid in PUBLIC_TRI_MODELS:
        cal_p = probe_dataset_profiled(cal_pt, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="pt_cal60")
        test_p = probe_dataset_profiled(test_pt, mid, tracker, dry_run=args.dry_run, workers=args.workers, label="pt_test240")
        calib = HeldOutCalibrator(cal_p)
        diag = evaluate_diagnosis_metrics(test_p)
        noncirc = compute_public_tri_noncircular_stats(test_p)

        regimes = {
            "Cap=2, c=0.5": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=2, di=2, c_ask=0.5),
            "Cap=2, c=1.5": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=2, di=2, c_ask=1.5),
            "Cap=2, c=3.0": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=2, di=2, c_ask=3.0),
            "Cap=3, c=1.5": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=3, di=2, c_ask=1.5),
            "Cap=1/2, c=1.5": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=1, di=2, c_ask=1.5),
            "Cap=1/3, c=1.5": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=1, di=3, c_ask=1.5),
            "Cap=1/3, c=3.0": eval_regime_with_sig(METHODS_PT, cal_p, test_p, calib, cap=1, di=3, c_ask=3.0),
        }
        eps = split_stratified_episodes(test_p, n_episodes=6, seed=2030)
        ep_res = eval_independent_episodes(
            ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI (Live)", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"],
            eps, calib, n_seeds=5, capacity=2, drain_interval=2, c_ask=1.5
        )
        pt_models_out[mid] = {
            "diagnosis": diag,
            "noncircular_stats": noncirc,
            "regimes": regimes,
            "episodes_c15": ep_res,
        }

    tracker.save()
    budget_snap = tracker.snapshot()
    pt_payload = {
        "suite": "Public-Tri (stellalisy/mediQ)",
        "n_cal": len(cal_pt),
        "n_test": len(test_pt),
        "models": pt_models_out,
        "budget": budget_snap,
    }

    (RESULTS_DIR / f"{prefix}benchmark_v2_metrics.json").write_text(json.dumps(v2_payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{prefix}public_tri_metrics.json").write_text(json.dumps(pt_payload, indent=2), encoding="utf-8")
    plot_v2_and_public_tri(v2_payload, pt_payload, prefix=prefix)

    # Write concise markdown reports
    v2_md = [
        f"# Benchmark v2 (Implicit-Override) Report (spoiler_hits={spoiler_hits}, N_cal={len(cal_v2)}, N_test={len(test_v2)})",
        "",
    ]
    for mid, mdata in v2_models_out.items():
        d = mdata["diagnosis"]
        v2_md.append(f"## `{mid}` (Missing-F1={d['missing_f1']}, Override-F1={d['override_f1']}, Collapse={d['entropy_collapse_pct']}%)")
        v2_md.append("| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 |")
        v2_md.append("|:---|---:|---:|---:|")
        for meth in METHODS_V2:
            r1 = mdata["regimes"]["Cap=2, c=1.5"][meth]
            r2 = mdata["regimes"]["Cap=2, c=2.5"][meth]
            r3 = mdata["regimes"]["Cap=1/3, c=1.5"][meth]
            v2_md.append(f"| `{meth}` | {r1['mean_utility']:.3f} ({r1['acc_pct']:.1f}%) | {r2['mean_utility']:.3f} ({r2['acc_pct']:.1f}%) | {r3['mean_utility']:.3f} ({r3['acc_pct']:.1f}%) |")
        v2_md.append("")
    (RESULTS_DIR / f"{prefix}benchmark_v2_report.md").write_text("\n".join(v2_md) + "\n", encoding="utf-8")

    pt_md = [
        f"# Public-Tri (MediQ Clinical Triage) Report (N_cal={len(cal_pt)}, N_test={len(test_pt)}, Spend=¥{budget_snap['estimated_rmb']:.2f})",
        "",
    ]
    for mid, mdata in pt_models_out.items():
        d = mdata["diagnosis"]
        nc = mdata["noncircular_stats"]
        pt_md.append(
            f"## `{mid}` (Missing-F1={d['missing_f1']}, Scope-F1={d['override_f1']}, Collapse={d['entropy_collapse_pct']}%, "
            f"Q2 Zero->Post={nc['Q2_underspec_easy']['zero_turn_acc_pct']}%->{nc['Q2_underspec_easy']['post_ask_acc_pct']}%, "
            f"Q4res Zero->Post={nc['Q4_resolvable']['zero_turn_acc_pct']}%->{nc['Q4_resolvable']['post_ask_acc_pct']}%)"
        )
        pt_md.append("| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 |")
        pt_md.append("|:---|---:|---:|---:|---:|")
        for meth in METHODS_PT:
            r1 = mdata["regimes"]["Cap=2, c=1.5"][meth]
            r2 = mdata["regimes"]["Cap=2, c=3.0"][meth]
            r3 = mdata["regimes"]["Cap=1/3, c=1.5"][meth]
            r4 = mdata["regimes"]["Cap=1/3, c=3.0"][meth]
            pt_md.append(
                f"| `{meth}` | {r1['mean_utility']:.3f} ({r1['acc_pct']:.1f}%) | {r2['mean_utility']:.3f} ({r2['acc_pct']:.1f}%) | "
                f"{r3['mean_utility']:.3f} ({r3['acc_pct']:.1f}%) | {r4['mean_utility']:.3f} ({r4['acc_pct']:.1f}%) |"
            )
        pt_md.append("")
    (RESULTS_DIR / f"{prefix}public_tri_report.md").write_text("\n".join(pt_md) + "\n", encoding="utf-8")
    print(f"Completed run_v2_and_public_tri (dry_run={args.dry_run}). Estimated spend: ¥{budget_snap['estimated_rmb']:.4f}")


if __name__ == "__main__":
    main()
