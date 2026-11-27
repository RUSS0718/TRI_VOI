import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import (
    DOMAINS,
    build_cscw_benchmark,
    build_cscw_expansion_batch,
    split_stratified_episodes,
)
from controller import (
    HeldOutCalibrator,
    evaluate_diagnosis_metrics,
    simulate_stream,
)
from llm_client import SiliconFlowAuthOrBalanceError
from overnight.overnight_lib import (
    OVERNIGHT_FIGS_DIR,
    OVERNIGHT_RESULTS_DIR,
    BudgetLimitReached,
    BudgetTracker,
    build_noisy_dataset,
    make_imperfect_user_sim,
    probe_dataset_budgeted,
    run_entropy_kt_stress,
    run_stage5_structural,
)
from run_all import (
    MAIN_METHODS,
    eval_independent_episodes,
    eval_methods_multi_seed,
)

PRIMARY_MODEL = "Qwen/Qwen3.5-35B-A3B"
STAGE1_SECONDARY_MODELS = [
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.5-Air",
    "Qwen/Qwen3-8B",
]
FIFTH_FAMILY_MODEL = "tencent/Hy4-preview"
ALL_5_MODELS = [PRIMARY_MODEL] + STAGE1_SECONDARY_MODELS + [FIFTH_FAMILY_MODEL]


def _save_stage(name: str, payload: Dict[str, Any], md_text: str, dry_run: bool) -> None:
    prefix = "dry_run_" if dry_run else ""
    jp = OVERNIGHT_RESULTS_DIR / f"{prefix}{name}.json"
    mp = OVERNIGHT_RESULTS_DIR / f"{prefix}{name}.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(md_text + "\n", encoding="utf-8")
    print(f"[Checkpoint] Saved {jp.name} & {mp.name}", flush=True)


def _eval_model_full_suite(
    model_id: str,
    cal_probed: List[Dict[str, Any]],
    test_probed: List[Dict[str, Any]],
    *,
    c_asks: tuple = (1.5, 2.5),
) -> Dict[str, Any]:
    calib = HeldOutCalibrator(cal_probed)
    diag = evaluate_diagnosis_metrics(test_probed)
    diag["model_id"] = model_id
    diag["short_name"] = model_id.split("/")[-1]
    diag["n_test"] = len(test_probed)

    cont: Dict[str, Any] = {}
    eps_res: Dict[str, Any] = {}
    episodes = split_stratified_episodes(test_probed, n_episodes=6, seed=2029)
    valid_eps = all(len(ep) > 0 for ep in episodes)

    for ca in c_asks:
        key = f"c_{ca:.1f}"
        cont[key] = eval_methods_multi_seed(
            MAIN_METHODS,
            test_probed,
            calib,
            n_seeds=5,
            capacity=2,
            drain_interval=2,
            drift_mode="sudden",
            c_ask=ca,
            compute_sig=True,
        )
        if valid_eps:
            eps_res[key] = eval_independent_episodes(
                MAIN_METHODS,
                episodes,
                calib,
                n_seeds=5,
                capacity=2,
                drain_interval=2,
                drift_mode="sudden",
                c_ask=ca,
            )

    # Also evaluate Cap=1/3 severe congestion vs Calibrated-Cascade / Joint-Greedy (+Naor & w/o Ext)
    sev = eval_methods_multi_seed(
        [
            "Naive-Cascade",
            "Joint-Greedy",
            "Joint-Greedy+Naor",
            "Calibrated-Cascade",
            "Calibrated-Cascade+Naor",
            "w/o Queue-Externality",
            "Tri-VoI-L2D",
        ],
        test_probed,
        calib,
        n_seeds=5,
        capacity=1,
        drain_interval=3,
        drift_mode="sudden",
        c_ask=1.5,
    )
    return {
        "model_id": model_id,
        "short_name": model_id.split("/")[-1],
        "n_cal": len(cal_probed),
        "n_test": len(test_probed),
        "diagnosis": diag,
        "continuous": cont,
        "episodes": eps_res,
        "severe_cap1_3": sev["rows"],
    }


def _plot_fig5_multimodel(model_summaries: List[Dict[str, Any]], dry_run: bool) -> List[str]:
    prefix = "dry_run_" if dry_run else ""
    names = [m["short_name"] for m in model_summaries]
    x = np.arange(len(names))
    w = 0.18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.1))

    for ax, ckey, title in [
        (ax1, "c_1.5", "(a) 5 Models (4 Families) N=360 (c_ask=1.5)"),
        (ax2, "c_2.5", "(b) 5 Models (4 Families) N=360 (c_ask=2.5)"),
    ]:
        for idx, (meth, col) in enumerate([
            ("Naive-Cascade", "#e7298a"),
            ("Joint-Greedy", "#d95f02"),
            ("Calibrated-Cascade", "#386cb0"),
            ("Tri-VoI-L2D", "#1b9e77"),
        ]):
            vals = []
            for ms in model_summaries:
                rmap = {r["method"]: r["mean_utility"] for r in ms["continuous"][ckey]["rows"]}
                vals.append(rmap[meth])
            ax.bar(x + (idx - 1.5) * w, vals, width=w, label=meth, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5, rotation=10)
        ax.set_ylabel("Mean Net Utility / Ticket")
        ax.set_title(title)
        ax.legend(fontsize=7.3)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        p = str(OVERNIGHT_FIGS_DIR / f"{prefix}fig5_multimodel_5families.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/overnight/figs/{prefix}fig5_multimodel_5families.{ext}")
    plt.close(fig)
    return saved


def _plot_fig6_noisy(stage3_models: List[Dict[str, Any]], dry_run: bool) -> List[str]:
    prefix = "dry_run_" if dry_run else ""
    names = [m["short_name"] for m in stage3_models]
    x = np.arange(len(names))
    w = 0.16
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0))

    methods = ["Direct-ACT", "Naive-Cascade", "Joint-Greedy", "Calibrated-Cascade", "Tri-VoI-L2D"]
    cols = ["#666666", "#e7298a", "#d95f02", "#386cb0", "#1b9e77"]
    for idx, (meth, col) in enumerate(zip(methods, cols)):
        u15 = [{r["method"]: r["mean_utility"] for r in m["continuous"]["c_1.5"]["rows"]}[meth] for m in stage3_models]
        u25 = [{r["method"]: r["mean_utility"] for r in m["continuous"]["c_2.5"]["rows"]}[meth] for m in stage3_models]
        ax1.bar(x + (idx - 2.0) * w, u15, width=w, label=meth, color=col)
        ax2.bar(x + (idx - 2.0) * w, u25, width=w, label=meth, color=col)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Mean Net Utility / Ticket")
    ax1.set_title("(a) Noisy Long-Form + Imperfect User (c_ask=1.5, N=360)")
    ax1.legend(fontsize=7.2)
    ax1.grid(axis="y", alpha=0.25)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("Mean Net Utility / Ticket")
    ax2.set_title("(b) Noisy Long-Form + Imperfect User (c_ask=2.5, N=360)")
    ax2.legend(fontsize=7.2)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        p = str(OVERNIGHT_FIGS_DIR / f"{prefix}fig6_noisy_longform_benchmark.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/overnight/figs/{prefix}fig6_noisy_longform_benchmark.{ext}")
    plt.close(fig)
    return saved


def _plot_fig7_kt_stress(kt_rows: List[Dict[str, Any]], dry_run: bool) -> List[str]:
    prefix = "dry_run_" if dry_run else ""
    models = sorted({r["model_id"] for r in kt_rows})
    fig, axes = plt.subplots(1, max(1, len(models)), figsize=(5.4 * max(1, len(models)), 3.9))
    if len(models) == 1:
        axes = [axes]

    ks = [3, 5, 10]
    for ax, mid in zip(axes, models):
        sub_m = [r for r in kt_rows if r["model_id"] == mid]
        for temp, col, ls in [(0.7, "#d95f02", "-"), (1.0, "#7570b3", "--")]:
            pts_c = [next(r["entropy_collapse_pct"] for r in sub_m if r["temperature"] == temp and r["K"] == k) for k in ks]
            pts_l = [next(r["low_entropy_pct"] for r in sub_m if r["temperature"] == temp and r["K"] == k) for k in ks]
            ax.plot(ks, pts_c, marker="o", linestyle=ls, linewidth=2.0, color=col, label=f"H=0 Collapse (T={temp})")
            ax.plot(ks, pts_l, marker="s", linestyle=":", linewidth=1.6, color=col, label=f"H<=0.45 Missed (T={temp})")
        f1_pct = sub_m[0]["disentangled_missing_f1"] * 100.0
        ax.axhline(100.0 - f1_pct, color="#1b9e77", linestyle="-.", linewidth=1.8, label=f"Tri-VoI Diag FN≈{100-f1_pct:.1f}%")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Underspec Failure / Collapse Rate (%)")
        ax.set_title(f"{mid.split('/')[-1]}: K×T Entropy Collapse")
        ax.legend(fontsize=7.0, loc="best")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        p = str(OVERNIGHT_FIGS_DIR / f"{prefix}fig7_entropy_kt_stress.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/overnight/figs/{prefix}fig7_entropy_kt_stress.{ext}")
    plt.close(fig)
    return saved


def _plot_fig8_phase_heatmap(phase_cells: List[Dict[str, Any]], dry_run: bool) -> List[str]:
    prefix = "dry_run_" if dry_run else ""
    mus = sorted({c["mu"] for c in phase_cells}, reverse=True)
    casks = sorted({c["c_ask"] for c in phase_cells})

    mat_cas = np.zeros((len(mus), len(casks)))
    mat_grd = np.zeros((len(mus), len(casks)))
    for c in phase_cells:
        i = mus.index(c["mu"])
        j = casks.index(c["c_ask"])
        mat_cas[i, j] = c["margin_vs_cascade"]
        mat_grd[i, j] = c["margin_vs_greedy"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for ax, mat, title in [
        (ax1, mat_cas, "(a) ΔUtil: Tri-VoI-L2D minus Calibrated-Cascade"),
        (ax2, mat_grd, "(b) ΔUtil: Tri-VoI-L2D minus Joint-Greedy"),
    ]:
        vmax = max(0.5, float(np.max(np.abs(mat))))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax * 0.4, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(casks)))
        ax.set_xticklabels([f"{c:.1f}" for c in casks], fontsize=8)
        ax.set_yticks(range(len(mus)))
        ax.set_yticklabels([f"μ={m:.2f}" for m in mus], fontsize=8)
        ax.set_xlabel("User Ask Burden c_ask")
        ax.set_ylabel("Specialist Service Capacity μ")
        ax.set_title(title, fontsize=10)
        for i in range(len(mus)):
            for j in range(len(casks)):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6.8, color="black")
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        p = str(OVERNIGHT_FIGS_DIR / f"{prefix}fig8_2d_phase_heatmap.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/overnight/figs/{prefix}fig8_2d_phase_heatmap.{ext}")
    plt.close(fig)
    return saved


def _plot_fig9_cv_lodo_bursty(stage5: Dict[str, Any], dry_run: bool) -> List[str]:
    prefix = "dry_run_" if dry_run else ""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.8, 3.9))

    # Panel (a): Calibration Sample Efficiency Curve
    cc = stage5["calibration_curve_n"]
    ns = [x["n_cal"] for x in cc]
    ax1.errorbar(ns, [x["tri_mean"] for x in cc], yerr=[x["tri_std"] for x in cc], marker="o", color="#1b9e77", linewidth=2.0, label="Tri-VoI-L2D (±std)")
    ax1.plot(ns, [x["cascade_mean"] for x in cc], marker="^", linestyle="--", color="#386cb0", linewidth=1.8, label="Calibrated-Cascade")
    ax1.set_xlabel("Held-Out Calibration Size N_cal")
    ax1.set_ylabel("Mean Net Utility / Ticket")
    ax1.set_title("(a) Calibration Sample Efficiency")
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.25)

    # Panel (b): LODO Zero-Shot Domain Transfer (Primary model)
    lodo0 = stage5["lodo_transfer"][0]["domains"]
    doms = DOMAINS
    x2 = np.arange(len(doms))
    w2 = 0.25
    ax2.bar(x2 - w2, [lodo0[d]["Naive-Cascade"] for d in doms], width=w2, label="Naive-Cascade", color="#e7298a")
    ax2.bar(x2, [lodo0[d]["Calibrated-Cascade"] for d in doms], width=w2, label="Calibrated-Cascade", color="#386cb0")
    ax2.bar(x2 + w2, [lodo0[d]["Tri-VoI-L2D"] for d in doms], width=w2, label="Tri-VoI-L2D (LODO)", color="#1b9e77")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([d[:7] for d in doms], fontsize=8)
    ax2.set_ylabel("Mean Net Utility / Ticket")
    ax2.set_title("(b) Leave-One-Domain-Out Zero-Shot Transfer")
    ax2.legend(fontsize=7.2)
    ax2.grid(axis="y", alpha=0.25)

    # Panel (c): Bursty Waves & 3-Expert Shift Fatigue
    sc_methods = ["Congested-L2D-Only", "Naive-Cascade", "Joint-Greedy", "Calibrated-Cascade", "Tri-VoI-L2D"]
    x3 = np.arange(len(sc_methods))
    w3 = 0.34
    ax3.bar(x3 - w3 / 2, [stage5["bursty_traffic_pool"][m]["mean_utility"] for m in sc_methods], width=w3, label="Bursty Poisson Waves", color="#1b9e77")
    ax3.bar(x3 + w3 / 2, [stage5["shift3_hetero_pool"][m]["mean_utility"] for m in sc_methods], width=w3, label="3-Expert Shift Fatigue", color="#7570b3")
    ax3.set_xticks(x3)
    ax3.set_xticklabels([m.replace("-Only", "").replace("-Cascade", "-Casc") for m in sc_methods], fontsize=7.2, rotation=15)
    ax3.set_ylabel("Mean Net Utility / Ticket")
    ax3.set_title("(c) Bursty Arrivals & 3-Expert Shift Pool")
    ax3.legend(fontsize=7.5)
    ax3.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    saved = []
    for ext in ("png", "pdf"):
        p = str(OVERNIGHT_FIGS_DIR / f"{prefix}fig9_cv_lodo_bursty.{ext}")
        plt.savefig(p, dpi=200)
        saved.append(f"results/overnight/figs/{prefix}fig9_cv_lodo_bursty.{ext}")
    plt.close(fig)
    return saved


def _write_master_report(
    state: Dict[str, Any],
    tracker: BudgetTracker,
    dry_run: bool,
    status_note: str,
) -> None:
    prefix = "dry_run_" if dry_run else ""
    snap = tracker.snapshot()
    lines = [
        f"# Overnight Paper-Complete Experiment Master Report — [{status_note}]",
        "",
        f"- **Budget & Telemetry**: Live Calls=`{snap['live_calls']}`, Cache Hits=`{snap['cached_hits']}`, "
        f"Dry Calls=`{snap['dry_calls']}`, Prompt Tokens=`{snap['prompt_tokens']:,}`, Completion Tokens=`{snap['completion_tokens']:,}`, "
        f"**Estimated Spend=`¥{snap['estimated_rmb']:.2f}` / `¥{snap['max_budget_rmb']:.2f}`**, Elapsed=`{snap['elapsed_min']} min`.",
        "",
    ]

    if "all_models_n360" in state:
        lines += [
            "## Stage 1 & 2: Full-Scale Cross-Model Evaluation (5 Models Across 4 Families, `N=360`, 6 Independent Episodes)",
            "",
            "| Backbone | N | Missing F1 | Ent Coll% | Direct (1.5) | Naive (1.5) | J-Greedy (1.5) | CalibCasc (1.5) | **Tri-VoI (1.5)** | CalibCasc (2.5) | **Tri-VoI (2.5)** | Severe Cap=1/3: Tri / Calib+Naor / w/oExt / Naive |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
        for ms in state["all_models_n360"]:
            r15 = {r["method"]: r["mean_utility"] for r in ms["continuous"]["c_1.5"]["rows"]}
            r25 = {r["method"]: r["mean_utility"] for r in ms["continuous"]["c_2.5"]["rows"]}
            sev = {r["method"]: r["mean_utility"] for r in ms["severe_cap1_3"]}
            dg = ms["diagnosis"]
            lines.append(
                f"| {ms['model_id']} | {ms['n_test']} | {dg['missing_f1']} | {dg['entropy_collapse_pct']} | "
                f"{r15['Direct-ACT']:.3f} | {r15['Naive-Cascade']:.3f} | {r15['Joint-Greedy']:.3f} | {r15['Calibrated-Cascade']:.3f} | **{r15['Tri-VoI-L2D']:.3f}** | "
                f"{r25['Calibrated-Cascade']:.3f} | **{r25['Tri-VoI-L2D']:.3f}** | "
                f"{sev['Tri-VoI-L2D']:.3f} / {sev['Calibrated-Cascade+Naor']:.3f} / {sev['w/o Queue-Externality']:.3f} / {sev['Naive-Cascade']:.3f} |"
            )
        lines.append("")

    if "stage3" in state:
        lines += [
            "## Stage 3: Realistic Noisy Multi-Paragraph Benchmark + Imperfect LLM User (`N=360`)",
            "",
            "| Backbone | Missing F1 | Ent Collapse% | Naive-Casc (1.5) | Joint-Greedy (1.5) | Calib-Casc (1.5) | **Tri-VoI (1.5)** | Calib-Casc (2.5) | **Tri-VoI (2.5)** | Boot p_Holm vs Naive (1.5) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for ms in state["stage3"]["models"]:
            r15 = {r["method"]: r for r in ms["continuous"]["c_1.5"]["rows"]}
            r25 = {r["method"]: r for r in ms["continuous"]["c_2.5"]["rows"]}
            dg = ms["diagnosis"]
            np_h = r15["Naive-Cascade"]["sig_vs_ours"].get("boot_p_holm", r15["Naive-Cascade"]["sig_vs_ours"]["boot_p"])
            lines.append(
                f"| {ms['model_id']} | {dg['missing_f1']} | {dg['entropy_collapse_pct']} | "
                f"{r15['Naive-Cascade']['mean_utility']:.3f} | {r15['Joint-Greedy']['mean_utility']:.3f} | "
                f"{r15['Calibrated-Cascade']['mean_utility']:.3f} | **{r15['Tri-VoI-L2D']['mean_utility']:.3f}** | "
                f"{r25['Calibrated-Cascade']['mean_utility']:.3f} | **{r25['Tri-VoI-L2D']['mean_utility']:.3f}** | {np_h} |"
            )
        lines.append("")

    if "stage4" in state:
        lines += [
            "## Stage 4: Entropy-Collapse Stress Test Across Sampling `K ∈ {3, 5, 10}` × Temperature `T ∈ {0.7, 1.0}`",
            "",
            "| Backbone | T | K | H=0 Collapse % | H<=0.45 Missed % | Mean Entropy (bits) | Majority Acc % | Disentangled Missing F1 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in state["stage4"]["rows"]:
            lines.append(
                f"| {r['short_name']} | {r['temperature']:.1f} | {r['K']} | {r['entropy_collapse_pct']} | {r['low_entropy_pct']} | "
                f"{r['mean_entropy_bits']} | {r['majority_acc_pct']} | {r['disentangled_missing_f1']} |"
            )
        lines.append("")

    if "stage5" in state:
        s5 = state["stage5"]
        cells = s5["phase_map_56pt"]
        n_strict_best = sum(1 for c in cells if c["winner"] == "Tri-VoI-L2D")
        n_win_cas = sum(1 for c in cells if c["margin_vs_cascade"] >= 0)
        n_win_cas_naor = sum(1 for c in cells if c.get("margin_vs_cascade_naor", 0) >= 0)
        n_win_grd = sum(1 for c in cells if c["margin_vs_greedy"] >= 0)
        n_win_grd_naor = sum(1 for c in cells if c.get("margin_vs_greedy_naor", 0) >= 0)
        n_win_wo_ext = sum(1 for c in cells if c.get("margin_vs_wo_ext", 0) >= 0)
        lines += [
            "## Stage 5: Deep Structural, Cross-Validation, LODO Transfer, Bursty/3-Expert, Subset & Specialist Stress",
            "",
            f"- **56-Point 2D Operating Phase Map (`7 loads × 8 c_ask`, 8 methods)**: `Tri-VoI-L2D` is strict overall winner on **{n_strict_best}/56** cells; "
            f"outperforms/ties `Calibrated-Cascade` (**{n_win_cas}/56**), `Calibrated-Cascade+Naor` (**{n_win_cas_naor}/56**), "
            f"`Joint-Greedy` (**{n_win_grd}/56**), `Joint-Greedy+Naor` (**{n_win_grd_naor}/56**), and `w/o Queue-Externality` (**{n_win_wo_ext}/56**).",
            "",
            "### 5-Fold Stratified Cross-Validation Across Available Models (420 items each)",
            "| Backbone | Naive-Cascade | Joint-Greedy | Calibrated-Cascade | **Tri-VoI-L2D** |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for cv in s5["cv5_cross_model"]:
            ms = cv["methods"]
            lines.append(
                f"| {cv['short_name']} | {ms['Naive-Cascade']['mean']}±{ms['Naive-Cascade']['std']} | "
                f"{ms['Joint-Greedy']['mean']}±{ms['Joint-Greedy']['std']} | "
                f"{ms['Calibrated-Cascade']['mean']}±{ms['Calibrated-Cascade']['std']} | "
                f"**{ms['Tri-VoI-L2D']['mean']}±{ms['Tri-VoI-L2D']['std']}** |"
            )

        lines += [
            "",
            "### Bursty Poisson Arrivals & 3-Expert Heterogeneous Shift-Fatigue Pool",
            "| Method | Bursty Util (Acc% / Wait) | 3-Expert Shift Util (Acc% / Wait) |",
            "| :--- | :---: | :---: |",
        ]
        for m in [
            "Direct-ACT",
            "Congested-L2D-Only",
            "Naive-Cascade",
            "Joint-Greedy",
            "Joint-Greedy+Naor",
            "Calibrated-Cascade",
            "Calibrated-Cascade+Naor",
            "Tri-VoI-L2D",
        ]:
            b = s5["bursty_traffic_pool"][m]
            s = s5["shift3_hetero_pool"][m]
            lines.append(f"| {m} | {b['mean_utility']:.3f} ({b['acc_pct']}% / W={b['mean_wait']}) | {s['mean_utility']:.3f} ({s['acc_pct']}% / W={s['mean_wait']}) |")

        if "decision_relevant_subset" in s5:
            dr = s5["decision_relevant_subset"]
            lines += [
                "",
                f"### Stage 5e: Decision-Relevant Subset (`cal={dr['n_cal_filtered']}/{dr['n_cal_orig']}`, `test={dr['n_test_filtered']}/{dr['n_test_orig']}`, Invariant Q2 Excluded)",
                "| Regime | Naive-Casc | Joint-Greedy | Joint-Greedy+Naor | Calib-Casc | Calib-Casc+Naor | w/o Externality | **Tri-VoI-L2D** |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for r_lbl, rm in dr["regimes"].items():
                lines.append(
                    f"| {r_lbl} | {rm['Naive-Cascade']['mean_utility']:.3f} | {rm['Joint-Greedy']['mean_utility']:.3f} | "
                    f"{rm['Joint-Greedy+Naor']['mean_utility']:.3f} | {rm['Calibrated-Cascade']['mean_utility']:.3f} | "
                    f"{rm['Calibrated-Cascade+Naor']['mean_utility']:.3f} | {rm['w/o Queue-Externality']['mean_utility']:.3f} | "
                    f"**{rm['Tri-VoI-L2D']['mean_utility']:.3f}** |"
                )

        if "realistic_specialist_stress" in s5:
            lines += [
                "",
                "### Stage 5f: Realistic Specialist Stress (Delayed Feedback & Unclarified-Ticket Rework)",
                "| Stress Condition | Congested-L2D | Joint-Greedy | Joint-Greedy+Naor | Calib-Casc | Calib-Casc+Naor | w/o Externality | **Tri-VoI-L2D** |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for s_lbl, sm in s5["realistic_specialist_stress"].items():
                lines.append(
                    f"| {s_lbl} | {sm['Congested-L2D-Only']['mean_utility']:.3f} | {sm['Joint-Greedy']['mean_utility']:.3f} | "
                    f"{sm['Joint-Greedy+Naor']['mean_utility']:.3f} | {sm['Calibrated-Cascade']['mean_utility']:.3f} | "
                    f"{sm['Calibrated-Cascade+Naor']['mean_utility']:.3f} | {sm['w/o Queue-Externality']['mean_utility']:.3f} | "
                    f"**{sm['Tri-VoI-L2D']['mean_utility']:.3f}** |"
                )

        lines += [
            "",
            f"### Error Taxonomy (`N={s5['error_taxonomy']['n_items']}`, Errors={s5['error_taxonomy']['total_errors']} / {s5['error_taxonomy']['error_rate_pct']}%)",
            "| Error Category | Count | % of Errors |",
            "| :--- | ---: | ---: |",
        ]
        for eb in s5["error_taxonomy"]["breakdown"]:
            lines.append(f"| {eb['category']} | {eb['count']} | {eb['pct_of_errors']}% |")

    mr_path = OVERNIGHT_RESULTS_DIR / f"{prefix}OVERNIGHT_MASTER_REPORT.md"
    mr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if "all_models_n360" in state:
        tex = [
            "% Overnight 5-Model (4 Families) N=360 Cross-Model IEEE Table",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Cross-model generalization across 5 models from 4 LLM families ($N=360$ test tickets $\\times$ 5 seeds each).}",
            "\\begin{tabular}{lccccccc}",
            "\\hline",
            "Backbone & Missing-F1 & Coll(\\%) & Naive(1.5) & Greedy(1.5) & CalibCasc(1.5) & TriVoI(1.5) & TriVoI(2.5) \\\\",
            "\\hline",
        ]
        for ms in state["all_models_n360"]:
            r15 = {r["method"]: r["mean_utility"] for r in ms["continuous"]["c_1.5"]["rows"]}
            r25 = {r["method"]: r["mean_utility"] for r in ms["continuous"]["c_2.5"]["rows"]}
            dg = ms["diagnosis"]
            tex.append(
                f"{ms['short_name']} & {dg['missing_f1']:.3f} & {dg['entropy_collapse_pct']:.1f} & "
                f"{r15['Naive-Cascade']:.3f} & {r15['Joint-Greedy']:.3f} & {r15['Calibrated-Cascade']:.3f} & "
                f"\\textbf{{{r15['Tri-VoI-L2D']:.3f}}} & \\textbf{{{r25['Tri-VoI-L2D']:.3f}}} \\\\"
            )
        tex += ["\\hline", "\\end{tabular}", "\\end{table*}"]
        (OVERNIGHT_RESULTS_DIR / f"{prefix}overnight_latex_tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Tri-VoI-L2D Overnight 5-Stage Paper-Complete Experiment Suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", type=str, default="1,2,3,4,5")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pace-sec", type=float, default=0.25)
    parser.add_argument("--max-budget-rmb", type=float, default=15.0)
    parser.add_argument("--smoke", type=int, default=0, help="Optional slice limit per split for rapid live smoke verification")
    args = parser.parse_args()

    stages = {int(x.strip()) for x in args.stages.split(",") if x.strip()}
    tracker = BudgetTracker(max_budget_rmb=args.max_budget_rmb)

    cal_full, test_180 = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    test_ext_180 = build_cscw_expansion_batch(n_per_quad=45, seed=2028, idx_offset=240)
    test_360 = test_180 + test_ext_180

    if args.smoke > 0:
        cal_items = cal_full[: args.smoke]
        test_items = test_360[: args.smoke]
    else:
        cal_items = cal_full
        test_items = test_360

    state: Dict[str, Any] = {}
    model_pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    summaries_by_model: Dict[str, Dict[str, Any]] = {}

    # Always load primary model pool first (100% cached from Phase 2)
    print("=== Loading Primary Model Pool (Qwen/Qwen3.5-35B-A3B) ===", flush=True)
    prim_cal = probe_dataset_budgeted(cal_items, PRIMARY_MODEL, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="cal")
    prim_test = probe_dataset_budgeted(test_items, PRIMARY_MODEL, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="test")
    model_pools[PRIMARY_MODEL] = {"cal": prim_cal, "test": prim_test}
    summaries_by_model[PRIMARY_MODEL] = _eval_model_full_suite(PRIMARY_MODEL, prim_cal, prim_test)

    try:
        # -----------------------------------------------------------------
        # STAGE 1: 4 existing backbones full N=360 + N_cal=60 alignment
        # -----------------------------------------------------------------
        if 1 in stages:
            print("\n=== STAGE 1: Aligning 4 Models to Full 60 cal + 360 test ===", flush=True)
            for mid in STAGE1_SECONDARY_MODELS:
                mc = probe_dataset_budgeted(cal_items, mid, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="cal60")
                mt = probe_dataset_budgeted(test_items, mid, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="test360")
                model_pools[mid] = {"cal": mc, "test": mt}
                summaries_by_model[mid] = _eval_model_full_suite(mid, mc, mt)

            st1_list = [summaries_by_model[m] for m in [PRIMARY_MODEL] + STAGE1_SECONDARY_MODELS]
            state["stage1"] = {"models": st1_list}
            state["all_models_n360"] = st1_list
            md1 = ["# Stage 1: 4-Model Full N=360 Report", ""] + [
                f"- **{m['model_id']}**: N={m['n_test']}, MissingF1={m['diagnosis']['missing_f1']}, "
                f"TriVoI(1.5)={ {r['method']: r['mean_utility'] for r in m['continuous']['c_1.5']['rows']}['Tri-VoI-L2D']:.3f}"
                for m in st1_list
            ]
            _save_stage("stage1_multimodel_n360", state["stage1"], "\n".join(md1), args.dry_run)
            _write_master_report(state, tracker, args.dry_run, "AFTER_STAGE_1")

        # -----------------------------------------------------------------
        # STAGE 2: 5th model (4th distinct family: tencent/Hy4-preview)
        # -----------------------------------------------------------------
        if 2 in stages:
            print(f"\n=== STAGE 2: 5th Model / 4th Family ({FIFTH_FAMILY_MODEL}, Full 60 cal + 360 test) ===", flush=True)
            lc = probe_dataset_budgeted(cal_items, FIFTH_FAMILY_MODEL, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="cal60")
            lt = probe_dataset_budgeted(test_items, FIFTH_FAMILY_MODEL, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="test360")
            model_pools[FIFTH_FAMILY_MODEL] = {"cal": lc, "test": lt}
            summaries_by_model[FIFTH_FAMILY_MODEL] = _eval_model_full_suite(FIFTH_FAMILY_MODEL, lc, lt)

            ordered = [summaries_by_model[m] for m in ALL_5_MODELS if m in summaries_by_model]
            state["stage2"] = {"hy4_preview": summaries_by_model[FIFTH_FAMILY_MODEL], "all_5_models": ordered}
            state["all_models_n360"] = ordered
            figs5 = _plot_fig5_multimodel(ordered, args.dry_run)
            state["stage2"]["figures"] = figs5
            _save_stage(
                "stage2_hy4_n360",
                state["stage2"],
                f"# Stage 2: 5th Family ({FIFTH_FAMILY_MODEL})\nMissingF1={summaries_by_model[FIFTH_FAMILY_MODEL]['diagnosis']['missing_f1']}",
                args.dry_run,
            )
            _write_master_report(state, tracker, args.dry_run, "AFTER_STAGE_2")

        # -----------------------------------------------------------------
        # STAGE 3: Realistic Multi-Paragraph Noisy Prose (360+60) + Imperfect User
        # -----------------------------------------------------------------
        if 3 in stages:
            print("\n=== STAGE 3: Realistic Long-Form Noisy Benchmark (60 cal + 360 test) + Imperfect LLM User ===", flush=True)
            noisy_cal_items = build_noisy_dataset(cal_items, tracker, rewriter_model=PRIMARY_MODEL, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec)
            noisy_test_items = build_noisy_dataset(test_items, tracker, rewriter_model=PRIMARY_MODEL, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec)
            imp_sim = make_imperfect_user_sim(tracker, sim_model=PRIMARY_MODEL, dry_run=args.dry_run, pace_sec=args.pace_sec)

            st3_models = []
            for mid in [PRIMARY_MODEL, FIFTH_FAMILY_MODEL]:
                nc = probe_dataset_budgeted(noisy_cal_items, mid, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="noisy_cal", user_sim_fn=imp_sim)
                nt = probe_dataset_budgeted(noisy_test_items, mid, tracker, dry_run=args.dry_run, workers=args.workers, pace_sec=args.pace_sec, label="noisy_test", user_sim_fn=imp_sim)
                st3_models.append(_eval_model_full_suite(mid, nc, nt))

            figs6 = _plot_fig6_noisy(st3_models, args.dry_run)
            state["stage3"] = {"models": st3_models, "figures": figs6}
            _save_stage(
                "stage3_noisy_longform_n360",
                state["stage3"],
                "\n".join([f"# Stage 3: Noisy Benchmark + Imperfect User"] + [f"- {m['model_id']}: MissingF1={m['diagnosis']['missing_f1']}" for m in st3_models]),
                args.dry_run,
            )
            _write_master_report(state, tracker, args.dry_run, "AFTER_STAGE_3")

        # -----------------------------------------------------------------
        # STAGE 4: Entropy-Collapse K x T Stress Test
        # -----------------------------------------------------------------
        if 4 in stages:
            print("\n=== STAGE 4: Entropy-Collapse K in {3,5,10} x T in {0.7,1.0} Stress Test ===", flush=True)
            f1_map = {
                m: summaries_by_model[m]["diagnosis"]["missing_f1"]
                for m in [PRIMARY_MODEL, FIFTH_FAMILY_MODEL]
                if m in summaries_by_model
            }
            kt_max = 120 if args.smoke <= 0 else min(12, len(test_items))
            kt_rows = run_entropy_kt_stress(
                test_items,
                [PRIMARY_MODEL, FIFTH_FAMILY_MODEL],
                tracker,
                diag_f1_by_model=f1_map,
                dry_run=args.dry_run,
                workers=args.workers,
                pace_sec=args.pace_sec,
                max_items=kt_max,
            )
            figs7 = _plot_fig7_kt_stress(kt_rows, args.dry_run)
            state["stage4"] = {"rows": kt_rows, "figures": figs7}
            _save_stage("stage4_entropy_kt_stress", state["stage4"], f"# Stage 4: KxT Stress ({len(kt_rows)} rows)", args.dry_run)
            _write_master_report(state, tracker, args.dry_run, "AFTER_STAGE_4")

        # -----------------------------------------------------------------
        # STAGE 5: Deep Offline Structural, 2D Phase Map, CV, LODO, Bursty, Pareto
        # -----------------------------------------------------------------
        if 5 in stages:
            print("\n=== STAGE 5: Deep Offline Structural & Robustness Suite (0 API Calls) ===", flush=True)
            s5 = run_stage5_structural(model_pools, PRIMARY_MODEL)
            figs8 = _plot_fig8_phase_heatmap(s5["phase_map_56pt"], args.dry_run)
            figs9 = _plot_fig9_cv_lodo_bursty(s5, args.dry_run)
            s5["figures"] = figs8 + figs9
            state["stage5"] = s5
            _save_stage("stage5_deep_structural", s5, "# Stage 5: Deep Structural Suite Completed", args.dry_run)
            _write_master_report(state, tracker, args.dry_run, "COMPLETED_ALL_5_STAGES")

    except (BudgetLimitReached, SiliconFlowAuthOrBalanceError) as exc:
        print(f"\n[SAFE STOP] {exc}", flush=True)
        _write_master_report(state, tracker, args.dry_run, f"STOPPED_SAFELY: {exc}")
    except Exception as exc:
        print(f"\n[ERROR] {exc}\n{traceback.format_exc()}", flush=True)
        _write_master_report(state, tracker, args.dry_run, f"ERROR_PARTIAL: {exc}")
        raise
    finally:
        tracker.save()
        snap = tracker.snapshot()
        print(
            f"\n[Overnight Summary] live_calls={snap['live_calls']} cache_hits={snap['cached_hits']} "
            f"est_rmb=¥{snap['estimated_rmb']:.4f} elapsed={snap['elapsed_min']}min",
            flush=True,
        )


if __name__ == "__main__":
    main()
