import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def _load(rel: str) -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / rel).read_text(encoding="utf-8"))


def _setup_rc() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })


def _save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGS_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig1_mdp_mechanism_and_regret(attr: Dict[str, Any], mtd: Dict[str, Any]) -> None:
    """Fig 1: (a) 13-suite +MDP gains under severe congestion; (b) 26-cell & 40-cell minimax regret frontier."""
    _setup_rc()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4), constrained_layout=True)

    # Panel (a): Severe-congestion (Cap=1/3, c=1.5) +MDP gains across 13 suites (exact logic as Section 0A)
    part_b_sev = [r for r in attr["part_b_fair_baseline_mdp"] if r["regime"] == "Cap=1/3, c=1.5"]
    part_a_sev = {r["suite"]: r["vals"] for r in attr["part_a_stepwise_ablation"] if r["regime"] == "Cap=1/3, c=1.5"}
    mdp_series: Dict[str, List[float]] = {
        "Tri-VoI-L2D -> Tri-VoI-MDP\n(Zero-Oracle)": [],
        "CC+Naor -> CC+MDP": [],
        "JG+Naor -> JG+MDP": [],
        "ACL+FD+Naor -> ACL+FD+MDP": [],
    }
    for r in part_b_sev:
        s = r["suite"]
        mdp_series["CC+Naor -> CC+MDP"].append(r["mdp_gain_cc"])
        mdp_series["JG+Naor -> JG+MDP"].append(r["mdp_gain_jg"])
        mdp_series["ACL+FD+Naor -> ACL+FD+MDP"].append(r["mdp_gain_acl"])
        mdp_series["Tri-VoI-L2D -> Tri-VoI-MDP\n(Zero-Oracle)"].append(
            round(part_a_sev[s]["Tri-VoI-MDP (Deploy-Self)"] - part_a_sev[s]["Tri-VoI-L2D"], 3)
        )
    for skey in ("setup_qwen35_to_real_ds", "setup_deepseek_generalist"):
        rm = mtd[skey]["Cap=1/3, c=1.5"]["methods"]
        mdp_series["CC+Naor -> CC+MDP"].append(
            round(rm["Calibrated-Cascade+MDP (2C-Self)"]["mean_utility"] - rm["Calibrated-Cascade+Naor"]["mean_utility"], 3)
        )
        mdp_series["JG+Naor -> JG+MDP"].append(
            round(rm["Joint-Greedy+MDP (2C-Self)"]["mean_utility"] - rm["Joint-Greedy+Naor"]["mean_utility"], 3)
        )
        mdp_series["ACL+FD+Naor -> ACL+FD+MDP"].append(
            round(rm["ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)"]["mean_utility"] - rm["ACL-2026-VoI (Live)+Full-Defer+Naor"]["mean_utility"], 3)
        )
        mdp_series["Tri-VoI-L2D -> Tri-VoI-MDP\n(Zero-Oracle)"].append(
            round(rm["Tri-VoI-MDP (Deploy-Self)"]["mean_utility"] - rm["Tri-VoI-L2D"]["mean_utility"], 3)
        )

    family_colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
    rng = np.random.default_rng(2027)
    labels = list(mdp_series.keys())
    ax1.axvline(0.0, color="#444444", ls="--", lw=1.1, zorder=1)
    for y_idx, (lbl, col) in enumerate(zip(labels, family_colors)):
        vals = np.array(mdp_series[lbl], dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax1.plot([vals.min(), vals.max()], [y_idx, y_idx], color=col, lw=1.6, alpha=0.45, zorder=2)
        ax1.scatter(vals, y_idx + jitter, color=col, alpha=0.82, s=30, edgecolors="white", linewidths=0.5, zorder=3)
        mean_v = float(vals.mean())
        pos_n = int((vals > 0).sum())
        ax1.scatter([mean_v], [y_idx], color=col, marker="D", s=58, edgecolors="#111111", linewidths=0.9, zorder=4)
        ax1.text(vals.max() + 0.03, y_idx, f"{pos_n}/13 (+{mean_v:.3f})", va="center", ha="left", fontsize=8, fontweight="bold", color="#222222")

    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_xlim(-0.25, 1.32)
    ax1.set_title("(a) Severe-Congestion (+MDP vs +Naor) Gains across 13 Suites")
    ax1.set_xlabel("Net Utility Gain at Cap=1/3, c_def=1.5 (Diamonds = 13-Suite Mean)")
    ax1.grid(axis="x", alpha=0.25)

    # Panel (b): Matched 2C-Self Minimax Regret Frontier (Proposition 5, Section 0B)
    cells_26: List[Dict[str, float]] = []
    for r in attr["part_e_matched_2c_self_attribution"]:
        v = r["vals"]
        cells_26.append({
            "CC+MDP (2C-S)": v["Calibrated-Cascade+MDP (2C-Self)"],
            "JG+MDP (2C-S)": v["Joint-Greedy+MDP (2C-Self)"],
            "ACL+FD+MDP (2C-S)": v[r["acl_2c_key"]],
            "Tri-VoI-MDP (Zero-Oracle)": v["Tri-VoI-MDP (Deploy-Self)"],
        })
    for skey in ("setup_qwen35_to_real_ds", "setup_deepseek_generalist"):
        for rg in ("Cap=2, c=1.5", "Cap=1/3, c=1.5"):
            rm = mtd[skey][rg]["methods"]
            cells_26.append({
                "CC+MDP (2C-S)": rm["Calibrated-Cascade+MDP (2C-Self)"]["mean_utility"],
                "JG+MDP (2C-S)": rm["Joint-Greedy+MDP (2C-Self)"]["mean_utility"],
                "ACL+FD+MDP (2C-S)": rm["ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)"]["mean_utility"],
                "Tri-VoI-MDP (Zero-Oracle)": rm["Tri-VoI-MDP (Deploy-Self)"]["mean_utility"],
            })

    pf = attr["part_f_c_ask_regime_map"]
    policies = [
        ("CC+MDP (2C-S)", "Calibrated-Cascade+MDP (2C-Self)", "#009E73"),
        ("JG+MDP (2C-S)", "Joint-Greedy+MDP (2C-Self)", "#E69F00"),
        ("ACL+FD+MDP (2C-S)", None, "#CC79A7"),
        ("Tri-VoI-MDP (Zero-Oracle)", "Tri-VoI-MDP (Deploy-Self)", "#0072B2"),
    ]
    for pname, pf_key, col in policies:
        u26 = float(np.mean([c[pname] for c in cells_26]))
        wr26 = float(max(max(c.values()) - c[pname] for c in cells_26))
        u40_vals, reg40_vals = [], []
        for r in pf:
            vv = r["vals"]
            k_r = r["acl_2c_key"] if pf_key is None else pf_key
            best40 = max(
                vv["Calibrated-Cascade+MDP (2C-Self)"],
                vv["Joint-Greedy+MDP (2C-Self)"],
                vv[r["acl_2c_key"]],
                vv["Tri-VoI-MDP (Deploy-Self)"],
            )
            u40_vals.append(vv[k_r])
            reg40_vals.append(best40 - vv[k_r])
        u40 = float(np.mean(u40_vals))
        wr40 = float(max(reg40_vals))

        ax2.plot([wr26, wr40], [u26, u40], color=col, ls=":", lw=1.2, alpha=0.65)
        ax2.scatter([wr26], [u26], color=col, marker="o", s=78, edgecolors="#111111", linewidths=0.8, label=f"{pname} (26-cell)", zorder=4)
        ax2.scatter([wr40], [u40], color="white", marker="s", s=72, edgecolors=col, linewidths=2.0, zorder=4)
        if "Tri-VoI-MDP" in pname:
            ax2.annotate(f"26c: ({wr26:.3f}, {u26:.3f})", (wr26, u26), xytext=(wr26 + 0.05, u26 + 0.01), fontsize=8, fontweight="bold", color=col)
            ax2.annotate(f"40c: ({wr40:.3f}, {u40:.3f})", (wr40, u40), xytext=(wr40 + 0.05, u40 - 0.03), fontsize=8, fontweight="bold", color=col)

    ax2.scatter([], [], color="#555555", marker="o", s=55, label="Filled circle = 26-cell cross-suite")
    ax2.scatter([], [], color="white", marker="s", s=55, edgecolors="#555555", linewidths=1.5, label="Open square = 40-cell c_ask sweep")
    ax2.invert_xaxis()
    ax2.set_title("(b) Prop 5: Matched 2C-Self Minimax Regret Frontier")
    ax2.set_xlabel("Worst-Case Regret (Lower is Better; Left-to-Right Inverted)")
    ax2.set_ylabel("Mean Net Utility (Higher is Better)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower left", frameon=True, fontsize=7.2)

    _save_fig(fig, "fig1_mdp_mechanism_and_regret")


def plot_fig3_efficiency_and_c_ask_phase(up: Dict[str, Any], attr: Dict[str, Any]) -> None:
    """Fig 3: (a) Exact SQLite 53.7%-57.8% token reduction; (b)-(c) Matched 2C-Self c_ask phase curves."""
    _setup_rc()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)

    # Panel (a): Token savings (Section 6)
    ax = axes[0]
    rows = up["two_call_tok_rows"]
    lbls = [f"{r['suite']}\n{r['model'].replace('-35B-A3B', '')}" for r in rows]
    y = np.arange(len(rows))
    h = 0.36
    tok_full = [r["realized_tok_full"] for r in rows]
    tok_mdp = [r["realized_tok_mdp"] for r in rows]
    ax.barh(y - h / 2, tok_full, height=h, color="#8C9BA5", label="5-Call Tri-VoI-L2D")
    ax.barh(y + h / 2, tok_mdp, height=h, color="#0072B2", label="2-Call Tri-VoI-MDP (Zero-Oracle)")
    for idx, r in enumerate(rows):
        du = r["tri_mdp_util"] - r["tri_full_util"]
        ax.text(
            r["realized_tok_mdp"] + 90,
            idx + h / 2,
            f"-{r['mdp_tok_reduction_pct']:.1f}% ({du:+.2f}u)",
            va="center",
            ha="left",
            fontsize=7.3,
            fontweight="bold",
            color="#005587",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 5800)
    ax.set_title("(a) Realized SQLite Tokens (Sec. 6)")
    ax.set_xlabel("Mean Realized Tokens / Item")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=7.3)

    # Panels (b) & (c): c_ask sweeps (Section 9F)
    pf = attr["part_f_c_ask_regime_map"]
    configs = [
        (axes[1], "v2 / DeepSeek-V3.2", "Cap=2", "(b) v2 / DeepSeek (Cap=2, c_ask Sweep)"),
        (axes[2], "MediQ-Holdout / Q3.5->RealDS", "Cap=1/3", "(c) MediQ-Holdout (Cap=1/3, c_ask Sweep)"),
    ]
    series_defs = [
        ("CC+MDP (2C-S)", "Calibrated-Cascade+MDP (2C-Self)", "#009E73", "s--"),
        ("JG+MDP (2C-S)", "Joint-Greedy+MDP (2C-Self)", "#E69F00", "^-."),
        ("ACL+FD+MDP (2C-S)", None, "#CC79A7", "v:"),
        ("Tri-VoI-MDP (Zero-Oracle)", "Tri-VoI-MDP (Deploy-Self)", "#0072B2", "o-"),
    ]
    for ax_sub, s_name, c_reg, title in configs:
        sub = sorted([r for r in pf if r["suite"] == s_name and r["cap_regime"] == c_reg], key=lambda x: x["c_ask"])
        cs = [r["c_ask"] for r in sub]
        for disp, k_name, col, fmt in series_defs:
            ys = [r["vals"][r["acl_2c_key"] if k_name is None else k_name] for r in sub]
            lw = 2.2 if "Tri-VoI-MDP" in disp else 1.5
            ax_sub.plot(cs, ys, fmt, color=col, lw=lw, ms=4.8, label=disp)
        ax_sub.set_title(title)
        ax_sub.set_xlabel("Clarification Cost $c_{\\mathrm{ask}}$")
        ax_sub.set_ylabel("Mean Net Utility")
        ax_sub.set_xticks([0.5, 1.5, 3.0, 4.5])
        ax_sub.grid(alpha=0.25)
        ax_sub.legend(loc="lower left", frameon=True, fontsize=7.2)

    _save_fig(fig, "fig3_efficiency_and_c_ask_phase")


def plot_fig4_mediq_disjoint_clinical_blind(mtd: Dict[str, Any]) -> None:
    """Fig 4: Organ-system-disjoint MediQ-Test-Disjoint Setup A & honest Setup B parity + split metrics."""
    _setup_rc()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.5), gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True)

    # Panel (a): Setup A (Qwen3.5 -> DeepSeek-V3.2) across 4 regimes
    regs = ["Cap=2, c=1.5", "Cap=2, c=3.0", "Cap=1/3, c=1.5", "Cap=1/3, c=3.0"]
    sa = mtd["setup_qwen35_to_real_ds"]
    methods_a = [
        ("KL-PBD+ (Warm)", "KL-PBD+ (Warm)", "#7F7F7F"),
        ("CC+MDP (2C-S)", "Calibrated-Cascade+MDP (2C-Self)", "#009E73"),
        ("JG+MDP (2C-S)", "Joint-Greedy+MDP (2C-Self)", "#E69F00"),
        ("ACL+FD+MDP (2C-S)", "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)", "#CC79A7"),
        ("Tri-VoI-L2D", "Tri-VoI-L2D", "#D55E00"),
        ("Tri-VoI-MDP (Zero-Oracle)", "Tri-VoI-MDP (Deploy-Self)", "#0072B2"),
    ]
    x = np.arange(len(regs))
    w = 0.13
    for i, (disp, mkey, col) in enumerate(methods_a):
        vals = [sa[rg]["methods"][mkey]["mean_utility"] for rg in regs]
        offset = (i - (len(methods_a) - 1) / 2.0) * w
        bars = ax1.bar(x + offset, vals, width=w * 0.92, color=col, label=disp)
        if "Tri-VoI-MDP" in disp:
            for b, val in zip(bars, vals):
                ax1.text(b.get_x() + b.get_width() / 2.0, val + 0.06, f"{val:.2f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#005587")

    ax1.annotate(
        "Severe congestion vs. Tri-VoI-L2D:\n+0.679 (p=0.0137) at c=1.5\n+0.785 (p=0.0057) at c=3.0",
        xy=(2.0 + 2.5 * w, 3.65),
        xytext=(1.7, 4.42),
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F7FC", edgecolor="#0072B2"),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.0),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(regs)
    ax1.set_ylim(0, 5.15)
    ax1.set_title("(a) MediQ-Test-Disjoint Setup A (Qwen3.5 -> Real DeepSeek)")
    ax1.set_ylabel("Mean Net Utility (N_test = 180, Disjoint Organs)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper right", frameon=True, ncol=2, fontsize=6.9)

    # Panel (b): Honest Setup B (DeepSeek Generalist) parity + Setup A clinical breakdown box
    sb = mtd["setup_deepseek_generalist"]
    regs_b = ["Cap=2, c=1.5", "Cap=1/3, c=1.5"]
    methods_b = [
        ("CC+MDP (2C-S)", "Calibrated-Cascade+MDP (2C-Self)", "#009E73"),
        ("JG+MDP (2C-S)", "Joint-Greedy+MDP (2C-Self)", "#E69F00"),
        ("ACL+FD+MDP (2C-S)", "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)", "#CC79A7"),
        ("Tri-VoI-L2D", "Tri-VoI-L2D", "#D55E00"),
        ("Tri-VoI-MDP (Zero-Oracle)", "Tri-VoI-MDP (Deploy-Self)", "#0072B2"),
    ]
    xb = np.arange(len(regs_b))
    wb = 0.15
    for i, (disp, mkey, col) in enumerate(methods_b):
        vals = [sb[rg]["methods"][mkey]["mean_utility"] for rg in regs_b]
        offset = (i - (len(methods_b) - 1) / 2.0) * wb
        bars = ax2.bar(xb + offset, vals, width=wb * 0.92, color=col, label=disp)
        for b, val in zip(bars, vals):
            if "Tri-VoI-MDP" in disp or (disp == "Tri-VoI-L2D" and val > 4.0) or (disp == "ACL+FD+MDP (2C-S)" and val > 2.8 and b.get_x() > 0.5):
                ax2.text(b.get_x() + b.get_width() / 2.0, val + 0.06, f"{val:.2f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#222222")

    mdp_c2 = sa["Cap=2, c=1.5"]["methods"]["Tri-VoI-MDP (Deploy-Self)"]
    mdp_sv = sa["Cap=1/3, c=1.5"]["methods"]["Tri-VoI-MDP (Deploy-Self)"]
    ax2.text(
        0.97,
        0.96,
        "Setup A Tri-VoI-MDP Clinical Metrics:\n"
        f"Cap=2: MedAcc {mdp_c2['raw_answer_acc_pct']:.1f}% | EscRec {mdp_c2['scope_escalation_recall_pct']:.1f}% ({mdp_c2['mean_realized_tokens']:.0f} tok)\n"
        f"Cap=1/3: MedAcc {mdp_sv['raw_answer_acc_pct']:.1f}% | EscRec {mdp_sv['scope_escalation_recall_pct']:.1f}%",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
    )
    ax2.set_xticks(xb)
    ax2.set_xticklabels(regs_b)
    ax2.set_ylim(0, 5.15)
    ax2.set_title("(b) Setup B Generalist Parity & Clinical Breakdown")
    ax2.set_ylabel("Mean Net Utility (Setup B, DeepSeek Generalist)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="center right", frameon=True, fontsize=7.0)

    _save_fig(fig, "fig4_mediq_disjoint_clinical_blind")


def main() -> None:
    attr = _load("attribution_and_dedup_audit.json")
    up = _load("deploy_self_canonical_upgrade.json")
    mtd = _load("mediq_test_disjoint_eval.json")
    plot_fig1_mdp_mechanism_and_regret(attr, mtd)
    plot_fig3_efficiency_and_c_ask_phase(up, attr)
    plot_fig4_mediq_disjoint_clinical_blind(mtd)
    print("Saved results/figs/fig{1_mdp_mechanism_and_regret,3_efficiency_and_c_ask_phase,4_mediq_disjoint_clinical_blind}.{pdf,png}")


if __name__ == "__main__":
    main()
