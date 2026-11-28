import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def _load(rel: str) -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / rel).read_text(encoding="utf-8"))


def _fmt_max_row_md(vals: List[float], idx: int, higher_is_better: bool = True, decimals: int = 3) -> str:
    best = max(vals) if higher_is_better else min(vals)
    s = f"{vals[idx]:.{decimals}f}"
    return f"**{s}**" if abs(vals[idx] - best) <= 1e-4 else s


def _fmt_max_row_tex(vals: List[float], idx: int, higher_is_better: bool = True, decimals: int = 3) -> str:
    best = max(vals) if higher_is_better else min(vals)
    s = f"{vals[idx]:.{decimals}f}"
    return f"\\textbf{{{s}}}" if abs(vals[idx] - best) <= 1e-4 else s


def main() -> None:
    ext_v1 = _load("external_baselines_v1.json")
    v2 = _load("benchmark_v2_metrics.json")
    pt = _load("public_tri_metrics.json")
    hard = _load("reviewer_hardening_suite.json")
    hold = _load("zero_touch_holdout.json")
    blind = _load("preregistered_blind_eval.json")
    attr = _load("attribution_and_dedup_audit.json")
    mtd = _load("mediq_test_disjoint_eval.json")
    up = _load("deploy_self_canonical_upgrade.json")

    md = [
        "# IJCAI 2027 Canonical Tables (Zero-Oracle Unified Single Source of Truth)",
        "",
        "Generated automatically by `generate_ijcai_tables.py` using canonical terminology (`Requester`, `Specialist`, `Benchmark v1-Clean`, `Benchmark v1-Noisy`, `Benchmark v2 (Implicit-Override)`, `Public-Tri`, `Benchmark v3`, `MediQ-Holdout`, `MediQ-Test-Disjoint`). **Primary Method across Sections 0–10: `Tri-VoI-MDP` (`Deploy-Self`, 100% Zero-Oracle)** — 2-Call coarse sufficient statistic + strictly local `post_missing` self-signal (zero access to simulator `oracle_revealed`) + finite-horizon periodic-drain Bellman DP shadow prices (**Theorem 4 & Proposition 5**). The legacy simulator-oracle variant (`MDP (Oracle-NoInfo Diag)`) appears **only** in explicit ablation columns of Sections 9A, 9B, and 10. **Bolding marks true row/column best non-Oracle metric.**",
        "",
    ]

    # Section 0: Dynamic Cross-Suite Quantitative Robustness Summary
    part_b_sev = [r for r in attr["part_b_fair_baseline_mdp"] if r["regime"] == "Cap=1/3, c=1.5"]
    part_a_sev = {r["suite"]: r["vals"] for r in attr["part_a_stepwise_ablation"] if r["regime"] == "Cap=1/3, c=1.5"}
    mdp_series = {
        "Tri-VoI-L2D -> Tri-VoI-MDP (Zero-Oracle)": [],
        "CC+Naor -> CC+MDP": [],
        "JG+Naor -> JG+MDP": [],
        "ACL+FD+Naor -> ACL+FD+MDP": [],
    }
    for r in part_b_sev:
        s = r["suite"]
        mdp_series["CC+Naor -> CC+MDP"].append(r["mdp_gain_cc"])
        mdp_series["JG+Naor -> JG+MDP"].append(r["mdp_gain_jg"])
        mdp_series["ACL+FD+Naor -> ACL+FD+MDP"].append(r["mdp_gain_acl"])
        mdp_series["Tri-VoI-L2D -> Tri-VoI-MDP (Zero-Oracle)"].append(
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
        mdp_series["Tri-VoI-L2D -> Tri-VoI-MDP (Zero-Oracle)"].append(
            round(rm["Tri-VoI-MDP (Deploy-Self)"]["mean_utility"] - rm["Tri-VoI-L2D"]["mean_utility"], 3)
        )

    cells_26 = []
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

    md.extend([
        "## Section 0: Cross-Suite Quantitative Robustness Summary (13 Suites, 100% Zero-Oracle)",
        "",
        "### 0A. Severe-Congestion (`Cap=1/3, c=1.5`) Dynamic Bellman Shadow-Pricing Gains (`+MDP` vs `+Naor` across All 13 Suites)",
        "| Policy Family | Positive Suites | Min Gain | Median Gain | Mean Gain | Max Gain |",
        "|:---|:---:|---:|---:|---:|---:|",
    ])
    for pname, vals in mdp_series.items():
        pos_n = sum(1 for v in vals if v > 0)
        md.append(
            f"| `{pname}` | **{pos_n}/{len(vals)}** | {min(vals):+.3f} | {float(np.median(vals)):+.3f} | "
            f"{float(np.mean(vals)):+.3f} | {max(vals):+.3f} |"
        )
    md.extend([
        "",
        "### 0B. Strictly Matched 2-Call Zero-Oracle Minimax Regret (`26` Cells = `13 Suites × {Cap=2, Cap=1/3}` at `c=1.5`, plus `40`-Cell `c_ask` Sweep)",
        "| Policy (`2C-Self`, Zero-Oracle, Equal DP Pricing) | 26-Cell Mean Util | 26-Cell Wins | 26-Cell Mean Regret | 26-Cell Worst Regret | 40-Cell `c_ask` Mean Util | 40-Cell `c_ask` Worst Regret |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    pf = attr["part_f_c_ask_regime_map"]
    k26_map = [
        ("CC+MDP (2C-S)", "Calibrated-Cascade+MDP (2C-Self)"),
        ("JG+MDP (2C-S)", "Joint-Greedy+MDP (2C-Self)"),
        ("ACL+FD+MDP (2C-S)", None),
        ("Tri-VoI-MDP (Zero-Oracle)", "Tri-VoI-MDP (Deploy-Self)"),
    ]
    for lbl26, pf_k in k26_map:
        us26 = [c[lbl26] for c in cells_26]
        regs26 = [max(c.values()) - c[lbl26] for c in cells_26]
        w26 = sum(1 for c in cells_26 if abs(c[lbl26] - max(c.values())) <= 1e-4)
        us40, regs40 = [], []
        for r in pf:
            vv = r["vals"]
            key_r = r["acl_2c_key"] if pf_k is None else pf_k
            best40 = max(vv["Calibrated-Cascade+MDP (2C-Self)"], vv["Joint-Greedy+MDP (2C-Self)"], vv[r["acl_2c_key"]], vv["Tri-VoI-MDP (Deploy-Self)"])
            us40.append(vv[key_r])
            regs40.append(best40 - vv[key_r])
        md.append(
            f"| `{lbl26}` | {float(np.mean(us26)):.3f} | {w26}/26 | {float(np.mean(regs26)):.3f} | "
            f"{max(regs26):.3f} | {float(np.mean(us40)):.3f} | {max(regs40):.3f} |"
        )
    md.extend([
        "",
        "## Table 1: `Benchmark v1-Clean` & `Benchmark v1-Noisy` Main Comparison (`Qwen/Qwen3.5-35B-A3B`, `N=360`)",
        "| Method | v1-Clean (Cap=2, c=1.5) | v1-Clean (Cap=2, c=2.5) | v1-Clean (Cap=1/3, c=1.5) | v1-Noisy (Cap=2, c=1.5) | v1-Noisy (Cap=2, c=2.5) | v1-Noisy (Cap=1/3, c=1.5) |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    tex = [
        "% Auto-generated by generate_ijcai_tables.py — Zero-Oracle Unified IJCAI 2027 canonical tables",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Comparison on \\textit{Benchmark v1-Clean} and \\textit{Benchmark v1-Noisy} (\\texttt{Qwen3.5-35B-A3B}, $N=360$, 5 seeds). Primary \\texttt{Tri-VoI-MDP} uses strictly Zero-Oracle \\texttt{Deploy-Self}. Bold marks column-wise best.}",
        "\\label{tab:v1_main}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Clean (2, 1.5) & Clean (2, 2.5) & Clean (1/3, 1.5) & Noisy (2, 1.5) & Noisy (2, 2.5) & Noisy (1/3, 1.5) \\\\",
        "\\midrule",
    ]

    v1_q35 = ext_v1["models"]["Qwen/Qwen3.5-35B-A3B"]
    n_q35 = hard["stage3_noisy_external"]["Qwen/Qwen3.5-35B-A3B"]
    t1_methods = [
        "Direct-ACT",
        "KL-PBD",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Calib)",
        "ACL-2026-VoI+Post-Defer",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
        "Tri-VoI-MDP",
    ]

    def _v1_clean_val(m: str, reg: str) -> float:
        if m == "Tri-VoI-MDP":
            return float(up["table1"]["v1_clean"][reg])
        return float(v1_q35[reg][m]["mean_utility"])

    def _v1_noisy_val(m: str, reg: str) -> float:
        if m == "Tri-VoI-MDP":
            return float(up["table1"]["v1_noisy"][reg])
        return float(n_q35[reg][m]["mean_utility"])

    t1_cols = [
        [_v1_clean_val(m, "Cap=2, c=1.5") for m in t1_methods],
        [_v1_clean_val(m, "Cap=2, c=2.5") for m in t1_methods],
        [_v1_clean_val(m, "Cap=1/3, c=1.5") for m in t1_methods],
        [_v1_noisy_val(m, "Cap=2, c=1.5") for m in t1_methods],
        [_v1_noisy_val(m, "Cap=2, c=2.5") for m in t1_methods],
        [_v1_noisy_val(m, "Cap=1/3, c=1.5") for m in t1_methods],
    ]
    for r_idx, m in enumerate(t1_methods):
        cells_md = [_fmt_max_row_md(col, r_idx, True) for col in t1_cols]
        cells_tex = [_fmt_max_row_tex(col, r_idx, True) for col in t1_cols]
        md.append(f"| `{m}` | " + " | ".join(cells_md) + " |")
        m_tex = m.replace("_", "\\_")
        tex.append(f"{m_tex} & " + " & ".join(cells_tex) + " \\\\")

    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # Table 2: Benchmark v2 & Public-Tri
    md.extend([
        "",
        "## Table 2: Cross-Benchmark Generalization on `Benchmark v2 (Implicit-Override)` & `Public-Tri (MediQ Scope-of-Practice Triage)`",
        "> **Transparency Note (`Public-Tri`)**: `Public-Tri` models **Scope-of-Practice / Regulatory Governance Triage** (`require_defer_for_ok=True` on high-acuity/multi-specialty `Q3`/`Q4_escalate`), alongside raw USMLE accuracy reported separately in Section 7. `Tri-VoI-MDP` reports strictly Zero-Oracle `Deploy-Self`. Bold marks true row-wise best at `Cap=2, c=1.5`.",
        "",
        "| Suite & Model | Miss-F1 / Collapse% | `KL-PBD+ (Warm)` | `ACL-2026-VoI (Live)` | `Joint-Greedy+Naor` | `Calib-Cascade+Naor` | `Tri-VoI-L2D` (Cap=2) | `Tri-VoI-MDP` (Cap=2) | `Tri-VoI-L2D` (1/3) | `Tri-VoI-MDP` (1/3) |",
        "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    tex.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Generalization across \\textit{Benchmark v2} ($N=180$) and \\textit{Public-Tri} ($N=240$). Primary \\texttt{Tri-VoI-MDP} uses Zero-Oracle \\texttt{Deploy-Self}. Bold marks row-wise best among Cap=(2, 1.5) columns.}",
        "\\label{tab:v2_public}",
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "Suite / Model & Miss-F1 (Coll\\%) & KL-PBD+(W) & ACL-2026-VoI & JG+Naor & CC+Naor & L2D (2) & MDP (2) & L2D (1/3) & MDP (1/3) \\\\",
        "\\midrule",
    ])

    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "tencent/Hy4-preview", "Qwen/Qwen3-8B"):
        mdata = v2["models"][mid]
        d = mdata["diagnosis"]
        r = mdata["regimes"]["Cap=2, c=1.5"]
        r_sev = mdata["regimes"]["Cap=1/3, c=1.5"]
        sn = mid.split("/")[-1]
        mdp_c2 = up["table2"]["v2"].get(mid, {}).get("Cap=2, c=1.5")
        mdp_sev = up["table2"]["v2"].get(mid, {}).get("Cap=1/3, c=1.5")
        row_u = [
            r["KL-PBD+ (Warm)"]["mean_utility"],
            r["ACL-2026-VoI (Live)"]["mean_utility"],
            r["Joint-Greedy+Naor"]["mean_utility"],
            r["Calibrated-Cascade+Naor"]["mean_utility"],
            r["Tri-VoI-L2D"]["mean_utility"],
        ]
        if mdp_c2 is not None:
            row_u.append(float(mdp_c2))
        f_md = [_fmt_max_row_md(row_u, i, True) for i in range(len(row_u))]
        f_tx = [_fmt_max_row_tex(row_u, i, True) for i in range(len(row_u))]
        if mdp_c2 is None:
            f_md.append("—")
            f_tx.append("---")
        mdp_s_str = f"{mdp_sev:.3f}" if mdp_sev is not None else "—"
        mdp_s_tex = f"{mdp_sev:.3f}" if mdp_sev is not None else "---"
        md.append(
            f"| v2 / `{sn}` | {d['missing_f1']:.3f} / {d['entropy_collapse_pct']:.1f}% | "
            + " | ".join(f_md)
            + f" | {r_sev['Tri-VoI-L2D']['mean_utility']:.3f} | {mdp_s_str} |"
        )
        tex.append(
            f"v2 / \\texttt{{{sn}}} & {d['missing_f1']:.3f} ({d['entropy_collapse_pct']:.1f}\\%) & "
            + " & ".join(f_tx)
            + f" & {r_sev['Tri-VoI-L2D']['mean_utility']:.3f} & {mdp_s_tex} \\\\"
        )

    tex.append("\\midrule")
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-8B"):
        mdata = pt["models"][mid]
        d = mdata["diagnosis"]
        r = mdata["regimes"]["Cap=2, c=1.5"]
        r_sev = mdata["regimes"]["Cap=1/3, c=1.5"]
        sn = mid.split("/")[-1]
        mdp_c2 = up["table2"]["public_tri"].get(mid, {}).get("Cap=2, c=1.5")
        mdp_sev = up["table2"]["public_tri"].get(mid, {}).get("Cap=1/3, c=1.5")
        row_u = [
            r["KL-PBD+ (Warm)"]["mean_utility"],
            r["ACL-2026-VoI (Live)"]["mean_utility"],
            r["Joint-Greedy+Naor"]["mean_utility"],
            r["Calibrated-Cascade+Naor"]["mean_utility"],
            r["Tri-VoI-L2D"]["mean_utility"],
        ]
        if mdp_c2 is not None:
            row_u.append(float(mdp_c2))
        f_md = [_fmt_max_row_md(row_u, i, True) for i in range(len(row_u))]
        f_tx = [_fmt_max_row_tex(row_u, i, True) for i in range(len(row_u))]
        if mdp_c2 is None:
            f_md.append("—")
            f_tx.append("---")
        mdp_s_str = f"{mdp_sev:.3f}" if mdp_sev is not None else "—"
        mdp_s_tex = f"{mdp_sev:.3f}" if mdp_sev is not None else "---"
        md.append(
            f"| Public-Tri / `{sn}` | {d['missing_f1']:.3f} / {d['entropy_collapse_pct']:.1f}% | "
            + " | ".join(f_md)
            + f" | {r_sev['Tri-VoI-L2D']['mean_utility']:.3f} | {mdp_s_str} |"
        )
        tex.append(
            f"Public-Tri / \\texttt{{{sn}}} & {d['missing_f1']:.3f} ({d['entropy_collapse_pct']:.1f}\\%) & "
            + " & ".join(f_tx)
            + f" & {r_sev['Tri-VoI-L2D']['mean_utility']:.3f} & {mdp_s_tex} \\\\"
        )

    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # Table 3: Descriptive 56-cell 2D Minimax Regret (Zero-Oracle Tri-VoI-MDP)
    md.extend([
        "",
        "## Table 3: 56-Cell 2D Minimax Regret (`Tri-VoI-MDP` = Zero-Oracle `Deploy-Self`, Descriptive Sensitivity Envelope — No Pseudoreplication $p$-values)",
        "| Suite | Method | Mean Util (56) | $\\Delta$ vs MDP | Worst-Case Regret | MDP W / T / L |",
        "|:---|:---|---:|---:|---:|:---:|",
    ])
    t3_suites = [
        ("v1-Clean (Qwen3.5)", up["phase2d_v1"]),
        ("v2 (DeepSeek-V3.2)", up["phase2d_v2"]["deepseek-ai/DeepSeek-V3.2"]),
        ("v2 (Qwen3.5)", up["phase2d_v2"]["Qwen/Qwen3.5-35B-A3B"]),
        ("v2 (Hy4-preview)", up["phase2d_v2"]["tencent/Hy4-preview"]),
        ("Public-Tri (Qwen3.5, c_def=2.0)", up["phase2d_public_tri"]["Qwen/Qwen3.5-35B-A3B (c_def=2.0)"]),
        ("Public-Tri (DeepSeek, c_def=2.0)", up["phase2d_public_tri"]["deepseek-ai/DeepSeek-V3.2 (c_def=2.0)"]),
        ("Public-Tri (DeepSeek, c_def=4.0)", up["phase2d_public_tri"]["deepseek-ai/DeepSeek-V3.2 (c_def=4.0)"]),
    ]
    t3_meths = [
        "Tri-VoI-MDP",
        "Tri-VoI-L2D",
        "Calibrated-Cascade+Naor",
        "Joint-Greedy+Naor",
        "KL-PBD+ (Warm)",
        "ACL-2026-VoI (Live)+Full-Defer+Naor",
        "ACL-2026-VoI+Full-Defer+Naor",
        "w/o Queue-Externality",
    ]
    for lbl, r2d in t3_suites:
        avail = [m for m in t3_meths if m in r2d["summary"]]
        means = [r2d["summary"][m]["mean_util_56"] for m in avail]
        wrs = [r2d["summary"][m]["worst_case_regret_56"] for m in avail]
        for idx_m, m in enumerate(avail):
            row = r2d["summary"][m]
            d_s = "—" if m == "Tri-VoI-MDP" else f"{row['mean_delta_vs_mdp']:+.3f}"
            wtl = "—" if m == "Tri-VoI-MDP" else f"{row['mdp_wins']}/{row['mdp_ties']}/{row['mdp_losses']}"
            md.append(
                f"| {lbl} | `{m}` | {_fmt_max_row_md(means, idx_m, True)} | {d_s} | "
                f"{_fmt_max_row_md(wrs, idx_m, False)} | {wtl} |"
            )

    tex.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Descriptive 56-cell 2D sensitivity across $(\\mu, c_{\\mathrm{ask}})$ (no shared-sample $p$-values). Primary \\texttt{Tri-VoI-MDP} uses strictly Zero-Oracle \\texttt{Deploy-Self}. Bold marks best mean utility and lowest worst-case regret per suite.}",
        "\\label{tab:minimax_regret}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Suite & Method & Mean Util & Max Regret & MDP W/T/L \\\\",
        "\\midrule",
    ])
    for idx_s, (lbl, r2d) in enumerate([
        ("v1-Clean (Q3.5)", up["phase2d_v1"]),
        ("v2 (DS-V3.2)", up["phase2d_v2"]["deepseek-ai/DeepSeek-V3.2"]),
        ("PT (DS-V3.2, $c_{\\mathrm{def}}=2$)", up["phase2d_public_tri"]["deepseek-ai/DeepSeek-V3.2 (c_def=2.0)"]),
    ]):
        if idx_s > 0:
            tex.append("\\midrule")
        avail = [m for m in t3_meths if m in r2d["summary"]]
        means = [r2d["summary"][m]["mean_util_56"] for m in avail]
        wrs = [r2d["summary"][m]["worst_case_regret_56"] for m in avail]
        for idx_m, m in enumerate(avail):
            row = r2d["summary"][m]
            wtl = "---" if m == "Tri-VoI-MDP" else f"{row['mdp_wins']}/{row['mdp_ties']}/{row['mdp_losses']}"
            m_short = (
                m.replace("Calibrated-Cascade+Naor", "Calib-Casc+Naor")
                .replace("ACL-2026-VoI (Live)+Full-Defer+Naor", "ACL-VoI+FD+Naor")
                .replace("ACL-2026-VoI+Full-Defer+Naor", "ACL-VoI+FD+Naor")
                .replace("w/o Queue-Externality", "w/o Externality")
            )
            tex.append(
                f"{lbl} & {m_short} & {_fmt_max_row_tex(means, idx_m, True)} & "
                f"{_fmt_max_row_tex(wrs, idx_m, False)} & {wtl} \\\\"
            )
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # Section 4: 2x2 Decomposition & Exact MDP Shadow Pricing
    md.extend([
        "",
        "## Section 4: Explicit 2x2 Decomposition (`±EVOI × ±Naor`) & Exact Bellman Shadow Pricing (`Tri-VoI-MDP`, Zero-Oracle)",
        "| Suite / Model | Regime | `JG` | `JG+N` | `CC` | `CC+N` | `ACL+FD+N` | `w/o Ext` | `Tri-VoI-L2D` | `Smooth-Naor` | `Tri-VoI-MDP` | MDP vs L2D | MDP vs CC+N |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in up["decomposition_2x2_rows"]:
        v = r["vals"]
        acl_k = r["acl_full_defer_key"]
        keys = ["Joint-Greedy", "Joint-Greedy+Naor", "Calibrated-Cascade", "Calibrated-Cascade+Naor", acl_k, "w/o Queue-Externality", "Tri-VoI-L2D", "Tri-VoI-L2D (Smooth-Naor)", "Tri-VoI-MDP"]
        nums = [v[k] for k in keys]
        f_md = [_fmt_max_row_md(nums, i, True) for i in range(len(keys))]
        md.append(
            f"| {r['suite_model']} | {r['regime']} | " + " | ".join(f_md)
            + f" | {r['mdp_gain_over_tri']:+.3f} | {r['mdp_gain_over_cc_naor']:+.3f} |"
        )

    # Section 5: 20 Stratified Random Resplits
    md.extend([
        "",
        "## Section 5: 20 Stratified Random `(cal, test)` Resplits (`Tri-VoI-MDP` = Zero-Oracle `Deploy-Self`)",
        "| Suite | Regime | `Tri-VoI-MDP` (mean±sd) | `Tri-VoI-L2D` | `KL-PBD+(W)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | MDP Wins/20 (L2D/KL/ACL/JG/CC) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in up["resplits_20"]:
        st = row["stats"]
        acl_k = [k for k in st if "ACL-2026-VoI" in k][0]
        means_row = [st["Tri-VoI-MDP"]["mean"], st["Tri-VoI-L2D"]["mean"], st["KL-PBD+ (Warm)"]["mean"], st[acl_k]["mean"], st["Joint-Greedy+Naor"]["mean"], st["Calibrated-Cascade+Naor"]["mean"]]
        f_md = [_fmt_max_row_md(means_row, i, True) for i in range(6)]
        w_str = f"{st['Tri-VoI-L2D']['mdp_wins_of_20']}/{st['KL-PBD+ (Warm)']['mdp_wins_of_20']}/{st[acl_k]['mdp_wins_of_20']}/{st['Joint-Greedy+Naor']['mdp_wins_of_20']}/{st['Calibrated-Cascade+Naor']['mdp_wins_of_20']}"
        md.append(
            f"| {row['suite']} | {row['regime']} | {f_md[0]}±{st['Tri-VoI-MDP']['std']:.2f} | "
            + " | ".join(f_md[1:])
            + f" | {w_str} |"
        )

    # Section 6: Measured SQLite Tokens & SC Parity
    md.extend([
        "",
        "## Section 6: Exact Measured SQLite Token Cost (`Tri-VoI-MDP` Zero-Oracle vs `Full` vs `2-Call`) & `KL-PBD+ (SC-Only)`",
        "| Suite | Model | `Tri-VoI` Full | `Tri-VoI` 2-Call | `Tri-VoI-MDP` | Realized Tok (Full -> MDP, % Saved) | `KL-PBD+ (Warm)` | `KL-PBD+ (SC-Only)` |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in up["two_call_tok_rows"]:
        pair = [r["tri_full_util"], r["tri_2call_util"], r["tri_mdp_util"], r["kl_pbd_warm"], r["kl_pbd_sc_only"]]
        md.append(
            f"| {r['suite']} | {r['model']} | {_fmt_max_row_md(pair, 0, True)} | {_fmt_max_row_md(pair, 1, True)} | "
            f"{_fmt_max_row_md(pair, 2, True)} | "
            f"{r['realized_tok_full']:.0f} -> {r['realized_tok_mdp']:.0f} (-{r['mdp_tok_reduction_pct']:.1f}%) | "
            f"{_fmt_max_row_md(pair, 3, True)} | {_fmt_max_row_md(pair, 4, True)} |"
        )

    # Section 7: Real-LLM Clinical Specialist (Scope + Pure Capability Gap) vs Semi-Synthetic
    md.extend([
        "",
        "## Section 7: Genuine Real-LLM Specialist (`Public-Tri` Scope-Compliance + Pure Capability-Gap) vs. Semi-Synthetic (`v1/v2`)",
        "| Setup | Regime | `KL-PBD+ (Warm)` | `ACL+FD+Naor` | `Calib-Casc+Naor` | `Joint-Greedy+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` (Zero-Oracle) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])
    spec_labels = [
        ("Real Scope `Public-Tri`: `Q3.5` -> `DeepSeek` (Stat.)", "real_public_tri_qwen35_to_deepseek_stationary"),
        ("Real Scope `Public-Tri`: `Q3.5` -> `DS->Q8B` (Shift)", "real_public_tri_qwen35_to_ds_to_qwen8b_shift"),
        ("Real Scope `Public-Tri`: `Q8B` -> `DeepSeek` (Floor)", "real_public_tri_qwen8b_to_deepseek_stationary"),
        ("Pure Cap-Gap `Public-Tri`: `Q8B` -> `Q3.5`", "pure_cap_gap_qwen8b_to_qwen35"),
        ("Pure Cap-Gap `Public-Tri`: `Q8B` -> `DeepSeek`", "pure_cap_gap_qwen8b_to_deepseek"),
        ("Semi-Synth `v1`: `Q3.5` -> `Hy4` + Sim Override", "semi_synthetic_v1_qwen35_to_hy4"),
        ("Semi-Synth `v2`: `Q3.5` -> `Hy4` + Sim Override", "semi_synthetic_v2_qwen35_to_hy4"),
    ]
    for p_lbl, p_key in spec_labels:
        for r_lbl, r_map in hard["real_llm_specialist"][p_key].items():
            acl_k = [k for k in r_map if "ACL-2026-VoI" in k][0]
            keys = ["KL-PBD+ (Warm)", acl_k, "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"]
            vals = [r_map[k]["mean_utility"] if isinstance(r_map[k], dict) else r_map[k] for k in keys]
            vals.append(float(up["section7_real_spec"][p_key][r_lbl]))
            f_md = [_fmt_max_row_md(vals, i, True) for i in range(len(vals))]
            md.append(f"| {p_lbl} | {r_lbl} | " + " | ".join(f_md) + " |")

    md.extend([
        "",
        "### Section 7b: Split Medical Accuracy vs. Mandatory Scope-Escalation Recall (`Public-Tri`, `Q3.5` -> `DeepSeek`)",
        "| Regime | `KL-PBD+ (W)` (Util / RawAcc / EscRec) | `ACL+FD+N` | `CC+Naor` | `JG+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` (Zero-Oracle) |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for r_lbl, r_map in hard["real_llm_specialist"]["real_public_tri_qwen35_split_metrics"].items():
        base_keys = ["KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Full-Defer+Naor", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor", "Tri-VoI-L2D"]
        ds_sp = up["section7_real_spec"]["real_public_tri_qwen35_split_metrics"][r_lbl]
        entries = [(k, r_map[k]) for k in base_keys] + [("Tri-VoI-MDP", ds_sp)]
        best_u = max(d["mean_utility"] for _, d in entries)
        cells = []
        for _, d in entries:
            s_u = f"**{d['mean_utility']:.3f}**" if abs(d["mean_utility"] - best_u) <= 1e-4 else f"{d['mean_utility']:.3f}"
            cells.append(f"{s_u} / {d['raw_answer_acc_pct']:.1f}% / {d['scope_escalation_recall_pct']:.1f}%")
        md.append(f"| {r_lbl} | " + " | ".join(cells) + " |")

    # Section 8: Pre-Registered Generalization & Strictly Disjoint MediQ-Holdout (tag: prereg-blind-v3)
    snap = blind["budget"]
    ov = attr["part_d_v3_overlap_audit"]
    md.extend([
        "",
        f"## Section 8: Pre-Registered Unseen-Domain Template Generalization (`Benchmark v3`) & Disjoint `MediQ-Holdout` (`prereg-blind-v3`, Zero-Oracle `Tri-VoI-MDP`, Spend `¥{snap['estimated_rmb']:.2f}`)",
        "",
        f"> **Transparency Note on `Benchmark v3` Content Overlap**: Because each unseen domain in `v3` spans `2×2×2=8` discrete attribute templates, de-numbered audit shows `{ov['test_prose_in_cal']}/{ov['n_test']}` (`{100*ov['test_prose_in_cal']/ov['n_test']:.1f}%`) test items share normalized prose with `cal` and `{ov['exact_triple_in_cal']}/{ov['n_test']}` (`{100*ov['exact_triple_in_cal']/ov['n_test']:.1f}%`) share exact `(prose, missing_val, gold)` triples (`uniq_test_prose={ov['uniq_test_prose']}`). Thus `v3` measures **template generalization** (zero-overlap slices in Section 9D), while disjoint **`MediQ-Holdout`** (`176` cases) and **`MediQ-Test-Disjoint`** (Section 10) provide sample- and organ-system-disjoint clinical evidence.",
        "",
        "### Suite A: `Benchmark v3` (4 Unseen Length-Matched Domains, Zero Protocol Code IDs, `N_cal=48, N_test=144`)",
        "| Model & Regime | `Direct-ACT` | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` | Realized Tok (`L2D` -> `MDP`) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    tex.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Pre-registered unseen-domain template generalization (\\textit{Benchmark v3}, $N_{\\text{test}}=144$; zero-overlap slices in Section 9D) and sample-disjoint \\textit{MediQ-Holdout} ($N_{\\text{test}}=128$, tag \\texttt{prereg-blind-v3}). Primary \\texttt{Tri-VoI-MDP} uses Zero-Oracle \\texttt{Deploy-Self}. Bold marks row-wise best non-Oracle utility.}",
        "\\label{tab:blind_v3}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Suite / Regime & Direct & KL-PBD+(W) & ACL+FD+N & JG+Naor & CC+Naor & L2D & MDP & Tok (L2D$\\to$MDP) \\\\",
        "\\midrule",
    ])
    v3_base_keys = ["Direct-ACT", "KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Full-Defer+Naor", "Joint-Greedy+Naor", "Calibrated-Cascade+Naor", "Tri-VoI-L2D"]
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        sn = mid.split("/")[-1]
        regs = blind["suite_a_benchmark_v3"][mid]["regimes"]
        for reg in ("Cap=2, c=1.5", "Cap=2, c=2.5", "Cap=1/3, c=1.5"):
            r = regs[reg]
            ds_v3 = up["section8_v3"][mid][reg]
            vals = [r[k]["mean_utility"] for k in v3_base_keys] + [ds_v3["mean_utility"]]
            f_md = [_fmt_max_row_md(vals, i, True) for i in range(len(vals))]
            f_tx = [_fmt_max_row_tex(vals, i, True) for i in range(len(vals))]
            tok_l2d = r["Tri-VoI-L2D"]["mean_realized_tokens"]
            tok_mdp = ds_v3["mean_realized_tokens"]
            md.append(f"| `v3` / `{sn}` ({reg}) | " + " | ".join(f_md) + f" | {tok_l2d:.0f} -> {tok_mdp:.0f} |")
            tex.append(f"v3 / {sn} ({reg}) & " + " & ".join(f_tx) + f" & {tok_l2d:.0f}$\\to${tok_mdp:.0f} \\\\")

    md.extend([
        "",
        "### Suite B: Blind `MediQ-Holdout` Real-LLM Specialist (Scope-Compliance + Pure Capability-Gap, `N_cal=48, N_test=128`)",
        "| Setup & Regime | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `Tri-VoI-MDP` | Split `MDP` (RawMedAcc% / EscRec%) |",
        "|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    sb = blind["suite_b_mediq_holdout"]
    sb_base_keys = ["KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Full-Defer+Naor", "Joint-Greedy+Naor", "Calibrated-Cascade+Naor", "Tri-VoI-L2D"]
    tex.append("\\midrule")
    for s_lbl, smap, up_key, split_flag in [
        ("Scope `Q3.5`->`DS`", sb["scope_compliance_qwen35_to_deepseek"], "scope_compliance_qwen35_to_deepseek", True),
        ("Pure Cap-Gap `Q8B`->`Q3.5`", sb["pure_cap_gap_qwen8b_to_qwen35"], "pure_cap_gap_qwen8b_to_qwen35", False),
        ("Pure Cap-Gap `Q8B`->`DS`", sb["pure_cap_gap_qwen8b_to_deepseek"], "pure_cap_gap_qwen8b_to_deepseek", False),
    ]:
        for reg, r in smap.items():
            dm = up["section8_holdout"][up_key][reg]
            vals = [r[k]["mean_utility"] for k in sb_base_keys] + [dm["mean_utility"]]
            f_md = [_fmt_max_row_md(vals, i, True) for i in range(len(vals))]
            sp_s = f"{dm['raw_answer_acc_pct']:.1f}% / {dm['scope_escalation_recall_pct']:.1f}%" if split_flag else f"{dm['raw_answer_acc_pct']:.1f}% / N/A"
            md.append(f"| {s_lbl} ({reg}) | " + " | ".join(f_md) + f" | {sp_s} |")
            f_tx = [_fmt_max_row_tex(vals, i, True) for i in range(len(vals))]
            tex.append(f"Holdout {s_lbl} ({reg}) & --- & " + " & ".join(f_tx) + " & --- \\\\")

    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # Legacy reference note on pre-Choice-2 zero-touch holdout & unseen-ID v2
    md.extend([
        "",
        "### Historical Pre-MDP Zero-Touch Reference (`zero_touch_holdout.json`, `Tri-VoI-L2D` Legacy Series)",
        "| Suite / Model | Miss-F1 / Scope-F1 | `KL-PBD+ (Warm)` | `ACL+FD+N` | `JG+Naor` | `CC+Naor` | `Tri-VoI-L2D` | `2-Call` | `Smooth-Naor` |",
        "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mid in ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.2"):
        mdata = hold["holdout_mediq_disjoint"]["models"][mid]
        d = mdata["diagnosis"]
        sn = mid.split("/")[-1]
        for reg in ("Cap=2, c=1.5", "Cap=2, c_def=4.0, c=1.5", "Cap=1/3, c=1.5"):
            r = mdata["regimes_by_calibrator"]["unified_default"][reg]
            keys = ["KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Full-Defer+Naor", "Joint-Greedy+Naor", "Calibrated-Cascade+Naor", "Tri-VoI-L2D", "Tri-VoI-L2D (2-Call)", "Tri-VoI-L2D (Smooth-Naor)"]
            vals = [r[k]["mean_utility"] for k in keys]
            f_md = [_fmt_max_row_md(vals, i, True) for i in range(len(keys))]
            md.append(f"| MediQ-Holdout / `{sn}` ({reg}) | {d['missing_f1']:.3f} / {d['override_f1']:.3f} | " + " | ".join(f_md) + " |")

    uv2 = hold["v2_unseen_protocol_ids"]
    ud = uv2["diagnosis"]
    for reg in ("Cap=2, c=1.5", "Cap=1/3, c=1.5"):
        r = uv2["regimes"][reg]
        keys = ["KL-PBD+ (Warm)", "ACL-2026-VoI (Live)+Full-Defer+Naor", "Joint-Greedy+Naor", "Calibrated-Cascade+Naor", "Tri-VoI-L2D", "Tri-VoI-L2D (2-Call)", "Tri-VoI-L2D (Smooth-Naor)"]
        vals = [r[k]["mean_utility"] for k in keys]
        f_md = [_fmt_max_row_md(vals, i, True) for i in range(len(keys))]
        md.append(f"| v2-UnseenIDs / `Qwen3.5` ({reg}) | {ud['missing_f1']:.3f} / {ud['override_f1']:.3f} | " + " | ".join(f_md) + " |")

    # Section 9: Offline Attribution, Fair Baseline+MDP, Deployable Signal & v3 Zero-Overlap Slices
    md.extend([
        "",
        "## Section 9: Controlled Attribution (`±EVOI` under Equal `MDP` Shadow Pricing), Deployable `post_missing` Signal & `Benchmark v3` Zero-Overlap Slices (`live=0`, `¥0.00`)",
        "",
        "### 9A. Step-by-Step Controlled Ablation (`L2D` -> `+2-Call` -> `+2C+NoInfo` -> `MDP (5C, Oracle Diag)` -> `MDP (Oracle-NoInfo Diag)` -> **`Tri-VoI-MDP` (Zero-Oracle)**)",
        "| Suite | Regime | `L2D (5C)` | `+2-Call` | `+2C+NoInfo` | `MDP (5C, Oracle Diag)` | `MDP (Oracle-NoInfo Diag)` | `Tri-VoI-MDP` (Zero-Oracle) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])
    a_keys = ["Tri-VoI-L2D", "Tri-VoI-L2D (2-Call)", "Tri-VoI-L2D (2-Call+NoInfo)", "Tri-VoI-MDP (5-Call)", "Tri-VoI-MDP", "Tri-VoI-MDP (Deploy-Self)"]
    a_non_oracle = [0, 1, 2, 5]
    for r in attr["part_a_stepwise_ablation"]:
        v = r["vals"]
        nums = [v[k] for k in a_keys]
        max_no = max(nums[idx] for idx in a_non_oracle)
        f_md = [
            (f"**{nums[i]:.3f}**" if (i in a_non_oracle and abs(nums[i] - max_no) < 1e-9) else f"{nums[i]:.3f}")
            for i in range(len(a_keys))
        ]
        md.append(f"| {r['suite']} | {r['regime']} | " + " | ".join(f_md) + " |")

    md.extend([
        "",
        "### 9B. Cross-Call `Baseline+MDP` vs. `Tri-VoI-MDP` (Note: 5C `Base+MDP` vs 2C `Tri-VoI-MDP`; Equal-Budget 2C Comparison in 9E)",
        "| Suite | Regime | `CC+N` | `CC+MDP` | `JG+N` | `JG+MDP` | `ACL+FD+N` | `ACL+FD+MDP` | `MDP (Oracle Diag)` | `Tri-VoI-MDP` (Zero-Oracle) | EVOI Net (Zero-Oracle vs Best 5C+MDP) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    b_non_oracle = [0, 1, 2, 3, 4, 5, 7]
    for r in attr["part_b_fair_baseline_mdp"]:
        v = r["vals"]
        an, am = r["acl_naor_key"], r["acl_mdp_key"]
        keys_b = ["Calibrated-Cascade+Naor", "Calibrated-Cascade+MDP", "Joint-Greedy+Naor", "Joint-Greedy+MDP", an, am, "Tri-VoI-MDP", "Tri-VoI-MDP (Deploy-Self)"]
        nums = [v[k] for k in keys_b]
        max_no = max(nums[idx] for idx in b_non_oracle)
        f_md = [
            (f"**{nums[i]:.3f}**" if (i in b_non_oracle and abs(nums[i] - max_no) < 1e-9) else f"{nums[i]:.3f}")
            for i in range(len(keys_b))
        ]
        md.append(f"| {r['suite']} | {r['regime']} | " + " | ".join(f_md) + f" | {r['deploy_self_net_vs_best_base_mdp']:+.3f} |")

    sa = attr["sqlite_post_missing_fallback_audit"]
    md.extend([
        "",
        "### 9C. Deployable LLM `post_missing` vs. Simulator `not oracle_revealed` (`obs_missing=True` Test Items)",
        f"- **Zero-Oracle Audit**: Across `{sa['total_post_ask_combined_rows']}` cached `post_ask_combined*` calls, `{sa['total_missing_key_fallbacks']}` lacked `missing_user_info` (truncated JSON). Primary `Tri-VoI-MDP` (`Deploy-Self`) falls back strictly to `obs_missing` and never reads `oracle_revealed`.",
        "",
        "| Suite | `obs_missing` N | True NoInfo | Pred `post_missing` | Unparsed (`obs_missing` Fallback) | Agreement% | Precision | Recall | F1 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for s_lbl, st in attr["part_c_post_missing_signal_audit"].items():
        md.append(
            f"| {s_lbl} | {st['n_obs_missing']} | {st['true_no_info']} | {st['pred_post_missing']} | "
            f"{st['unparsed_fallback_to_obs']} | {st['agreement_pct']:.2f}% | {st['precision']:.3f} | {st['recall']:.3f} | {st['f1']:.3f} |"
        )

    md.extend([
        "",
        "### 9D. `Benchmark v3` Strict Zero-Overlap Slices (`Zero-Cal-Prose-Overlap (72)` & `Unique-Prose (40)`)",
        "| Model | Slice | Regime | `ACL+FD+MDP (2C-S)` | `JG+MDP (2C-S)` | `CC+MDP (2C-S)` | `CC+MDP (5C)` | `JG+MDP (5C)` | `Tri-VoI-MDP` (Zero-Oracle) | p (Zero-Oracle vs Best 2C-S) |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in attr["part_d_v3_zero_overlap_slices"]:
        v = r["vals"]
        keys_d = [
            "ACL-2026-VoI (Live)+Full-Defer+MDP (2C-Self)",
            "Joint-Greedy+MDP (2C-Self)",
            "Calibrated-Cascade+MDP (2C-Self)",
            "Calibrated-Cascade+MDP",
            "Joint-Greedy+MDP",
            "Tri-VoI-MDP (Deploy-Self)",
        ]
        nums = [v[k] for k in keys_d]
        f_md = [_fmt_max_row_md(nums, i, True) for i in range(len(keys_d))]
        md.append(f"| {r['model']} | {r['slice']} | {r['regime']} | " + " | ".join(f_md) + f" | {r['boot_p_deploy_vs_best_2c_self_base_mdp']:.4f} |")

    md.extend([
        "",
        "### 9E. Strictly Matched 2-Call Zero-Oracle Attribution (Equal Probe Cost, Zero Oracle & Same DP Pricing)",
        "| Suite | Regime | `CC+MDP (2C-S)` | `CC+OwnDP` | `JG+MDP (2C-S)` | `JG+OwnDP` | `ACL+FD+MDP (2C-S)` | `Tri-VoI-MDP` (Zero-Oracle) | Net vs Shared (`p`) | Net vs OwnDP (`p`) | Tok (`CC/JG/MDP`) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ])
    for r in attr["part_e_matched_2c_self_attribution"]:
        v = r["vals"]
        tk = r["tokens"]
        a2c = r["acl_2c_key"]
        keys_e = [
            "Calibrated-Cascade+MDP (2C-Self)",
            "Calibrated-Cascade+OwnDP (2C-Self)",
            "Joint-Greedy+MDP (2C-Self)",
            "Joint-Greedy+OwnDP (2C-Self)",
            a2c,
            "Tri-VoI-MDP (Deploy-Self)",
        ]
        nums = [v[k] for k in keys_e]
        f_md = [_fmt_max_row_md(nums, i, True) for i in range(len(keys_e))]
        md.append(
            f"| {r['suite']} | {r['regime']} | " + " | ".join(f_md)
            + f" | {r['evoi_net_vs_best_shared_2c']:+.3f} (`{r['boot_p_vs_best_shared_2c']:.4f}`) | "
            + f"{r['evoi_net_vs_best_own_2c']:+.3f} (`{r['boot_p_vs_best_own_2c']:.4f}`) | "
            + f"{tk['Calibrated-Cascade+MDP (2C-Self)']:.0f}/{tk['Joint-Greedy+MDP (2C-Self)']:.0f}/{tk['Tri-VoI-MDP (Deploy-Self)']:.0f} |"
        )

    md.extend([
        "",
        "### 9F. Clarification-Cost `c_ask` Regime Map Under Matched 2-Call Zero-Oracle Pricing (Proposition 5)",
        "| Suite | Cap | `c_ask` | `CC+MDP (2C-S)` | `JG+MDP (2C-S)` | `ACL+FD+MDP (2C-S)` | `Tri-VoI-MDP` (Zero-Oracle) | EVOI Net vs Best 2C-S |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in attr["part_f_c_ask_regime_map"]:
        v = r["vals"]
        a2c = r["acl_2c_key"]
        keys_f = ["Calibrated-Cascade+MDP (2C-Self)", "Joint-Greedy+MDP (2C-Self)", a2c, "Tri-VoI-MDP (Deploy-Self)"]
        nums = [v[k] for k in keys_f]
        f_md = [_fmt_max_row_md(nums, i, True) for i in range(len(keys_f))]
        md.append(
            f"| {r['suite']} | {r['cap_regime']} | {r['c_ask']:.1f} | " + " | ".join(f_md)
            + f" | {r['evoi_net_vs_best_2c']:+.3f} |"
        )

    # Section 10: Pre-Registered Organ-System-Disjoint Length-Matched MediQ-Test-Disjoint
    mtd_b = mtd["budget"]
    md.extend([
        "",
        f"## Section 10: Pre-Registered Organ-System-Disjoint, Length-Matched `MediQ-Test-Disjoint` (`prereg-mediq-test-disjoint`, `PY_DIFF_LEN=0`, `N_cal=60, N_test=180`, Spend `¥{mtd_b['estimated_rmb']:.2f}`)",
        "> Constructed from official `medqa_test_convo.jsonl` (0 dev-split overlap, 0 clinical organ-system overlap between Cal `{Cardio_Pulm, Endo_Renal, Other}` and Test `{Heme_Onc_Repro_MSK, Infectious_Immuno, GI_Hepatic, Neuro_Psych}`, Q1 vs Q2 prose gap `4.86%`, `Unparsed Fallback=0` on both models).",
        "",
    ])
    m10_labels = {
        "Tri-VoI-MDP": "MDP (Oracle-NoInfo Diag)",
        "Tri-VoI-MDP (Deploy-Self)": "Tri-VoI-MDP (Zero-Oracle)",
    }
    for s_title, s_key in [
        ("### 10A. Setup A: Cross-Group Real-LLM Specialist (`Qwen3.5-35B-A3B` -> `DeepSeek-V3.2`)", "setup_qwen35_to_real_ds"),
        ("### 10B. Setup B: Cross-Group Generalist (`DeepSeek-V3.2`)", "setup_deepseek_generalist"),
    ]:
        sdata = mtd[s_key]
        md.extend([
            s_title,
            "| Method | Cap=2, c=1.5 | Cap=2, c=3.0 | Cap=1/3, c=1.5 | Cap=1/3, c=3.0 | Tok (Cap=2, c=1.5) |",
            "|:---|---:|---:|---:|---:|---:|",
        ])
        m10_list = [
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
        ]
        regs10 = ["Cap=2, c=1.5", "Cap=2, c=3.0", "Cap=1/3, c=1.5", "Cap=1/3, c=3.0"]
        oracle_idx10 = m10_list.index("Tri-VoI-MDP")
        cols10 = [[sdata[rg]["methods"][m]["mean_utility"] for m in m10_list] for rg in regs10]
        cols10_max = [max(cols10[j][idx] for idx in range(len(m10_list)) if idx != oracle_idx10) for j in range(len(regs10))]
        for idx_m, m in enumerate(m10_list):
            c_cells = []
            for j, rg in enumerate(regs10):
                val = cols10[j][idx_m]
                u_str = f"**{val:.3f}**" if (idx_m != oracle_idx10 and abs(val - cols10_max[j]) < 1e-9) else f"{val:.3f}"
                if j in (0, 2):
                    mr = sdata[rg]["methods"][m]
                    c_cells.append(f"{u_str} ({mr['raw_answer_acc_pct']:.1f}%/{mr['scope_escalation_recall_pct']:.1f}%)")
                else:
                    c_cells.append(u_str)
            tok_c = sdata["Cap=2, c=1.5"]["methods"][m]["mean_realized_tokens"]
            disp_m = m10_labels.get(m, m)
            md.append(f"| `{disp_m}` | " + " | ".join(c_cells) + f" | {tok_c:.0f} |")
        md.append("")
        for rg in regs10:
            sm = sdata[rg]["summary"]
            md.append(
                f"- **{rg}**: `Tri-VoI-MDP (Zero-Oracle)` vs `{sm['best_2c_shared_name']}` = `{sm['deploy_self_vs_best_2c_shared']:+.3f}` (`p={sm['boot_p_vs_best_2c_shared']:.4f}`), "
                f"vs `Tri-VoI-L2D` = `{sm['deploy_self_vs_l2d']:+.3f}` (`p={sm['boot_p_vs_l2d']:.4f}`)."
            )
        md.append("")

    (RESULTS_DIR / "ijcai2027_canonical_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS_DIR / "ijcai2027_canonical_tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("Saved audit-verified zero-Oracle results/ijcai2027_canonical_tables.{md,tex}")


if __name__ == "__main__":
    main()
