import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark import (
    build_cscw_benchmark,
    build_cscw_expansion_batch,
    split_stratified_episodes,
)
from controller import HeldOutCalibrator, probe_dataset, simulate_stream
from run_all import block_paired_bootstrap, holm_bonferroni


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

MODELS_5 = [
    "Qwen/Qwen3.5-35B-A3B",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.5-Air",
    "Qwen/Qwen3-8B",
    "tencent/Hy4-preview",
]

METHODS_V1_EXT = [
    "Direct-ACT",
    "Congested-L2D-Only",
    "KL-PBD",
    "KL-PBD+",
    "KL-PBD+ (Warm)",
    "ACL-2026-VoI (Calib)",
    "ACL-2026-VoI+Post-Defer",
    "Joint-Greedy+Naor",
    "Calibrated-Cascade+Naor",
    "Tri-VoI-L2D",
]


def eval_regime(
    cal_p: List[Dict[str, Any]],
    test_p: List[Dict[str, Any]],
    calib: HeldOutCalibrator,
    *,
    cap: int,
    di: int,
    c_ask: float,
    unclarified_spec_penalty: bool = False,
    delayed_feedback: bool = False,
) -> Dict[str, Any]:
    raw: Dict[str, List[Dict[str, Any]]] = {m: [] for m in METHODS_V1_EXT}
    for s in range(5):
        for m in METHODS_V1_EXT:
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
                    delayed_feedback=delayed_feedback,
                    unclarified_spec_penalty=unclarified_spec_penalty,
                    cal_probed=cal_p,
                )
            )
    ours_vec = np.mean([r["per_item_utils"] for r in raw["Tri-VoI-L2D"]], axis=0)
    boot_ps: Dict[str, float] = {}
    out_rows: Dict[str, Any] = {}
    for m in METHODS_V1_EXT:
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
            boot_ps[m] = float(bp["boot_p"])
        out_rows[m] = row
    holm = holm_bonferroni(boot_ps)
    for m, pv in holm.items():
        out_rows[m]["boot_p_holm"] = pv
    return out_rows


def eval_glm_shrinkage(glm_all: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_q = defaultdict(list)
    for r in glm_all:
        by_q[r["item"]["quadrant"]].append(r)

    ablation_grid = {}
    for k_val in (2.0, 5.0, 10.0):
        k_rows = []
        for n_cal in (60, 120, 180):
            u2_list, usev_list, cas_sev_list = [], [], []
            for rep in range(10):
                rr = random.Random(4200 + rep * 13 + n_cal)
                cal_s, test_s = [], []
                per_q = n_cal // 4
                for qk in sorted(by_q.keys()):
                    bkt = list(by_q[qk])
                    rr.shuffle(bkt)
                    take = per_q if qk in ("Q1_clear_easy", "Q2_underspec_easy", "Q3_clear_hard") else (per_q // 2)
                    cal_s.extend(bkt[:take])
                    test_s.extend(bkt[take:])
                rr.shuffle(test_s)
                eb = HeldOutCalibrator(cal_s, k=k_val, k_fine=k_val)
                u2_list.append(simulate_stream("Tri-VoI-L2D", test_s, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=rep)["mean_utility"])
                usev_list.append(simulate_stream("Tri-VoI-L2D", test_s, eb, capacity=1, drain_interval=3, c_ask=1.5, seed=rep)["mean_utility"])
                cas_sev_list.append(simulate_stream("Calibrated-Cascade+Naor", test_s, eb, capacity=1, drain_interval=3, c_ask=1.5, seed=rep)["mean_utility"])
            k_rows.append({
                "n_cal": len(cal_s),
                "tri_cap2_mean": round(float(np.mean(u2_list)), 3),
                "tri_cap2_std": round(float(np.std(u2_list, ddof=1)), 3),
                "tri_sev_mean": round(float(np.mean(usev_list)), 3),
                "tri_sev_std": round(float(np.std(usev_list, ddof=1)), 3),
                "casc_naor_sev_mean": round(float(np.mean(cas_sev_list)), 3),
            })
        ablation_grid[f"k={int(k_val)}"] = k_rows

    # 5-fold CV over all 420 items
    rng_cv = random.Random(777)
    folds = [[] for _ in range(5)]
    for qk in sorted(by_q.keys()):
        bkt = list(by_q[qk])
        rng_cv.shuffle(bkt)
        for i, r in enumerate(bkt):
            folds[i % 5].append(r)
    cv5 = {}
    for k_val in (2.0, 5.0, 10.0):
        c2, csv, cas_sv = [], [], []
        for fi in range(5):
            test_f = list(folds[fi])
            cal_f = [r for j in range(5) if j != fi for r in folds[j]]
            eb = HeldOutCalibrator(cal_f, k=k_val, k_fine=k_val)
            c2.append(float(np.mean([simulate_stream("Tri-VoI-L2D", test_f, eb, capacity=2, drain_interval=2, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)])))
            csv.append(float(np.mean([simulate_stream("Tri-VoI-L2D", test_f, eb, capacity=1, drain_interval=3, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)])))
            cas_sv.append(float(np.mean([simulate_stream("Calibrated-Cascade+Naor", test_f, eb, capacity=1, drain_interval=3, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)])))
        cv5[f"k={int(k_val)}"] = {
            "tri_cap2": round(float(np.mean(c2)), 3),
            "tri_sev": round(float(np.mean(csv)), 3),
            "casc_naor_sev": round(float(np.mean(cas_sv)), 3),
        }
    return {"n_cal_grid": ablation_grid, "cv5": cv5}


def main() -> None:
    cal_items, test_180 = build_cscw_benchmark(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
    test_ext = build_cscw_expansion_batch(n_per_quad=45, seed=2028, idx_offset=240)
    test_items = test_180 + test_ext

    all_model_res: Dict[str, Any] = {}
    prim_cal, prim_test, prim_calib = None, None, None
    glm_all = None

    for mid in MODELS_5:
        cal_p = probe_dataset(cal_items, mid, dry_run=False, workers=1, label="cal60")
        test_p = probe_dataset(test_items, mid, dry_run=False, workers=1, label="test360")
        calib = HeldOutCalibrator(cal_p)
        if mid == "Qwen/Qwen3.5-35B-A3B":
            prim_cal, prim_test, prim_calib = cal_p, test_p, calib
        if mid == "zai-org/GLM-4.5-Air":
            glm_all = cal_p + test_p

        regimes = {
            "Cap=2, c=1.5": eval_regime(cal_p, test_p, calib, cap=2, di=2, c_ask=1.5),
            "Cap=2, c=2.5": eval_regime(cal_p, test_p, calib, cap=2, di=2, c_ask=2.5),
            "Cap=1/3, c=1.5": eval_regime(cal_p, test_p, calib, cap=1, di=3, c_ask=1.5),
        }
        all_model_res[mid] = regimes

    assert prim_cal is not None and prim_test is not None and prim_calib is not None and glm_all is not None

    # Stage 5f stress suite with U_defer^rework active on primary model
    stage5f_rework = {
        "Delayed Feedback (Cap=2)": eval_regime(prim_cal, prim_test, prim_calib, cap=2, di=2, c_ask=1.5, delayed_feedback=True, unclarified_spec_penalty=False),
        "Delayed Feedback (Cap=1/3)": eval_regime(prim_cal, prim_test, prim_calib, cap=1, di=3, c_ask=1.5, delayed_feedback=True, unclarified_spec_penalty=False),
        "Unclarified Rework (Cap=2)": eval_regime(prim_cal, prim_test, prim_calib, cap=2, di=2, c_ask=1.5, delayed_feedback=False, unclarified_spec_penalty=True),
        "Unclarified Rework (Cap=1/3)": eval_regime(prim_cal, prim_test, prim_calib, cap=1, di=3, c_ask=1.5, delayed_feedback=False, unclarified_spec_penalty=True),
        "Delayed + Rework (Cap=2)": eval_regime(prim_cal, prim_test, prim_calib, cap=2, di=2, c_ask=1.5, delayed_feedback=True, unclarified_spec_penalty=True),
        "Delayed + Rework (Cap=1/3)": eval_regime(prim_cal, prim_test, prim_calib, cap=1, di=3, c_ask=1.5, delayed_feedback=True, unclarified_spec_penalty=True),
    }

    # 6 independent 60-ticket episodes on primary model (c_ask=1.5 and 2.5)
    episodes = split_stratified_episodes(prim_test, n_episodes=6, seed=2029)
    ep_summary = {}
    for c_ask in (1.5, 2.5):
        m_ep = {m: [] for m in METHODS_V1_EXT}
        for ep_i, ep in enumerate(episodes):
            for m in METHODS_V1_EXT:
                us = [simulate_stream(m, ep, prim_calib, capacity=2, drain_interval=2, c_ask=c_ask, seed=ep_i * 10 + s, cal_probed=prim_cal)["mean_utility"] for s in range(3)]
                m_ep[m].append(float(np.mean(us)))
        ep_summary[f"c_ask={c_ask}"] = {
            m: {
                "mean": round(float(np.mean(v)), 3),
                "wins_vs_tri": sum(1 for i in range(6) if m_ep["Tri-VoI-L2D"][i] > v[i] + 1e-9),
            }
            for m, v in m_ep.items()
        }

    glm_ablation = eval_glm_shrinkage(glm_all)

    out = {
        "models": all_model_res,
        "stage5f_rework": stage5f_rework,
        "episodes_primary": ep_summary,
        "glm_shrinkage_ablation": glm_ablation,
    }
    (RESULTS_DIR / "external_baselines_v1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# Offline External Baselines & Ablations on Benchmark v1 (0 API Calls)", ""]
    md.append("## 1. 5-Model External Baseline Comparison (`N=360` Test)")
    for mid, regs in all_model_res.items():
        md.append(f"### `{mid}`")
        md.append("| Method | Cap=2, c=1.5 | Cap=2, c=2.5 | Cap=1/3, c=1.5 |")
        md.append("|:---|---:|---:|---:|")
        for m in METHODS_V1_EXT:
            r1 = regs["Cap=2, c=1.5"][m]
            r2 = regs["Cap=2, c=2.5"][m]
            r3 = regs["Cap=1/3, c=1.5"][m]
            md.append(f"| `{m}` | {r1['mean_utility']:.3f} ({r1['acc_pct']:.1f}%) | {r2['mean_utility']:.3f} ({r2['acc_pct']:.1f}%) | {r3['mean_utility']:.3f} ({r3['acc_pct']:.1f}%) |")
        md.append("")

    md.append("## 2. Stage 5f with Rework-Aware $U_{\\text{defer}}^{\\text{rework}}$ (`Qwen3.5-35B-A3B`)")
    md.append("| Scenario | `Tri-VoI-L2D` | `Calib-Casc+Naor` | `Joint-Greedy+Naor` | `KL-PBD+ (Warm)` | `ACL-2026-VoI+Post` |")
    md.append("|:---|---:|---:|---:|---:|---:|")
    for s_lbl, smap in stage5f_rework.items():
        md.append(
            f"| {s_lbl} | **{smap['Tri-VoI-L2D']['mean_utility']:.3f}** | {smap['Calibrated-Cascade+Naor']['mean_utility']:.3f} | "
            f"{smap['Joint-Greedy+Naor']['mean_utility']:.3f} | {smap['KL-PBD+ (Warm)']['mean_utility']:.3f} | {smap['ACL-2026-VoI+Post-Defer']['mean_utility']:.3f} |"
        )
    md.append("")
    md.append("## 3. GLM-4.5-Air Small-Calibration Shrinkage Ablation (`k` x `n_cal`)")
    md.append("| Prior `k` | `n_cal=60` (Cap=1/3) | `n_cal=120` (Cap=1/3) | `n_cal=180` (Cap=1/3) | 5-Fold CV (Cap=1/3) | `Calib-Casc+Naor` 5-Fold CV |")
    md.append("|:---|---:|---:|---:|---:|---:|")
    for klbl, krows in glm_ablation["n_cal_grid"].items():
        cv = glm_ablation["cv5"][klbl]
        md.append(
            f"| `{klbl}` | {krows[0]['tri_sev_mean']:.3f}±{krows[0]['tri_sev_std']:.3f} | "
            f"{krows[1]['tri_sev_mean']:.3f}±{krows[1]['tri_sev_std']:.3f} | "
            f"{krows[2]['tri_sev_mean']:.3f}±{krows[2]['tri_sev_std']:.3f} | "
            f"**{cv['tri_sev']:.3f}** | {cv['casc_naor_sev']:.3f} |"
        )

    (RESULTS_DIR / "external_baselines_v1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Saved results/external_baselines_v1.{json,md}")


if __name__ == "__main__":
    main()
