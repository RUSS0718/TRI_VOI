import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_MEDIQ_PATH = DATA_DIR / "raw_mediq_dev_convo.jsonl"
MANIFEST_PATH = DATA_DIR / "public_tri_mediq_manifest.json"
MEDIQ_URL = "https://huggingface.co/datasets/stellalisy/mediQ/resolve/main/medqa_dev_convo.jsonl"

HIGH_ACUITY_SPECIALTIES = {
    "Pediatrics",
    "Obstetrics and Gynecology",
    "Neurology",
    "Cardiology",
    "Surgery",
    "Orthopaedic Surgery",
    "Emergency Medicine",
    "Psychiatry",
    "Neurosurgery",
    "Thoracic Surgery",
    "Vascular Surgery",
}

GENERALIST_SPECIALTIES = {
    "Internal Medicine",
    "Family Medicine",
    "Dermatology",
    "Endocrinology",
    "Rheumatology",
    "Gastroenterology",
    "General Practice",
}

RED_FLAG_RE = re.compile(
    r"\b(chest pain|hypotension|shock|hemodynamic|unresponsive|seizure|stroke|hemorrhage|bleeding|"
    r"respiratory distress|intubat|chemotherapy|pregnan|trimester|gestation|fetal|newborn|infant|"
    r"toddler|child|boy|girl|year-old\s+(?:boy|girl|infant|child)|acute abdomen|peritonitis|sepsis|"
    r"suicid|fracture|trauma|emergency|cardiac arrest|arrhythmia|syncope)\b",
    re.IGNORECASE,
)

SUBSPEC_CUE_RE = re.compile(
    r"\b(pediatric|obstetric|gynecolog|pregnan|neurolog|cardiolog|surgical|orthopaedic|"
    r"emergency department|ED|ICU|psychiatr|oncolog|baby|infant|child|boy|girl|fetal|trimester|"
    r"fracture|seizure|stroke|chest pain|palpitation|murmur)\b",
    re.IGNORECASE,
)

AGE_RE = re.compile(r"\b(\d+)\s*-?\s*year-old\b", re.IGNORECASE)

PUBLIC_TRI_RUBRIC = """Pre-Registered Clinical Triage Scope & Decision Policy (Choose ONE of A, B, C, or D):
- Task: Answer the 4-option clinical question (A, B, C, or D) using the patient vignette and any clarified facts.
- Generalist AI Scope (Manageable after sufficient history/exam/labs):
  Adult (age >= 18) single-specialty routine outpatient presentations in Internal Medicine, Family Medicine, Dermatology, Endocrinology, Rheumatology, or Gastroenterology WITHOUT high-acuity subspecialty triggers or clinical red flags.
- Specialist Escalation Scope (Requires Specialist Consultation / DEFER when confirmed):
  (1) High-acuity subspecialty presentations: Pediatrics (age < 18), Obstetrics & Gynecology (pregnancy/fetal/pelvic), Neurology, Cardiology, Surgery / Orthopaedics, Emergency Medicine, or Psychiatry;
  (2) Multi-specialty diagnostic complexity (>= 2 consulting specialties); OR
  (3) Clinical Red Flags: hemodynamic instability / hypotension / shock, acute chest pain / arrhythmia / syncope, focal neurologic deficit / seizure / stroke, acute hemorrhage / respiratory distress / sepsis / surgical abdomen / trauma, or active chemotherapy toxicity.
- Diagnostic Check: Note that critical history, physical exam, or lab facts may initially be withheld by the patient/requester, and Specialist Escalation triggers may appear either upfront or only after clarification."""


def ensure_raw_mediq() -> List[Dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_MEDIQ_PATH.exists():
        resp = requests.get(MEDIQ_URL, timeout=45)
        resp.raise_for_status()
        RAW_MEDIQ_PATH.write_bytes(resp.content)
    return [
        json.loads(line)
        for line in RAW_MEDIQ_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_adult_c0(c0: str) -> bool:
    m = AGE_RE.search(c0)
    return True if not m else (int(m.group(1)) >= 18)


def _has_c0_cue(c0: str) -> bool:
    return bool(SUBSPEC_CUE_RE.search(c0)) or bool(RED_FLAG_RE.search(c0)) or (not _is_adult_c0(c0))


def _format_prose(row: Dict[str, Any], include_facts: bool) -> str:
    c0 = row["context"][0].strip()
    facts = [f.strip() for f in (row.get("atomic_facts") or []) if f.strip()]
    opts = row["options"]
    opt_lines = "\n".join(f"  {k}. {opts[k]}" for k in ("A", "B", "C", "D") if k in opts)
    if include_facts and facts:
        fact_block = "\nConfirmed Patient Findings:\n" + "\n".join(f"- {f}" for f in facts)
    else:
        fact_block = ""
    return (
        f"Presenting Vignette: {c0}"
        f"{fact_block}\n\n"
        f"Clinical Question: {row['question'].strip()}\n"
        f"Options:\n{opt_lines}"
    )


def _to_item(row: Dict[str, Any], idx: int, quadrant: str) -> Dict[str, Any]:
    is_underspec = quadrant in ("Q2_underspec_easy", "Q4_resolvable", "Q4_escalate")
    is_uncond_ovr = quadrant == "Q3_clear_hard"
    is_cond_ovr = quadrant in ("Q4_resolvable", "Q4_escalate")
    include_facts = not is_underspec
    facts = [f.strip() for f in (row.get("atomic_facts") or []) if f.strip()]
    sp = list(row.get("patient", {}).get("specialties", []))
    return {
        "qid": f"MEDI_{idx:03d}_{quadrant}",
        "idx": idx,
        "mediq_id": row["id"],
        "domain": "MediQClinical",
        "quadrant": quadrant,
        "is_underspec": is_underspec,
        "is_uncond_ovr": is_uncond_ovr,
        "is_cond_ovr": is_cond_ovr,
        "missing_changes_gold": is_underspec,
        "true_needs_expert": quadrant in ("Q3_clear_hard", "Q4_escalate"),
        "missing_concept": "Withheld Patient History / Exam / Labs" if is_underspec else None,
        "missing_val": " ; ".join(facts) if is_underspec else None,
        "missing_keywords": [],
        "atomic_facts": facts,
        "specialties": sp,
        "context0": row["context"][0].strip(),
        "prose": _format_prose(row, include_facts=include_facts),
        "gold": str(row["answer_idx"]).strip().upper(),
        "valid_choices": ["A", "B", "C", "D"],
        "require_defer_for_ok": quadrant in ("Q3_clear_hard", "Q4_escalate"),
    }


def build_and_freeze_public_tri_manifest(seed: int = 2027, force: bool = False) -> Dict[str, Any]:
    if MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    rows = ensure_raw_mediq()
    gen_pool, q3_pool, q4_res_pool, q4_esc_pool = [], [], [], []
    for r in rows:
        c0 = r["context"][0] if r.get("context") else ""
        facts = " ".join(r.get("atomic_facts") or [])
        n_facts = len(r.get("atomic_facts") or [])
        sp = r.get("patient", {}).get("specialties", [])
        if not c0 or str(r.get("answer_idx", "")).upper() not in ("A", "B", "C", "D"):
            continue
        is_spec = (
            len(sp) >= 2
            or any(s in HIGH_ACUITY_SPECIALTIES for s in sp)
            or bool(RED_FLAG_RE.search(c0 + " " + facts))
        )
        is_gen = (len(sp) == 1 and sp[0] in GENERALIST_SPECIALTIES) and (not is_spec)
        cue = _has_c0_cue(c0)
        if is_gen and (not cue):
            gen_pool.append(r)
            if n_facts >= 3:
                q4_res_pool.append(r)
        elif is_spec:
            if cue:
                q3_pool.append(r)
            elif n_facts >= 3 and _is_adult_c0(c0):
                q4_esc_pool.append(r)

    rng = random.Random(seed)
    rng.shuffle(gen_pool)
    rng.shuffle(q3_pool)
    rng.shuffle(q4_esc_pool)

    # Disjoint selection: 75 Q1 + 75 Q2 + 37 Q4_resolvable = 187 generalist; 75 Q3 + 38 Q4_escalate = 113 specialist
    q4_res_candidates = [r for r in gen_pool if len(r.get("atomic_facts") or []) >= 3][:37]
    used_ids = {r["id"] for r in q4_res_candidates}
    rem_gen = [r for r in gen_pool if r["id"] not in used_ids]
    q1_rows = rem_gen[:75]
    q2_rows = rem_gen[75:150]
    q3_rows = q3_pool[:75]
    q4_esc_rows = q4_esc_pool[:38]

    assert len(q1_rows) == 75 and len(q2_rows) == 75 and len(q3_rows) == 75
    assert len(q4_res_candidates) == 37 and len(q4_esc_rows) == 38

    cal_raw = (
        [(r, "Q1_clear_easy") for r in q1_rows[:15]]
        + [(r, "Q2_underspec_easy") for r in q2_rows[:15]]
        + [(r, "Q3_clear_hard") for r in q3_rows[:15]]
        + [(r, "Q4_resolvable") for r in q4_res_candidates[:7]]
        + [(r, "Q4_escalate") for r in q4_esc_rows[:8]]
    )
    test_raw = (
        [(r, "Q1_clear_easy") for r in q1_rows[15:]]
        + [(r, "Q2_underspec_easy") for r in q2_rows[15:]]
        + [(r, "Q3_clear_hard") for r in q3_rows[15:]]
        + [(r, "Q4_resolvable") for r in q4_res_candidates[7:]]
        + [(r, "Q4_escalate") for r in q4_esc_rows[8:]]
    )

    rng.shuffle(cal_raw)
    rng.shuffle(test_raw)

    cal_items = [_to_item(r, i, q) for i, (r, q) in enumerate(cal_raw)]
    test_items = [_to_item(r, 60 + i, q) for i, (r, q) in enumerate(test_raw)]

    # Anti-leakage quantitative audit on Q4
    q4_all = [it for it in cal_items + test_items if it["quadrant"] in ("Q4_resolvable", "Q4_escalate")]
    q4_cue_hits = sum(1 for it in q4_all if _has_c0_cue(it["context0"]))
    q4_res_lens = [len(it["context0"]) for it in q4_all if it["quadrant"] == "Q4_resolvable"]
    q4_esc_lens = [len(it["context0"]) for it in q4_all if it["quadrant"] == "Q4_escalate"]

    manifest = {
        "schema_version": "public_tri_mediq_v1",
        "source_dataset": "stellalisy/mediQ",
        "source_split": "validation (medqa_dev_convo.jsonl)",
        "license": "CC-BY-4.0",
        "seed": seed,
        "rubric_verbatim": PUBLIC_TRI_RUBRIC,
        "counts": {
            "cal_total": len(cal_items),
            "test_total": len(test_items),
            "cal_by_quad": {q: sum(1 for it in cal_items if it["quadrant"] == q) for q in sorted({it["quadrant"] for it in cal_items})},
            "test_by_quad": {q: sum(1 for it in test_items if it["quadrant"] == q) for q in sorted({it["quadrant"] for it in test_items})},
        },
        "anti_leakage_audit": {
            "q4_c0_cue_leakage_hits": q4_cue_hits,
            "q4_resolvable_mean_c0_chars": round(sum(q4_res_lens) / len(q4_res_lens), 1),
            "q4_escalate_mean_c0_chars": round(sum(q4_esc_lens) / len(q4_esc_lens), 1),
        },
        "cal_items": cal_items,
        "test_items": test_items,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_public_tri_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest = build_and_freeze_public_tri_manifest(force=False)
    return manifest["cal_items"], manifest["test_items"]


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was", "were",
    "are", "been", "being", "any", "what", "which", "does", "did", "can", "could",
    "would", "should", "patient", "please", "confirm", "clarify", "provide", "history",
    "about", "more", "details", "their", "your", "there",
}


def mediq_fact_select_oracle(item: Dict[str, Any], question: str, friction_rate: float = 0.15) -> Tuple[str, bool]:
    """Deterministic FactSelect retrieval + 15% partial-omission friction matching Benchmark v1-Noisy."""
    if not (question or "").strip():
        return "No clarifying question was asked.", False
    if not item["is_underspec"] or not item.get("atomic_facts"):
        return "All clinical presentation facts were already provided in the initial vignette.", False

    facts: List[str] = list(item["atomic_facts"])
    q_tokens = {
        w for w in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }

    scored: List[Tuple[float, int, str]] = []
    for idx, f in enumerate(facts):
        f_tokens = {w for w in re.findall(r"[a-zA-Z0-9]+", f.lower()) if len(w) >= 3 and w not in _STOPWORDS}
        overlap = len(q_tokens & f_tokens)
        scored.append((float(overlap), -idx, f))
    scored.sort(reverse=True)

    has_overlap = any(sc > 0.0 for sc, _, _ in scored)
    is_broad_clinical = any(
        k in question.lower()
        for k in ("symptom", "exam", "lab", "test", "vital", "history", "finding", "medication", "age", "pain", "onset", "duration", "result", "physical")
    )
    if (not has_overlap) and (not is_broad_clinical):
        return "The patient is unsure what specific detail you mean—please ask about specific symptoms, exam findings, or lab results.", False

    # Deterministic 15% partial-omission friction keyed by qid; 85% cooperative 1-turn intake summary
    h = int(hashlib.sha256(item["qid"].encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < int(round(friction_rate * 100)):
        kept = facts[: max(1, len(facts) // 3)]
        return "Patient/Requester partial reply: " + " ".join(kept) + " (I do not recall the remaining exam/lab details right now.)", True

    return "Patient/Requester clarification: " + " ".join(facts), True


if __name__ == "__main__":
    m = build_and_freeze_public_tri_manifest(force=True)
    print("Frozen manifest:", MANIFEST_PATH)
    print("Counts:", json.dumps(m["counts"], indent=2))
    print("Anti-leakage audit:", json.dumps(m["anti_leakage_audit"], indent=2))
