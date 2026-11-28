import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_V3_PATH = DATA_DIR / "benchmark_v3_blind_manifest.json"

DOMAINS_V3 = ["LegalContract", "BioSafetyLab", "CloudSecOps", "AviationDispatch"]

DOMAIN_RUBRIC_V3 = """Cooperative Triage & Escalation Rubric (Choose exactly ONE of A, B, or C):
1. [LegalContract] Required Intake Fields: (1) Counterparty Entity (Domestic Corp / Foreign Sovereign), (2) Liability Cap Tier (Capped <=1x / Uncapped), (3) Indemnity Scope Confirmed (Standard / Consequential-Damages Included).
   - A (Standard Auto-Execute): Domestic Corp, Liability Cap Capped <=1x, and Indemnity Scope = Standard.
   - B (Senior Associate Review): Foreign Sovereign OR Uncapped Liability Cap, with Indemnity Scope = Standard.
   - C (General Counsel Escalation): Any contract with Indemnity Scope = Consequential-Damages Included.
   - Internal Specialist Advisory Note: Complex sovereign-arbitration waivers or confirmed consequential-damages indemnity exposures require Specialist Counsel sign-off under confidential executive guidelines.

2. [BioSafetyLab] Required Intake Fields: (1) Containment Level (Level 1 Routine / Level 2 Restricted), (2) Culture Volume (Micro-Assay <=50mL / Bulk Culture >50mL), (3) Aerosol-Generation Risk Confirmed (No / Yes).
   - A (Routine Bench Approval): Level 1 Routine, Micro-Assay <=50mL, and Aerosol Risk = No.
   - B (Containment Hood Review): Level 2 Restricted OR Bulk Culture >50mL, with Aerosol Risk = No.
   - C (Negative-Pressure Suite): Any experiment with Aerosol Risk = Yes.
   - Internal Specialist Advisory Note: Dual-use regulatory oversight holds or confirmed aerosol-generating assays require Institutional Biosafety Officer specialist routing overriding standard bench rules.

3. [CloudSecOps] Required Intake Fields: (1) Blast Radius (Single Tenant / Multi-Tenant), (2) Environment (Staging / Production), (3) Active Secret Exposure Confirmed (No / Yes).
   - A (Async Patch Queue): Single Tenant, Staging environment, and Secret Exposure = No.
   - B (On-Call SRE Mitigation): Production OR Multi-Tenant scope, with Secret Exposure = No.
   - C (Emergency Key Rotation & Lockdown): Any incident with Secret Exposure = Yes.
   - Internal Specialist Advisory Note: Regulator audit freezes or confirmed active credential/secret exposure trigger Principal Security Architect escalation overriding standard triage tiers.

4. [AviationDispatch] Required Intake Fields: (1) Runway Surface (Dry / Wet), (2) Sector Length (Short-Haul / Long-Haul), (3) Hydraulic Pressure Warning Confirmed (No / Yes).
   - A (Normal Flight Release): Dry runway, Short-Haul leg, and Hydraulic Warning = No.
   - B (Performance Dispatch Recalc): Wet runway OR Long-Haul leg, with Hydraulic Warning = No.
   - C (Maintenance Hold & Tail Swap): Any departure with Hydraulic Warning = Yes.
   - Internal Specialist Advisory Note: Active airworthiness directive restrictions or confirmed hydraulic pressure warnings trigger Chief Fleet Dispatcher specialist routing overriding normal release rules.
"""

_CODE_ID_RE = re.compile(r"\b[A-Z]{2,5}-\d+\b")


def _build_ticket_v3(
    idx: int,
    quadrant: str,
    rng: random.Random,
) -> Dict[str, Any]:
    domain = DOMAINS_V3[idx % len(DOMAINS_V3)]
    qid = f"V3_{idx:03d}_{quadrant}"

    is_underspec = quadrant in ("Q2_underspec_easy", "Q4_resolvable", "Q4_escalate")
    is_uncond_ovr = (quadrant == "Q3_clear_hard")
    is_cond_ovr = quadrant in ("Q4_resolvable", "Q4_escalate")

    if domain == "LegalContract":
        entity = rng.choice(["Domestic Corp", "Foreign Sovereign"])
        cap_tier = rng.choice(["Capped <=1x", "Uncapped"])
        if quadrant == "Q4_resolvable":
            indemnity = "Standard"
        elif quadrant == "Q4_escalate":
            indemnity = "Consequential-Damages Included"
        else:
            indemnity = rng.choice(["Standard", "Consequential-Damages Included"])

        if indemnity == "Consequential-Damages Included":
            base_ans = "C"
        elif entity == "Foreign Sovereign" or cap_tier == "Uncapped":
            base_ans = "B"
        else:
            base_ans = "A"
        base_default = "B" if (entity == "Foreign Sovereign" or cap_tier == "Uncapped") else "A"

        missing_concept = "Indemnity Scope"
        missing_val = f"Indemnity Scope = {indemnity}"
        missing_keywords = ["indemnity", "consequential", "damages", "liability scope", "clause"]
        tag_id = f"Contract #{3100 + idx}"

        if is_underspec:
            prose = (
                f"Matter Summary ({tag_id}): Counterparty is {entity}; contract specifies {cap_tier} liability ceiling; "
                f"docket arrived via standard regional commercial review desk."
            )
        else:
            ind_phrase = (
                "confirmed indemnity covers consequential damages (Indemnity Scope = Consequential-Damages Included)"
                if indemnity == "Consequential-Damages Included"
                else "confirmed indemnity is strictly capped at direct loss (Indemnity Scope = Standard)"
            )
            prose = (
                f"Matter Summary ({tag_id}): Counterparty is {entity}; contract specifies {cap_tier} liability ceiling; "
                f"{ind_phrase}."
            )

        if is_uncond_ovr:
            prose += (
                " Matter file carries an active sovereign-arbitration waiver notice under confidential executive "
                "counsel guidelines overriding standard sign-off tiers."
            )
            gold = "B" if base_ans != "B" else "A"
        elif is_cond_ovr:
            prose += (
                " Internal Advisory: Specialist Counsel escalation applies when confirmed Indemnity Scope includes "
                "consequential damages; standard-indemnity files follow normal tiering."
            )
            if indemnity == "Consequential-Damages Included":
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    elif domain == "BioSafetyLab":
        level = rng.choice(["Level 1 Routine", "Level 2 Restricted"])
        vol = rng.choice(["Micro-Assay <=50mL", "Bulk Culture >50mL"])
        if quadrant == "Q4_resolvable":
            aerosol = "No"
        elif quadrant == "Q4_escalate":
            aerosol = "Yes"
        else:
            aerosol = rng.choice(["No", "Yes"])

        if aerosol == "Yes":
            base_ans = "C"
        elif level == "Level 2 Restricted" or vol == "Bulk Culture >50mL":
            base_ans = "B"
        else:
            base_ans = "A"
        base_default = "B" if (level == "Level 2 Restricted" or vol == "Bulk Culture >50mL") else "A"

        missing_concept = "Aerosol-Generation Risk"
        missing_val = f"Aerosol Risk = {aerosol}"
        missing_keywords = ["aerosol", "vapor", "mist", "nebul", "generation"]
        tag_id = f"Assay #{5200 + idx}"

        if is_underspec:
            prose = (
                f"Lab Protocol ({tag_id}): Scheduled {level} bench procedure using {vol}; "
                f"bench log registered during morning core facility intake cycle."
            )
        else:
            aer_phrase = (
                "verified sonication creates respirable hazard (Aerosol Risk = Yes)"
                if aerosol == "Yes"
                else "verified closed-tube handling prevents dispersion (Aerosol Risk = No)"
            )
            prose = (
                f"Lab Protocol ({tag_id}): Scheduled {level} bench procedure using {vol}; "
                f"{aer_phrase}."
            )

        if is_uncond_ovr:
            prose += (
                " Sample strain carries a pending dual-use regulatory oversight hold requiring Institutional "
                "Biosafety Officer specialist assignment overriding standard bench rules."
            )
            gold = "B" if base_ans != "B" else "A"
        elif is_cond_ovr:
            prose += (
                " Internal Advisory: Specialist Biosafety Officer containment routing applies when confirmed "
                "aerosol generation is present; non-aerosol work follows normal bench rules."
            )
            if aerosol == "Yes":
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    elif domain == "CloudSecOps":
        blast = rng.choice(["Single Tenant", "Multi-Tenant"])
        env = rng.choice(["Staging", "Production"])
        if quadrant == "Q4_resolvable":
            secret = "No"
        elif quadrant == "Q4_escalate":
            secret = "Yes"
        else:
            secret = rng.choice(["No", "Yes"])

        if secret == "Yes":
            base_ans = "C"
        elif env == "Production" or blast == "Multi-Tenant":
            base_ans = "B"
        else:
            base_ans = "A"
        base_default = "B" if (env == "Production" or blast == "Multi-Tenant") else "A"

        missing_concept = "Active Secret Exposure"
        missing_val = f"Secret Exposure = {secret}"
        missing_keywords = ["secret", "credential", "key", "token", "exposure", "leak"]
        tag_id = f"Security Ticket #{7400 + idx}"

        if is_underspec:
            prose = (
                f"Security Alert ({tag_id}): Alert scoped to {blast} workload running inside {env}; "
                f"event forwarded from automated cluster telemetry pipeline."
            )
        else:
            sec_phrase = (
                "confirmed live signing credentials appeared in egress logs (Secret Exposure = Yes)"
                if secret == "Yes"
                else "confirmed no credentials or API keys left memory scope (Secret Exposure = No)"
            )
            prose = (
                f"Security Alert ({tag_id}): Alert scoped to {blast} workload running inside {env}; "
                f"{sec_phrase}."
            )

        if is_uncond_ovr:
            prose += (
                " Cluster belongs to a regulator audit freeze zone requiring Principal Security Architect "
                "specialist escalation overriding standard incident tiers."
            )
            gold = "B" if base_ans != "B" else "A"
        elif is_cond_ovr:
            prose += (
                " Internal Advisory: Principal Security Architect specialist escalation triggers when confirmed "
                "active secret exposure occurs; unexposed alerts follow normal triage."
            )
            if secret == "Yes":
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    else:  # AviationDispatch
        runway = rng.choice(["Dry", "Wet"])
        sector = rng.choice(["Short-Haul", "Long-Haul"])
        if quadrant == "Q4_resolvable":
            hyd = "No"
        elif quadrant == "Q4_escalate":
            hyd = "Yes"
        else:
            hyd = rng.choice(["No", "Yes"])

        if hyd == "Yes":
            base_ans = "C"
        elif runway == "Wet" or sector == "Long-Haul":
            base_ans = "B"
        else:
            base_ans = "A"
        base_default = "B" if (runway == "Wet" or sector == "Long-Haul") else "A"

        missing_concept = "Hydraulic Pressure Warning"
        missing_val = f"Hydraulic Warning = {hyd}"
        missing_keywords = ["hydraulic", "pressure", "warning", "sensor", "line"]
        tag_id = f"Flight #{8800 + idx}"

        if is_underspec:
            prose = (
                f"Dispatch Brief ({tag_id}): Departure planned for {sector} leg on {runway} runway surface; "
                f"flight plan queued at central regional operations desk."
            )
        else:
            hyd_phrase = (
                "telemetry reports green-system pressure drop caution (Hydraulic Warning = Yes)"
                if hyd == "Yes"
                else "preflight sensors show nominal hydraulic pressure (Hydraulic Warning = No)"
            )
            prose = (
                f"Dispatch Brief ({tag_id}): Departure planned for {sector} leg on {runway} runway surface; "
                f"{hyd_phrase}."
            )

        if is_uncond_ovr:
            prose += (
                " Tail frame carries an active airworthiness directive restriction requiring Chief Fleet "
                "Dispatcher specialist release overriding normal flight clearance rules."
            )
            gold = "B" if base_ans != "B" else "A"
        elif is_cond_ovr:
            prose += (
                " Internal Advisory: Chief Fleet Dispatcher specialist routing triggers when confirmed hydraulic "
                "pressure warnings are active; nominal departures follow standard release."
            )
            if hyd == "Yes":
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    return {
        "qid": qid,
        "idx": idx,
        "domain": domain,
        "quadrant": quadrant,
        "is_underspec": is_underspec,
        "is_uncond_ovr": is_uncond_ovr,
        "is_cond_ovr": is_cond_ovr,
        "missing_changes_gold": is_underspec,
        "true_needs_expert": (quadrant in ("Q3_clear_hard", "Q4_escalate")),
        "missing_concept": missing_concept if is_underspec else None,
        "missing_val": missing_val if is_underspec else None,
        "missing_keywords": missing_keywords,
        "prose": f"Domain: {domain}\n{prose}",
        "gold": gold,
    }


def user_oracle_v3(item: Dict[str, Any], question: str) -> Tuple[str, bool]:
    """Return (reply_text, revealed_new_info) for Benchmark v3 unseen domains."""
    if not (question or "").strip():
        return "No clarifying question was asked.", False
    if not item["is_underspec"] or not item["missing_val"]:
        return "All available intake fields were already stated; I do not hold internal specialist advisory decisions.", False
    q_low = question.lower()
    if any(kw in q_low for kw in item["missing_keywords"]):
        return f"Clarification provided: {item['missing_val']}.", True
    return "Please specify which missing intake attribute you need confirmed (indemnity scope, aerosol risk, secret exposure, or hydraulic warning).", False


def build_and_freeze_v3_manifest(
    n_cal_per_quad: int = 12,
    n_test_per_quad: int = 36,
    seed: int = 2029,
    force: bool = False,
) -> Dict[str, Any]:
    """Generate and freeze Benchmark v3 blind manifest (48 cal + 144 test = 192 items)."""
    if MANIFEST_V3_PATH.exists() and not force:
        return json.loads(MANIFEST_V3_PATH.read_text(encoding="utf-8"))

    rng = random.Random(seed)
    cal_items: List[Dict[str, Any]] = []
    test_items: List[Dict[str, Any]] = []
    idx = 0

    def _add_quad(quad: str, n_c: int, n_t: int) -> None:
        nonlocal idx
        buf = []
        for _ in range(n_c + n_t):
            buf.append(_build_ticket_v3(idx, quad, rng))
            idx += 1
        rng.shuffle(buf)
        cal_items.extend(buf[:n_c])
        test_items.extend(buf[n_c:])

    _add_quad("Q1_clear_easy", n_cal_per_quad, n_test_per_quad)
    _add_quad("Q2_underspec_easy", n_cal_per_quad, n_test_per_quad)
    _add_quad("Q3_clear_hard", n_cal_per_quad, n_test_per_quad)
    n_c4 = n_cal_per_quad // 2
    n_t4 = n_test_per_quad // 2
    _add_quad("Q4_resolvable", n_c4, n_t4 + (n_test_per_quad % 2))
    _add_quad("Q4_escalate", n_cal_per_quad - n_c4, n_t4)

    rng.shuffle(cal_items)
    rng.shuffle(test_items)

    all_items = cal_items + test_items
    q1_lens = [len(it["prose"]) for it in all_items if it["quadrant"] == "Q1_clear_easy"]
    q2_lens = [len(it["prose"]) for it in all_items if it["quadrant"] == "Q2_underspec_easy"]
    q1_mean = sum(q1_lens) / len(q1_lens)
    q2_mean = sum(q2_lens) / len(q2_lens)
    code_hits = _CODE_ID_RE.findall(DOMAIN_RUBRIC_V3 + "\n" + "\n".join(it["prose"] for it in all_items))

    assert len(code_hits) == 0, f"Protocol code ID leakage detected in v3: {code_hits}"
    assert abs(q1_mean - q2_mean) < 35.0, f"Length imbalance Q1={q1_mean:.1f} vs Q2={q2_mean:.1f}"

    manifest = {
        "schema_version": "benchmark_v3_blind_zero_codes_v1",
        "domains": DOMAINS_V3,
        "seed": seed,
        "counts": {
            "cal_total": len(cal_items),
            "test_total": len(test_items),
        },
        "anti_leakage_audit": {
            "protocol_code_id_hits": len(code_hits),
            "q1_mean_prose_chars": round(q1_mean, 1),
            "q2_mean_prose_chars": round(q2_mean, 1),
            "abs_q1_q2_char_diff": round(abs(q1_mean - q2_mean), 1),
        },
        "rubric_verbatim": DOMAIN_RUBRIC_V3,
        "cal_items": cal_items,
        "test_items": test_items,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_V3_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_benchmark_v3_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    man = build_and_freeze_v3_manifest(force=False)
    return man["cal_items"], man["test_items"]


if __name__ == "__main__":
    m = build_and_freeze_v3_manifest(force=True)
    print(json.dumps({"counts": m["counts"], "anti_leakage_audit": m["anti_leakage_audit"]}, indent=2))
