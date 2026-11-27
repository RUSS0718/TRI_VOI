import random
from typing import Any, Dict, List, Tuple


DOMAINS = ["ClinicalTriage", "FinCompliance", "ITIncident", "LogisticsSLA"]

DOMAIN_RUBRIC = """Cooperative Decision-Making Policy Rubric (Choose exactly ONE of A, B, or C):
1. [ClinicalTriage] Required Intake Fields: (1) Patient Age Group, (2) Body Temperature, (3) Chest Pain Status (Yes/No).
   - A (Home Self-Care): Adult (18-64), temperature < 38.5C, and Chest Pain = No.
   - B (Urgent Clinic): Pediatric (<18) or Elderly (>=65) or temperature >= 38.5C, with Chest Pain = No.
   - C (Emergency Room): Any patient with Chest Pain = Yes.
   - Internal Advisory Note: If an active specialist protocol (e.g., Protocol Onco-B7, Cardiac-Watch CW-7) applies to the patient's confirmed condition, confidential specialist rules override standard triage.

2. [FinCompliance] Required Intake Fields: (1) Account Risk Tier, (2) Transfer Jurisdiction (Domestic/Cross-Border), (3) Transaction Amount (USD).
   - A (Auto-Approve): Standard Tier, Domestic jurisdiction, Amount <= $10,000.
   - B (Enhanced Review): Standard Tier + Domestic with Amount > $10,000; OR Cross-Border / High-Risk Tier with Amount <= $5,000.
   - C (Freeze & Escalate): Cross-Border / High-Risk Tier with Amount > $5,000.
   - Internal Advisory Note: If an active sanctions/treasury bulletin (e.g., Bulletin CB-19, Sanctions Matrix M-14) applies to the transaction's confirmed jurisdiction/account, confidential compliance rules override standard limits.

3. [ITIncident] Required Intake Fields: (1) Service Scope (Internal Tool vs. Customer-Facing), (2) Outage Severity (Single User / Partial Outage / Full Outage).
   - A (Standard Ticket Queue): Internal Tool affecting a Single User.
   - B (P2 On-Call Response): Customer-Facing Partial Outage, or Internal Tool Partial Outage.
   - C (P1 War Room): Any Full Outage (or Customer-Facing outage with active SLA escalation).
   - Internal Advisory Note: If an active SRE runbook exception (e.g., Runbook LPM-4, Escalation Clause VIP-9) applies to the confirmed service scope, confidential routing rules override standard queues.

4. [LogisticsSLA] Required Intake Fields: (1) Shipping Tier (Standard Freight vs. Express Tier), (2) Hazardous Material Status (Hazmat Yes/No).
   - A (Ground 3-Day): Standard Freight and Hazmat = No.
   - B (Air Next-Day): Express Tier and Hazmat = No.
   - C (Specialized Hazmat Carrier): Any shipment with Hazmat = Yes.
   - Internal Advisory Note: If an active carrier waiver/notice (e.g., Carrier Notice PA-12, Customs Waiver PX-8) applies to the confirmed hazmat status/account, confidential dispatch rules override standard SLAs.
"""


def _build_ticket(
    idx: int,
    quadrant: str,
    rng: random.Random,
    *,
    strict_voi: bool = False,
    implicit_override: bool = False,
) -> Dict[str, Any]:
    domain = DOMAINS[idx % len(DOMAINS)]
    qid = f"T_{idx:03d}_{quadrant}"

    is_underspec = quadrant in ("Q2_underspec_easy", "Q4_resolvable", "Q4_escalate")
    is_uncond_ovr = (quadrant == "Q3_clear_hard")
    is_cond_ovr = quadrant in ("Q4_resolvable", "Q4_escalate")

    if domain == "ClinicalTriage":
        age = rng.choice([29, 36, 44, 71, 78])
        age_str = f"{age}-year-old adult" if age < 65 else f"{age}-year-old senior"
        temp = rng.choice([36.8, 37.2, 38.9, 39.3])
        if quadrant == "Q4_resolvable":
            chest = "No"
        elif quadrant == "Q4_escalate":
            chest = "Yes"
        else:
            chest = rng.choice(["No", "Yes"])

        if chest == "Yes":
            base_ans = "C"
        elif age >= 65 or temp >= 38.5:
            base_ans = "B"
        else:
            base_ans = "A"

        missing_concept = "Chest Pain Status"
        missing_val = f"Chest Pain = {chest}"
        missing_keywords = ["chest", "pain", "cardiac", "symptom"]
        missing_changes_gold = is_underspec

        # Natural prose WITHOUT spoiler tags
        patient_tag = f"Patient #{4100 + idx}"
        if is_underspec:
            prose = (
                f"Intake Note ({patient_tag}): {age_str} presenting at triage desk with recorded temperature {temp}C "
                f"and mild fatigue since this morning."
            )
        else:
            cp_phrase = "reports acute chest tightness and pain" if chest == "Yes" else "denies any chest pain or tightness"
            prose = (
                f"Intake Note ({patient_tag}): {age_str} presenting at triage desk with temperature {temp}C; "
                f"patient {cp_phrase}."
            )

        if is_uncond_ovr:
            if implicit_override:
                prose += " Patient chart carries active Specialist Track Onco-B7 routing tag."
            else:
                prose += " Flagged under confidential Specialist Oncology Protocol Onco-B7 (overrides standard triage)."
            gold = "B" if base_ans != "B" else "A"
        elif is_cond_ovr:
            if implicit_override:
                prose += (
                    " Note: Cardiac-Watch CW-7 clinical pathway applies when Chest Pain = Yes; "
                    "otherwise follow standard triage."
                )
            else:
                prose += (
                    " Note: Confidential Cardiac-Watch CW-7 specialist protocol is active and overrides standard triage "
                    "ONLY if Chest Pain = Yes; if Chest Pain = No, follow standard triage."
                )
            if chest == "Yes":
                base_ans_default = "B" if (age >= 65 or temp >= 38.5) else "A"
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_ans_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    elif domain == "FinCompliance":
        tier = rng.choice(["Standard", "High-Risk"])
        amt = rng.choice([2400, 4200, 7800, 14500])
        if strict_voi and quadrant == "Q2_underspec_easy":
            tier = "Standard"
        if quadrant == "Q4_resolvable":
            region = "Domestic"
            tier = "Standard"
        elif quadrant == "Q4_escalate":
            region = "Cross-Border"
        else:
            region = rng.choice(["Domestic", "Cross-Border"])

        if tier == "Standard" and region == "Domestic":
            base_ans = "A" if amt <= 10000 else "B"
        else:
            base_ans = "B" if amt <= 5000 else "C"

        missing_concept = "Transfer Jurisdiction"
        missing_val = f"Transfer Jurisdiction = {region}"
        missing_keywords = ["jurisdiction", "domestic", "cross-border", "region", "country"]
        missing_changes_gold = True if is_cond_ovr else (is_underspec and tier == "Standard")

        tx_tag = f"TX-{8200 + idx}"
        if is_underspec:
            prose = (
                f"Compliance Ticket ({tx_tag}): {tier} Tier corporate client initiated a wire transfer of ${amt:,} USD "
                f"for vendor invoice settlement."
            )
        else:
            prose = (
                f"Compliance Ticket ({tx_tag}): {tier} Tier corporate client initiated a {region} wire transfer "
                f"of ${amt:,} USD for vendor invoice settlement."
            )

        if is_uncond_ovr:
            if implicit_override:
                prose += " Account carries internal Compliance Matrix M-14 flag."
            else:
                prose += " Account carries confidential Sanctions Matrix M-14 override flag requiring internal review."
            gold = "C" if base_ans != "C" else "B"
        elif is_cond_ovr:
            if implicit_override:
                prose += (
                    " Note: Treasury Bulletin CB-19 routing applies for Cross-Border wires; "
                    "Domestic wires follow standard rules."
                )
            else:
                prose += (
                    " Note: Confidential Treasury Bulletin CB-19 is in effect and supersedes standard limits "
                    "ONLY for Cross-Border wires; Domestic wires follow standard public rules."
                )
            if region == "Cross-Border":
                if tier == "Standard":
                    base_ans_default = "A" if amt <= 10000 else "B"
                else:
                    base_ans_default = "B" if amt <= 5000 else "C"
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_ans_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    elif domain == "ITIncident":
        sev = rng.choice(["Single User", "Partial Outage", "Full Outage"])
        if strict_voi and quadrant == "Q2_underspec_easy":
            sev = "Single User"
        if quadrant == "Q4_resolvable":
            scope = "Internal Tool"
            sev = rng.choice(["Single User", "Partial Outage"])
        elif quadrant == "Q4_escalate":
            scope = "Customer-Facing"
        else:
            scope = rng.choice(["Internal Tool", "Customer-Facing"])

        if scope == "Internal Tool" and sev == "Single User":
            base_ans = "A"
        elif sev == "Full Outage":
            base_ans = "C"
        else:
            base_ans = "B"

        missing_concept = "Service Scope"
        missing_val = f"Service Scope = {scope}"
        missing_keywords = ["scope", "internal", "customer", "facing", "service"]
        missing_changes_gold = True if is_cond_ovr else (is_underspec and sev == "Single User")

        inc_tag = f"INC-{3100 + idx}"
        if is_underspec:
            prose = (
                f"Ops Dispatch ({inc_tag}): Alert triggered for service degradation classified as {sev} "
                f"starting 12 minutes ago."
            )
        else:
            prose = (
                f"Ops Dispatch ({inc_tag}): Alert triggered on {scope} module with severity {sev} "
                f"starting 12 minutes ago."
            )

        if is_uncond_ovr:
            if implicit_override:
                prose += " Linked to SRE Escalation Clause VIP-9 custom routing."
            else:
                prose += " Linked to confidential Escalation Clause VIP-9 (undocumented custom routing applies)."
            gold = "C" if base_ans != "C" else "B"
        elif is_cond_ovr:
            if implicit_override:
                prose += (
                    " Note: SRE Runbook LPM-4 routing applies when the affected module is "
                    "Customer-Facing; Internal Tool alerts follow standard queues."
                )
            else:
                prose += (
                    " Note: Confidential SRE Runbook LPM-4 override applies ONLY if the affected module is "
                    "Customer-Facing; Internal Tool alerts follow standard queue rules."
                )
            if scope == "Customer-Facing":
                base_ans_default = "A" if sev == "Single User" else ("C" if sev == "Full Outage" else "B")
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_ans_default)][0]
            else:
                gold = base_ans
        else:
            gold = base_ans

    else:  # LogisticsSLA
        tier = rng.choice(["Standard Freight", "Express Tier"])
        if quadrant == "Q4_resolvable":
            hazmat = "No"
        elif quadrant == "Q4_escalate":
            hazmat = "Yes"
        else:
            hazmat = rng.choice(["No", "Yes"])

        if hazmat == "Yes":
            base_ans = "C"
        elif tier == "Express Tier":
            base_ans = "B"
        else:
            base_ans = "A"

        missing_concept = "Hazardous Material Status"
        missing_val = f"Hazmat = {hazmat}"
        missing_keywords = ["hazmat", "hazardous", "material", "dangerous", "cargo"]
        missing_changes_gold = is_underspec

        ship_tag = f"SHP-{6500 + idx}"
        if is_underspec:
            prose = (
                f"Shipment Request ({ship_tag}): {tier} domestic parcel batch (240 kg) queued for hub dispatch."
            )
        else:
            hz_phrase = "containing hazardous lithium materials (Hazmat = Yes)" if hazmat == "Yes" else "certified non-hazardous (Hazmat = No)"
            prose = (
                f"Shipment Request ({ship_tag}): {tier} domestic parcel batch (240 kg), {hz_phrase}, queued for dispatch."
            )

        if is_uncond_ovr:
            if implicit_override:
                prose += " Marked with Customs Waiver PX-8 specialist dispatch handling."
            else:
                prose += " Marked with confidential Customs Waiver PX-8 (internal specialist routing supersedes SLA)."
            gold = "B" if base_ans != "B" else "A"
        elif is_cond_ovr:
            if implicit_override:
                prose += (
                    " Note: Carrier Notice PA-12 handling applies when Hazmat = Yes; "
                    "non-hazardous batches follow standard SLA."
                )
            else:
                prose += (
                    " Note: Confidential Carrier Notice PA-12 overrides standard SLA ONLY if Hazmat = Yes; "
                    "non-hazardous batches follow standard SLA."
                )
            if hazmat == "Yes":
                base_ans_default = "B" if tier == "Express Tier" else "A"
                gold = [c for c in ("A", "B", "C") if c not in (base_ans, base_ans_default)][0]
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
        "missing_changes_gold": missing_changes_gold,
        "true_needs_expert": (quadrant in ("Q3_clear_hard", "Q4_escalate")),
        "missing_concept": missing_concept if is_underspec else None,
        "missing_val": missing_val if is_underspec else None,
        "missing_keywords": missing_keywords,
        "prose": f"Domain: {domain}\n{prose}",
        "gold": gold,
    }


def user_oracle(item: Dict[str, Any], question: str) -> Tuple[str, bool]:
    """Return (reply_text, revealed_new_info). User knows omitted intake field, never internal overrides."""
    if not (question or "").strip():
        return "No clarifying question was asked.", False
    if not item["is_underspec"] or not item["missing_val"]:
        return "All available requester intake details were already included; I have no knowledge of internal policy overrides.", False
    q_low = question.lower()
    if any(kw in q_low for kw in item["missing_keywords"]):
        return f"Clarification provided: {item['missing_val']}.", True
    return "Could you clarify which specific intake detail is missing? (e.g., chest pain, jurisdiction, service scope, or hazmat status)", False


def build_cscw_benchmark(
    n_cal_per_quad: int = 15,
    n_test_per_quad: int = 45,
    seed: int = 2027,
    *,
    strict_voi: bool = False,
    implicit_override: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build stratified Held-Out Calibration (60) and Test (180) sets (50/50 split inside Q4)."""
    rng = random.Random(seed)
    cal_items: List[Dict[str, Any]] = []
    test_items: List[Dict[str, Any]] = []
    idx = 0

    def _add_quad(quad: str, n_cal: int, n_test: int) -> None:
        nonlocal idx
        buf = []
        for _ in range(n_cal + n_test):
            buf.append(
                _build_ticket(
                    idx,
                    quad,
                    rng,
                    strict_voi=strict_voi,
                    implicit_override=implicit_override,
                )
            )
            idx += 1
        rng.shuffle(buf)
        cal_items.extend(buf[:n_cal])
        test_items.extend(buf[n_cal:])

    _add_quad("Q1_clear_easy", n_cal_per_quad, n_test_per_quad)
    _add_quad("Q2_underspec_easy", n_cal_per_quad, n_test_per_quad)
    _add_quad("Q3_clear_hard", n_cal_per_quad, n_test_per_quad)

    # Q4 Conditional Escalation: exactly 50% Q4_resolvable, 50% Q4_escalate
    n_q4_cal_half = n_cal_per_quad // 2
    n_q4_test_half = n_test_per_quad // 2
    _add_quad("Q4_resolvable", n_q4_cal_half, n_q4_test_half + (n_test_per_quad % 2))
    _add_quad("Q4_escalate", n_cal_per_quad - n_q4_cal_half, n_q4_test_half)

    rng.shuffle(cal_items)
    rng.shuffle(test_items)
    return cal_items, test_items


def build_cscw_benchmark_v2(
    n_cal_per_quad: int = 15,
    n_test_per_quad: int = 45,
    seed: int = 2027,
    strict_voi: bool = True,
    implicit_override: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build v2 benchmark ensuring 100% decision-relevant Q2 fields and implicit override cues."""
    return build_cscw_benchmark(
        n_cal_per_quad=n_cal_per_quad,
        n_test_per_quad=n_test_per_quad,
        seed=seed,
        strict_voi=strict_voi,
        implicit_override=implicit_override,
    )


def build_cscw_expansion_batch(
    n_per_quad: int = 45,
    seed: int = 2028,
    idx_offset: int = 240,
) -> List[Dict[str, Any]]:
    """Generate 180 additional independent test tickets (Q4 split 22/23 to balance combined N=360 to 45/45)."""
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    idx = idx_offset

    def _add(quad: str, count: int) -> None:
        nonlocal idx
        for _ in range(count):
            items.append(_build_ticket(idx, quad, rng))
            idx += 1

    _add("Q1_clear_easy", n_per_quad)
    _add("Q2_underspec_easy", n_per_quad)
    _add("Q3_clear_hard", n_per_quad)
    n_res = n_per_quad // 2
    n_esc = n_per_quad - n_res
    _add("Q4_resolvable", n_res)
    _add("Q4_escalate", n_esc)
    rng.shuffle(items)
    return items


def split_stratified_episodes(
    items: List[Dict[str, Any]],
    n_episodes: int = 6,
    seed: int = 2029,
) -> List[List[Dict[str, Any]]]:
    """Partition items into disjoint, quadrant-stratified episodes."""
    rng = random.Random(seed)
    by_quad: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        quad = it["quadrant"] if "quadrant" in it else it["item"]["quadrant"]
        by_quad.setdefault(quad, []).append(it)

    episodes: List[List[Dict[str, Any]]] = [[] for _ in range(n_episodes)]
    offset = 0
    for quad in sorted(by_quad.keys()):
        bucket = list(by_quad[quad])
        rng.shuffle(bucket)
        for i, it in enumerate(bucket):
            episodes[(offset + i) % n_episodes].append(it)
        offset += len(bucket)

    for ep in episodes:
        rng.shuffle(ep)
    return episodes
