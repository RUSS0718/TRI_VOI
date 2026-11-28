from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from benchmark import DOMAIN_RUBRIC, user_oracle
from benchmark_v3 import DOMAIN_RUBRIC_V3, DOMAINS_V3, user_oracle_v3
from controller import extract_answer, parse_json_obj, shannon_entropy
from external_baselines import build_acl_voi_belief_prompt
from public_tri import PUBLIC_TRI_HOLDOUT_RUBRIC, PUBLIC_TRI_RUBRIC, mediq_fact_select_oracle


def get_item_profile(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return prompt rubric, valid choices, and requester oracle for Benchmark v1/v2 or Public-Tri."""
    if item.get("domain") == "MediQHoldout":
        return {
            "rubric": item.get("custom_rubric") or PUBLIC_TRI_HOLDOUT_RUBRIC,
            "choices": ("A", "B", "C", "D"),
            "oracle_fn": mediq_fact_select_oracle,
            "is_mediq": True,
        }
    if item.get("domain") == "MediQClinical":
        return {
            "rubric": PUBLIC_TRI_RUBRIC,
            "choices": ("A", "B", "C", "D"),
            "oracle_fn": mediq_fact_select_oracle,
            "is_mediq": True,
        }
    if item.get("domain") in DOMAINS_V3:
        return {
            "rubric": item.get("custom_rubric") or DOMAIN_RUBRIC_V3,
            "choices": ("A", "B", "C"),
            "oracle_fn": user_oracle_v3,
            "is_mediq": False,
            "is_v3": True,
        }
    return {
        "rubric": item.get("custom_rubric") or DOMAIN_RUBRIC,
        "choices": ("A", "B", "C"),
        "oracle_fn": user_oracle,
        "is_mediq": False,
        "is_v3": False,
    }


def build_wave_a_specs(item: Dict[str, Any], dry_run: bool = False) -> List[Dict[str, Any]]:
    """Build exact Turn-1 prompt specifications (3 stochastic samples, 1 greedy, 1 source diag)."""
    prof = get_item_profile(item)
    rubric = prof["rubric"]
    choices = prof["choices"]
    c_list_str = ", ".join(choices[:-1]) + f", or {choices[-1]}"
    c_json_str = "|".join(f'"{c}"' for c in choices)
    dry_meta = f" [DRY_META:{item['quadrant']}:{item['gold']}]" if dry_run else ""

    base_prompt = (
        f"{rubric}\nTicket:\n{item['prose']}{dry_meta}\n\n"
        f"Choose the best action {c_list_str}. Respond in JSON ONLY: "
        f"{{\"answer\": {c_json_str}, \"rationale\": \"brief\"}}."
    )

    if prof["is_mediq"]:
        diag_prompt = (
            f"{rubric}\nTicket:\n{item['prose']}{dry_meta}\n\n"
            "Audit the clinical presentation against the Pre-Registered Clinical Triage Scope:\n"
            "1. missing_user_info: true if decisive patient history, physical exam, or lab/imaging details are omitted from the Presenting Vignette.\n"
            "2. potential_override: true if the case could fall under Specialist Escalation Scope (high-acuity subspecialty, multi-specialty complexity, or clinical red flags) either upfront or pending withheld findings.\n"
            "3. override_confirmed: true ONLY if a Specialist Escalation trigger is ALREADY confirmed present in the Presenting Vignette.\n"
            "4. clarifying_question: if missing_user_info is true, write one specific clinical question asking the patient/requester for missing history/exam/lab details; else empty string.\n"
            "Respond in JSON ONLY: "
            "{\"missing_user_info\": true|false, \"potential_override\": true|false, \"override_confirmed\": true|false, \"clarifying_question\": \"...\"}."
        )
    elif prof.get("is_v3"):
        diag_prompt = (
            f"{rubric}\nTicket:\n{item['prose']}{dry_meta}\n\n"
            "Audit the ticket against the domain's Required Intake Fields and Internal Specialist Advisory Notes:\n"
            "1. missing_user_info: true if ANY Required Intake Field for this domain is omitted/unstated in the Ticket.\n"
            "2. potential_override: true if the Ticket triggers or conditionally mentions an Internal Specialist Advisory rule.\n"
            "3. override_confirmed: true ONLY if an unconditional or confirmed conditional Internal Specialist Advisory rule applies in the Ticket.\n"
            "4. clarifying_question: if missing_user_info is true, write one short specific question asking for that missing field; else empty string.\n"
            "Respond in JSON ONLY: "
            "{\"missing_user_info\": true|false, \"potential_override\": true|false, \"override_confirmed\": true|false, \"clarifying_question\": \"...\"}."
        )
    else:
        diag_prompt = (
            f"{rubric}\nTicket:\n{item['prose']}{dry_meta}\n\n"
            "Audit the ticket against the domain's Required Intake Fields and Internal Advisory Rules:\n"
            "1. missing_user_info: true if ANY Required Intake Field for this domain is omitted/unstated in the Ticket.\n"
            "2. potential_override: true if the Ticket mentions an active confidential specialist protocol, bulletin, runbook, or waiver.\n"
            "3. override_confirmed: true ONLY if an unconditional override applies OR the trigger condition of a conditional protocol is ALREADY confirmed present in the Ticket.\n"
            "4. clarifying_question: if missing_user_info is true, write one short specific question asking for that missing field; else empty string.\n"
            "Respond in JSON ONLY: "
            "{\"missing_user_info\": true|false, \"potential_override\": true|false, \"override_confirmed\": true|false, \"clarifying_question\": \"...\"}."
        )

    specs = []
    for s_idx in range(3):
        specs.append({
            "messages": [{"role": "user", "content": base_prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
            "tag": f"probe_s{s_idx}",
            "seed_idx": s_idx,
        })
    specs.append({
        "messages": [{"role": "user", "content": base_prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "tag": "greedy_act",
        "seed_idx": 0,
    })
    specs.append({
        "messages": [{"role": "user", "content": diag_prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "tag": "diag_init",
        "seed_idx": 0,
    })
    return specs


def build_wave_b_specs(
    item: Dict[str, Any],
    clarify_q: str,
    *,
    dry_run: bool = False,
    include_acl_voi: bool = True,
) -> Tuple[str, bool, List[Dict[str, Any]]]:
    """Build Turn-2 prompt specs (post_ask_combined, post_ask_s1, and optional acl_voi_belief)."""
    if not clarify_q.strip():
        return "No clarifying question was asked.", False, []

    prof = get_item_profile(item)
    rubric = prof["rubric"]
    choices = prof["choices"]
    c_list_str = ", ".join(choices[:-1]) + f", or {choices[-1]}"
    c_json_str = "|".join(f'"{c}"' for c in choices)
    dry_meta = f" [DRY_META:{item['quadrant']}:{item['gold']}]" if dry_run else ""

    oracle_reply, oracle_revealed = prof["oracle_fn"](item, clarify_q)

    if prof["is_mediq"]:
        post_prompt = (
            f"{rubric}\nPresenting Vignette:\n{item['prose']}{dry_meta}\n\n"
            f"[Patient/Requester Clarified Findings]:\n{oracle_reply}\n\n"
            "Carefully integrate the Presenting Vignette and [Patient/Requester Clarified Findings], then evaluate:\n"
            "1. rationale: 1-2 concise clinical sentences determining (a) the best diagnostic/management option among A, B, C, D, and (b) whether ANY Specialist Escalation Scope rule is triggered (high-acuity subspecialty: Pediatrics/OB-GYN/Neurology/Cardiology/Surgery/EM/Psychiatry; >=2 consulting specialties; OR clinical red flag).\n"
            f"2. answer: best clinical option {c_list_str}.\n"
            "3. missing_user_info: true if decisive clinical details are still missing.\n"
            "4. override_confirmed: true ONLY if a Specialist Escalation Scope rule is confirmed present after combining the vignette and clarified findings.\n"
            "Respond in JSON ONLY (rationale first): "
            f"{{\"rationale\": \"...\", \"answer\": {c_json_str}, \"missing_user_info\": true|false, \"override_confirmed\": true|false}}."
        )
        post_tag = "post_ask_combined_v2"
    else:
        post_prompt = (
            f"{rubric}\nTicket:\n{item['prose']}{dry_meta}\n"
            f"[Requester Follow-up]: {oracle_reply}\n\n"
            "Re-evaluate after the requester's reply:\n"
            f"- answer: best choice {c_list_str}.\n"
            "- missing_user_info: true if a required intake field is still missing.\n"
            "- override_confirmed: true if a confidential specialist protocol/override applies to the confirmed ticket.\n"
            "Respond in JSON ONLY: "
            f"{{\"answer\": {c_json_str}, \"missing_user_info\": true|false, \"override_confirmed\": true|false}}."
        )
        post_tag = "post_ask_combined"

    specs = [
        {
            "messages": [{"role": "user", "content": post_prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
            "tag": post_tag,
            "seed_idx": 0,
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{rubric}\nTicket:\n{item['prose']}{dry_meta}\n[Requester Follow-up]: {oracle_reply}\nRespond in JSON ONLY: {{\"answer\": {c_json_str}}}.",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "tag": "post_ask_s1",
            "seed_idx": 1,
        },
    ]

    if include_acl_voi:
        acl_prompt = build_acl_voi_belief_prompt(rubric, f"{item['prose']}{dry_meta}", clarify_q, choices=choices)
        specs.append({
            "messages": [{"role": "user", "content": acl_prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
            "tag": "acl_voi_belief",
            "seed_idx": 0,
        })

    return oracle_reply, oracle_revealed, specs


def assemble_probed_record(
    item: Dict[str, Any],
    model_id: str,
    responses_by_tag: Dict[str, Dict[str, Any]],
    *,
    oracle_reply: str = "No clarifying question was asked.",
    oracle_revealed: bool = False,
) -> Dict[str, Any]:
    """Assemble standardized probed dict from cached/live responses."""
    prof = get_item_profile(item)
    choices = prof["choices"]

    samples = [extract_answer(responses_by_tag[f"probe_s{i}"]["content"], choices) for i in range(3)]
    r_greedy = responses_by_tag["greedy_act"]
    greedy_ans = extract_answer(r_greedy["content"], choices)

    r_diag = responses_by_tag["diag_init"]
    diag_obj = parse_json_obj(r_diag["content"])
    obs_missing = bool(diag_obj.get("missing_user_info", False))
    obs_pot_ovr = bool(diag_obj.get("potential_override", False))
    obs_conf_ovr = bool(diag_obj.get("override_confirmed", False))
    clarify_q = str(diag_obj.get("clarifying_question", "")).strip()

    tag_tokens: Dict[str, int] = {
        t: int(responses_by_tag[t].get("prompt_tokens", 0)) + int(responses_by_tag[t].get("completion_tokens", 0))
        for t in ("probe_s0", "probe_s1", "probe_s2", "greedy_act", "diag_init")
    }
    tag_tokens["post_ask_combined"] = 0
    tag_tokens["post_ask_s1"] = 0
    tag_tokens["acl_voi_belief"] = 0
    tokens = sum(tag_tokens.values())

    r_post = responses_by_tag.get("post_ask_combined_v2") or responses_by_tag.get("post_ask_combined")
    if clarify_q and r_post is not None:
        post_obj = parse_json_obj(r_post["content"])
        post_ans = extract_answer(r_post["content"], choices)
        post_ovr_conf = bool(post_obj.get("override_confirmed", obs_conf_ovr))
        r_ps1 = responses_by_tag["post_ask_s1"]
        post_s1 = extract_answer(r_ps1["content"], choices)
        t_post = int(r_post.get("prompt_tokens", 0)) + int(r_post.get("completion_tokens", 0))
        t_ps1 = int(r_ps1.get("prompt_tokens", 0)) + int(r_ps1.get("completion_tokens", 0))
        tag_tokens["post_ask_combined"] = t_post
        tag_tokens["post_ask_s1"] = t_ps1
        tokens += t_post + t_ps1
    else:
        post_ans = greedy_ans
        post_s1 = samples[0]
        post_ovr_conf = obs_conf_ovr

    acl_belief_raw = ""
    if clarify_q and "acl_voi_belief" in responses_by_tag:
        r_acl = responses_by_tag["acl_voi_belief"]
        acl_belief_raw = str(r_acl.get("content", ""))
        tag_tokens["acl_voi_belief"] = int(r_acl.get("prompt_tokens", 0)) + int(r_acl.get("completion_tokens", 0))

    return {
        "item": item,
        "model_id": model_id,
        "valid_choices": list(choices),
        "samples": samples,
        "entropy": shannon_entropy(samples),
        "maj_count": Counter(samples).most_common(1)[0][1],
        "greedy_ans": greedy_ans,
        "obs_missing": obs_missing,
        "obs_pot_ovr": obs_pot_ovr,
        "obs_conf_ovr": obs_conf_ovr,
        "clarify_q": clarify_q,
        "oracle_reply": oracle_reply,
        "oracle_revealed": oracle_revealed,
        "post_ans": post_ans,
        "post_s1": post_s1,
        "post_ovr_conf": post_ovr_conf,
        "acl_belief_raw": acl_belief_raw,
        "tokens": tokens,
        "tag_tokens": tag_tokens,
    }
