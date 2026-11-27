import hashlib
import json
import math
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark import DOMAIN_RUBRIC, DOMAINS, split_stratified_episodes, user_oracle
from controller import (
    DiscountedBetaTracker,
    HeldOutCalibrator,
    evaluate_diagnosis_metrics,
    extract_answer,
    parse_json_obj,
    shannon_entropy,
    simulate_stream,
)
from llm_client import RESULTS_DIR, call_llm

OVERNIGHT_RESULTS_DIR = RESULTS_DIR / "overnight"
OVERNIGHT_FIGS_DIR = OVERNIGHT_RESULTS_DIR / "figs"
OVERNIGHT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OVERNIGHT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Conservative upper-bound SiliconFlow pricing: (RMB per 1M input tokens, RMB per 1M output tokens)
MODEL_PRICING_RMB = {
    "Qwen/Qwen3-8B": (0.20, 0.60),
    "inclusionAI/Ling-flash-2.0": (0.60, 2.40),
    "tencent/Hy4-preview": (0.80, 3.00),
    "Qwen/Qwen3.5-35B-A3B": (0.70, 2.80),
    "zai-org/GLM-4.5-Air": (0.80, 4.00),
    "deepseek-ai/DeepSeek-V3.2": (2.00, 3.00),
}
DEFAULT_PRICING_RMB = (1.50, 4.00)


class BudgetLimitReached(RuntimeError):
    """Raised cleanly when live non-cached spend reaches --max-budget-rmb."""


class BudgetTracker:
    """Thread-safe cumulative token & estimated RMB tracker for non-cached API calls."""

    def __init__(self, max_budget_rmb: float = 15.0, state_path: Optional[Path] = None):
        self.max_budget_rmb = float(max_budget_rmb)
        self.state_path = state_path or (OVERNIGHT_RESULTS_DIR / "budget_state.json")
        self._lock = threading.Lock()
        self.live_calls = 0
        self.cached_hits = 0
        self.dry_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_rmb = 0.0
        self.by_model: Dict[str, Dict[str, Any]] = {}
        self.started_at = time.time()

    def check_budget(self) -> None:
        with self._lock:
            if self.estimated_rmb >= self.max_budget_rmb:
                raise BudgetLimitReached(
                    f"Budget ceiling reached: estimated_rmb={self.estimated_rmb:.3f} >= max={self.max_budget_rmb:.3f}"
                )

    def record(self, model_id: str, pt: int, ct: int, cached: bool, dry_run: bool) -> None:
        with self._lock:
            if dry_run:
                self.dry_calls += 1
                return
            if cached:
                self.cached_hits += 1
                return
            pin, pout = MODEL_PRICING_RMB.get(model_id, DEFAULT_PRICING_RMB)
            cost = (pt / 1e6) * pin + (ct / 1e6) * pout
            self.live_calls += 1
            self.prompt_tokens += int(pt)
            self.completion_tokens += int(ct)
            self.estimated_rmb += cost
            bm = self.by_model.setdefault(
                model_id,
                {"live_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "estimated_rmb": 0.0},
            )
            bm["live_calls"] += 1
            bm["prompt_tokens"] += int(pt)
            bm["completion_tokens"] += int(ct)
            bm["estimated_rmb"] += cost
            if self.live_calls % 20 == 0:
                self._save_unlocked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> Dict[str, Any]:
        return {
            "max_budget_rmb": round(self.max_budget_rmb, 3),
            "estimated_rmb": round(self.estimated_rmb, 4),
            "live_calls": self.live_calls,
            "cached_hits": self.cached_hits,
            "dry_calls": self.dry_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_min": round((time.time() - self.started_at) / 60.0, 2),
            "by_model": {
                m: {
                    "live_calls": v["live_calls"],
                    "prompt_tokens": v["prompt_tokens"],
                    "completion_tokens": v["completion_tokens"],
                    "estimated_rmb": round(v["estimated_rmb"], 4),
                }
                for m, v in sorted(self.by_model.items())
            },
        }

    def _save_unlocked(self) -> None:
        self.state_path.write_text(json.dumps(self._snapshot_unlocked(), indent=2), encoding="utf-8")

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()


def paced_call_llm(
    messages: List[Dict[str, str]],
    *,
    model_id: str,
    temperature: float,
    max_tokens: int,
    tag: str,
    seed_idx: int,
    dry_run: bool,
    tracker: BudgetTracker,
    pace_sec: float = 0.25,
) -> Dict[str, Any]:
    """Call llm_client with pre-call budget check and gentle pacing on non-cached live calls."""
    if not dry_run:
        tracker.check_budget()
    resp = call_llm(
        messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        tag=tag,
        seed_idx=seed_idx,
        dry_run=dry_run,
    )
    is_cached = bool(resp.get("cached", False))
    tracker.record(model_id, resp["prompt_tokens"], resp["completion_tokens"], is_cached, dry_run)
    if (not is_cached) and (not dry_run) and pace_sec > 0:
        time.sleep(pace_sec)
    return resp


def probe_item_budgeted(
    item: Dict[str, Any],
    model_id: str,
    tracker: BudgetTracker,
    *,
    dry_run: bool = False,
    pace_sec: float = 0.25,
    user_sim_fn: Optional[Callable[[Dict[str, Any], str], Tuple[str, bool]]] = None,
) -> Dict[str, Any]:
    """7-call probe preserving exact prompts/tags/seeds of controller.probe_item for 100% cache reuse."""
    dry_meta = f" [DRY_META:{item['quadrant']}:{item['gold']}]" if dry_run else ""
    base_prompt = (
        f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n\n"
        "Choose the best action A, B, or C. Respond in JSON ONLY: "
        "{\"answer\": \"A\"|\"B\"|\"C\", \"rationale\": \"brief\"}."
    )

    samples = []
    tokens = 0
    lat_total = 0.0
    for s_idx in range(3):
        r = paced_call_llm(
            [{"role": "user", "content": base_prompt}],
            model_id=model_id,
            temperature=0.7,
            max_tokens=1024,
            tag=f"probe_s{s_idx}",
            seed_idx=s_idx,
            dry_run=dry_run,
            tracker=tracker,
            pace_sec=pace_sec,
        )
        samples.append(extract_answer(r["content"]))
        tokens += r["prompt_tokens"] + r["completion_tokens"]
        lat_total += float(r.get("latency", 0.0))

    r_greedy = paced_call_llm(
        [{"role": "user", "content": base_prompt}],
        model_id=model_id,
        temperature=0.0,
        max_tokens=1024,
        tag="greedy_act",
        seed_idx=0,
        dry_run=dry_run,
        tracker=tracker,
        pace_sec=pace_sec,
    )
    greedy_ans = extract_answer(r_greedy["content"])
    tokens += r_greedy["prompt_tokens"] + r_greedy["completion_tokens"]
    lat_greedy = float(r_greedy.get("latency", 0.0))
    lat_total += lat_greedy

    diag_prompt = (
        f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n\n"
        "Audit the ticket against the domain's Required Intake Fields and Internal Advisory Rules:\n"
        "1. missing_user_info: true if ANY Required Intake Field for this domain is omitted/unstated in the Ticket.\n"
        "2. potential_override: true if the Ticket mentions an active confidential specialist protocol, bulletin, runbook, or waiver.\n"
        "3. override_confirmed: true ONLY if an unconditional override applies OR the trigger condition of a conditional protocol is ALREADY confirmed present in the Ticket.\n"
        "4. clarifying_question: if missing_user_info is true, write one short specific question asking for that missing field; else empty string.\n"
        "Respond in JSON ONLY: "
        "{\"missing_user_info\": true|false, \"potential_override\": true|false, \"override_confirmed\": true|false, \"clarifying_question\": \"...\"}."
    )
    r_diag = paced_call_llm(
        [{"role": "user", "content": diag_prompt}],
        model_id=model_id,
        temperature=0.0,
        max_tokens=1024,
        tag="diag_init",
        seed_idx=0,
        dry_run=dry_run,
        tracker=tracker,
        pace_sec=pace_sec,
    )
    diag_obj = parse_json_obj(r_diag["content"])
    tokens += r_diag["prompt_tokens"] + r_diag["completion_tokens"]
    lat_diag = float(r_diag.get("latency", 0.0))
    lat_total += lat_diag

    obs_missing = bool(diag_obj.get("missing_user_info", False))
    obs_pot_ovr = bool(diag_obj.get("potential_override", False))
    obs_conf_ovr = bool(diag_obj.get("override_confirmed", False))
    clarify_q = str(diag_obj.get("clarifying_question", "")).strip()

    sim_fn = user_sim_fn or user_oracle
    if clarify_q:
        oracle_reply, oracle_revealed = sim_fn(item, clarify_q)

        post_prompt = (
            f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n"
            f"[Requester Follow-up]: {oracle_reply}\n\n"
            "Re-evaluate after the requester's reply:\n"
            "- answer: best choice A, B, or C.\n"
            "- missing_user_info: true if a required intake field is still missing.\n"
            "- override_confirmed: true if a confidential specialist protocol/override applies to the confirmed ticket.\n"
            "Respond in JSON ONLY: "
            "{\"answer\": \"A\"|\"B\"|\"C\", \"missing_user_info\": true|false, \"override_confirmed\": true|false}."
        )
        r_post = paced_call_llm(
            [{"role": "user", "content": post_prompt}],
            model_id=model_id,
            temperature=0.0,
            max_tokens=1024,
            tag="post_ask_combined",
            seed_idx=0,
            dry_run=dry_run,
            tracker=tracker,
            pace_sec=pace_sec,
        )
        post_obj = parse_json_obj(r_post["content"])
        post_ans = extract_answer(r_post["content"])
        post_ovr_conf = bool(post_obj.get("override_confirmed", obs_conf_ovr))
        tok_post = r_post["prompt_tokens"] + r_post["completion_tokens"]
        tokens += tok_post
        lat_post = float(r_post.get("latency", 0.0))
        lat_total += lat_post

        r_post_s1 = paced_call_llm(
            [
                {
                    "role": "user",
                    "content": f"{DOMAIN_RUBRIC}\nTicket:\n{item['prose']}{dry_meta}\n[Requester Follow-up]: {oracle_reply}\nRespond in JSON ONLY: {{\"answer\": \"A\"|\"B\"|\"C\"}}.",
                }
            ],
            model_id=model_id,
            temperature=0.7,
            max_tokens=1024,
            tag="post_ask_s1",
            seed_idx=1,
            dry_run=dry_run,
            tracker=tracker,
            pace_sec=pace_sec,
        )
        post_s1 = extract_answer(r_post_s1["content"])
        tokens += r_post_s1["prompt_tokens"] + r_post_s1["completion_tokens"]
        lat_total += float(r_post_s1.get("latency", 0.0))
    else:
        oracle_reply = "No clarifying question was asked."
        oracle_revealed = False
        post_ans = greedy_ans
        post_s1 = samples[0]
        post_ovr_conf = obs_conf_ovr
        tok_post = 0
        lat_post = 0.0

    return {
        "item": item,
        "model_id": model_id,
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
        "tokens": tokens,
        "tokens_greedy": r_greedy["prompt_tokens"] + r_greedy["completion_tokens"],
        "tokens_diag": r_diag["prompt_tokens"] + r_diag["completion_tokens"],
        "tokens_post": tok_post,
        "latency_total": round(lat_total, 3),
        "latency_greedy": round(lat_greedy, 3),
        "latency_diag": round(lat_diag, 3),
        "latency_post": round(lat_post, 3),
    }


def probe_dataset_budgeted(
    items: List[Dict[str, Any]],
    model_id: str,
    tracker: BudgetTracker,
    *,
    dry_run: bool = False,
    workers: int = 2,
    pace_sec: float = 0.25,
    label: str = "",
    user_sim_fn: Optional[Callable[[Dict[str, Any], str], Tuple[str, bool]]] = None,
) -> List[Dict[str, Any]]:
    out_map: Dict[int, Dict[str, Any]] = {}
    if dry_run or workers <= 1:
        for i, it in enumerate(items):
            out_map[i] = probe_item_budgeted(
                it, model_id, tracker, dry_run=dry_run, pace_sec=pace_sec, user_sim_fn=user_sim_fn
            )
            if (i + 1) % 30 == 0 or (i + 1) == len(items):
                snap = tracker.snapshot()
                print(
                    f"[{model_id}{' ' + label if label else ''}] probed {i+1}/{len(items)} | "
                    f"live={snap['live_calls']} cache={snap['cached_hits']} est_rmb=¥{snap['estimated_rmb']:.2f}",
                    flush=True,
                )
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    probe_item_budgeted,
                    it,
                    model_id,
                    tracker,
                    dry_run=False,
                    pace_sec=pace_sec,
                    user_sim_fn=user_sim_fn,
                ): i
                for i, it in enumerate(items)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                out_map[idx] = fut.result()
                done += 1
                if done % 30 == 0 or done == len(items):
                    snap = tracker.snapshot()
                    print(
                        f"[{model_id}{' ' + label if label else ''}] probed {done}/{len(items)} | "
                        f"live={snap['live_calls']} cache={snap['cached_hits']} est_rmb=¥{snap['estimated_rmb']:.2f}",
                        flush=True,
                    )
    tracker.save()
    return [out_map[i] for i in range(len(items))]


# =====================================================================
# Stage 3: Realistic Multi-Paragraph Noisy Prose + Imperfect User Sim
# =====================================================================

_NOISY_DISTRACTOR_CONTEXT = {
    "ClinicalTriage": (
        "[Nurse Shift Handoff Thread — Desk B, 08:42]\n"
        "Shift note from RN Ramirez: Reception queue has 4 routine immunization follow-ups pending lab printer toner swap. "
        "Patient chart synced from outpatient portal with prior allergy panel (penicillin-free, seasonal rhinitis documented 2024). "
        "Vitals cuff #3 re-zeroed before triage intake; accompanying family member asked about parking validation.\n"
        "Primary Intake Entry:\n"
    ),
    "FinCompliance": (
        "[Corporate Treasury Queue — Slack #aml-escalations, 14:15]\n"
        "Automated ledger check passed IBAN checksum validation and matched historical vendor master record (onboarding date 2023-09). "
        "Note from Ops Analyst Lee: Q4 procurement batch contains 17 recurring SaaS renewals and 3 hardware deposits; ERP sync latency 120ms.\n"
        "Escalated Wire Summary:\n"
    ),
    "ITIncident": (
        "[PagerDuty -> #ops-triage Thread, 03:19 UTC]\n"
        "Auto-telemetry snapshot: Kubernetes ingress p95 latency spiked 18ms around US-West node rotation; backup CronJob completed cleanly. "
        "Night-shift SRE pinged DBA on standby regarding unrelated read-replica lag from yesterday's schema backfill.\n"
        "Active Alert Payload:\n"
    ),
    "LogisticsSLA": (
        "[Regional Hub Dispatch Memo — Dock 7, 17:05]\n"
        "Carrier manifest synced with automated barcode scanner #14; pallet wrap inspection verified intact and weather advisory mild along I-80 corridor. "
        "Supervisor Choi noted two empty return trailers scheduled for overnight deadhead repositioning.\n"
        "Shipment Order Record:\n"
    ),
}

_SPOILER_ATTRIBUTION_PHRASES = [
    "chest pain = yes",
    "chest pain = no",
    "denies any chest pain",
    "denies chest pain",
    "reports acute chest",
    "no chest pain",
    "transfer jurisdiction = domestic",
    "transfer jurisdiction = cross-border",
    "initiated a domestic wire",
    "initiated a cross-border wire",
    "service scope = internal tool",
    "service scope = customer-facing",
    "triggered on internal tool",
    "triggered on customer-facing",
    "hazmat = yes",
    "hazmat = no",
    "certified non-hazardous",
    "containing hazardous lithium",
]


def _has_underspec_spoiler(rewritten: str, item: Dict[str, Any]) -> bool:
    low = rewritten.lower()
    if item.get("missing_val") and item["missing_val"].lower() in low:
        return True
    for ph in _SPOILER_ATTRIBUTION_PHRASES:
        if ph in low:
            return True
    return False


def _deterministic_noisy_fallback(item: Dict[str, Any]) -> str:
    body = item["prose"]
    if body.startswith("Domain: "):
        _, body = body.split("\n", 1)
    prefix = _NOISY_DISTRACTOR_CONTEXT.get(item["domain"], "[Workplace Thread]\n")
    footer = (
        "\n[Automated Audit Metadata]: Ticket hash verified; requester notified of standard SLA turnaround window. "
        "Please route strictly according to confirmed intake fields and any active internal advisory notices."
    )
    return f"Domain: {item['domain']}\n{prefix}{body}{footer}"


def rewrite_ticket_noisy(
    item: Dict[str, Any],
    tracker: BudgetTracker,
    *,
    rewriter_model: str = "Qwen/Qwen3.5-35B-A3B",
    dry_run: bool = False,
    pace_sec: float = 0.25,
    implicit_override: bool = False,
) -> Dict[str, Any]:
    """Rewrite a ticket into realistic 2-3 paragraph workplace prose with harmless distractors & zero spoiler leakage."""
    out = dict(item)
    out["is_noisy_longform"] = True
    if dry_run:
        out["prose"] = _deterministic_noisy_fallback(item)
        return out

    body = item["prose"]
    if body.startswith("Domain: "):
        _, body = body.split("\n", 1)

    guard_rule = (
        f"CRITICAL CONSTRAINT: This ticket is intentionally underspecified about '{item['missing_concept']}'. "
        f"You MUST NOT mention, guess, imply, deny, or confirm '{item['missing_concept']}' in the patient/transaction/alert/shipment facts! "
        "However, keep any conditional policy sentence (e.g., 'ONLY if ...') intact."
        if item["is_underspec"]
        else "Preserve every factual attribute (age, temperature, chest pain, risk tier, jurisdiction, amount, scope, severity, shipping tier, hazmat status) accurately."
    )
    ovr_rule = (
        "Paraphrase any internal protocol, bulletin, runbook, or waiver notice naturally into workplace context WITHOUT using the words 'confidential' or 'override'."
        if implicit_override
        else "Preserve any confidential specialist protocol/bulletin/runbook/waiver notice verbatim if present."
    )
    prompt = (
        f"Rewrite the following {item['domain']} cooperative triage ticket into a realistic 2-to-3 paragraph workplace record "
        "(e.g., nurse handoff note, compliance email, Slack ops thread, or hub dispatch memo) by weaving in realistic harmless background noise "
        "(timestamps, colleague chatter, secondary system logs).\n"
        f"{guard_rule}\n"
        f"{ovr_rule}\n"
        "Output ONLY the rewritten workplace text (no preamble or markdown fences).\n\n"
        f"Original Ticket:\n{body}"
    )
    r = paced_call_llm(
        [{"role": "user", "content": prompt}],
        model_id=rewriter_model,
        temperature=0.4,
        max_tokens=1024,
        tag="noisy_rewrite_v1",
        seed_idx=0,
        dry_run=False,
        tracker=tracker,
        pace_sec=pace_sec,
    )
    text = (r.get("content") or "").strip()
    if (len(text) < 80) or (item["is_underspec"] and _has_underspec_spoiler(text, item)):
        out["prose"] = _deterministic_noisy_fallback(item)
        out["rewrite_fallback"] = True
    else:
        out["prose"] = f"Domain: {item['domain']}\n{text}"
        out["rewrite_fallback"] = False
    return out


def build_noisy_dataset(
    items: List[Dict[str, Any]],
    tracker: BudgetTracker,
    *,
    rewriter_model: str = "Qwen/Qwen3.5-35B-A3B",
    dry_run: bool = False,
    workers: int = 2,
    pace_sec: float = 0.25,
) -> List[Dict[str, Any]]:
    out_map: Dict[int, Dict[str, Any]] = {}
    if dry_run or workers <= 1:
        for i, it in enumerate(items):
            out_map[i] = rewrite_ticket_noisy(it, tracker, rewriter_model=rewriter_model, dry_run=dry_run, pace_sec=pace_sec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(rewrite_ticket_noisy, it, tracker, rewriter_model=rewriter_model, dry_run=False, pace_sec=pace_sec): i
                for i, it in enumerate(items)
            }
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                out_map[idx] = fut.result()
                done += 1
                if done % 60 == 0 or done == len(items):
                    print(f"[Noisy Rewrite] {done}/{len(items)} done", flush=True)
    return [out_map[i] for i in range(len(items))]


def make_imperfect_user_sim(
    tracker: BudgetTracker,
    *,
    sim_model: str = "Qwen/Qwen3.5-35B-A3B",
    dry_run: bool = False,
    pace_sec: float = 0.25,
) -> Callable[[Dict[str, Any], str], Tuple[str, bool]]:
    """Create an imperfect natural-language requester simulator (with 15% non-answer friction)."""

    def _sim(item: Dict[str, Any], clarify_q: str) -> Tuple[str, bool]:
        if not (clarify_q or "").strip():
            return ("No clarifying question was asked.", False)
        if not item["is_underspec"] or not item["missing_val"]:
            return (
                "Honestly I already pasted everything from my side—we don't have visibility into internal specialist overrides or bulletins.",
                False,
            )
        q_low = (clarify_q or "").lower()
        if not any(kw in q_low for kw in item["missing_keywords"]):
            return (
                "Sorry, could you clarify which intake field you mean? Our form only shows the case summary I shared.",
                False,
            )
        h = int(hashlib.sha256(item["qid"].encode("utf-8")).hexdigest(), 16) % 100
        if h < 15:
            # 15% realistic human distraction / partial answer failing to resolve the missing field
            return (
                "Just stepped away from my desk—let me ping the requester/client to double check that detail, I can't confirm it yet.",
                False,
            )
        if dry_run:
            return (f"Sure, checking the intake sheet: {item['missing_val']}.", True)

        prompt = (
            f"You are a busy frontline employee responding in 1-2 short natural sentences to a triage bot's question.\n"
            f"Bot Question: {clarify_q}\n"
            f"Fact you know: {item['missing_val']}.\n"
            "Write a realistic, conversational reply (do NOT use 'Field = Value' key-value formatting; speak naturally) that clearly states this fact."
        )
        r = paced_call_llm(
            [{"role": "user", "content": prompt}],
            model_id=sim_model,
            temperature=0.3,
            max_tokens=256,
            tag="imperfect_user_reply_v1",
            seed_idx=0,
            dry_run=False,
            tracker=tracker,
            pace_sec=pace_sec,
        )
        reply = (r.get("content") or "").strip()
        if not reply:
            reply = f"Sure, checking my notes: {item['missing_val']}."
        return reply, True

    return _sim


# =====================================================================
# Stage 4: Entropy-Collapse K x T Stress Test
# =====================================================================


def run_entropy_kt_stress(
    test_360_items: List[Dict[str, Any]],
    models: List[str],
    tracker: BudgetTracker,
    *,
    diag_f1_by_model: Dict[str, float],
    dry_run: bool = False,
    workers: int = 2,
    pace_sec: float = 0.25,
    max_items: int = 120,
) -> List[Dict[str, Any]]:
    """Test K in {3, 5, 10} x T in {0.7, 1.0} on stratified underspecified tickets."""
    q2 = [it for it in test_360_items if it["quadrant"] == "Q2_underspec_easy"][:60]
    q4r = [it for it in test_360_items if it["quadrant"] == "Q4_resolvable"][:30]
    q4e = [it for it in test_360_items if it["quadrant"] == "Q4_escalate"][:30]
    sub_items = (q2 + q4r + q4e)[:max_items]

    rows: List[Dict[str, Any]] = []
    for mid in models:
        for temp in (0.7, 1.0):
            def _probe_one(it: Dict[str, Any]) -> Dict[str, Any]:
                dry_meta = f" [DRY_META:{it['quadrant']}:{it['gold']}]" if dry_run else ""
                prompt = (
                    f"{DOMAIN_RUBRIC}\nTicket:\n{it['prose']}{dry_meta}\n\n"
                    "Choose the best action A, B, or C. Respond in JSON ONLY: "
                    "{\"answer\": \"A\"|\"B\"|\"C\", \"rationale\": \"brief\"}."
                )
                ans_list = []
                for s_idx in range(10):
                    r = paced_call_llm(
                        [{"role": "user", "content": prompt}],
                        model_id=mid,
                        temperature=temp,
                        max_tokens=1024,
                        tag=f"kt_stress_T{temp:.1f}_s{s_idx}",
                        seed_idx=s_idx,
                        dry_run=dry_run,
                        tracker=tracker,
                        pace_sec=pace_sec,
                    )
                    ans_list.append(extract_answer(r["content"]))
                return {"gold": it["gold"], "ans_list": ans_list}

            recs: List[Dict[str, Any]] = []
            if dry_run or workers <= 1:
                for i, it in enumerate(sub_items):
                    recs.append(_probe_one(it))
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_probe_one, it) for it in sub_items]
                    for fut in as_completed(futs):
                        recs.append(fut.result())

            for k_val in (3, 5, 10):
                collapse_cnt = 0
                low_ent_cnt = 0
                ent_sum = 0.0
                maj_ok_cnt = 0
                for rec in recs:
                    sub = rec["ans_list"][:k_val]
                    ent = shannon_entropy(sub)
                    ent_sum += ent
                    if len(set(sub)) == 1:
                        collapse_cnt += 1
                    if ent <= 0.45:
                        low_ent_cnt += 1
                    maj_label = Counter(sub).most_common(1)[0][0]
                    if maj_label == rec["gold"]:
                        maj_ok_cnt += 1
                n_it = max(1, len(recs))
                rows.append(
                    {
                        "model_id": mid,
                        "short_name": mid.split("/")[-1],
                        "n_items": len(recs),
                        "temperature": temp,
                        "K": k_val,
                        "entropy_collapse_pct": round(100.0 * collapse_cnt / n_it, 2),
                        "low_entropy_pct": round(100.0 * low_ent_cnt / n_it, 2),
                        "mean_entropy_bits": round(ent_sum / n_it, 4),
                        "majority_acc_pct": round(100.0 * maj_ok_cnt / n_it, 2),
                        "disentangled_missing_f1": diag_f1_by_model.get(mid, 0.0),
                    }
                )
        tracker.save()
    return rows


# =====================================================================
# Stage 5: Deep Offline Structural & Statistical Suite (0 API Calls)
# =====================================================================


def _simulate_bursty_or_shift_stream(
    method: str,
    test_probed: List[Dict[str, Any]],
    calibrator: HeldOutCalibrator,
    *,
    mode: str = "bursty",  # "bursty" | "shift3"
    seed: int = 0,
    c_ask: float = 1.5,
    gamma_queue: float = 0.85,
) -> Dict[str, Any]:
    """Evaluate Tri-VoI-L2D vs baselines under bursty Poisson arrivals or 3-expert shift fatigue."""
    rng = random.Random(9000 + seed)
    n = len(test_probed)
    if mode == "shift3":
        queues = {"Senior": 0, "Mid": 0, "Junior": 0}
        caps = {"Senior": 1, "Mid": 1, "Junior": 2}
        costs = {"Senior": 3.5, "Mid": 2.2, "Junior": 1.3}
        trackers = {
            "Senior": DiscountedBetaTracker(9.0, 0.6, 0.92),
            "Mid": DiscountedBetaTracker(8.5, 1.0, 0.92),
            "Junior": DiscountedBetaTracker(8.0, 1.2, 0.92),
        }
        lam_ema = {"Senior": 0.12, "Mid": 0.22, "Junior": 0.30}
        drain_interval = 2
    else:
        queues = {"Spec": 0}
        caps = {"Spec": 2}
        costs = {"Spec": 2.0}
        trackers = {"Spec": DiscountedBetaTracker(8.5, 1.0, 0.92)}
        lam_ema = {"Spec": 0.35}
        drain_interval = 2

    use_ext_flag = method in ("Tri-VoI-L2D", "Calibrated-Cascade+Naor", "Joint-Greedy+Naor")
    utils = np.zeros(n, dtype=float)
    ok_cnt = 0
    wait_sum = 0
    def_cnt = 0
    ask_cnt = 0

    for t, rec in enumerate(test_probed):
        it = rec["item"]
        r_ok, r_err = 10.0, -10.0

        # Bursty background external arrivals create transient congestion waves in "bursty" mode
        if mode == "bursty":
            if t % drain_interval == 0:
                queues["Spec"] = max(0, queues["Spec"] - caps["Spec"])
            if (t % 45) in range(10, 22):
                # Poisson-like spike wave: extra external backlog
                queues["Spec"] += 1 if (rng.random() < 0.65) else 0
            true_acc = {"Spec": 0.60 if t >= (n // 2) else 0.92}
        else:
            if t % drain_interval == 0:
                for m in queues:
                    queues[m] = max(0, queues[m] - caps[m])
            # Staggered shift fatigue: Mid drops in middle third, Junior drops in final third, Senior steady
            true_acc = {
                "Senior": 0.94,
                "Mid": 0.62 if (n // 3 <= t < 2 * n // 3) else 0.89,
                "Junior": 0.58 if (t >= n // 2) else 0.86,
            }

        def best_spec(use_ext: bool = False) -> Tuple[str, float]:
            bm = list(queues.keys())[0]
            bu = -1e9
            for m in queues:
                pe = trackers[m].mean()
                qm = queues[m]
                mu_m = caps[m] / float(drain_interval)
                rho_m = min(lam_ema[m] / max(1e-6, mu_m), 1.5)
                cong = gamma_queue * (qm + rho_m * (qm + 1.0)) if (use_ext and (qm >= caps[m] or rho_m >= 0.6)) else gamma_queue * qm
                um = pe * r_ok + (1.0 - pe) * r_err - costs[m] - cong
                if um > bu:
                    bu = um
                    bm = m
            return bm, bu

        unanimous = rec["maj_count"] == 3
        obs_m, obs_p, obs_c = rec["obs_missing"], rec["obs_pot_ovr"], rec["obs_conf_ovr"]
        ent = rec["entropy"]
        did_ask = False
        did_def = False
        chosen = list(queues.keys())[0]
        ans = rec["greedy_ans"]

        if method == "Direct-ACT":
            pass
        elif method == "Congested-L2D-Only":
            pa = calibrator.get_p_act(obs_m, obs_p, unanimous)
            bm, ud = best_spec(False)
            if ud > (pa * r_ok + (1.0 - pa) * r_err):
                did_def = True
                chosen = bm
        elif method == "Naive-Cascade":
            if ent > 0.45:
                did_ask = True
                ans = rec["post_ans"]
                if rec["post_ans"] != rec["post_s1"] or rec["post_ovr_conf"]:
                    did_def = True
                    chosen, _ = best_spec(False)
            elif obs_c or not unanimous:
                did_def = True
                chosen, _ = best_spec(False)
        elif method in ("Calibrated-Cascade", "Calibrated-Cascade+Naor"):
            pa = calibrator.get_p_act(obs_m, obs_p, unanimous)
            bm, ud = best_spec(use_ext_flag)
            if obs_m:
                did_ask = True
                pp = calibrator.p_post_ovr if rec["post_ovr_conf"] else calibrator.p_post_clear
                if ud > (pp * r_ok + (1.0 - pp) * r_err):
                    did_def = True
                    chosen = bm
                else:
                    ans = rec["post_ans"]
            elif ud > (pa * r_ok + (1.0 - pa) * r_err):
                did_def = True
                chosen = bm
        elif method in ("Joint-Greedy", "Joint-Greedy+Naor"):
            pa = calibrator.get_p_act(obs_m, obs_p, unanimous)
            pr, po, pn = calibrator.get_post_ask_probs(obs_m, obs_p)
            ua = pa * r_ok + (1.0 - pa) * r_err
            uc = calibrator.p_post_clear * r_ok + (1.0 - calibrator.p_post_clear) * r_err
            uo = calibrator.p_post_ovr * r_ok + (1.0 - calibrator.p_post_ovr) * r_err
            bm, ud = best_spec(use_ext_flag)
            if obs_m and (pr * uc + po * uo + pn * ua - c_ask) > max(ua, ud):
                did_ask = True
                ans = rec["post_ans"]
            elif ud > ua:
                did_def = True
                chosen = bm
        else:
            # Tri-VoI-L2D
            pa = calibrator.get_p_act(obs_m, obs_p, unanimous)
            pr, po, pn = calibrator.get_post_ask_probs(obs_m, obs_p)
            ua = pa * r_ok + (1.0 - pa) * r_err
            uc = calibrator.p_post_clear * r_ok + (1.0 - calibrator.p_post_clear) * r_err
            uo = calibrator.p_post_ovr * r_ok + (1.0 - calibrator.p_post_ovr) * r_err
            bm, ud = best_spec(use_ext_flag)
            u_ask = pr * max(uc, ud) + po * max(uo, ud) + pn * max(ua, ud) - c_ask
            if u_ask > max(ua, ud):
                did_ask = True
                pp = calibrator.p_post_ovr if rec["post_ovr_conf"] else calibrator.p_post_clear
                bm2, ud2 = best_spec(use_ext_flag)
                if ud2 > (pp * r_ok + (1.0 - pp) * r_err):
                    did_def = True
                    chosen = bm2
                else:
                    ans = rec["post_ans"]
            elif ud > ua:
                did_def = True
                chosen = bm

        for m in queues:
            lam_ema[m] = 0.90 * lam_ema[m] + 0.10 * (1.0 if (did_def and chosen == m) else 0.0)

        su = -c_ask if did_ask else 0.0
        if did_ask:
            ask_cnt += 1
        if did_def:
            def_cnt += 1
            wait_sum += queues[chosen]
            su -= costs[chosen] + gamma_queue * queues[chosen]
            queues[chosen] += 1
            is_ok = rng.random() < true_acc[chosen]
            trackers[chosen].update(is_ok)
        else:
            is_ok = ans == it["gold"]
        su += r_ok if is_ok else r_err
        utils[t] = su
        ok_cnt += int(is_ok)

    return {
        "mean_utility": float(np.mean(utils)),
        "acc_pct": 100.0 * ok_cnt / n,
        "mean_wait": wait_sum / max(1, def_cnt),
        "ask_rate_pct": 100.0 * ask_cnt / n,
    }


def _simulate_hyperparam_tri(
    test_probed: List[Dict[str, Any]],
    calibrator: HeldOutCalibrator,
    *,
    lam_forget: float = 0.92,
    gamma_model: float = 0.85,
    gamma_env: float = 0.85,
    capacity: int = 2,
    drain_interval: int = 2,
    c_ask: float = 1.5,
    seed: int = 0,
) -> float:
    rng = random.Random(5000 + seed)
    n = len(test_probed)
    q = 0
    lam_ema = 0.35
    tr = DiscountedBetaTracker(8.5, 1.0, lam_forget)
    utils = np.zeros(n, dtype=float)
    for t, rec in enumerate(test_probed):
        if t % drain_interval == 0:
            q = max(0, q - capacity)
        true_acc = 0.60 if t >= (n // 2) else 0.92
        pe = tr.mean()
        mu = capacity / float(drain_interval)
        rho = min(lam_ema / max(1e-6, mu), 1.5)
        cong = gamma_model * (q + rho * (q + 1.0)) if (q >= capacity or rho >= 0.6) else gamma_model * q
        ud = pe * 10.0 - (1.0 - pe) * 10.0 - 2.0 - cong
        pa = calibrator.get_p_act(rec["obs_missing"], rec["obs_pot_ovr"], rec["maj_count"] == 3)
        pr, po, pn = calibrator.get_post_ask_probs(rec["obs_missing"], rec["obs_pot_ovr"])
        ua = pa * 10.0 - (1.0 - pa) * 10.0
        uc = calibrator.p_post_clear * 10.0 - (1.0 - calibrator.p_post_clear) * 10.0
        uo = calibrator.p_post_ovr * 10.0 - (1.0 - calibrator.p_post_ovr) * 10.0
        did_ask = did_def = False
        ans = rec["greedy_ans"]
        if (pr * max(uc, ud) + po * max(uo, ud) + pn * max(ua, ud) - c_ask) > max(ua, ud):
            did_ask = True
            pp = calibrator.p_post_ovr if rec["post_ovr_conf"] else calibrator.p_post_clear
            if ud > (pp * 10.0 - (1.0 - pp) * 10.0):
                did_def = True
            else:
                ans = rec["post_ans"]
        elif ud > ua:
            did_def = True
        lam_ema = 0.90 * lam_ema + 0.10 * (1.0 if did_def else 0.0)
        su = -c_ask if did_ask else 0.0
        if did_def:
            su -= 2.0 + gamma_env * q
            q += 1
            ok = rng.random() < true_acc
            tr.update(ok)
        else:
            ok = ans == rec["item"]["gold"]
        su += 10.0 if ok else -10.0
        utils[t] = su
    return float(np.mean(utils))


def _audit_error_taxonomy(
    test_probed: List[Dict[str, Any]],
    calibrator: HeldOutCalibrator,
    *,
    c_ask: float = 1.5,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(5000 + seed)
    n = len(test_probed)
    q = 0
    lam_ema = 0.35
    tr = DiscountedBetaTracker(8.5, 1.0, 0.92)
    cats = Counter()
    tot_err = 0

    for t, rec in enumerate(test_probed):
        it = rec["item"]
        quad = it["quadrant"]
        if t % 2 == 0:
            q = max(0, q - 2)
        true_acc = 0.60 if t >= (n // 2) else 0.92
        pe = tr.mean()
        rho = min(lam_ema / 1.0, 1.5)
        cong = 0.85 * (q + rho * (q + 1.0)) if (q >= 2 or rho >= 0.6) else 0.85 * q
        ud = pe * 10.0 - (1.0 - pe) * 10.0 - 2.0 - cong
        pa = calibrator.get_p_act(rec["obs_missing"], rec["obs_pot_ovr"], rec["maj_count"] == 3)
        pr, po, pn = calibrator.get_post_ask_probs(rec["obs_missing"], rec["obs_pot_ovr"])
        ua = pa * 10.0 - (1.0 - pa) * 10.0
        uc = calibrator.p_post_clear * 10.0 - (1.0 - calibrator.p_post_clear) * 10.0
        uo = calibrator.p_post_ovr * 10.0 - (1.0 - calibrator.p_post_ovr) * 10.0

        did_ask = did_def = False
        ans = rec["greedy_ans"]
        if (pr * max(uc, ud) + po * max(uo, ud) + pn * max(ua, ud) - c_ask) > max(ua, ud):
            did_ask = True
            pp = calibrator.p_post_ovr if rec["post_ovr_conf"] else calibrator.p_post_clear
            if ud > (pp * 10.0 - (1.0 - pp) * 10.0):
                did_def = True
            else:
                ans = rec["post_ans"]
        elif ud > ua:
            did_def = True

        lam_ema = 0.90 * lam_ema + 0.10 * (1.0 if did_def else 0.0)
        if did_def:
            q += 1
            ok = rng.random() < true_acc
            tr.update(ok)
        else:
            ok = ans == it["gold"]

        if not ok:
            tot_err += 1
            if did_def:
                cats["Specialist_Fatigue_or_Noise_Error"] += 1
            elif it["is_underspec"] and not did_ask:
                if not rec["obs_missing"]:
                    cats["Diagnoser_FN_Missed_Underspec"] += 1
                else:
                    cats["EVOI_Declined_ASK_Under_Cost_or_Queue"] += 1
            elif did_ask and not rec["oracle_revealed"]:
                cats["Clarifying_Question_or_User_Failure"] += 1
            elif did_ask and rec["oracle_revealed"]:
                if it["true_needs_expert"] and not rec["post_ovr_conf"]:
                    cats["PostASK_Override_Detection_FN"] += 1
                else:
                    cats["PostASK_Public_Rubric_Reasoning_Error"] += 1
            elif it["is_uncond_ovr"]:
                cats["Uncond_Override_FN_or_Congestion_ACT"] += 1
            else:
                cats["Clear_Easy_Direct_ACT_Error"] += 1

    return {
        "n_items": n,
        "total_errors": tot_err,
        "error_rate_pct": round(100.0 * tot_err / n, 2),
        "breakdown": [
            {
                "category": k,
                "count": v,
                "pct_of_errors": round(100.0 * v / max(1, tot_err), 1),
            }
            for k, v in cats.most_common()
        ],
    }


def run_stage5_structural(
    all_model_pools: Dict[str, Dict[str, List[Dict[str, Any]]]],
    primary_model: str,
) -> Dict[str, Any]:
    """Run 5a-5d deep structural, CV, LODO, bursty/shift, sensitivity, Pareto, and error taxonomy (0 API calls)."""
    prim_cal = all_model_pools[primary_model]["cal"]
    prim_test = all_model_pools[primary_model]["test"]
    prim_calib = HeldOutCalibrator(prim_cal)

    # 5a. 56-point 2D Phase Map: 7 loads (mu) x 8 c_ask x 10 seeds
    load_grid = [
        ("mu=4.00", 4, 1),
        ("mu=2.00", 2, 1),
        ("mu=1.00", 2, 2),
        ("mu=0.75", 3, 4),
        ("mu=0.50", 1, 2),
        ("mu=0.33", 1, 3),
        ("mu=0.25", 1, 4),
    ]
    cask_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    phase_methods = [
        "Tri-VoI-L2D",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "w/o Queue-Externality",
        "Naive-Cascade",
        "Congested-L2D-Only",
    ]
    phase_cells = []
    for l_lbl, cap, d_int in load_grid:
        mu_val = round(cap / float(d_int), 2)
        for ca in cask_grid:
            m_utils = {}
            for meth in phase_methods:
                rs = [
                    simulate_stream(
                        meth, prim_test, prim_calib, capacity=cap, drain_interval=d_int, drift_mode="sudden", c_ask=ca, seed=s
                    )["mean_utility"]
                    for s in range(10)
                ]
                m_utils[meth] = round(float(np.mean(rs)), 4)
            best_m = max(phase_methods, key=lambda x: m_utils[x])
            phase_cells.append(
                {
                    "load_label": l_lbl,
                    "mu": mu_val,
                    "c_ask": ca,
                    "utils": m_utils,
                    "margin_vs_cascade": round(m_utils["Tri-VoI-L2D"] - m_utils["Calibrated-Cascade"], 4),
                    "margin_vs_cascade_naor": round(m_utils["Tri-VoI-L2D"] - m_utils["Calibrated-Cascade+Naor"], 4),
                    "margin_vs_greedy": round(m_utils["Tri-VoI-L2D"] - m_utils["Joint-Greedy"], 4),
                    "margin_vs_greedy_naor": round(m_utils["Tri-VoI-L2D"] - m_utils["Joint-Greedy+Naor"], 4),
                    "margin_vs_wo_ext": round(m_utils["Tri-VoI-L2D"] - m_utils["w/o Queue-Externality"], 4),
                    "margin_vs_best_baseline": round(
                        m_utils["Tri-VoI-L2D"] - max(m_utils[x] for x in phase_methods if x != "Tri-VoI-L2D"), 4
                    ),
                    "winner": best_m,
                }
            )

    # 5b. 5-Fold Cross-Validation across ALL models (420 items per model) + Calibration Curve on primary
    cv5_rows = []
    cv_methods = ["Naive-Cascade", "Joint-Greedy", "Calibrated-Cascade", "Tri-VoI-L2D"]
    for mid, pool in all_model_pools.items():
        combined = pool["cal"] + pool["test"]
        rng_cv = random.Random(777)
        by_q = defaultdict(list)
        for r in combined:
            by_q[r["item"]["quadrant"]].append(r)
        folds: List[List[Dict[str, Any]]] = [[] for _ in range(5)]
        for qk in sorted(by_q.keys()):
            bkt = list(by_q[qk])
            rng_cv.shuffle(bkt)
            for idx, r in enumerate(bkt):
                folds[idx % 5].append(r)

        m_scores: Dict[str, List[float]] = {m: [] for m in cv_methods}
        for f_idx in range(5):
            test_f = list(folds[f_idx])
            rng_cv.shuffle(test_f)
            cal_f = [r for k in range(5) if k != f_idx for r in folds[k]]
            cc_f = HeldOutCalibrator(cal_f)
            for m in cv_methods:
                us = [simulate_stream(m, test_f, cc_f, capacity=2, drain_interval=2, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)]
                m_scores[m].append(float(np.mean(us)))
        cv5_rows.append(
            {
                "model_id": mid,
                "short_name": mid.split("/")[-1],
                "n_total": len(combined),
                "methods": {
                    m: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v, ddof=1)), 3)}
                    for m, v in m_scores.items()
                },
            }
        )

    prim_combined = prim_cal + prim_test
    cal_curve_rows = []
    for n_cal in (16, 32, 48, 60, 100, 140):
        tri_vals = []
        cas_vals = []
        for rep in range(10):
            rr = random.Random(1200 + rep * 31 + n_cal)
            by_q = defaultdict(list)
            for r in prim_combined:
                by_q[r["item"]["quadrant"]].append(r)
            cal_s, test_s = [], []
            per_q = max(1, n_cal // 4)
            for qk in sorted(by_q.keys()):
                bkt = list(by_q[qk])
                rr.shuffle(bkt)
                take = per_q if qk.startswith("Q1") or qk.startswith("Q2") or qk.startswith("Q3") else max(1, per_q // 2)
                cal_s.extend(bkt[:take])
                test_s.extend(bkt[take:])
            rr.shuffle(test_s)
            cc_s = HeldOutCalibrator(cal_s)
            tri_vals.append(simulate_stream("Tri-VoI-L2D", test_s, cc_s, capacity=2, drain_interval=2, c_ask=1.5, seed=rep)["mean_utility"])
            cas_vals.append(simulate_stream("Calibrated-Cascade", test_s, cc_s, capacity=2, drain_interval=2, c_ask=1.5, seed=rep)["mean_utility"])
        cal_curve_rows.append(
            {
                "n_cal": len(cal_s),
                "tri_mean": round(float(np.mean(tri_vals)), 3),
                "tri_std": round(float(np.std(tri_vals, ddof=1)), 3),
                "cascade_mean": round(float(np.mean(cas_vals)), 3),
            }
        )

    # 5c. Leave-One-Domain-Out (LODO) Zero-Shot Domain Transfer across all models
    lodo_rows = []
    for mid, pool in all_model_pools.items():
        combined = pool["cal"] + pool["test"]
        dom_res: Dict[str, Any] = {}
        for hold_dom in DOMAINS:
            train_d = [r for r in combined if r["item"]["domain"] != hold_dom]
            test_d = [r for r in combined if r["item"]["domain"] == hold_dom]
            cc_d = HeldOutCalibrator(train_d)
            tri_u = float(np.mean([simulate_stream("Tri-VoI-L2D", test_d, cc_d, capacity=2, drain_interval=2, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)]))
            cas_u = float(np.mean([simulate_stream("Calibrated-Cascade", test_d, cc_d, capacity=2, drain_interval=2, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)]))
            nai_u = float(np.mean([simulate_stream("Naive-Cascade", test_d, cc_d, capacity=2, drain_interval=2, c_ask=1.5, seed=s)["mean_utility"] for s in range(3)]))
            dom_res[hold_dom] = {
                "n_test": len(test_d),
                "Tri-VoI-L2D": round(tri_u, 3),
                "Calibrated-Cascade": round(cas_u, 3),
                "Naive-Cascade": round(nai_u, 3),
            }
        lodo_rows.append({"model_id": mid, "short_name": mid.split("/")[-1], "domains": dom_res})

    # 5d-1. Bursty Poisson Traffic Waves + 3-Expert Heterogeneous Shift-Fatigue Pool
    bursty_summary = {}
    shift3_summary = {}
    stress_methods = [
        "Direct-ACT",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
    ]
    for m in stress_methods:
        rb = [_simulate_bursty_or_shift_stream(m, prim_test, prim_calib, mode="bursty", seed=s) for s in range(10)]
        r3 = [_simulate_bursty_or_shift_stream(m, prim_test, prim_calib, mode="shift3", seed=s) for s in range(10)]
        bursty_summary[m] = {
            "mean_utility": round(float(np.mean([x["mean_utility"] for x in rb])), 3),
            "acc_pct": round(float(np.mean([x["acc_pct"] for x in rb])), 2),
            "mean_wait": round(float(np.mean([x["mean_wait"] for x in rb])), 2),
        }
        shift3_summary[m] = {
            "mean_utility": round(float(np.mean([x["mean_utility"] for x in r3])), 3),
            "acc_pct": round(float(np.mean([x["acc_pct"] for x in r3])), 2),
            "mean_wait": round(float(np.mean([x["mean_wait"] for x in r3])), 2),
        }

    # 5d-2. Hyperparameter robustness: lambda_forget & gamma_model
    lam_sens = []
    for lv in (0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 1.00):
        vals = [_simulate_hyperparam_tri(prim_test, prim_calib, lam_forget=lv, seed=s) for s in range(5)]
        lam_sens.append({"lambda": lv, "mean_utility": round(float(np.mean(vals)), 3)})
    gamma_sens = []
    for gv in (0.30, 0.50, 0.85, 1.20, 1.80):
        vals = [_simulate_hyperparam_tri(prim_test, prim_calib, gamma_model=gv, gamma_env=0.85, seed=s) for s in range(5)]
        gamma_sens.append({"gamma_model": gv, "mean_utility": round(float(np.mean(vals)), 3)})

    # 5d-3. Token & Latency Cost-Efficiency Pareto on primary N=360
    pareto_rows = []
    avg_tok_greedy = float(np.mean([r["tokens_greedy"] for r in prim_test]))
    avg_tok_diag = float(np.mean([r["tokens_diag"] for r in prim_test]))
    avg_tok_post = float(np.mean([r["tokens_post"] for r in prim_test]))
    avg_tok_probe3 = float(np.mean([r["tokens"] - r["tokens_greedy"] - r["tokens_diag"] - r["tokens_post"] * 2 for r in prim_test]))
    avg_lat_g = float(np.mean([r["latency_greedy"] for r in prim_test]))
    avg_lat_d = float(np.mean([r["latency_diag"] for r in prim_test]))
    avg_lat_p = float(np.mean([r["latency_post"] for r in prim_test]))
    for m in [
        "Direct-ACT",
        "Entropy-ASK (Proxy)",
        "Consensus-DEFER (Proxy)",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "Tri-VoI-L2D",
    ]:
        sr = simulate_stream(m, prim_test, prim_calib, capacity=2, drain_interval=2, c_ask=1.5, seed=0)
        ask_f = sr["ask_rate_pct"] / 100.0
        if m == "Direct-ACT":
            tok = avg_tok_greedy
            lat = avg_lat_g
        elif m == "Entropy-ASK (Proxy)":
            tok = avg_tok_probe3 + ask_f * avg_tok_post
            lat = 3.0 * avg_lat_g + ask_f * avg_lat_p
        elif m == "Consensus-DEFER (Proxy)":
            tok = avg_tok_probe3
            lat = 3.0 * avg_lat_g
        elif m == "Naive-Cascade":
            tok = avg_tok_probe3 + avg_tok_diag + ask_f * (2.0 * avg_tok_post)
            lat = 3.0 * avg_lat_g + avg_lat_d + ask_f * (2.0 * avg_lat_p)
        else:
            tok = avg_tok_greedy + avg_tok_probe3 + avg_tok_diag + ask_f * avg_tok_post
            lat = avg_lat_g + avg_lat_d + ask_f * avg_lat_p
        pareto_rows.append(
            {
                "method": m,
                "mean_utility": round(sr["mean_utility"], 3),
                "acc_pct": round(sr["acc_pct"], 2),
                "avg_tokens_per_ticket": round(tok, 1),
                "est_latency_s_per_ticket": round(lat, 2),
                "util_per_1k_tokens": round(sr["mean_utility"] / max(0.1, tok / 1000.0), 3),
            }
        )

    # 5d-4. Error taxonomy
    err_tax = _audit_error_taxonomy(prim_test, prim_calib, c_ask=1.5, seed=0)

    # 5e. Decision-Relevant Subset (excluding decision-invariant Q2 tickets)
    dr_cal = [r for r in prim_cal if (not r["item"]["is_underspec"]) or r["item"].get("missing_changes_gold", True)]
    dr_test = [r for r in prim_test if (not r["item"]["is_underspec"]) or r["item"].get("missing_changes_gold", True)]
    dr_calib = HeldOutCalibrator(dr_cal)
    dr_methods = [
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "w/o Queue-Externality",
        "Tri-VoI-L2D",
    ]
    dr_regimes = [
        ("Cap=2, c=1.5", 2, 2, 1.5),
        ("Cap=2, c=2.5", 2, 2, 2.5),
        ("Cap=1/3, c=1.5", 1, 3, 1.5),
    ]
    dr_results: Dict[str, Any] = {
        "n_cal_orig": len(prim_cal),
        "n_cal_filtered": len(dr_cal),
        "n_test_orig": len(prim_test),
        "n_test_filtered": len(dr_test),
        "regimes": {},
    }
    for r_lbl, r_cap, r_di, r_ca in dr_regimes:
        r_map = {}
        for m in dr_methods:
            rs = [
                simulate_stream(
                    m, dr_test, dr_calib, capacity=r_cap, drain_interval=r_di, drift_mode="sudden", c_ask=r_ca, seed=s
                )
                for s in range(5)
            ]
            r_map[m] = {
                "mean_utility": round(float(np.mean([x["mean_utility"] for x in rs])), 3),
                "acc_pct": round(float(np.mean([x["acc_pct"] for x in rs])), 2),
                "mean_wait": round(float(np.mean([x["mean_wait"] for x in rs])), 2),
                "ask_rate_pct": round(float(np.mean([x["ask_rate_pct"] for x in rs])), 1),
            }
        dr_results["regimes"][r_lbl] = r_map

    # 5f. Realistic Specialist Simulation Stress (Delayed Feedback + Unclarified-Ticket Rework Penalty)
    spec_methods = [
        "Direct-ACT",
        "Congested-L2D-Only",
        "Naive-Cascade",
        "Joint-Greedy",
        "Joint-Greedy+Naor",
        "Calibrated-Cascade",
        "Calibrated-Cascade+Naor",
        "w/o Queue-Externality",
        "Tri-VoI-L2D",
    ]
    spec_cfgs = [
        ("Delayed Feedback (Cap=2)", 2, 2, True, False),
        ("Delayed Feedback (Cap=1/3)", 1, 3, True, False),
        ("Unclarified Rework (Cap=2)", 2, 2, False, True),
        ("Unclarified Rework (Cap=1/3)", 1, 3, False, True),
        ("Delayed + Rework (Cap=2)", 2, 2, True, True),
        ("Delayed + Rework (Cap=1/3)", 1, 3, True, True),
    ]
    spec_stress: Dict[str, Any] = {}
    for s_lbl, s_cap, s_di, d_fb, u_pen in spec_cfgs:
        s_map = {}
        for m in spec_methods:
            rs = [
                simulate_stream(
                    m,
                    prim_test,
                    prim_calib,
                    capacity=s_cap,
                    drain_interval=s_di,
                    drift_mode="sudden",
                    c_ask=1.5,
                    seed=s,
                    delayed_feedback=d_fb,
                    unclarified_spec_penalty=u_pen,
                )
                for s in range(5)
            ]
            s_map[m] = {
                "mean_utility": round(float(np.mean([x["mean_utility"] for x in rs])), 3),
                "acc_pct": round(float(np.mean([x["acc_pct"] for x in rs])), 2),
                "mean_wait": round(float(np.mean([x["mean_wait"] for x in rs])), 2),
            }
        spec_stress[s_lbl] = s_map

    return {
        "phase_map_56pt": phase_cells,
        "cv5_cross_model": cv5_rows,
        "calibration_curve_n": cal_curve_rows,
        "lodo_transfer": lodo_rows,
        "bursty_traffic_pool": bursty_summary,
        "shift3_hetero_pool": shift3_summary,
        "lambda_sensitivity": lam_sens,
        "gamma_sensitivity": gamma_sens,
        "token_latency_pareto": pareto_rows,
        "error_taxonomy": err_tax,
        "decision_relevant_subset": dr_results,
        "realistic_specialist_stress": spec_stress,
    }
