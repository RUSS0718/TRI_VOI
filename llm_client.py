import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import winreg
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CALLS_LOG = RESULTS_DIR / "calls.jsonl"

_LOCK = threading.Lock()
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


class SiliconFlowAuthOrBalanceError(RuntimeError):
    """Raised immediately on HTTP 401/402 from SiliconFlow."""


def load_api_key() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
            val, _ = winreg.QueryValueEx(k, "API_KEY")
            if isinstance(val, str) and val.strip():
                return val.strip()
    except OSError:
        pass
    env_val = os.environ.get("API_KEY", "").strip()
    if env_val:
        return env_val
    raise SiliconFlowAuthOrBalanceError("No API_KEY found in HKCU\\Environment or process env.")


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip() if text else ""


def _cache_path_for(model_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)
    return RESULTS_DIR / f"cache_{safe}.sqlite"


def _ensure_db(db_path: Path) -> None:
    with _LOCK:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_v1 (
                key TEXT PRIMARY KEY,
                model TEXT,
                tag TEXT,
                content TEXT,
                finish_reason TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                latency REAL,
                created_at REAL
            )
            """
        )
        conn.commit()
        conn.close()


def _make_key(
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    tag: str,
    seed_idx: int,
) -> str:
    payload = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "temperature": round(temperature, 4),
            "max_tokens": max_tokens,
            "tag": tag,
            "seed_idx": seed_idx,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(db_path: Path, key: str) -> Optional[Dict[str, Any]]:
    _ensure_db(db_path)
    with _LOCK:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        row = conn.execute(
            "SELECT content, finish_reason, prompt_tokens, completion_tokens, latency FROM cache_v1 WHERE key=?",
            (key,),
        ).fetchone()
        conn.close()
    if row is None:
        return None
    return {
        "content": row[0],
        "finish_reason": row[1],
        "prompt_tokens": row[2],
        "completion_tokens": row[3],
        "latency": row[4],
        "cached": True,
    }


def _cache_put(
    db_path: Path,
    key: str,
    model_id: str,
    tag: str,
    content: str,
    finish_reason: str,
    pt: int,
    ct: int,
    dt: float,
) -> None:
    _ensure_db(db_path)
    with _LOCK:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.execute(
            """
            INSERT OR REPLACE INTO cache_v1
            (key, model, tag, content, finish_reason, prompt_tokens, completion_tokens, latency, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key, model_id, tag, content, finish_reason, pt, ct, dt, time.time()),
        )
        conn.commit()
        conn.close()


def _log_call(record: Dict[str, Any]) -> None:
    with _LOCK:
        with open(CALLS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mock_response(model_id: str, messages: List[Dict[str, str]], tag: str, seed_idx: int) -> Dict[str, Any]:
    """Deterministic realistic mock for --dry-run verification only."""
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    h = int(hashlib.sha256(f"{model_id}|{user_text}|{tag}|{seed_idx}".encode("utf-8")).hexdigest(), 16)
    h_model = int(hashlib.sha256(f"{model_id}|{user_text}|base".encode("utf-8")).hexdigest(), 16)

    m_quad = re.search(r"\[DRY_META:([A-Za-z0-9_]+):([A-C])\]", user_text)
    quad = m_quad.group(1) if m_quad else "Q1_clear_easy"
    true_ans = m_quad.group(2) if m_quad else "A"
    user_revealed = "[Requester Follow-up]:" in user_text and "Clarification provided:" in user_text
    choices = ["A", "B", "C"]

    is_missing = quad in ("Q2_underspec_easy", "Q4_resolvable", "Q4_escalate") and not user_revealed
    is_pot_ovr = quad in ("Q3_clear_hard", "Q4_resolvable", "Q4_escalate")
    is_confirmed_ovr = (quad == "Q3_clear_hard") or (quad == "Q4_escalate" and user_revealed)

    # Model-dependent accuracy scale (8B slightly lower than 35B/DeepSeek)
    base_easy_ok = 84 if "8B" in model_id else 92

    if tag == "diag_init":
        # Simulate ~88% accurate source diagnosis on natural language tickets
        det_miss = is_missing if (h % 100) < 88 else (not is_missing)
        det_pot = is_pot_ovr if ((h >> 4) % 100) < 90 else (not is_pot_ovr)
        det_conf = (quad == "Q3_clear_hard") if ((h >> 8) % 100) < 88 else False
        q_ask = "Could you confirm the missing required triage/compliance field?" if det_miss else ""
        content = json.dumps(
            {
                "missing_user_info": det_miss,
                "potential_override": det_pot,
                "override_confirmed": det_conf,
                "clarifying_question": q_ask,
            }
        )
    elif tag == "post_ask_combined":
        if is_confirmed_ovr:
            ans = true_ans if (h % 100) < 38 else choices[(choices.index(true_ans) + 1) % 3]
        else:
            ans = true_ans if (h % 100) < base_easy_ok else choices[(choices.index(true_ans) + 1) % 3]
        det_conf = is_confirmed_ovr if ((h >> 6) % 100) < 90 else (not is_confirmed_ovr)
        content = json.dumps(
            {
                "answer": ans,
                "missing_user_info": False,
                "override_confirmed": det_conf,
            }
        )
    else:
        # Probe / greedy / post_ask_s1
        if tag.startswith("post_ask"):
            p_ok = 38 if is_confirmed_ovr else base_easy_ok
            ans = true_ans if (h % 100) < p_ok else choices[(choices.index(true_ans) + 1 + seed_idx) % 3]
        elif is_missing:
            # High-consensus entropy collapse: 72% of underspecified items collapse deterministically onto a single guess
            if (h_model % 100) < 72:
                ans = choices[h_model % 3]
            else:
                ans = choices[(h_model + seed_idx) % 3]
        elif quad == "Q3_clear_hard":
            ans = true_ans if (h % 100) < 38 else choices[(choices.index(true_ans) + 1 + seed_idx) % 3]
        else:
            ans = true_ans if (h % 100) < base_easy_ok else choices[(choices.index(true_ans) + 1) % 3]
        content = json.dumps({"answer": ans, "rationale": "Evaluated against policy rubric."})

    return {
        "content": content,
        "finish_reason": "stop",
        "prompt_tokens": max(110, len(user_text) // 4),
        "completion_tokens": 42,
        "latency": 0.005,
        "cached": False,
        "dry_run": True,
    }


def call_llm(
    messages: List[Dict[str, str]],
    *,
    model_id: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tag: str = "default",
    seed_idx: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return _mock_response(model_id, messages, tag, seed_idx)

    eff_max = max(max_tokens, 1024)
    db_path = _cache_path_for(model_id)
    key = _make_key(model_id, messages, temperature, eff_max, tag, seed_idx)
    cached = _cache_get(db_path, key)
    if cached is not None:
        return cached

    api_key = load_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": eff_max,
    }
    if any(k in model_id.lower() for k in ["qwen3", "deepseek-v3.2", "glm-4.5", "glm-5", "hy4"]):
        payload["enable_thinking"] = False

    for attempt in range(8):
        t0 = time.time()
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        except requests.RequestException as exc:
            if attempt == 7:
                raise RuntimeError(f"SiliconFlow request failed for {model_id}: {exc}") from exc
            time.sleep(min(30.0, 2.5 * (2**attempt)))
            continue

        dt = time.time() - t0
        if resp.status_code in (401, 402):
            raise SiliconFlowAuthOrBalanceError(f"SiliconFlow HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 400 and "enable_thinking" in resp.text.lower() and "enable_thinking" in payload:
            payload.pop("enable_thinking", None)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == 7:
                raise RuntimeError(f"SiliconFlow HTTP {resp.status_code} ({model_id}): {resp.text[:300]}")
            wait_s = min(62.0, 5.0 * (2**attempt)) if resp.status_code == 429 else min(30.0, 3.0 * (2**attempt))
            time.sleep(wait_s)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"SiliconFlow HTTP {resp.status_code} ({model_id}): {resp.text[:300]}")

        data = resp.json()
        choice = data["choices"][0]
        raw_content = choice["message"].get("content") or ""
        cleaned = strip_thinking(raw_content)
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        fr = str(choice.get("finish_reason", "stop"))

        _cache_put(db_path, key, model_id, tag, cleaned, fr, pt, ct, dt)
        _log_call(
            {
                "ts": time.time(),
                "model": model_id,
                "tag": tag,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "latency": round(dt, 3),
            }
        )
        return {
            "content": cleaned,
            "finish_reason": fr,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency": dt,
            "cached": False,
        }

    raise RuntimeError(f"Exhausted retries calling {model_id}.")
