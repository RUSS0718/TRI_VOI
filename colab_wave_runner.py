import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark import build_cscw_benchmark_v2
from controller import parse_json_obj
from llm_client import CALLS_LOG, _cache_get, _cache_path_for, _ensure_db, _make_key
from probe_profiles import build_wave_a_specs, build_wave_b_specs
from public_tri import load_public_tri_dataset


BASE_DIR = Path(__file__).resolve().parent
BATCH_DIR = BASE_DIR / "results" / "colab_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(model_id: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def get_suite_items(suite: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if suite in ("v2", "both"):
        cal_v2, test_v2 = build_cscw_benchmark_v2(n_cal_per_quad=15, n_test_per_quad=45, seed=2027)
        items.extend(cal_v2)
        items.extend(test_v2)
    if suite in ("public_tri", "both"):
        cal_pt, test_pt = load_public_tri_dataset()
        items.extend(cal_pt)
        items.extend(test_pt)
    return items


def export_wave_a(model_id: str, suite: str, out_path: Path) -> int:
    db_path = _cache_path_for(model_id)
    _ensure_db(db_path)
    items = get_suite_items(suite)
    rows: List[str] = []
    seen_keys = set()
    for it in items:
        for spec in build_wave_a_specs(it, dry_run=False):
            key = _make_key(
                model_id,
                spec["messages"],
                spec["temperature"],
                spec["max_tokens"],
                spec["tag"],
                spec["seed_idx"],
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if _cache_get(db_path, key) is not None:
                continue
            rows.append(
                json.dumps(
                    {
                        "key": key,
                        "model": model_id,
                        "messages": spec["messages"],
                        "temperature": spec["temperature"],
                        "max_tokens": spec["max_tokens"],
                        "tag": spec["tag"],
                        "seed_idx": spec["seed_idx"],
                    },
                    ensure_ascii=False,
                )
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"[Wave A Export] model={model_id} suite={suite} uncached={len(rows)} -> {out_path}")
    return len(rows)


def export_wave_b(model_id: str, suite: str, out_path: Path) -> int:
    db_path = _cache_path_for(model_id)
    _ensure_db(db_path)
    items = get_suite_items(suite)
    rows: List[str] = []
    seen_keys = set()
    missing_diag = 0
    for it in items:
        specs_a = build_wave_a_specs(it, dry_run=False)
        diag_spec = [s for s in specs_a if s["tag"] == "diag_init"][0]
        d_key = _make_key(
            model_id,
            diag_spec["messages"],
            diag_spec["temperature"],
            diag_spec["max_tokens"],
            diag_spec["tag"],
            diag_spec["seed_idx"],
        )
        cached_d = _cache_get(db_path, d_key)
        if cached_d is None:
            missing_diag += 1
            continue
        diag_obj = parse_json_obj(cached_d["content"])
        clarify_q = str(diag_obj.get("clarifying_question", "")).strip()
        if not clarify_q:
            continue
        _, _, specs_b = build_wave_b_specs(it, clarify_q, dry_run=False, include_acl_voi=True)
        for spec in specs_b:
            key = _make_key(
                model_id,
                spec["messages"],
                spec["temperature"],
                spec["max_tokens"],
                spec["tag"],
                spec["seed_idx"],
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if _cache_get(db_path, key) is not None:
                continue
            rows.append(
                json.dumps(
                    {
                        "key": key,
                        "model": model_id,
                        "messages": spec["messages"],
                        "temperature": spec["temperature"],
                        "max_tokens": spec["max_tokens"],
                        "tag": spec["tag"],
                        "seed_idx": spec["seed_idx"],
                    },
                    ensure_ascii=False,
                )
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"[Wave B Export] model={model_id} suite={suite} uncached={len(rows)} missing_diag={missing_diag} -> {out_path}")
    return len(rows)


def merge_sqlite(model_id: str, remote_sqlite: Path) -> int:
    local_db = _cache_path_for(model_id)
    _ensure_db(local_db)
    src = sqlite3.connect(str(remote_sqlite), timeout=60.0)
    dst = sqlite3.connect(str(local_db), timeout=60.0)
    rows = src.execute(
        "SELECT key, model, tag, content, finish_reason, prompt_tokens, completion_tokens, latency, created_at FROM cache_v1"
    ).fetchall()
    inserted = 0
    log_lines = []
    for row in rows:
        cur = dst.execute(
            """
            INSERT OR IGNORE INTO cache_v1
            (key, model, tag, content, finish_reason, prompt_tokens, completion_tokens, latency, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        if cur.rowcount > 0:
            inserted += 1
            log_lines.append(
                json.dumps(
                    {
                        "ts": row[8] or time.time(),
                        "model": row[1],
                        "tag": row[2],
                        "prompt_tokens": row[5],
                        "completion_tokens": row[6],
                        "latency": round(float(row[7] or 0.0), 3),
                        "backend": "colab_vllm",
                    },
                    ensure_ascii=False,
                )
            )
    dst.commit()
    src.close()
    dst.close()
    if log_lines:
        with open(CALLS_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
    print(f"[Merge SQLite] {remote_sqlite.name} -> {local_db.name}: total_rows={len(rows)}, newly_inserted={inserted}")
    return inserted


def main() -> None:
    p = argparse.ArgumentParser(description="Two-Wave Colab Batch Exporter & Merger")
    p.add_argument("action", choices=["export_a", "export_b", "merge"])
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="both", choices=["v2", "public_tri", "both"])
    p.add_argument("--path", required=True)
    args = p.parse_args()

    path = Path(args.path)
    if args.action == "export_a":
        export_wave_a(args.model, args.suite, path)
    elif args.action == "export_b":
        export_wave_b(args.model, args.suite, path)
    else:
        merge_sqlite(args.model, path)


if __name__ == "__main__":
    main()
