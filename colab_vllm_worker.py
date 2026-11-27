import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
PROMPTS_PATH = Path(os.environ.get("PROMPTS_PATH", "/content/wave_prompts.jsonl"))
OUT_DB = Path(os.environ.get("OUT_DB", "/content/out_cache.sqlite"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def ensure_vllm():
    try:
        import vllm  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("Installing vllm...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm"])


def ensure_db(db_path: Path):
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
    return conn


def main():
    ensure_vllm()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
    print(f"GPU={gpu_name} VRAM={vram_gb:.1f}GB MODEL={MODEL_ID}", flush=True)

    # Zero 4-bit quantization rule: use BF16 for <=16B or >=70GB VRAM, FP8 only for 35B on 40-80GB if needed
    quant = None
    dtype = "bfloat16"
    max_len = 4096
    if "35B" in MODEL_ID and vram_gb < 75.0:
        raise RuntimeError(f"VRAM {vram_gb:.1f}GB insufficient for zero-4bit {MODEL_ID}; fallback to SiliconFlow.")

    raw_rows = [json.loads(l) for l in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    conn = ensure_db(OUT_DB)
    existing_keys = {r[0] for r in conn.execute("SELECT key FROM cache_v1").fetchall()}
    todo = [r for r in raw_rows if r["key"] not in existing_keys]
    print(f"Loaded {len(raw_rows)} prompts, {len(existing_keys)} cached, {len(todo)} remaining.", flush=True)
    if not todo:
        conn.close()
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = LLM(
        model=MODEL_ID,
        dtype=dtype,
        quantization=quant,
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    groups = defaultdict(list)
    for row in todo:
        groups[(round(float(row["temperature"]), 4), int(row["max_tokens"]))].append(row)

    total_done = 0
    for (temp, max_tok), grp in groups.items():
        sp = SamplingParams(
            temperature=temp,
            max_tokens=max_tok,
            top_p=0.95 if temp > 0.0 else 1.0,
        )
        for i in range(0, len(grp), BATCH_SIZE):
            chunk = grp[i : i + BATCH_SIZE]
            formatted = []
            for row in chunk:
                try:
                    txt = tokenizer.apply_chat_template(
                        row["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    txt = tokenizer.apply_chat_template(
                        row["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                formatted.append(txt)

            t0 = time.time()
            outs = llm.generate(formatted, sp)
            dt_per = (time.time() - t0) / max(1, len(chunk))
            now = time.time()
            for row, out in zip(chunk, outs):
                raw_txt = out.outputs[0].text if out.outputs else ""
                cleaned = THINK_RE.sub("", raw_txt).strip()
                fr = str(out.outputs[0].finish_reason or "stop") if out.outputs else "stop"
                pt = len(out.prompt_token_ids or [])
                ct = len(out.outputs[0].token_ids or []) if out.outputs else 0
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_v1
                    (key, model, tag, content, finish_reason, prompt_tokens, completion_tokens, latency, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row["key"], row["model"], row["tag"], cleaned, fr, pt, ct, dt_per, now),
                )
            conn.commit()
            total_done += len(chunk)
            print(f"Completed {total_done}/{len(todo)}", flush=True)

    conn.close()
    print("Finished vLLM wave successfully.", flush=True)


if __name__ == "__main__":
    main()
