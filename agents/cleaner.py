from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_jsonl, with_run_id, write_json, write_jsonl


def _norm(text: str, lowercase: bool, strip_ws: bool) -> str:
    if lowercase:
        text = text.lower()
    if strip_ws:
        text = re.sub(r"\s+", " ", text.strip())
    return text


def run(config: dict[str, Any]) -> None:
    """Dataset Cleaner: normalize, dedupe, filter by length."""
    paths = config["paths"]
    clean_cfg = config.get("cleaner", {})

    raw_path = Path(paths["raw_dir"]) / "corpus.jsonl"
    clean_dir = Path(paths["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / "corpus.jsonl"
    stats_path = clean_dir / "stats.json"

    rows = read_jsonl(raw_path)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dropped: dict[str, int] = {"too_short": 0, "too_long": 0, "duplicate": 0, "missing_text": 0}

    min_chars = int(clean_cfg.get("min_chars", 1))
    max_chars = int(clean_cfg.get("max_chars", 100_000))
    lowercase = bool(clean_cfg.get("lowercase", False))
    strip_ws = bool(clean_cfg.get("strip_extra_whitespace", True))
    dedupe = bool(clean_cfg.get("dedupe", True))

    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            dropped["missing_text"] += 1
            continue
        normalized = _norm(text, lowercase, strip_ws)
        if len(normalized) < min_chars:
            dropped["too_short"] += 1
            continue
        if len(normalized) > max_chars:
            dropped["too_long"] += 1
            continue
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if dedupe and key in seen:
            dropped["duplicate"] += 1
            continue
        if dedupe:
            seen.add(key)
        new_row = {**row, "text": normalized}
        out.append(new_row)

    write_jsonl(clean_path, out)

    stats = {
        "stage": "cleaner",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(rows),
        "output_rows": len(out),
        "dropped": dropped,
        "clean_path": str(clean_path.as_posix()),
    }
    write_json(stats_path, with_run_id(config, stats))
