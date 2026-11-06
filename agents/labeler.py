from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_jsonl, write_json, write_jsonl


def _dummy_summary(text: str) -> str:
    """Cheap synthetic 'label' for demo pipelines — replace with an LLM or human workflow."""
    words = text.split()
    if not words:
        return ""
    tail = " ".join(words[-8:]) if len(words) > 8 else text
    return f"In short: {tail}"


def run(config: dict[str, Any]) -> None:
    """Labeling Agent: build instruction–response (SFT) rows from clean text."""
    paths = config["paths"]
    label_cfg = config.get("labeling", {})

    clean_path = Path(paths["clean_dir"]) / "corpus.jsonl"
    labeled_dir = Path(paths["labeled_dir"])
    labeled_dir.mkdir(parents=True, exist_ok=True)
    sft_path = labeled_dir / "sft.jsonl"
    stats_path = labeled_dir / "stats.json"

    instruction = str(label_cfg.get("instruction", "Answer based on the context."))
    input_key = str(label_cfg.get("input_key", "instruction"))
    output_key = str(label_cfg.get("output_key", "response"))

    rows = read_jsonl(clean_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        text = row.get("text", "")
        response = _dummy_summary(str(text))
        out.append(
            {
                "id": row.get("id"),
                input_key: instruction,
                "input": text,
                output_key: response,
                "meta": {"source_row": row},
            }
        )

    write_jsonl(sft_path, out)
    stats = {
        "stage": "labeler",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labeled_rows": len(out),
        "sft_path": str(sft_path.as_posix()),
    }
    write_json(stats_path, stats)
