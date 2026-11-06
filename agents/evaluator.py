from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_json, read_jsonl, write_json


def run(config: dict[str, Any]) -> None:
    """Evaluation Agent: simple dataset sanity checks + stub metrics."""
    paths = config["paths"]
    eval_cfg = config.get("evaluation", {})

    sft_path = Path(paths["labeled_dir"]) / "sft.jsonl"
    training_log_path = Path(paths["artifacts_dir"]) / "training_log.json"

    rows = read_jsonl(sft_path)
    min_samples = int(eval_cfg.get("min_train_samples", 1))

    issues: list[str] = []
    if len(rows) < min_samples:
        issues.append(f"Too few labeled samples: {len(rows)} < {min_samples}")

    label_cfg = config.get("labeling", {})
    output_key = str(label_cfg.get("output_key", "response"))
    empty_outputs = sum(1 for r in rows if not str(r.get(output_key, "")).strip())
    if empty_outputs:
        issues.append(f"Empty outputs: {empty_outputs}")

    training_meta = {}
    if training_log_path.exists():
        training_meta = read_json(training_log_path)

    passed = len(issues) == 0
    report = {
        "stage": "evaluation_agent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "sample_count": len(rows),
        "issues": issues,
        "training": training_meta,
        "notes": "Replace with task metrics, perplexity, or LLM-as-judge as you scale.",
    }
    out_path = Path(paths["artifacts_dir"]) / "eval_report.json"
    write_json(out_path, report)
