from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_json, read_jsonl, write_json
from .metrics import aggregate_metrics, summarize_row_metrics


def _check_gates(
    row_metrics: dict[str, float],
    gates: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    cr = row_metrics["compression_ratio"]
    tr = row_metrics["token_recall"]
    f1 = row_metrics["token_f1"]

    lo_cr = float(gates.get("min_compression_ratio", 0.02))
    hi_cr = float(gates.get("max_compression_ratio", 0.9))
    lo_tr = float(gates.get("min_token_recall", 0.1))
    hi_f1 = float(gates.get("max_token_f1", 0.85))

    if cr < lo_cr:
        issues.append(f"compression_ratio {cr} < {lo_cr}")
    if cr > hi_cr:
        issues.append(f"compression_ratio {cr} > {hi_cr}")
    if tr < lo_tr:
        issues.append(f"token_recall {tr} < {lo_tr}")
    if f1 > hi_f1:
        issues.append(f"token_f1 {f1} > {hi_f1} (near-copy)")
    return issues


def run(config: dict[str, Any]) -> None:
    """Evaluation Agent: dataset gates + summarization overlap metrics."""
    paths = config["paths"]
    eval_cfg = config.get("evaluation", {})

    sft_path = Path(paths["labeled_dir"]) / "sft.jsonl"
    dpo_path = Path(paths["labeled_dir"]) / "dpo.jsonl"
    training_log_path = Path(paths["artifacts_dir"]) / "training_log.json"

    rows = read_jsonl(sft_path)
    min_samples = int(eval_cfg.get("min_train_samples", 1))
    gates = eval_cfg.get("gates", {})
    label_cfg = config.get("labeling", {})
    output_key = str(label_cfg.get("output_key", "response"))

    issues: list[str] = []
    if len(rows) < min_samples:
        issues.append(f"Too few labeled samples: {len(rows)} < {min_samples}")

    empty_outputs = sum(1 for r in rows if not str(r.get(output_key, "")).strip())
    min_nonempty = float(gates.get("min_nonempty_rate", 1.0))
    nonempty_rate = 1.0 if not rows else 1.0 - (empty_outputs / len(rows))
    if nonempty_rate < min_nonempty:
        issues.append(f"Nonempty rate {nonempty_rate:.2f} < {min_nonempty}")

    per_row: list[dict[str, Any]] = []
    row_metric_list: list[dict[str, float]] = []
    row_failures = 0
    for row in rows:
        inp = str(row.get("input", ""))
        resp = str(row.get(output_key, ""))
        m = summarize_row_metrics(inp, resp)
        row_metric_list.append(m)
        row_issues = _check_gates(m, gates)
        if row_issues:
            row_failures += 1
        per_row.append({"id": row.get("id"), "metrics": m, "issues": row_issues})

    max_row_fail_fraction = float(gates.get("max_row_fail_fraction", 0.5))
    fail_fraction = (row_failures / len(rows)) if rows else 0.0
    if fail_fraction > max_row_fail_fraction:
        issues.append(
            f"Too many low-quality rows: {row_failures}/{len(rows)} "
            f"({fail_fraction:.0%}) > allowed {max_row_fail_fraction:.0%}"
        )

    dpo_stats: dict[str, Any] = {}
    if dpo_path.exists():
        dpo_rows = read_jsonl(dpo_path)
        bad_pairs = sum(
            1
            for r in dpo_rows
            if not str(r.get("chosen", "")).strip() or not str(r.get("rejected", "")).strip()
        )
        identical = sum(
            1
            for r in dpo_rows
            if str(r.get("chosen", "")).strip() == str(r.get("rejected", "")).strip()
        )
        dpo_stats = {
            "pairs": len(dpo_rows),
            "invalid_pairs": bad_pairs,
            "identical_chosen_rejected": identical,
        }
        if identical > 0:
            issues.append(f"DPO has {identical} identical chosen/rejected pairs")

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
        "aggregate_metrics": aggregate_metrics(row_metric_list),
        "row_failures": row_failures,
        "row_fail_fraction": round(fail_fraction, 4),
        "per_row": per_row,
        "dpo": dpo_stats,
        "training": training_meta,
    }
    out_path = Path(paths["artifacts_dir"]) / "eval_report.json"
    write_json(out_path, report)
