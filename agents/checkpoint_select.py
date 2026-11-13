from __future__ import annotations

from pathlib import Path
from typing import Any


def _checkpoint_step(path: Path) -> int:
    if path.name.startswith("checkpoint-"):
        try:
            return int(path.name.split("-", 1)[1])
        except (ValueError, IndexError):
            return 0
    if (path / "config.json").exists():
        return 10**9
    return -1


def list_checkpoint_candidates(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []

    candidates: list[Path] = []
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and (child / "config.json").exists():
            candidates.append(child)
    if (checkpoint_dir / "config.json").exists():
        candidates.append(checkpoint_dir)
    if not candidates and (checkpoint_dir / "dry_run_checkpoint.json").exists():
        candidates.append(checkpoint_dir)
    return candidates


def select_checkpoint(
    training_meta: dict[str, Any],
    *,
    eval_passed: bool,
    aggregate_metrics: dict[str, Any],
) -> dict[str, Any]:
    ckpt_raw = training_meta.get("checkpoint_dir")
    if not ckpt_raw:
        return {
            "path": None,
            "kind": "missing",
            "deployable": False,
            "reason": "training_log has no checkpoint_dir",
        }

    ckpt_dir = Path(str(ckpt_raw))
    candidates = list_checkpoint_candidates(ckpt_dir)
    if not candidates:
        return {
            "path": str(ckpt_dir.as_posix()),
            "kind": "missing",
            "deployable": False,
            "reason": "no weight files under checkpoint_dir",
        }

    selected = max(candidates, key=_checkpoint_step)
    has_weights = (selected / "config.json").exists()
    is_stub = (selected / "dry_run_checkpoint.json").exists() and not has_weights

    rouge = aggregate_metrics.get("rouge_l_f1", {}).get("mean")
    quality_score = float(rouge) if rouge is not None else None

    deployable = bool(eval_passed and has_weights)
    return {
        "path": str(selected.as_posix()),
        "kind": "dry_run_stub" if is_stub else ("weights" if has_weights else "unknown"),
        "criterion": "newest_checkpoint_subdir",
        "eval_passed": eval_passed,
        "quality_score": quality_score,
        "deployable": deployable,
        "reason": "stub only (dry_run)" if is_stub else "selected newest checkpoint with weights",
    }
