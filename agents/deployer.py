from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_json, with_run_id, write_json


def _copy_checkpoint(src: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    if (src / "config.json").exists():
        for item in src.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
            copied.append(item.name)
        return copied

    stub = src / "dry_run_checkpoint.json"
    if stub.exists():
        shutil.copy2(stub, dest / "dry_run_checkpoint.json")
        return ["dry_run_checkpoint.json"]

    return copied


def run(config: dict[str, Any]) -> None:
    """Deployment Agent: package model card + best checkpoint from eval selection."""
    paths = config["paths"]
    deploy_cfg = config.get("deployment", {})

    pkg_dir = Path(deploy_cfg.get("package_dir") or Path(paths["artifacts_dir"]) / "deploy")
    pkg_dir.mkdir(parents=True, exist_ok=True)

    training_log = Path(paths["artifacts_dir"]) / "training_log.json"
    eval_report = Path(paths["artifacts_dir"]) / "eval_report.json"

    t = read_json(training_log) if training_log.exists() else {}
    e = read_json(eval_report) if eval_report.exists() else {}
    selection = e.get("selected_checkpoint") or {}

    model_bundle_dir = pkg_dir / "model"
    packaged_files: list[str] = ["model_card.json", "release_manifest.json", "checkpoint_selection.json"]

    checkpoint_selection = {
        "selected": selection,
        "fallback_checkpoint_dir": t.get("checkpoint_dir"),
    }
    write_json(pkg_dir / "checkpoint_selection.json", with_run_id(config, checkpoint_selection))

    copied: list[str] = []
    src_path = selection.get("path") or t.get("checkpoint_dir")
    if src_path:
        src = Path(str(src_path))
        if selection.get("deployable"):
            copied = _copy_checkpoint(src, model_bundle_dir)
        elif (src / "dry_run_checkpoint.json").exists():
            copied = _copy_checkpoint(src, model_bundle_dir)

    model_card = with_run_id(
        config,
        {
            "title": "Small LLM Training Factory — demo bundle",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training": t,
            "evaluation": {
                "passed": e.get("passed"),
                "aggregate_metrics": e.get("aggregate_metrics"),
                "selected_checkpoint": selection,
            },
            "checkpoint_bundle_dir": str(model_bundle_dir.as_posix()) if copied else None,
            "usage": (
                "Use model/ under the deploy package when deployable; "
                "otherwise retrain with training.dry_run false."
            ),
        },
    )

    write_json(pkg_dir / "model_card.json", model_card)
    if copied:
        packaged_files.append("model/")
        packaged_files.extend(f"model/{name}" for name in copied)

    release = with_run_id(
        config,
        {
            "stage": "deployment_agent",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_dir": str(pkg_dir.as_posix()),
            "checkpoint_path": selection.get("path"),
            "deployable": selection.get("deployable", False),
            "files": packaged_files,
        },
    )
    write_json(pkg_dir / "release_manifest.json", release)
