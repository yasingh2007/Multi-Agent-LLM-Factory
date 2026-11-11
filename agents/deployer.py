from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_json, with_run_id, write_json


def run(config: dict[str, Any]) -> None:
    """Deployment Agent: package pointers + model card for a release bundle."""
    paths = config["paths"]
    deploy_cfg = config.get("deployment", {})

    pkg_dir = Path(deploy_cfg.get("package_dir") or Path(paths["artifacts_dir"]) / "deploy")
    pkg_dir.mkdir(parents=True, exist_ok=True)

    training_log = Path(paths["artifacts_dir"]) / "training_log.json"
    eval_report = Path(paths["artifacts_dir"]) / "eval_report.json"

    t = read_json(training_log) if training_log.exists() else {}
    e = read_json(eval_report) if eval_report.exists() else {}

    model_card = with_run_id(
        config,
        {
            "title": "Small LLM Training Factory — demo bundle",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training": t,
            "evaluation": e,
            "usage": "Point your inference server at checkpoint_dir from training_log; quantize as needed.",
        },
    )

    write_json(pkg_dir / "model_card.json", model_card)
    release = with_run_id(
        config,
        {
            "stage": "deployment_agent",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_dir": str(pkg_dir.as_posix()),
            "files": ["model_card.json"],
        },
    )
    write_json(pkg_dir / "release_manifest.json", release)
