from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_json, write_json


STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("collect", "Dataset Collector"),
    ("clean", "Dataset Cleaner"),
    ("label", "Labeling Agent"),
    ("train", "Training Agent"),
    ("evaluate", "Evaluation Agent"),
    ("deploy", "Deployment Agent"),
)

VALID_STAGES = frozenset(name for name, _ in STAGE_ORDER)


def git_short_sha(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def hydrate_run_id_from_artifacts(config: dict[str, Any]) -> str | None:
    if config.get("run_id"):
        return str(config["run_id"])

    artifacts = Path(config["paths"]["artifacts_dir"])
    run_path = artifacts / "run.json"
    if run_path.exists():
        rid = read_json(run_path).get("run_id")
        if rid:
            config["run_id"] = rid
            return str(rid)

    for candidate in (
        artifacts / "eval_report.json",
        Path(config["paths"]["labeled_dir"]) / "stats.json",
        Path(config["paths"]["raw_dir"]) / "manifest.json",
    ):
        if candidate.exists():
            rid = read_json(candidate).get("run_id")
            if rid:
                config["run_id"] = rid
                return str(rid)
    return None


def ensure_run_id(config: dict[str, Any], *, project_root: Path | None = None) -> str:
    existing = config.get("run_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_short_sha(project_root)
    run_id = f"{ts}-{sha}" if sha else ts
    config["run_id"] = run_id
    return run_id


def write_run_record(config: dict[str, Any], *, stages: list[str], config_path: Path) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    record = {
        "run_id": config.get("run_id"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.as_posix()),
        "stages": stages,
    }
    write_json(artifacts / "run.json", record)


def parse_only(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        return None
    unknown = set(names) - VALID_STAGES
    if unknown:
        valid = ", ".join(sorted(VALID_STAGES))
        raise SystemExit(f"Unknown stage(s): {', '.join(sorted(unknown))}. Valid: {valid}")
    # Preserve canonical pipeline order
    order = {name: i for i, (name, _) in enumerate(STAGE_ORDER)}
    return sorted(names, key=lambda n: order[n])


def print_pipeline_summary(config: dict[str, Any]) -> bool | None:
    """Print recap from artifacts. Returns eval passed if report exists, else None."""
    paths = config["paths"]
    run_id = config.get("run_id")
    print("--- Pipeline summary ---")
    if run_id:
        print(f"  run_id: {run_id}")

    clean_stats_path = Path(paths["clean_dir"]) / "stats.json"
    if clean_stats_path.exists():
        clean = read_json(clean_stats_path)
        print(f"  clean rows: {clean.get('output_rows', '?')} (in {clean.get('input_rows', '?')})")

    label_stats_path = Path(paths["labeled_dir"]) / "stats.json"
    if label_stats_path.exists():
        label = read_json(label_stats_path)
        print(f"  labeled rows: {label.get('labeled_rows', '?')} ({label.get('mode_effective', '?')})")
        print(f"  llm_ok / fallback: {label.get('llm_success', 0)} / {label.get('heuristic_fallback', 0)}")
        if label.get("dpo_rows") is not None:
            print(f"  dpo pairs: {label.get('dpo_rows', 0)}")

    eval_path = Path(paths["artifacts_dir"]) / "eval_report.json"
    passed: bool | None = None
    if eval_path.exists():
        ev = read_json(eval_path)
        passed = bool(ev.get("passed"))
        status = "PASSED" if passed else "FAILED"
        print(f"  evaluation: {status}")
        if ev.get("issues"):
            for issue in ev["issues"][:5]:
                print(f"    - {issue}")
            extra = len(ev["issues"]) - 5
            if extra > 0:
                print(f"    - ... and {extra} more")

    print("------------------------")
    return passed


def exit_code_for_eval(ran_evaluate: bool, config: dict[str, Any]) -> int:
    if not ran_evaluate:
        return 0
    eval_path = Path(config["paths"]["artifacts_dir"]) / "eval_report.json"
    if not eval_path.exists():
        return 1
    ev = read_json(eval_path)
    return 0 if ev.get("passed") else 1
