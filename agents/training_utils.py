from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def data_provenance(paths: dict[str, Any], objective: str) -> dict[str, Any]:
    labeled = Path(paths["labeled_dir"])
    sft_path = labeled / "sft.jsonl"
    dpo_path = labeled / "dpo.jsonl"
    out: dict[str, Any] = {
        "sft.jsonl": {"path": str(sft_path.as_posix()), "sha256": sha256_file(sft_path)},
    }
    if objective == "dpo" or dpo_path.exists():
        out["dpo.jsonl"] = {"path": str(dpo_path.as_posix()), "sha256": sha256_file(dpo_path)}
    return out


def hyperparams_snapshot(train_cfg: dict[str, Any], objective: str) -> dict[str, Any]:
    keys = (
        "dry_run",
        "objective",
        "base_model",
        "output_dir",
        "num_train_epochs",
        "per_device_train_batch_size",
        "learning_rate",
        "max_length",
        "seed",
    )
    snap = {k: train_cfg[k] for k in keys if k in train_cfg}
    snap["objective"] = objective
    if objective == "dpo":
        snap["dpo"] = dict(train_cfg.get("dpo", {}))
    return snap


def resolve_checkpoint_dir(train_cfg: dict[str, Any], artifacts_dir: Path, objective: str) -> Path:
    base = Path(train_cfg.get("output_dir") or artifacts_dir / "checkpoints")
    return base / objective
