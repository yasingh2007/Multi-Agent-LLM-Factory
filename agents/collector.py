from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import ensure_dir, read_jsonl, write_json, write_jsonl


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run(config: dict[str, Any]) -> None:
    """Dataset Collector: seed or synthesize raw corpus + manifest."""
    paths = config["paths"]
    raw_dir = Path(paths["raw_dir"])
    ensure_dir(raw_dir)

    collector_cfg = config.get("collector", {})
    seed_file = collector_cfg.get("seed_file")

    corpus_path = raw_dir / "corpus.jsonl"
    manifest_path = raw_dir / "manifest.json"

    if seed_file:
        src = Path(seed_file)
        if not src.exists():
            raise FileNotFoundError(f"Collector seed file not found: {src}")
        shutil.copy2(src, corpus_path)
    else:
        # Minimal synthetic fallback if no seed is configured
        write_jsonl(
            corpus_path,
            [
                {"id": "syn-1", "text": "Sample line one.", "source": "synthetic"},
                {"id": "syn-2", "text": "Sample line two.", "source": "synthetic"},
            ],
        )

    rows = read_jsonl(corpus_path)
    created_at = datetime.now(timezone.utc).isoformat()
    content_hash = _sha256_file(corpus_path)

    manifest: dict[str, Any] = {
        "stage": "collector",
        "created_at": created_at,
        "row_count": len(rows),
        "corpus_path": str(corpus_path.as_posix()),
        "content_sha256": content_hash,
        "extra": collector_cfg.get("manifest_extra", {}),
    }
    write_json(manifest_path, manifest)
