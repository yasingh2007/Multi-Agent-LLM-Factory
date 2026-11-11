from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import read_jsonl, with_run_id, write_json, write_jsonl
from .llm_client import LLMError, chat_completion, resolve_llm_settings


def _dummy_summary(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    tail = " ".join(words[-8:]) if len(words) > 8 else text
    return f"In short: {tail}"


def _weak_rejected(text: str) -> str:
    words = text.split()
    if len(words) >= 4:
        return " ".join(words[:3]) + "..."
    return "Not sure."


def _rejected_response(text: str, strategy: str) -> str:
    if strategy == "tail":
        return _dummy_summary(text)
    return _weak_rejected(text)


def _format_prompt(instruction: str, text: str) -> str:
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{text}\n\n"
        f"### Response:\n"
    )


def _label_with_llm(
    instruction: str,
    text: str,
    llm_cfg: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    settings = resolve_llm_settings(llm_cfg)
    system = str(
        llm_cfg.get(
            "system_prompt",
            "You label training data. Reply with only the final answer—no quotes or explanation.",
        )
    )
    user_template = str(
        llm_cfg.get(
            "user_template",
            "{instruction}\n\nContext:\n{input}\n\nAnswer:",
        )
    )
    user = user_template.format(instruction=instruction, input=text)
    content = chat_completion(settings=settings, system=system, user=user)
    meta = {"provider": "llm", "model": settings["model"]}
    return content, meta


def run(config: dict[str, Any]) -> None:
    """Labeling Agent: SFT rows + optional DPO pairs (chosen vs rejected)."""
    paths = config["paths"]
    label_cfg = config.get("labeling", {})

    clean_path = Path(paths["clean_dir"]) / "corpus.jsonl"
    labeled_dir = Path(paths["labeled_dir"])
    labeled_dir.mkdir(parents=True, exist_ok=True)
    sft_path = labeled_dir / "sft.jsonl"
    dpo_path = labeled_dir / "dpo.jsonl"
    stats_path = labeled_dir / "stats.json"

    instruction = str(label_cfg.get("instruction", "Answer based on the context."))
    input_key = str(label_cfg.get("input_key", "instruction"))
    output_key = str(label_cfg.get("output_key", "response"))
    mode = str(label_cfg.get("mode", "heuristic")).lower()
    llm_cfg = label_cfg.get("llm", {})
    dpo_cfg = label_cfg.get("dpo", {})
    dpo_enabled = bool(dpo_cfg.get("enabled", True))

    rows = read_jsonl(clean_path)
    sft_out: list[dict[str, Any]] = []
    dpo_out: list[dict[str, Any]] = []

    llm_ok = 0
    llm_errors: list[str] = []
    heuristic_fallback = 0

    use_llm = mode == "llm"
    llm_settings = resolve_llm_settings(llm_cfg) if use_llm else {}
    if use_llm and not llm_settings.get("api_key"):
        use_llm = False
        heuristic_fallback = len(rows)

    for row in rows:
        text = str(row.get("text", ""))
        label_meta: dict[str, Any] = {"source_row": row}

        if use_llm:
            try:
                response, provider_meta = _label_with_llm(instruction, text, llm_cfg)
                label_meta["labeler"] = provider_meta
                llm_ok += 1
            except LLMError as e:
                llm_errors.append(f"{row.get('id')}: {e}")
                response = _dummy_summary(text)
                label_meta["labeler"] = {"provider": "heuristic_fallback", "error": str(e)}
                heuristic_fallback += 1
        else:
            response = _dummy_summary(text)
            label_meta["labeler"] = {"provider": "heuristic"}

        sft_out.append(
            {
                "id": row.get("id"),
                input_key: instruction,
                "input": text,
                output_key: response,
                "meta": label_meta,
            }
        )

        if dpo_enabled:
            rejected_strategy = str(dpo_cfg.get("rejected_strategy", "weak"))
            chosen = response
            rejected = _rejected_response(text, rejected_strategy)
            dpo_out.append(
                {
                    "id": row.get("id"),
                    "prompt": _format_prompt(instruction, text),
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "task": "summarization",
                        "source_id": row.get("id"),
                    },
                }
            )

    write_jsonl(sft_path, sft_out)
    if dpo_enabled:
        write_jsonl(dpo_path, dpo_out)

    stats = {
        "stage": "labeler",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_requested": mode,
        "mode_effective": "llm" if use_llm else "heuristic",
        "labeled_rows": len(sft_out),
        "dpo_rows": len(dpo_out) if dpo_enabled else 0,
        "llm_success": llm_ok,
        "llm_error_count": len(llm_errors),
        "heuristic_fallback": heuristic_fallback,
        "llm_errors_sample": llm_errors[:5],
        "sft_path": str(sft_path.as_posix()),
        "dpo_path": str(dpo_path.as_posix()) if dpo_enabled else None,
    }
    write_json(stats_path, with_run_id(config, stats))
