from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import ensure_dir, read_jsonl, with_run_id, write_json
from .training_utils import data_provenance, hyperparams_snapshot, resolve_checkpoint_dir, sha256_file


def _base_log_fields(
    config: dict[str, Any],
    *,
    objective: str,
    mode: str,
    ckpt_dir: Path,
    sample_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    paths = config["paths"]
    labeled = Path(paths["labeled_dir"])
    sft_path = labeled / "sft.jsonl"
    dpo_path = labeled / "dpo.jsonl"

    log: dict[str, Any] = {
        "stage": "training_agent",
        "mode": mode,
        "objective": objective,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": sample_count,
        "checkpoint_dir": str(ckpt_dir.as_posix()),
        "hyperparams": hyperparams_snapshot(train_cfg, objective),
        "data_provenance": data_provenance(paths, objective),
    }
    if extra:
        log.update(extra)
    return log


def _dry_run_train(
    config: dict[str, Any],
    *,
    objective: str,
    ckpt_dir: Path,
) -> dict[str, Any]:
    paths = config["paths"]
    labeled = Path(paths["labeled_dir"])
    if objective == "dpo":
        data_path = labeled / "dpo.jsonl"
        message = (
            "Dry run (dpo): install torch, transformers, datasets, trl and set training.dry_run false."
        )
    else:
        data_path = labeled / "sft.jsonl"
        message = (
            "Dry run (sft): install torch, transformers, datasets and set training.dry_run false."
        )

    rows = read_jsonl(data_path)
    ensure_dir(ckpt_dir)
    stub_path = ckpt_dir / "dry_run_checkpoint.json"
    stub_path.write_text(
        json.dumps(
            {"status": "dry_run", "objective": objective, "rows": len(rows), "data_sha256": sha256_file(data_path)},
            indent=2,
        ),
        encoding="utf-8",
    )

    return _base_log_fields(
        config,
        objective=objective,
        mode="dry_run",
        ckpt_dir=ckpt_dir,
        sample_count=len(rows),
        extra={"message": message},
    )


def _real_sft_train(config: dict[str, Any], sft_path: Path, ckpt_dir: Path) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "training.dry_run is false but transformers/torch/datasets are not installed. "
            "pip install torch transformers datasets accelerate"
        ) from e

    train_cfg = config.get("training", {})
    label_cfg = config.get("labeling", {})
    input_key = str(label_cfg.get("input_key", "instruction"))
    output_key = str(label_cfg.get("output_key", "response"))

    base_model = str(train_cfg.get("base_model", "sshleifer/tiny-gpt2"))
    max_length = int(train_cfg.get("max_length", 128))

    rows = read_jsonl(sft_path)

    def format_example(row: dict[str, Any]) -> dict[str, str]:
        instr = str(row.get(input_key, ""))
        inp = str(row.get("input", ""))
        out = str(row.get(output_key, ""))
        text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        return {"text": text}

    ds = Dataset.from_list([format_example(r) for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[str]]):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForCausalLM.from_pretrained(base_model)
    ensure_dir(ckpt_dir)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        logging_steps=1,
        save_strategy="epoch",
        seed=int(train_cfg.get("seed", 42)),
    )

    class CausalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = CausalTrainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    return _base_log_fields(
        config,
        objective="sft",
        mode="real",
        ckpt_dir=ckpt_dir,
        sample_count=len(rows),
        extra={"base_model": base_model},
    )


def _real_dpo_train(config: dict[str, Any], dpo_path: Path, ckpt_dir: Path) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "training.objective is dpo but trl/transformers/torch/datasets are not installed. "
            "pip install torch transformers datasets accelerate trl"
        ) from e

    train_cfg = config.get("training", {})
    dpo_cfg = train_cfg.get("dpo", {})

    base_model = str(train_cfg.get("base_model", "sshleifer/tiny-gpt2"))
    max_length = int(dpo_cfg.get("max_length", train_cfg.get("max_length", 128)))
    max_prompt_length = int(dpo_cfg.get("max_prompt_length", max_length // 2))

    rows = read_jsonl(dpo_path)
    if not rows:
        raise RuntimeError(f"No DPO rows found at {dpo_path}")

    ds = Dataset.from_list(
        [
            {
                "prompt": str(r.get("prompt", "")),
                "chosen": str(r.get("chosen", "")),
                "rejected": str(r.get("rejected", "")),
            }
            for r in rows
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    ensure_dir(ckpt_dir)

    training_args = DPOConfig(
        output_dir=str(ckpt_dir),
        beta=float(dpo_cfg.get("beta", 0.1)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        logging_steps=1,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        save_strategy="epoch",
        seed=int(train_cfg.get("seed", 42)),
    )

    try:
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            tokenizer=tokenizer,
        )
    trainer.train()
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    return _base_log_fields(
        config,
        objective="dpo",
        mode="real",
        ckpt_dir=ckpt_dir,
        sample_count=len(rows),
        extra={"base_model": base_model},
    )


def run(config: dict[str, Any]) -> None:
    """Training Agent: SFT or DPO fine-tune (or dry-run stub)."""
    paths = config["paths"]
    train_cfg = config.get("training", {})
    objective = str(train_cfg.get("objective", "sft")).lower()
    if objective not in {"sft", "dpo"}:
        raise ValueError(f"training.objective must be 'sft' or 'dpo', got {objective!r}")

    artifacts = Path(paths["artifacts_dir"])
    ckpt_dir = resolve_checkpoint_dir(train_cfg, artifacts, objective)
    labeled = Path(paths["labeled_dir"])
    sft_path = labeled / "sft.jsonl"
    dpo_path = labeled / "dpo.jsonl"

    dry_run = bool(train_cfg.get("dry_run", True))

    if dry_run:
        training_log = _dry_run_train(config, objective=objective, ckpt_dir=ckpt_dir)
    elif objective == "dpo":
        training_log = _real_dpo_train(config, dpo_path, ckpt_dir)
    else:
        training_log = _real_sft_train(config, sft_path, ckpt_dir)

    log_path = artifacts / "training_log.json"
    write_json(log_path, with_run_id(config, training_log))
