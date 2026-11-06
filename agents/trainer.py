from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._io import ensure_dir, read_jsonl, write_json


def _dry_run_train(config: dict[str, Any], sft_path: Path, ckpt_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(sft_path)
    training_log = {
        "stage": "training_agent",
        "mode": "dry_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "message": "Dry run: install torch+transformers and set training.dry_run false for real fine-tuning.",
        "checkpoint_dir": str(ckpt_dir.as_posix()),
    }
    ensure_dir(ckpt_dir)
    stub_path = ckpt_dir / "dry_run_checkpoint.json"
    stub_path.write_text(json.dumps({"status": "dry_run", "rows": len(rows)}, indent=2), encoding="utf-8")
    return training_log


def _real_train(config: dict[str, Any], sft_path: Path, ckpt_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from datasets import Dataset
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
    )

    class CausalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = CausalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    return {
        "stage": "training_agent",
        "mode": "real",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "sample_count": len(rows),
        "checkpoint_dir": str(ckpt_dir.as_posix()),
    }


def run(config: dict[str, Any]) -> None:
    """Training Agent: fine-tune a small causal LM or record a dry run."""
    paths = config["paths"]
    train_cfg = config.get("training", {})
    sft_path = Path(paths["labeled_dir"]) / "sft.jsonl"
    ckpt_dir = Path(train_cfg.get("output_dir") or Path(paths["artifacts_dir"]) / "checkpoints")
    ckpt_dir = Path(ckpt_dir)

    dry_run = bool(train_cfg.get("dry_run", True))

    if dry_run:
        training_log = _dry_run_train(config, sft_path, ckpt_dir)
    else:
        training_log = _real_train(config, sft_path, ckpt_dir)

    log_path = Path(paths["artifacts_dir"]) / "training_log.json"
    write_json(log_path, training_log)
