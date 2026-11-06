# Small LLM Training Factory

A **multi-agent pipeline** for collecting data, cleaning it, labeling, training small models, evaluating, and packaging for deployment. Each stage is a separate Python module with a clear artifact contract.

## Agents

| Agent | Role | Reads | Writes |
|-------|------|-------|--------|
| **Dataset Collector** | Ingests raw sources into a versioned corpus + manifest | Config, optional URL list / files | `data/raw/corpus.jsonl`, `data/raw/manifest.json` |
| **Dataset Cleaner** | Normalize, dedupe, filter | Raw corpus + manifest | `data/clean/corpus.jsonl`, `data/clean/stats.json` |
| **Labeling Agent** | Build SFT (and optional preference) examples | Clean corpus | `data/labeled/sft.jsonl`, `data/labeled/stats.json` |
| **Training Agent** | Fine-tune (or stub dry-run) | Labeled JSONL, config | `artifacts/checkpoints/`, `artifacts/training_log.json` |
| **Evaluation Agent** | Simple overlap / length checks + optional metrics | Labeled file, checkpoint dir | `artifacts/eval_report.json` |
| **Deployment Agent** | Package model card + file list for release | Checkpoint + eval report | `artifacts/deploy/` |

## Quick start

```bash
cd small-llm-training-factory
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --config config/default.yaml
```

Stages run in order. Training defaults to **`dry_run: true`** (no GPU, writes a stub checkpoint). Set `training.dry_run: false` and install `torch` + `transformers` to run a minimal fine-tune on CPU (slow but real).

## Sample data

- **`data/raw/sample_corpus.jsonl`**: noisy duplicate lines (simulates collector output).
- After the pipeline, **`data/clean/`** and **`data/labeled/sft.jsonl`** hold cleaned and instruction-formatted rows.

## Configuration

Edit **`config/default.yaml`**: paths, cleaner rules, labeling templates, training knobs.

## Extending

- Swap **Collector** to pull from APIs or object storage; keep the same JSONL + manifest shape.
- Add **human review** between Labeling and Training by exporting a CSV and merging approved rows back into `data/labeled/`.
