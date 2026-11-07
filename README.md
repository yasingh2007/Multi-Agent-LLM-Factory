# Small LLM Training Factory

A **multi-agent pipeline** for collecting data, cleaning it, labeling, training small models, evaluating, and packaging for deployment. Each stage is a separate Python module with a clear artifact contract.

## Agents

| Agent | Role | Reads | Writes |
|-------|------|-------|--------|
| **Dataset Collector** | Ingests raw sources into a versioned corpus + manifest | Config, optional URL list / files | `data/raw/corpus.jsonl`, `data/raw/manifest.json` |
| **Dataset Cleaner** | Normalize, dedupe, filter | Raw corpus + manifest | `data/clean/corpus.jsonl`, `data/clean/stats.json` |
| **Labeling Agent** | SFT via **heuristic** or **OpenAI-compatible LLM**; writes **DPO** pairs | Clean corpus | `data/labeled/sft.jsonl`, `data/labeled/dpo.jsonl`, `stats.json` |
| **Training Agent** | Fine-tune (or stub dry-run) | Labeled JSONL, config | `artifacts/checkpoints/`, `artifacts/training_log.json` |
| **Evaluation Agent** | Compression, token recall, ROUGE-L, quality gates | SFT (+ DPO sanity) | `artifacts/eval_report.json` |
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

### LLM labeling (OpenAI-compatible)

1. Copy **`.env.example`** → set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` for Groq, Together, Ollama, etc.).
2. Set `labeling.mode: llm` or run **`config/llm.yaml`**:

```bash
$env:OPENAI_API_KEY="sk-..."
python scripts/run_pipeline.py --config config/llm.yaml
```

If the key is missing, the labeler **falls back to heuristic** and records errors in `data/labeled/stats.json`.

### DPO / preference JSONL

With `labeling.dpo.enabled: true`, each row gets:

- **`prompt`**: instruction + input (chat template friendly)
- **`chosen`**: preferred summary (LLM or heuristic)
- **`rejected`**: weak summary (`rejected_strategy: weak` | `tail`)

See **`data/labeled/sample_dpo.schema.jsonl`** for one-line schema reference. Feed `dpo.jsonl` into TRL `DPOTrainer` or your preference optimizer.

### Evaluation metrics

`artifacts/eval_report.json` includes per-row and aggregate:

- **compression_ratio** — response length / input length  
- **token_recall** — input tokens present in the summary  
- **token_f1** / **rouge_l_f1** — overlap without extra dependencies  

Tune thresholds under **`evaluation.gates`**; the run **fails** (`passed: false`) when gates are breached.

## Extending

- Swap **Collector** to pull from APIs or object storage; keep the same JSONL + manifest shape.
- Add **human review** between Labeling and Training by exporting a CSV and merging approved rows back into `data/labeled/`.
