from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import clean, collect, deploy, evaluate, label, train


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent training factory pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)

    collect(config)
    clean(config)
    label(config)
    train(config)
    evaluate(config)
    deploy(config)

    print("Pipeline complete.")
    print(f"  Raw:    {config['paths']['raw_dir']}/corpus.jsonl")
    print(f"  Clean:  {config['paths']['clean_dir']}/corpus.jsonl")
    print(f"  Labeled: {config['paths']['labeled_dir']}/sft.jsonl")
    print(f"  Artifacts: {config['paths']['artifacts_dir']}/")


if __name__ == "__main__":
    main()
