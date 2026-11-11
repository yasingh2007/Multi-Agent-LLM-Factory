from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import clean, collect, deploy, evaluate, label, train
from agents.run_context import (
    STAGE_ORDER,
    ensure_run_id,
    exit_code_for_eval,
    hydrate_run_id_from_artifacts,
    parse_only,
    print_pipeline_summary,
    write_run_record,
)

STAGE_RUNNERS = {
    "collect": collect,
    "clean": clean,
    "label": label,
    "train": train,
    "evaluate": evaluate,
    "deploy": deploy,
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stages_to_run(only: list[str] | None) -> list[str]:
    if only is None:
        return [name for name, _ in STAGE_ORDER]
    return only


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent training factory pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument(
        "--only",
        metavar="STAGES",
        help="Comma-separated stages to run (in pipeline order): "
        + ", ".join(name for name, _ in STAGE_ORDER),
    )
    parser.add_argument(
        "--summary-only",
        "--dry-run",
        action="store_true",
        dest="summary_only",
        help="Print summary from existing artifacts; do not run stages.",
    )
    parser.add_argument("--quiet", action="store_true", help="Skip end-of-run summary output.")
    args = parser.parse_args()

    config = load_config(args.config.resolve())

    if args.summary_only:
        hydrate_run_id_from_artifacts(config)
        if not args.quiet:
            print_pipeline_summary(config)
        raise SystemExit(exit_code_for_eval(ran_evaluate=True, config=config))

    ensure_run_id(config, project_root=ROOT)

    only = parse_only(args.only)
    selected = stages_to_run(only)
    write_run_record(config, stages=selected, config_path=args.config.resolve())

    for stage in selected:
        runner = STAGE_RUNNERS[stage]
        runner(config)

    print("Pipeline complete.")
    print(f"  run_id:  {config.get('run_id')}")
    print(f"  Stages:  {', '.join(selected)}")
    print(f"  Raw:     {config['paths']['raw_dir']}/corpus.jsonl")
    print(f"  Clean:   {config['paths']['clean_dir']}/corpus.jsonl")
    print(f"  Labeled: {config['paths']['labeled_dir']}/sft.jsonl")
    if config.get("labeling", {}).get("dpo", {}).get("enabled", False):
        print(f"  DPO:     {config['paths']['labeled_dir']}/dpo.jsonl")
    print(f"  Artifacts: {config['paths']['artifacts_dir']}/")

    if not args.quiet:
        print_pipeline_summary(config)

    ran_evaluate = "evaluate" in selected
    raise SystemExit(exit_code_for_eval(ran_evaluate=ran_evaluate, config=config))


if __name__ == "__main__":
    main()
