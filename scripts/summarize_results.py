#!/usr/bin/env python
"""Summarize experiment results into a CSV table.

The training scripts save runs as:
    SAVE_ROOT / wandb_run_group / experiment_name / {flags.json, train.csv, eval.csv}

This script recursively finds all experiment directories under a result root,
extracts env_name from flags.json, extracts a success metric from eval.csv, and
writes one row per experiment.
"""

import argparse
import csv
import json
from pathlib import Path


SUCCESS_KEYWORDS = ("success", "success_rate", "is_success")

# You can set your result directory here and run this script without arguments.
# For example: DEFAULT_RESULT_DIR = Path("cdp_eval_result")
DEFAULT_RESULT_DIR = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize env_name and success rate for all experiments in a result directory."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Root directory containing experiment outputs, e.g. cdp_eval_result.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <result_dir>/summary.csv.",
    )
    parser.add_argument(
        "--metric-column",
        default=None,
        help="Exact eval.csv column to use, e.g. evaluation/success.",
    )
    parser.add_argument(
        "--mode",
        choices=("last", "best"),
        default="last",
        help="Use the last logged value or the best value from eval.csv.",
    )
    return parser.parse_args()


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def read_eval_rows(path):
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def find_success_column(fieldnames, metric_column=None):
    if not fieldnames:
        return None
    if metric_column is not None:
        return metric_column if metric_column in fieldnames else None

    candidates = []
    for name in fieldnames:
        lowered = name.lower()
        if any(keyword in lowered for keyword in SUCCESS_KEYWORDS):
            candidates.append(name)

    if not candidates:
        return None

    # Prefer the most direct metric names when several success-related columns exist.
    exact_priority = (
        "evaluation/success",
        "evaluation/success_rate",
        "evaluation/is_success",
        "success",
        "success_rate",
        "is_success",
    )
    lowered_to_original = {name.lower(): name for name in candidates}
    for preferred in exact_priority:
        if preferred in lowered_to_original:
            return lowered_to_original[preferred]

    return candidates[0]


def select_metric(rows, metric_column, mode):
    values = []
    for row in rows:
        value = to_float(row.get(metric_column))
        step = to_float(row.get("step"))
        if value is not None:
            values.append((value, step))

    if not values:
        return None, None

    if mode == "best":
        return max(values, key=lambda item: item[0])

    return values[-1]


def find_experiment_root(eval_path, result_dir):
    for directory in (eval_path.parent, *eval_path.parents):
        if directory == result_dir.parent:
            break
        if (directory / "flags.json").is_file() or (directory / "config" / "flags.json").is_file():
            return directory
        if directory == result_dir:
            break
    return eval_path.parent


def find_flags_path(exp_dir):
    candidates = (
        exp_dir / "flags.json",
        exp_dir / "config" / "flags.json",
        exp_dir / "config.json",
        exp_dir / "config" / "config.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    return exp_dir / "flags.json"


def experiment_outputs(result_dir):
    for eval_path in sorted(result_dir.rglob("eval.csv")):
        if eval_path.is_file():
            yield find_experiment_root(eval_path, result_dir), eval_path


def summarize_experiment(exp_dir, eval_path, result_dir, metric_column=None, mode="last"):
    flags = read_json(find_flags_path(exp_dir))
    rows = read_eval_rows(eval_path)
    fieldnames = rows[0].keys() if rows else []
    success_column = find_success_column(fieldnames, metric_column)

    success_rate = None
    selected_step = None
    if success_column is not None:
        success_rate, selected_step = select_metric(rows, success_column, mode)

    agent = flags.get("agent", {})
    if not isinstance(agent, dict):
        agent = {}

    return {
        "experiment": exp_dir.name,
        "relative_path": exp_dir.relative_to(result_dir).as_posix(),
        "eval_path": eval_path.relative_to(result_dir).as_posix(),
        "env_name": flags.get("env_name", ""),
        "seed": flags.get("seed", ""),
        "agent_name": agent.get("agent_name", ""),
        "step": "" if selected_step is None else int(selected_step),
        "success_rate": "" if success_rate is None else success_rate,
        "success_column": "" if success_column is None else success_column,
        "eval_rows": len(rows),
    }


def write_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "relative_path",
        "eval_path",
        "env_name",
        "seed",
        "agent_name",
        "step",
        "success_rate",
        "success_column",
        "eval_rows",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.result_dir is None:
        raise SystemExit(
            "Please provide result_dir or set DEFAULT_RESULT_DIR in scripts/summarize_results.py"
        )
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve() if args.output else result_dir / "summary.csv"

    if not result_dir.exists():
        raise SystemExit(f"Result directory does not exist: {result_dir}")

    rows = [
        summarize_experiment(exp_dir, eval_path, result_dir, args.metric_column, args.mode)
        for exp_dir, eval_path in experiment_outputs(result_dir)
    ]

    write_summary(rows, output_path)
    print(f"Wrote {len(rows)} experiments to {output_path}")


if __name__ == "__main__":
    main()
