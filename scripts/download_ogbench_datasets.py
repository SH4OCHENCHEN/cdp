"""Download OGBench datasets used by ``run_baselines_non_visual.sh``.

This script uses ``ogbench.download_datasets`` directly, so it does not create
MuJoCo/Gymnasium environments. Single-task OGBench tasks reuse these same base
dataset files, so downloading this list covers the OGBench environments used by
``scripts/run_baselines_non_visual.sh``.

The D4RL Adroit tasks in that baseline script are not OGBench datasets and are
therefore intentionally not listed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASELINE_NON_VISUAL_DATASET_NAMES = (
    # Offline OGBench manipulation tasks.
    'cube-double-play-v0',
    'cube-triple-play-v0',
    'puzzle-3x3-play-v0',
    'puzzle-4x4-play-v0',
    'scene-play-v0',
    # Online OGBench locomotion tasks.
    'antmaze-large-navigate-v0',
    'humanoidmaze-medium-navigate-v0',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download OGBench datasets used by scripts/run_baselines_non_visual.sh.'
    )
    parser.add_argument(
        '--dataset_dir',
        default='~/.ogbench/data',
        help='Directory where OGBench .npz files are stored. Default: ~/.ogbench/data',
    )
    parser.add_argument(
        '--only_missing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Skip datasets whose train and val .npz files already exist. Default: true',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print the datasets that would be downloaded without downloading them.',
    )
    return parser.parse_args()


def missing_dataset_names(dataset_names: tuple[str, ...], dataset_dir: Path) -> list[str]:
    missing = []
    for name in dataset_names:
        train_path = dataset_dir / f'{name}.npz'
        val_path = dataset_dir / f'{name}-val.npz'
        if not train_path.exists() or not val_path.exists():
            missing.append(name)
    return missing


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    dataset_names = list(BASELINE_NON_VISUAL_DATASET_NAMES)

    if args.only_missing:
        dataset_names = missing_dataset_names(BASELINE_NON_VISUAL_DATASET_NAMES, dataset_dir)

    print(f'OGBench datasets listed: {len(BASELINE_NON_VISUAL_DATASET_NAMES)}')
    print(f'Datasets to download: {len(dataset_names)}')
    print(f'Dataset directory: {dataset_dir}')

    if args.dry_run:
        for dataset_name in dataset_names:
            print(dataset_name)
        return 0

    if not dataset_names:
        print('All datasets are already present.')
        return 0

    try:
        import ogbench
    except ImportError:
        print('Failed to import ogbench. Install dependencies with: pip install -r requirements.txt', file=sys.stderr)
        return 1

    ogbench.download_datasets(dataset_names, dataset_dir=str(dataset_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
