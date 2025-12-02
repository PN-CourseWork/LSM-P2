"""
Runtime configuration utilities for job array sweeps.

Expands YAML configs with static, grid_sweep, and matrix_sweep sections
into concrete parameter combinations, selected by LSB_JOBINDEX.
"""

import itertools
import os
from pathlib import Path
from typing import Any

import yaml


def expand_runtime(cfg: dict) -> list[dict]:
    """
    Expand a runtime config into all parameter combinations.

    Config format:
        static: {key: value, ...}        # Applied to all combinations
        grid_sweep: {key: [v1, v2], ...}  # Cartesian product
        matrix_sweep: [{k1: v1, k2: v2}, ...]  # Explicit combinations

    Returns list of dicts, one per combination.
    """
    static = cfg.get("static", {}) or {}
    grid = cfg.get("grid_sweep") or {}
    matrix = cfg.get("matrix_sweep") or []

    # Expand grid sweep (Cartesian product)
    if grid:
        keys = list(grid.keys())
        vals = [grid[k] for k in keys]
        grid_combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    else:
        grid_combos = [{}]

    # Matrix sweep (explicit combinations)
    mat_combos = matrix if matrix else [{}]

    # Combine: static + grid + matrix
    combos = []
    for g in grid_combos:
        for m in mat_combos:
            c = {}
            c.update(static)
            c.update(g)
            c.update(m)
            combos.append(c)

    return combos


def load_runtime_config(
    config_path: str | Path,
    experiment: str,
    index: int | None = None,
) -> dict[str, Any]:
    """
    Load a runtime config and return parameters for the given index.

    Args:
        config_path: Path to YAML config file
        experiment: Name of experiment in the YAML
        index: 1-based index (default: from LSB_JOBINDEX env var)

    Returns:
        Dict of parameters for the selected combination
    """
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)

    if experiment not in all_cfg:
        available = list(all_cfg.keys())
        raise SystemExit(f"Experiment '{experiment}' not found. Available: {available}")

    cfg = all_cfg[experiment]
    combos = expand_runtime(cfg)

    # Get index from arg or environment
    if index is None:
        index = int(os.environ.get("LSB_JOBINDEX", "1"))

    if not (1 <= index <= len(combos)):
        raise SystemExit(f"Index {index} out of range 1..{len(combos)}")

    return combos[index - 1]


def get_num_tasks(config_path: str | Path, experiment: str) -> int:
    """Return the number of tasks (combinations) in an experiment."""
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)
    return len(expand_runtime(all_cfg[experiment]))


def list_experiments(config_path: str | Path) -> None:
    """Print all experiments and their task counts."""
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)

    print(f"Config: {config_path}")
    print("-" * 60)
    for name, cfg in all_cfg.items():
        n = len(expand_runtime(cfg))
        print(f"  {name}: {n} tasks")


def list_tasks(config_path: str | Path, experiment: str) -> None:
    """Print all tasks in an experiment (for debugging)."""
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)

    cfg = all_cfg[experiment]
    combos = expand_runtime(cfg)

    print(f"Experiment: {experiment}")
    print(f"Tasks: {len(combos)}")
    print("-" * 60)
    for i, c in enumerate(combos, 1):
        print(f"  {i}: {c}")
