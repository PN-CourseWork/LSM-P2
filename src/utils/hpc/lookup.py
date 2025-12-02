"""
Runtime argument resolver for LSF Job Arrays.

Usage:
    python -m utils.hpc.lookup --config experiments.yaml --group my_experiment --index $LSB_JOBINDEX

Output:
    --arg1 value1 --arg2 value2 ...
"""
import sys
import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List

# Reuse existing config loader
from .jobgen import load_config


def get_combinations(group_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for a group."""
    static_args = group_config.get("static_args", {})
    sweep = group_config.get("sweep", {})
    sweep_paired = group_config.get("sweep_paired", {})

    # 1. Paired Sweep
    paired_combinations = [{}]
    if sweep_paired:
        keys = list(sweep_paired.keys())
        values = list(sweep_paired.values())
        # Validation should happen at generation time, but safety check here
        if len(set(len(v) for v in values)) > 1:
             raise ValueError("Paired sweep values must have same length")
        paired_combinations = [dict(zip(keys, v)) for v in zip(*values)]

    # 2. Regular Sweep (Cartesian Product)
    regular_combinations = [{}]
    if sweep:
        keys = list(sweep.keys())
        values = list(sweep.values())
        regular_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 3. Combine
    all_combos = []
    for paired in paired_combinations:
        for regular in regular_combinations:
            # Merge: Static < Paired < Regular (Regular takes precedence if conflict)
            # Note: You might want to enforce distinct keys for safety
            combo = {**static_args, **paired, **regular}
            all_combos.append(combo)
            
    return all_combos


def format_args(params: Dict[str, Any]) -> str:
    """Format dictionary into command line arguments."""
    args = []
    for key, value in params.items():
        # Handle flags (boolean True)
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
            
        # Handle standard key-value pairs
        args.append(f"--{key}")
        args.append(str(value))
        
    return " ".join(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--group", type=str, required=True, help="Group name in config")
    parser.add_argument("--index", type=int, required=True, help="Job array index (1-based)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found", file=sys.stderr)
        sys.exit(1)

    if args.group not in config:
        print(f"Error: Group '{args.group}' not found in config", file=sys.stderr)
        sys.exit(1)

    group_config = config[args.group]
    
    try:
        combinations = get_combinations(group_config)
    except Exception as e:
        print(f"Error processing combinations: {e}", file=sys.stderr)
        sys.exit(1)

    # LSF indices are 1-based
    idx = args.index - 1

    if idx < 0 or idx >= len(combinations):
        print(f"Error: Index {args.index} out of bounds (1-{len(combinations)})", file=sys.stderr)
        sys.exit(1)

    # Output only the arguments string to stdout
    print(format_args(combinations[idx]))


if __name__ == "__main__":
    main()
