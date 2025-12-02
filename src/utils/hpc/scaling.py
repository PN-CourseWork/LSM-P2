"""
Scaling Experiment Job Generator
================================

Generates LSF job scripts and pack files for scaling experiments.
Reads experiment definitions from experiments.yaml and groups them
by resource requirements (node count).

Usage:
    uv run python -m utils.hpc.scaling --list
    uv run python -m utils.hpc.scaling -G jacobi_strong
    uv run python -m utils.hpc.scaling -G jacobi_strong -G fmg_weak --dry-run
    uv run python -m utils.hpc.scaling --all
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import tomllib
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Experiments" / "06-scaling"))

from runtime_config import expand_runtime


def load_utils_conf() -> dict:
    """Load .utils.conf from project root."""
    conf_path = PROJECT_ROOT / ".utils.conf"
    if not conf_path.exists():
        raise FileNotFoundError(f"Config not found: {conf_path}")
    with open(conf_path, "rb") as f:
        return tomllib.load(f)


def load_experiments(config_path: Path) -> dict:
    """Load experiments from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_tasks(experiment_config: dict) -> int:
    """Count number of tasks for an experiment."""
    return len(expand_runtime(experiment_config))


def parse_node_count(name: str) -> int | None:
    """Extract node count from experiment name."""
    for suffix in ["_1node", "_2node", "_3node", "_4node"]:
        if name.endswith(suffix):
            return int(suffix[1])
    return None


def get_base_name(name: str) -> str:
    """Get base experiment name without node suffix."""
    for suffix in ["_1node", "_2node", "_3node", "_4node"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def group_experiments(experiments: dict) -> dict[str, list[tuple[str, int, int]]]:
    """Group experiments by base name. Returns {base: [(full_name, nodes, tasks), ...]}."""
    groups = {}
    for name, config in experiments.items():
        nodes = parse_node_count(name)
        if nodes is None:
            continue
        base = get_base_name(name)
        tasks = count_tasks(config)
        groups.setdefault(base, []).append((name, nodes, tasks))

    # Sort by node count
    for base in groups:
        groups[base].sort(key=lambda x: x[1])
    return groups


def generate_jobscript(
    nodes: int,
    experiments: list[tuple[str, int]],  # [(name, tasks), ...]
    conf: dict,
    runner_script: str,
    config_path: str,
    hpc_dir: str,
) -> str:
    """Generate a jobscript for a specific node count."""
    lsf = conf["hpc"]["lsf"]
    cores = nodes * lsf["cores_per_node"]
    total_tasks = sum(t for _, t in experiments)

    # Build index mapping
    mapping_lines = []
    offset = 0
    for name, tasks in experiments:
        start = offset + 1
        end = offset + tasks
        mapping_lines.append(f"#   {start:3d}-{end:3d}: {name} ({tasks} tasks)")
        offset += tasks

    mapping_code = []
    offset = 0
    for i, (name, tasks) in enumerate(experiments):
        end = offset + tasks
        if i == 0:
            mapping_code.append(f'if   [ $IDX -le {end} ]; then EXP="{name}"; LOCAL_IDX=$IDX')
        elif i == len(experiments) - 1:
            mapping_code.append(f'else EXP="{name}"; LOCAL_IDX=$((IDX-{offset}))')
        else:
            mapping_code.append(f'elif [ $IDX -le {end} ]; then EXP="{name}"; LOCAL_IDX=$((IDX-{offset}))')
        offset += tasks
    mapping_code.append("fi")

    nps = lsf["ranks_per_socket"]

    return f"""#!/bin/bash
#BSUB -J scaling_{nodes}node[1-{total_tasks}]
#BSUB -q {lsf['queue']}
#BSUB -W {lsf['walltime']}
#BSUB -n {cores}
#BSUB -R "span[ptile={lsf['cores_per_node']}]"
#BSUB -R "rusage[mem={lsf['mem']}]"
#BSUB -o {hpc_dir}/logs/{nodes}node_%J_%I.out
#BSUB -e {hpc_dir}/logs/{nodes}node_%J_%I.err

# Index mapping ({total_tasks} total tasks):
{chr(10).join(mapping_lines)}

IDX=$LSB_JOBINDEX
{chr(10).join(mapping_code)}

module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null
cd $LSB_SUBCWD

echo "========================================"
echo "$EXP task $LOCAL_IDX (global $IDX)"
echo "========================================"

MOPTS="--map-by ppr:{nps}:package --bind-to core"

mpirun $MOPTS uv run python {runner_script} \\
    --runtime-config {config_path} --experiment $EXP --index $LOCAL_IDX --numba
"""


def generate_pack(jobscripts: list[Path], generated_dir: Path) -> str:
    """Generate pack file content with relative paths."""
    lines = ["#!/bin/bash", "# Generated scaling experiment submission", ""]
    for script in jobscripts:
        rel_path = script.relative_to(generated_dir)
        lines.append(f"bsub < {rel_path}")
    return "\n".join(lines)


def interactive_scaling():
    """Interactive questionnaire for selecting and generating scaling jobs."""
    import questionary

    # Load config
    conf = load_utils_conf()
    hpc_conf = conf["hpc"]

    hpc_dir = PROJECT_ROOT / hpc_conf["hpc_dir"]
    experiments_path = hpc_dir / hpc_conf["experiments_config"]

    if not experiments_path.exists():
        print(f"Error: Experiments config not found: {experiments_path}")
        return

    experiments = load_experiments(experiments_path)
    groups = group_experiments(experiments)

    # Build choices with task counts
    choices = []
    for base in sorted(groups.keys()):
        total = sum(t for _, _, t in groups[base])
        node_info = ", ".join(f"{n}n:{t}" for _, n, t in groups[base])
        choices.append(questionary.Choice(
            title=f"{base} ({total} tasks: {node_info})",
            value=base,
        ))

    # Multi-select questionnaire
    selected = questionary.checkbox(
        "Select experiment groups to generate:",
        choices=choices,
        instruction="(Space to select, Enter to confirm)",
    ).ask()

    if not selected:
        print("No groups selected.")
        return

    # Generate
    print(f"\nGenerating jobs for: {', '.join(selected)}")
    generate_scaling_jobs(selected, conf)


def generate_scaling_jobs(selected_groups: list[str], conf: dict | None = None):
    """Generate scaling jobs for selected groups."""
    if conf is None:
        conf = load_utils_conf()

    hpc_conf = conf["hpc"]
    hpc_dir = PROJECT_ROOT / hpc_conf["hpc_dir"]
    experiments_path = hpc_dir / hpc_conf["experiments_config"]
    generated_dir = hpc_dir / "generated"
    logs_dir = hpc_dir / "logs"
    runner_script = hpc_conf["runner_script"]

    experiments = load_experiments(experiments_path)
    groups = group_experiments(experiments)

    # Validate
    for g in selected_groups:
        if g not in groups:
            print(f"Error: Unknown group '{g}'")
            return

    # Collect experiments by node count
    by_nodes: dict[int, list[tuple[str, int]]] = {}
    for base in selected_groups:
        for name, nodes, tasks in groups[base]:
            by_nodes.setdefault(nodes, []).append((name, tasks))

    for nodes in by_nodes:
        by_nodes[nodes].sort()

    # Create directories
    generated_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate jobscripts
    jobscripts = []
    for nodes in sorted(by_nodes.keys()):
        exps = by_nodes[nodes]
        total = sum(t for _, t in exps)

        script_content = generate_jobscript(
            nodes=nodes,
            experiments=exps,
            conf=conf,
            runner_script=runner_script,
            config_path=str(experiments_path.relative_to(PROJECT_ROOT)),
            hpc_dir=str(hpc_dir.relative_to(PROJECT_ROOT)),
        )

        script_path = generated_dir / f"{nodes}node.sub"
        jobscripts.append(script_path)

        print(f"  {nodes}-node: {len(exps)} experiments, {total} tasks")
        for name, tasks in exps:
            print(f"    - {name}: {tasks}")

        with open(script_path, "w") as f:
            f.write(script_content)

    # Generate pack file
    pack_name = "_".join(sorted(selected_groups)) if len(selected_groups) <= 3 else "scaling"
    pack_path = generated_dir / f"{pack_name}.pack"
    pack_content = generate_pack(jobscripts, generated_dir)

    with open(pack_path, "w") as f:
        f.write(pack_content)

    print(f"\n✓ Generated {len(jobscripts)} jobscripts + {pack_path.name}")
    print(f"  To submit: cd {hpc_dir.relative_to(PROJECT_ROOT)}/generated && source {pack_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate scaling experiment jobs")
    parser.add_argument("-G", "--group", action="append", help="Experiment group(s)")
    parser.add_argument("--all", action="store_true", help="All groups")
    parser.add_argument("--list", action="store_true", help="List available groups")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_scaling()
        return

    # Load config
    conf = load_utils_conf()
    hpc_conf = conf["hpc"]

    hpc_dir = PROJECT_ROOT / hpc_conf["hpc_dir"]
    experiments_path = hpc_dir / hpc_conf["experiments_config"]
    generated_dir = hpc_dir / "generated"
    logs_dir = hpc_dir / "logs"
    runner_script = hpc_conf["runner_script"]

    if not experiments_path.exists():
        print(f"Error: Experiments config not found: {experiments_path}")
        sys.exit(1)

    experiments = load_experiments(experiments_path)
    groups = group_experiments(experiments)

    # List mode
    if args.list:
        print("Available experiment groups:\n")
        for base, exps in sorted(groups.items()):
            total = sum(t for _, _, t in exps)
            print(f"  {base} ({total} tasks)")
            for name, nodes, tasks in exps:
                print(f"    - {name}: {tasks} tasks, {nodes} node(s)")
        print(f"\nTotal: {len(groups)} groups, {sum(sum(t for _, _, t in e) for e in groups.values())} tasks")
        return

    # Determine groups
    if args.all:
        selected = list(groups.keys())
    elif args.group:
        selected = args.group
        for g in selected:
            if g not in groups:
                print(f"Error: Unknown group '{g}'. Available: {', '.join(sorted(groups.keys()))}")
                sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Collect experiments by node count across selected groups
    by_nodes: dict[int, list[tuple[str, int]]] = {}
    for base in selected:
        for name, nodes, tasks in groups[base]:
            by_nodes.setdefault(nodes, []).append((name, tasks))

    # Sort experiments within each node group for consistent ordering
    for nodes in by_nodes:
        by_nodes[nodes].sort()

    # Generate jobscripts
    generated_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobscripts = []
    for nodes in sorted(by_nodes.keys()):
        exps = by_nodes[nodes]
        total = sum(t for _, t in exps)

        script_content = generate_jobscript(
            nodes=nodes,
            experiments=exps,
            conf=conf,
            runner_script=runner_script,
            config_path=str(experiments_path.relative_to(PROJECT_ROOT)),
            hpc_dir=str(hpc_dir.relative_to(PROJECT_ROOT)),
        )

        script_path = generated_dir / f"{nodes}node.sub"
        jobscripts.append(script_path)

        print(f"{nodes}-node: {len(exps)} experiments, {total} tasks")
        for name, tasks in exps:
            print(f"  - {name}: {tasks}")

        if args.dry_run:
            print(f"\n--- {script_path.name} ---")
            print(script_content[:500] + "..." if len(script_content) > 500 else script_content)
        else:
            with open(script_path, "w") as f:
                f.write(script_content)
            print(f"  -> {script_path}")

    # Generate pack file
    pack_name = "_".join(sorted(selected)) if len(selected) <= 3 else "scaling"
    pack_path = generated_dir / f"{pack_name}.pack"
    pack_content = generate_pack(jobscripts, generated_dir)

    print(f"\nPack file: {pack_path.name}")
    if args.dry_run:
        print(pack_content)
    else:
        with open(pack_path, "w") as f:
            f.write(pack_content)
        print(f"  -> {pack_path}")
        print(f"\nTo submit: cd {generated_dir.relative_to(PROJECT_ROOT)} && source {pack_path.name}")


if __name__ == "__main__":
    main()
