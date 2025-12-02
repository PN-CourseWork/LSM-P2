# Running Job Arrays with Different Runtime Arguments for a Python Script (LSF)

This guide shows how to run **parameter sweeps** on an HPC cluster using **LSF job arrays**, with all runtime logic handled inside Python.
Each array element becomes an independent job, and Python maps the job index to a concrete configuration.

---

## Core Idea

1. Submit a job array: `#BSUB -J myjob[1-N]`.
2. LSF sets `LSB_JOBINDEX` (`1..N`) for each array element.
3. Your Python script uses that index to select a configuration.
4. Each array element is a **separate job** with its own resources, logs, and exit code.

---

## 1. Example LSF Job Script

Save as `job.lsf`:

```bash
#!/bin/bash
#BSUB -J jacobi_strong[1-8]
#BSUB -q hpcintro
#BSUB -W 00:10
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/jacobi.%J.%I.out
#BSUB -e logs/jacobi.%J.%I.err

module purge
module load mpi

mpiexec --bind-to core --map-by ppr:12:package -n 8 \
  uv run python jacobi_runner.py --runtime-config runtime/jacobi_strong.yaml
```

Submit with:

```bash
bsub < job.lsf
```

---

## 2. Python Runner Logic (index → parameters)

Create or modify `jacobi_runner.py`:

```python
import os, argparse, itertools, yaml

def expand_runtime(cfg):
    static = cfg.get("static", {}) or {}
    grid   = cfg.get("grid_sweep") or {}
    matrix = cfg.get("matrix_sweep") or []

    if grid:
        keys = list(grid.keys())
        vals = [grid[k] for k in keys]
        grid_combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    else:
        grid_combos = [dict()]

    mat_combos = matrix if matrix else [dict()]

    combos = []
    for g in grid_combos:
        for m in mat_combos:
            c = {}
            c.update(static)
            c.update(g)
            c.update(m)
            combos.append(c)
    return combos

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int)
    p.add_argument("--strategy")
    p.add_argument("--communicator")
    p.add_argument("--tol", type=float)
    p.add_argument("--max_iter", type=int)
    p.add_argument("--ranks", type=int)
    p.add_argument("--runtime-config", type=str)
    p.add_argument("--index", type=int)
    return p.parse_args()

def resolve_params(args):
    if args.N is not None and args.strategy is not None:
        return dict(
            N=args.N,
            strategy=args.strategy,
            communicator=args.communicator or "custom",
            tol=args.tol if args.tol is not None else 0.0,
            max_iter=args.max_iter if args.max_iter is not None else 100,
            ranks=args.ranks if args.ranks is not None else 1,
        )

    if not args.runtime_config:
        raise SystemExit("Provide --runtime-config for array mode or explicit flags.")

    with open(args.runtime_config) as f:
        cfg = yaml.safe_load(f)

    combos = expand_runtime(cfg)
    idx = args.index if args.index is not None else int(os.environ.get("LSB_JOBINDEX", "1"))

    if not (1 <= idx <= len(combos)):
        raise SystemExit(f"Index {idx} out of range 1..{len(combos)}")

    return combos[idx - 1]

def main():
    args = parse_args()
    cfg = resolve_params(args)
    print("Using configuration:", cfg)
    # Call solver here using cfg[...]

if __name__ == "__main__":
    main()
```

---

## 3. Runtime Configuration (YAML)

Create `runtime/jacobi_strong.yaml`:

```yaml
static:
  tol: 0.0
  max_iter: 100
  communicator: "custom"

grid_sweep:
  ranks: [1, 8, 12, 24]
  N: [144]
  strategy: ["sliced", "cubic"]

matrix_sweep: []
```

Array length here is **8**, matching `#BSUB -J jacobi_strong[1-8]`.

---

## 4. Running Locally (simulate array index)

```bash
uv run python jacobi_runner.py --runtime-config runtime/jacobi_strong.yaml --index 3
```

---

## 5. Notes

- Each array element is a separate job.
- Outputs include both job ID and index (%J.%I).
- Ensure array length matches number of runtime combinations.
- Log chosen parameters for reproducibility.

---

## Summary

- YAML defines sweeps.
- LSF provides job indices.
- Python maps index → parameters.
- Each element runs independently.

This pattern scales from small experiments to large parameter sweeps cleanly.
