#!/bin/bash
# Generic LSF+MPI job template driven entirely by environment variables.
#
# Required env vars (set by your generator):
#   QUEUE, WALLTIME, N_CORES, MEM_PER_CORE, PTILE
#   JOB_NAME, OUT_FILE, ERR_FILE
#   WORKDIR, MODULES
#   RANKS, MAP_BY (or MPIRUN_EXTRA), RUNNER_SCRIPT
#   ARGS_FILE (JSON list of runtime-arg dicts for this array)
#   NUMBA_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS (optional)

# ----------------- Scheduler directives --------------------------------------
# These variables should already be concrete strings in the generated script.
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${N_CORES}
#BSUB -R "rusage[mem=${MEM_PER_CORE}]"
#BSUB -R "span[ptile=${PTILE}]"
#BSUB -J ${JOB_NAME}
#BSUB -o ${OUT_FILE}
#BSUB -e ${ERR_FILE}

# ----------------- Setup -----------------------------------------------------
cd "${WORKDIR:-$LSB_SUBCWD}"

if [ -n "${MODULES:-}" ]; then
  module purge
  module load ${MODULES}
fi

[ -n "${NUMBA_NUM_THREADS:-}" ] && export NUMBA_NUM_THREADS
[ -n "${OMP_NUM_THREADS:-}" ] && export OMP_NUM_THREADS
[ -n "${MKL_NUM_THREADS:-}" ] && export MKL_NUM_THREADS

MPI_LAUNCHER="${MPI_LAUNCHER:-mpiexec}"

# If you want to override everything, set MPIRUN_EXTRA from outside.
if [ -z "${MPIRUN_EXTRA:-}" ]; then
  MPIRUN_EXTRA="${MAP_BY:-} --bind-to core --report-bindings"
fi

# ----------------- Resolve runtime args from ARGS_FILE -----------------------
if [ -z "${ARGS_FILE:-}" ]; then
  echo "ERROR: ARGS_FILE not set"
  exit 1
fi

IDX=${LSB_JOBINDEX:-1}

RUNTIME_ARGS=$(python - << 'EOF'
import json, os

path = os.environ["ARGS_FILE"]
idx  = int(os.environ.get("LSB_JOBINDEX", "1"))  # 1-based

with open(path) as f:
    data = json.load(f)

if not isinstance(data, list):
    raise SystemExit("ARGS_FILE must contain a JSON list")

if not (1 <= idx <= len(data)):
    raise SystemExit(f"Job index {idx} out of range 1..{len(data)}")

entry = data[idx - 1]  # 0-based

parts = []
for k, v in entry.items():
    if isinstance(v, bool):
        if v:
            parts.append(f"--{k}")
    else:
        parts.append(f"--{k} {v}")
print(" ".join(parts))
EOF
)

echo "Running array index ${IDX} with:"
echo "  RANKS        = ${RANKS}"
echo "  RUNNER       = ${RUNNER_SCRIPT}"
echo "  RUNTIME_ARGS = ${RUNTIME_ARGS}"
echo

# ----------------- Run -------------------------------------------------------
set -e

${MPI_LAUNCHER} ${MPIRUN_EXTRA} -n ${RANKS} \
  ${RUNNER_SCRIPT} ${RUNTIME_ARGS}
