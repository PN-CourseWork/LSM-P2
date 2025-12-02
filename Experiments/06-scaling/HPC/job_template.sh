#!/bin/bash
# Generic LSF + MPI template driven by env variables and a parameter file

# ----------------- Scheduler directives --------------------------------------
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${N_CORES}
#BSUB -R "rusage[mem=${MEM_PER_CORE}]"
#BSUB -R "span[ptile=${PTILE}]"
#BSUB -J ${JOB_NAME}        # must already include [1-M] when generated
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

# Optional: build extra mpiexec flags from MAP_BY if you want
MPIRUN_EXTRA="${MPIRUN_EXTRA:-${MAP_BY} --bind-to core --report-bindings}"

# ----------------- Resolve runtime args from index ---------------------------
if [ -z "${ARGS_FILE:-}" ]; then
  echo "ERROR: ARGS_FILE not set"
  exit 1
fi

IDX=${LSB_JOBINDEX:-1}

# Small helper that reads ARGS_FILE and prints a CLI string for this index.
# You can keep this generic and reuse it for all projects.
RUNTIME_ARGS=$(python - << 'EOF'
import json, os, sys

path = os.environ["ARGS_FILE"]
idx  = int(os.environ.get("LSB_JOBINDEX", "1"))  # 1-based

with open(path) as f:
    data = json.load(f)

entry = data[idx - 1]  # 0-based list

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

echo "Running job index $IDX with:"
echo "  RANKS       = ${RANKS}"
echo "  RUNNER      = ${RUNNER_SCRIPT}"
echo "  RUNTIME_ARGS= ${RUNTIME_ARGS}"

# ----------------- Run -------------------------------------------------------
${MPI_LAUNCHER} ${MPIRUN_EXTRA} -n ${RANKS} \
  ${RUNNER_SCRIPT} ${RUNTIME_ARGS}
