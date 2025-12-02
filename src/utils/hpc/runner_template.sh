#!/bin/sh
# ============================================================================
# Universal LSF Runner Template
# ============================================================================
# This script is submitted via 'bsub' with CLI arguments for resources.
# It expects the following environment variables to be set:
#   SWEEP_CONFIG    : Path to the YAML configuration file
#   SWEEP_GROUP     : Name of the configuration group
#   SWEEP_MAP_FILE  : Path to the index mapping file (.idx)
#   SWEEP_CMD       : The python command prefix (e.g., "uv run python script.py")
# ============================================================================

# 1. Environment Setup
cd $LSB_SUBCWD || exit 1
module purge
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null
uv sync >& /dev/null

# 2. Validation
if [ -z "$SWEEP_CONFIG" ] || [ -z "$SWEEP_MAP_FILE" ]; then
    echo "Error: Missing required environment variables (SWEEP_CONFIG, SWEEP_MAP_FILE)"
    exit 1
fi

# 3. Index Mapping
# Map the Local Array Index ($LSB_JOBINDEX) to the Global Config Index
GLOBAL_IDX=$(sed -n "${LSB_JOBINDEX}p" "$SWEEP_MAP_FILE")

if [ -z "$GLOBAL_IDX" ]; then
    echo "Error: Could not map LSB_JOBINDEX=$LSB_JOBINDEX using $SWEEP_MAP_FILE"
    exit 1
fi

echo "--- Job Setup ---"
echo "Job ID:       $LSB_JOBID[$LSB_JOBINDEX]"
echo "Group:        $SWEEP_GROUP"
echo "Global Index: $GLOBAL_IDX"
echo "Config:       $SWEEP_CONFIG"

# 4. Argument Resolution
RUNTIME_ARGS=$(uv run python -m src.utils.hpc.lookup \
    --config "$SWEEP_CONFIG" \
    --group "$SWEEP_GROUP" \
    --index "$GLOBAL_IDX")

if [ $? -ne 0 ]; then
    echo "Error resolving arguments"
    exit 1
fi

echo "Args:         $RUNTIME_ARGS"
echo "-----------------"

# 5. Execution
export NUMBA_NUM_THREADS=1
$SWEEP_CMD $RUNTIME_ARGS
