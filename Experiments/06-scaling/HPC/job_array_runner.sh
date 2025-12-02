#BSUB -J jacobi_2n_2s
#BSUB -q hpcintro
#BSUB -W 0:10
#BSUB -M 8GB
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -N

# ==================================================================================================
# Template job script for parameter sweeping in job-array (With same requested resources)
# ==================================================================================================

# Default values (for testing)
N=64
STRAT="sliced"
COMM="numpy"
MAX_ITER=100

# MPI mapping options
# NPS (#ranks/socket)
# MPI parameters: NP (#ranks), NPN (#ranks/node), NPS (#ranks/socket)
NPS=12
MOPTS="--report-binding --map-by ppr:$NPS:package --bind-to core"

# Python Runner arguments
SOLVER_ARGS=" --N $N_SIZE --strategy $STRAT --communicator $COMM --max-iter $MAX_ITER --numba"
MLFLOW_ARGS="--job-name $LSB_JOBINDEX --experiment-name $EXPERIMENT_NAME"

SCRIPT="Experiments/06-scaling/jacobi_runner.py"

cd $LSB_SUBCWD || exit 1
mkdir -p {LSF_OUTPUT_DIR}
# load the MPI module
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 >& /dev/null
uv sync 

export NUMBA_NUM_THREADS=$SPECIFIED_NUMBA_THREADS

# Run solver 
mpirun $MOPTS uv run python $SCRIPT $SOLVER_ARGS $MLFLOW_ARGS


