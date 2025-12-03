#!/bin/bash
# Submit all HPC experiments (70 total jobs)
# Run from project root: ./Experiments/HPC-jobs/submit_all.sh

set -e
cd "$(dirname "$0")/../.."

echo "Submitting all experiments from $(pwd)"

bsub < Experiments/HPC-jobs/submit_scaling_24.sh
bsub < Experiments/HPC-jobs/submit_scaling_48.sh
bsub < Experiments/HPC-jobs/submit_scaling_72.sh
bsub < Experiments/HPC-jobs/submit_scaling_96.sh
bsub < Experiments/HPC-jobs/submit_scaling_144.sh
bsub < Experiments/HPC-jobs/submit_weak_scaling.sh
bsub < Experiments/HPC-jobs/submit_fmg_strong.sh
bsub < Experiments/HPC-jobs/submit_fmg_weak.sh

echo "All experiments submitted!"
