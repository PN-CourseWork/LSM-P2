#!/bin/bash
# Submit LSF job packs

if [ -z "$1" ]; then
    echo "Usage: $0 <pack_file>"
    exit 1
fi

PACK_FILE=$1

if [ ! -f "$PACK_FILE" ]; then
    echo "Error: Pack file '$PACK_FILE' not found."
    exit 1
fi

# Ensure python/uv is available in the path or modules are loaded if needed
# This script is executed on the login node to submit the jobs.
# The jobs themselves will run on compute nodes.
# Since 'uv run' is part of the command in the pack file, 
# we assume the environment where this is submitted has uv accessible 
# OR that the user's .bashrc sets it up on compute nodes.

echo "Submitting jobs from $PACK_FILE to LSF..."
bsub -pack "$PACK_FILE"

echo "Jobs submitted."