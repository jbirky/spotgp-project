#!/bin/bash
#SBATCH --job-name=spotgp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   sbatch scripts/run_fit.slurm configs/my_star.yaml
#
# Or with a custom image path:
#   SPOTGP_SIF=/path/to/spotgp.sif sbatch scripts/run_fit.slurm configs/my_star.yaml

CONFIG=${1:?Usage: sbatch run_fit.slurm <config.yaml>}

SPOTGP_SIF=${SPOTGP_SIF:-$HOME/containers/spotgp.sif}

mkdir -p logs results

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Config:    $CONFIG"
echo "Container: $SPOTGP_SIF"
echo "Started:   $(date)"
echo "---"

apptainer exec \
    --bind $PWD:/work \
    --bind /ourdisk:/ourdisk \
    $SPOTGP_SIF \
    python /work/scripts/run_fit.py /work/$CONFIG

echo "---"
echo "Finished: $(date)"
