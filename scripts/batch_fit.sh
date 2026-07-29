#!/bin/bash
# Submit SLURM job arrays for a directory of spotgp config files.
#
# Usage:
#   bash scripts/batch_fit.sh configs/
#   bash scripts/batch_fit.sh configs/ --partition=gpu --gres=gpu:1 --time=02:00:00
#
# The first argument is a directory (or glob) of YAML configs.
# Any additional arguments are forwarded as extra SBATCH flags.

set -euo pipefail

CONFIG_DIR="${1:?Usage: batch_fit.sh <config_dir> [extra sbatch flags...]}"
shift
EXTRA_SBATCH_FLAGS=("$@")

# Collect config files
mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -o -name '*.yml' | sort)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No YAML files found in $CONFIG_DIR" >&2
    exit 1
fi

echo "Found ${#CONFIGS[@]} config files in $CONFIG_DIR"

# Write the file list so the array job can index into it
FILELIST=$(mktemp "${CONFIG_DIR}/.batch_filelist_XXXXXX")
printf '%s\n' "${CONFIGS[@]}" > "$FILELIST"

mkdir -p logs results

# Detect if GPU resources were requested
GPU_REQUESTED=false
for flag in "${EXTRA_SBATCH_FLAGS[@]}"; do
    if [[ "$flag" == *gres=gpu* || "$flag" == *partition=gpu* ]]; then
        GPU_REQUESTED=true
        break
    fi
done

# Build the GPU env preamble
GPU_ENV=""
if $GPU_REQUESTED; then
    GPU_ENV='export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}'
fi

SPOTGP_SIF="${SPOTGP_SIF:-$HOME/containers/spotgp.sif}"

# Submit as a SLURM job array
JOB_ID=$(sbatch \
    --job-name=spotgp-batch \
    --array=1-${#CONFIGS[@]} \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time=04:00:00 \
    --output=logs/spotgp_%A_%a.out \
    --error=logs/spotgp_%A_%a.err \
    "${EXTRA_SBATCH_FLAGS[@]}" \
    --parsable \
    --wrap "$(cat <<EOF
${GPU_ENV}
CONFIG=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$FILELIST")
echo "Job \$SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}: \$CONFIG"
echo "Node: \$SLURM_NODELIST"
echo "Started: \$(date)"
echo "---"
apptainer exec \\
    --bind \$PWD:/work \\
    --bind /ourdisk:/ourdisk \\
    ${SPOTGP_SIF} \\
    python /work/scripts/run_fit.py /work/\$CONFIG
echo "---"
echo "Finished: \$(date)"
EOF
)")

echo "Submitted job array ${JOB_ID} (${#CONFIGS[@]} tasks)"
echo "File list: $FILELIST"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      logs/spotgp_${JOB_ID}_*.{out,err}"
