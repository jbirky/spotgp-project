#!/bin/bash
#SBATCH --job-name=spotgp-app
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PORT=${1:-8501}
SPOTGP_SIF=${SPOTGP_SIF:-$HOME/containers/spotgp.sif}
NODE=$(hostname)

mkdir -p logs

echo "=========================================="
echo "spotgp Explorer running on $NODE:$PORT"
echo ""
echo "Connect with:"
echo "  ssh -NL $PORT:$NODE:$PORT $(whoami)@schooner.oscer.ou.edu"
echo ""
echo "Then open: http://localhost:$PORT"
echo "=========================================="

apptainer exec \
    --bind $PWD:/work \
    --bind /ourdisk:/ourdisk \
    $SPOTGP_SIF \
    streamlit run /work/scripts/app.py \
        --server.port=$PORT \
        --server.address=0.0.0.0 \
        --server.headless=true
