#!/bin/bash
# Run the spotgp explorer as its own SLURM job on the awesomestars node,
# for when you don't want to go through an Open OnDemand Jupyter session.
#
# Usage:
#   sbatch scripts/ood/spotgp_app.slurm [project_name]
#
# Then read logs/spotgp-app_<jobid>.out for the URL. If you are logged into
# OnDemand in the same browser, the printed /node/ URL works directly;
# otherwise use the SSH tunnel command that the log prints.

#SBATCH --job-name=spotgp-app
#SBATCH --partition=awesomestars
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PROJECT_NAME="${1:-${SPOTGP_PROJECT:-default}}"

SPOTGP_ROOT="${SPOTGP_ROOT:-/ourdisk/hpc/astrogroup/shared/spotgp-project}"
export SPOTGP_ROOT
export SPOTGP_PROJECT_ROOT="${SPOTGP_PROJECT_ROOT:-/ourdisk/hpc/astrogroup/$USER}"

mkdir -p logs

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')

echo "SSH tunnel alternative:"
echo "  ssh -NL $PORT:$(hostname):$PORT $USER@schooner.oscer.ou.edu"
echo "  then open http://localhost:$PORT/node/$(hostname -s)/$PORT/"
echo

exec "$SPOTGP_ROOT/scripts/spotgp_app.sh" "$PROJECT_NAME" --port "$PORT"
