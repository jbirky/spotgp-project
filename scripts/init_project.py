"""Initialize a new spotgp analysis project directory.

Creates a standalone project directory with its own configs, data, results,
and DVC pipeline that references the spotgp-project scripts by path.

Usage:
    python scripts/init_project.py ~/projects/kepler-411
    python scripts/init_project.py ./my-project --no-git --no-dvc
"""

import argparse
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPTS_DIR)

PARAMS_TEMPLATE = "configs: {}\n"

GITIGNORE_TEMPLATE = """\
# Run outputs
results/*
!results/.gitkeep

# DVC-managed light-curve cache
data/lightcurves/

# SLURM logs
logs/*.out
logs/*.err

# Python
__pycache__/
*.pyc

# Experiment tracking
wandb/
mlruns/

# DVC local storage
.dvc-storage/
"""


def _dvc_template(scripts_dir):
    return f"""\
stages:
  download:
    foreach: ${{configs}}
    do:
      cmd: python {scripts_dir}/fetch_lightcurves.py --config ${{item}} --out data/lightcurves/${{key}}.npz
      deps:
        - ${{item}}
      outs:
        - data/lightcurves/${{key}}.npz

  fit:
    foreach: ${{configs}}
    do:
      cmd: python {scripts_dir}/run_fit.py ${{item}} --output-dir results/${{key}}
      deps:
        - ${{item}}
        - data/lightcurves/${{key}}.npz
      outs:
        - results/${{key}}/result.h5
      plots:
        - results/${{key}}/plots:
            cache: false
      metrics:
        - results/${{key}}/metrics.json:
            cache: false

  index:
    cmd: python {scripts_dir}/build_index.py results --out results/results_index.csv
    always_changed: true
    outs:
      - results/results_index.csv:
          cache: false
"""


def _makefile_template(scripts_dir):
    return f"""\
SPOTGP_SCRIPTS = {scripts_dir}
CONFIG         ?= configs/example.yaml
PORT           ?= 8501

.PHONY: run validate app fetch index clean

run:
\tpython $(SPOTGP_SCRIPTS)/run_fit.py $(CONFIG)

validate:
\tpython $(SPOTGP_SCRIPTS)/run_fit.py $(CONFIG) --validate

app:
\tstreamlit run $(SPOTGP_SCRIPTS)/app.py --server.port=$(PORT)

fetch:
\tpython $(SPOTGP_SCRIPTS)/fetch_lightcurves.py --params params.yaml

index:
\tpython $(SPOTGP_SCRIPTS)/build_index.py results --out results/results_index.csv

clean:
\trm -rf logs/*.out logs/*.err results/*/ results/*.h5
"""


def _batch_template(scripts_dir):
    return f"""\
#!/bin/bash
# Submit SLURM job arrays for a directory of spotgp config files.
#
# Usage:
#   bash batch_fit.sh configs/
#   bash batch_fit.sh configs/ --partition=gpu --gres=gpu:1 --time=02:00:00

set -euo pipefail

SPOTGP_SCRIPTS="{scripts_dir}"
CONFIG_DIR="${{1:?Usage: batch_fit.sh <config_dir> [extra sbatch flags...]}}"
shift
EXTRA_SBATCH_FLAGS=("$@")

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -o -name '*.yml' | sort)

if [[ ${{#CONFIGS[@]}} -eq 0 ]]; then
    echo "No YAML files found in $CONFIG_DIR" >&2
    exit 1
fi

echo "Found ${{#CONFIGS[@]}} config files in $CONFIG_DIR"

FILELIST=$(mktemp "${{CONFIG_DIR}}/.batch_filelist_XXXXXX")
printf '%s\\n' "${{CONFIGS[@]}}" > "$FILELIST"

mkdir -p logs results

GPU_REQUESTED=false
for flag in "${{EXTRA_SBATCH_FLAGS[@]}}"; do
    if [[ "$flag" == *gres=gpu* || "$flag" == *partition=gpu* ]]; then
        GPU_REQUESTED=true
        break
    fi
done

GPU_ENV=""
if $GPU_REQUESTED; then
    GPU_ENV='export XLA_PYTHON_CLIENT_MEM_FRACTION=${{XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}}'
fi

SPOTGP_SIF="${{SPOTGP_SIF:-$HOME/containers/spotgp.sif}}"

JOB_ID=$(sbatch \\
    --job-name=spotgp-batch \\
    --array=1-${{#CONFIGS[@]}} \\
    --nodes=1 \\
    --ntasks=1 \\
    --cpus-per-task=4 \\
    --mem=8G \\
    --time=04:00:00 \\
    --output=logs/spotgp_%A_%a.out \\
    --error=logs/spotgp_%A_%a.err \\
    "${{EXTRA_SBATCH_FLAGS[@]}}" \\
    --parsable \\
    --wrap "$(cat <<EOF
${{GPU_ENV}}
CONFIG=\\$(sed -n "\\${{SLURM_ARRAY_TASK_ID}}p" "$FILELIST")
echo "Job \\$SLURM_ARRAY_TASK_ID / ${{#CONFIGS[@]}}: \\$CONFIG"
echo "Node: \\$SLURM_NODELIST"
echo "Started: \\$(date)"
echo "---"
python $SPOTGP_SCRIPTS/run_fit.py \\$CONFIG
echo "---"
echo "Finished: \\$(date)"
EOF
)")

echo "Submitted job array ${{JOB_ID}} (${{#CONFIGS[@]}} tasks)"
echo "File list: $FILELIST"
echo "Monitor:   squeue -j ${{JOB_ID}}"
echo "Logs:      logs/spotgp_${{JOB_ID}}_*.{{out,err}}"
"""


def init_project(target_dir, init_git=True, init_dvc=True):
    target_dir = os.path.abspath(target_dir)

    for d in ("configs", "data/lightcurves", "results", "logs"):
        os.makedirs(os.path.join(target_dir, d), exist_ok=True)

    open(os.path.join(target_dir, "results", ".gitkeep"), "a").close()

    def _write_if_missing(rel, content):
        path = os.path.join(target_dir, rel)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)

    _write_if_missing("params.yaml", PARAMS_TEMPLATE)
    _write_if_missing("dvc.yaml", _dvc_template(SCRIPTS_DIR))
    _write_if_missing(".gitignore", GITIGNORE_TEMPLATE)
    _write_if_missing("Makefile", _makefile_template(SCRIPTS_DIR))
    _write_if_missing("batch_fit.sh", _batch_template(SCRIPTS_DIR))

    if init_git:
        if not os.path.isdir(os.path.join(target_dir, ".git")):
            subprocess.run(["git", "init"], cwd=target_dir,
                           capture_output=True)

    if init_dvc:
        if not os.path.isdir(os.path.join(target_dir, ".dvc")):
            try:
                subprocess.run(["dvc", "init"], cwd=target_dir,
                               capture_output=True)
            except FileNotFoundError:
                print("Note: DVC not available; skipping `dvc init`")

    print(f"Initialized spotgp project: {target_dir}")
    print(f"  Scripts: {SCRIPTS_DIR}")
    print()
    print("Next steps:")
    print(f"  cd {target_dir}")
    print(f"  streamlit run {SCRIPTS_DIR}/app.py    # launch the GUI")
    print(f"  # or: make app")


def main():
    ap = argparse.ArgumentParser(
        description="Initialize a new spotgp analysis project")
    ap.add_argument("target", help="Directory to create the project in")
    ap.add_argument("--no-git", action="store_true",
                    help="Skip git init")
    ap.add_argument("--no-dvc", action="store_true",
                    help="Skip dvc init")
    args = ap.parse_args()

    init_project(args.target,
                 init_git=not args.no_git,
                 init_dvc=not args.no_dvc)


if __name__ == "__main__":
    main()
