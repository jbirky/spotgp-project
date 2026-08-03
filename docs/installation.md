# Installation

## Requirements

- Python 3.9+
- [spotgp](https://github.com/jbirky/spotgp) with JAX support
- Dependencies listed in `requirements.txt`

## Local install

Clone the repository and install dependencies:

```bash
git clone https://github.com/jbirky/spotgp-project.git
cd spotgp-project
pip install -r requirements.txt
```

This installs spotgp (with JAX), the fitting backends (dynesty), the
interactive explorer (Streamlit), experiment tracking (wandb), and all
I/O and pipeline dependencies.

## HPC / container install

On an HPC cluster with Apptainer, pull the pre-built container instead of
installing locally:

```bash
mkdir -p ~/containers
apptainer pull ~/containers/spotgp.sif docker://ghcr.io/<org>/spotgp:latest
```

Set a custom container path when running:

```bash
SPOTGP_SIF=/path/to/spotgp.sif sbatch scripts/run_fit.slurm configs/my_star.yaml
```

To serve the interactive explorer to a whole group through the browser — one
shared install, per-user project directories, no local setup — see
[Running on OSCER](oscer.md).

## Optional dependencies

These are commented out in `requirements.txt` — uncomment or install
manually as needed:

| Package | Purpose |
|---------|---------|
| `blackjax` | NUTS sampler (alternative to dynesty) |
| `mlflow` | Experiment tracking with MLflow |

### Setting up wandb

wandb is included in the default dependencies. To use it:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Authenticate: `wandb login` (paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize))
3. Add a `wandb:` section to your config YAML (see [YAML config guide](yaml-guide.md#experiment-tracking))

### Setting up MLflow

1. Install: `pip install mlflow`
2. Add an `mlflow:` section to your config YAML with the tracking URI

## Running tests

```bash
make test        # or: pytest -v
```

Config-schema and data-helper tests run with only numpy and pyyaml. The
end-to-end smoke test (a MAP fit on a tiny synthetic light curve) runs when
spotgp and jax are available and skips otherwise.

## Project structure

The `spotgp-project` repository contains the scripts, tests, and docs.
Analysis work happens in **separate project directories** that you create
with `init_project.py` (see [Project Setup](project-setup.md)):

```bash
python scripts/init_project.py ~/projects/kepler-411
cd ~/projects/kepler-411
make app    # launch the GUI pointed at this project
```

Each project gets its own `configs/`, `data/`, `results/`, `params.yaml`,
and `dvc.yaml`, while sharing the spotgp-project scripts.

### Repository layout

```
spotgp-project/
├── scripts/
│   ├── run_fit.py         config-driven fit runner
│   ├── data_utils.py      shared light-curve loading helpers
│   ├── app.py             interactive Streamlit explorer
│   ├── fetch_lightcurves.py  bulk parallel light-curve downloader
│   ├── build_index.py     aggregate pipeline runs into a CSV index
│   ├── init_project.py    scaffold a new analysis project
│   ├── batch_fit.sh       SLURM job array submission script
│   ├── run_fit.slurm      SLURM job script for single fits
│   └── run_app.slurm      SLURM job script for the explorer app
├── tests/                 config and end-to-end smoke tests
├── docs/                  project website (MkDocs)
├── requirements.txt       dependency list
└── Makefile               shortcuts for common commands
```

### Project directory layout

Created by `init_project.py` (or by working directly in `spotgp-project`):

```
my-project/
├── configs/               YAML run configurations
├── data/
│   └── lightcurves/       cached .npz files (DVC download stage)
├── results/               one directory per run (tracked by DVC)
│   └── results_index.csv  aggregated results (DVC index stage)
├── logs/                  SLURM job logs
├── params.yaml            registered configs for dvc repro
├── dvc.yaml               DVC pipeline (download → fit → index)
├── Makefile               shortcuts referencing spotgp-project scripts
└── batch_fit.sh           SLURM job array submission
```

## Make shortcuts

From the `spotgp-project` directory (or a project created with
`init_project.py`):

```bash
make run CONFIG=configs/example.yaml            # local python
make validate CONFIG=configs/example.yaml        # check config only
make app                                         # streamlit explorer
make test                                        # run the test suite (spotgp-project only)
make init PROJECT_DIR=~/projects/my-star         # create a new project
```

From the `spotgp-project` directory only:

```bash
make run-container CONFIG=configs/example.yaml   # apptainer
make submit CONFIG=configs/example.yaml          # sbatch
make docs                                        # build docs site
make serve                                       # preview docs locally
make shell                                       # shell in container
```
