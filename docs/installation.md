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
interactive explorer (Streamlit), and all I/O and pipeline dependencies.

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

## Optional dependencies

These are commented out in `requirements.txt` — uncomment or install
manually as needed:

| Package | Purpose |
|---------|---------|
| `blackjax` | NUTS sampler (alternative to dynesty) |
| `wandb` | Experiment tracking with Weights & Biases |
| `mlflow` | Experiment tracking with MLflow |

### Setting up wandb

1. Install: `pip install wandb`
2. Create a free account at [wandb.ai](https://wandb.ai)
3. Authenticate: `wandb login` (paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize))
4. Add a `wandb:` section to your config YAML (see [YAML config guide](yaml-guide.md#experiment-tracking))

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

```
spotgp-project/
├── configs/           YAML run configurations
├── data/              input light curve files
├── results/           one directory per run (tracked by DVC)
├── logs/              SLURM job logs
├── scripts/
│   ├── run_fit.py     config-driven fit runner
│   ├── data_utils.py  shared light-curve loading helpers
│   ├── app.py         interactive Streamlit explorer
│   ├── run_fit.slurm  SLURM job script for HPC clusters
│   └── run_app.slurm  SLURM job script for the explorer app
├── tests/             config and end-to-end smoke tests
├── docs/              project website (MkDocs)
├── dvc.yaml           DVC pipeline definition
├── params.yaml        active config + run directory for dvc repro
├── requirements.txt   pinned dependency list
└── Makefile           shortcuts for common commands
```

## Make shortcuts

```bash
make run CONFIG=configs/example.yaml            # local python
make run-container CONFIG=configs/example.yaml   # apptainer
make submit CONFIG=configs/example.yaml          # sbatch
make validate CONFIG=configs/example.yaml        # check config only
make app                                         # streamlit explorer
make test                                        # run the test suite
make docs                                        # build docs site
make serve                                       # preview docs locally
make shell                                       # shell in container
```
