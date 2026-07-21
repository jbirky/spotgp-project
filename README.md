# spotgp-project

Template for running [spotgp](https://github.com/jbirky/spotgp) GP fits with
config-driven experiments, HDF5 results, DVC pipelines, and optional experiment
tracking via wandb or MLflow.

## Quick start

```bash
git clone https://github.com/jbirky/spotgp-project.git
cd spotgp-project
pip install -r requirements.txt
```

Create a config, validate it, and run:

```bash
cp configs/example.yaml configs/my_star.yaml
python scripts/run_fit.py configs/my_star.yaml --validate
python scripts/run_fit.py configs/my_star.yaml
```

Launch the interactive explorer:

```bash
make app
```

## Documentation

Full documentation is available on the project website:

- [**Installation**](docs/installation.md) — setup, dependencies, HPC containers, project structure
- [**Interactive Explorer**](docs/gui-guide.md) — using the Streamlit GUI to tune models and run fits
- [**YAML Config Guide**](docs/yaml-guide.md) — config reference for the command-line runner and DVC pipelines
- [**Projects**](docs/projects/index.md) — contributed analyses and results

Build and preview the docs locally:

```bash
make serve       # preview at localhost:8000
make docs        # build static site to site/
```

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
