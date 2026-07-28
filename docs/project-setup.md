# Project Setup

Analysis projects live in their own directories, separate from the
`spotgp-project` package. Each project gets its own configs, data cache,
results, and DVC pipeline while sharing the same scripts and spotgp library.

```
~/packages/
  spotgp-project/          # the code (scripts, docs, tests)

~/projects/
  kepler-411/              # one analysis project
    configs/
    data/lightcurves/
    results/
    params.yaml
    dvc.yaml
  sun-as-a-star/           # another project
    configs/
    ...
```

## Creating a project

Use `init_project.py` to scaffold a new project directory:

```bash
python scripts/init_project.py ~/projects/kepler-411
```

This creates:

```
kepler-411/
├── configs/               YAML run configurations
├── data/
│   └── lightcurves/       cached .npz files (DVC download stage)
├── results/               one directory per run (tracked by DVC)
├── logs/                  SLURM job logs
├── params.yaml            registered configs for dvc repro
├── dvc.yaml               DVC pipeline (download -> fit -> index)
├── Makefile               shortcuts (run, app, fetch, etc.)
├── batch_fit.sh           SLURM job array submission script
└── .gitignore
```

The generated `dvc.yaml` and `Makefile` reference the `spotgp-project`
scripts by absolute path, so commands work from the project directory
without any extra setup.

Options:

| Flag | Description |
|------|-------------|
| `--no-git` | Skip `git init` |
| `--no-dvc` | Skip `dvc init` |

If the target directory already exists, only missing files are created;
existing files are not overwritten.

## Working in a project

After creating a project, `cd` into it and work from there:

```bash
cd ~/projects/kepler-411
```

### GUI

Launch the interactive explorer from the project directory:

```bash
make app
# or: streamlit run /path/to/spotgp-project/scripts/app.py
```

The GUI's **Project directory** field (top of the sidebar) defaults to the
current working directory. All file paths -- configs, data, results,
`params.yaml` -- resolve relative to this directory.

To point the GUI at a different project without restarting, edit the
**Project directory** field in the sidebar. You can also set it on the
command line:

```bash
streamlit run /path/to/spotgp-project/scripts/app.py -- --project-dir ~/projects/kepler-411
```

### Command-line fits

Run a fit from the project directory:

```bash
make run CONFIG=configs/my_star.yaml
# or: python /path/to/spotgp-project/scripts/run_fit.py configs/my_star.yaml
```

All paths in the config (data, output, results) resolve relative to the
current working directory. To run from elsewhere, use `--project-dir`:

```bash
python /path/to/spotgp-project/scripts/run_fit.py \
    configs/my_star.yaml --project-dir ~/projects/kepler-411
```

### DVC pipeline

The generated `dvc.yaml` is ready to use:

```bash
dvc repro                   # run all stages
dvc metrics show            # print metrics
dvc plots show              # view diagnostics
```

Register new targets by adding entries to `params.yaml` (or use the GUI's
**Add object to pipeline** button):

```yaml
configs:
  KIC_7286309: configs/KIC_7286309.yaml
  my_star: configs/my_star.yaml
```

### Bulk download

Fetch all registered light curves in parallel:

```bash
make fetch
# or: python /path/to/spotgp-project/scripts/fetch_lightcurves.py --params params.yaml
```

### Batch fitting on HPC

Submit a SLURM job array for all configs in the project:

```bash
bash batch_fit.sh configs/
bash batch_fit.sh configs/ --partition=gpu --gres=gpu:1 --time=02:00:00
```

### Results index

Aggregate all runs into a CSV:

```bash
make index
```

## Make targets

The generated `Makefile` provides shortcuts for common operations:

```bash
make run CONFIG=configs/my_star.yaml   # run a single fit
make validate CONFIG=configs/my_star.yaml  # check config only
make app                               # launch the GUI
make fetch                             # bulk download light curves
make index                             # aggregate results CSV
make clean                             # remove logs and results
```

## Path conventions

All paths in config files and `params.yaml` are **project-relative**:

```yaml
# configs/my_star.yaml
data:
  path: data/lightcurves/my_star.npz   # relative to project root

output:
  save_dir: results                     # relative to project root
```

```yaml
# params.yaml
configs:
  my_star: configs/my_star.yaml        # relative to project root
```

The scripts resolve these paths against the working directory (or
`--project-dir` if specified). The GUI resolves them against the
**Project directory** field.

Absolute paths also work and are used as-is.

## Using spotgp-project directly

You can still run everything from within the `spotgp-project` directory
itself -- the default behavior is unchanged. The project directory defaults
to the current working directory, so existing workflows that use
`spotgp-project/configs/`, `spotgp-project/results/`, etc. continue to
work without modification.
