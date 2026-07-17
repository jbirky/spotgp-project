# spotgp-project

Template for running [spotgp](https://github.com/jbirky/spotgp) GP fits with
config-driven experiments, HDF5 results, DVC pipelines, and optional experiment
tracking via wandb or MLflow.

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
├── dvc.yaml           DVC pipeline definition
├── params.yaml        active config + run directory for dvc repro
├── requirements.txt   pinned dependency list
└── Makefile           shortcuts for common commands
```

Each fit writes a self-contained run directory:

```
results/TIC_441420236_TrapezoidSymmetric_a3f7c2/
├── result.h5          data, model, solver state, MAP result, samples
├── config.yaml        copy of the resolved config
├── metrics.json       scalar metrics (MAP, evidence, posterior summaries)
└── plots/             ACF comparison, GP prediction, corner plot
```

`result.h5` also embeds the full config, the spotgp version, the seed, and a
timestamp as root attributes, so every result file is reproducible on its own.

## Quick start

### 1. Create your project

Fork or clone this template, then install dependencies:

```bash
git clone https://github.com/<your-username>/spotgp-project.git
cd spotgp-project
pip install -r requirements.txt
```

On an HPC cluster with Apptainer, pull the pre-built container instead:

```bash
mkdir -p ~/containers
apptainer pull ~/containers/spotgp.sif docker://ghcr.io/<org>/spotgp:latest
```

### 2. Create a config

Copy the example and edit it for your star:

```bash
cp configs/example.yaml configs/tic441420236.yaml
```

Set `star_name` to a TIC ID (e.g. `TIC 441420236`), KIC ID
(e.g. `KIC 8462852`), or EPIC ID. The runner downloads the light curve
automatically from MAST via lightkurve, inferring the mission from the ID
(TIC → TESS/SPOC, KIC → Kepler, EPIC → K2).

```yaml
star_name: TIC 441420236
seed: 42                  # optional: reproducible restarts and sampling

data:
  sectors: [1, 2]         # sectors/quarters to download (omit for all)
  normalize: true         # normalize flux to per-segment median
  zero_mean: false        # subtract mean after normalization
  downsample: 1           # keep every Nth point

model:
  envelope: TrapezoidSymmetricEnvelope
  envelope_params:
    lspot: 15.0
    tau_spot: 5.0
  visibility: VisibilityFunction
  visibility_params:
    peq: 3.3
    kappa: 0.1
    inc: 1.31
  sigma_k: 0.005

bounds:
  peq: [1.0, 10.0]
  kappa: [0.0, 0.5]
  log_sigma_k: [-5.0, -1.0]

solver:
  kernel_type: analytic
  matrix_solver: cholesky_banded

fitting:
  map:
    nopt: 10
  sampling:
    sampler: dynesty
    nlive: 500

output:
  save_dir: results
```

To use a local file instead of downloading, set `path` under `data`:

```yaml
data:
  path: data/lightcurve.csv   # CSV columns: time, flux, flux_err
  normalize: true
  zero_mean: false
```

Local files can be CSV (columns read in order: time, flux, flux_err) or
NumPy `.npz` (with keys `time`, `flux`, `flux_err`).

**Envelope types:** `TrapezoidSymmetricEnvelope`, `TrapezoidAsymmetricEnvelope`,
`ExponentialEnvelope`, `ExponentialAsymmetricEnvelope`, `SkewedGaussianEnvelope`

**Visibility types:** `VisibilityFunction`, `EdgeOnVisibilityFunction`,
`FullGeometryVisibilityFunction`

**Samplers:** `blackjax` (NUTS), `dynesty` (nested sampling)

Optional extras (see `configs/example.yaml` for the full annotated schema):

```yaml
priors:                  # Gaussian priors on top of the bounds prior
  peq:
    mu: 3.3
    sigma: 0.3

fitting:
  acf_init:              # cheap ACF fit to initialize the MAP restarts
    nopt: 5
```

### 3. Validate the config

```bash
python scripts/run_fit.py configs/tic441420236.yaml --validate
```

This checks the config structure and class names (listing the valid options
on a typo) without downloading data or running anything — useful before
submitting to a cluster queue.

### 4. Run the fit

```bash
python scripts/run_fit.py configs/tic441420236.yaml
```

Results are saved to a run directory whose name encodes the star, envelope
type, and a hash of the config, so different parameters never collide:

```
results/TIC_441420236_TrapezoidSymmetric_a3f7c2/
```

Use `--output-dir` (or `output.run_dir` in the config) to override the
directory. When sampling runs, `metrics.json` includes per-parameter
posterior medians and 16/84-percentile uncertainties alongside the
log-evidence.

### 5. Inspect results

```python
import h5py

with h5py.File("results/TIC_441420236_TrapezoidSymmetric_a3f7c2/result.h5") as f:
    print(f.attrs["config_yaml"])       # full config used for this run
    print(dict(f["map"].attrs))         # MAP parameter values
    t = f["data/time"][:]
    y = f["data/flux"][:]
```

## Interactive explorer

```bash
make app          # streamlit run scripts/app.py
```

The Streamlit app downloads light curves (TESS/Kepler/K2), visualizes the
kernel, PSD, harmonic components, and single-spot flux for hand-tuned
parameters, and runs MAP fits with a configurable set of free parameters
and bounds. Once you find a good starting point, **Export config** in the
sidebar writes the current settings to `configs/<star>.yaml`, ready for
`run_fit.py` or the DVC pipeline.

On an HPC cluster, run it through SLURM and tunnel the port:

```bash
sbatch scripts/run_app.slurm
```

## Running on an HPC cluster

Submit a SLURM job that runs inside the Apptainer container:

```bash
sbatch scripts/run_fit.slurm configs/tic441420236.yaml
```

Adjust resource requests in `scripts/run_fit.slurm`:

```bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
```

Set a custom container path:

```bash
SPOTGP_SIF=/path/to/spotgp.sif sbatch scripts/run_fit.slurm configs/my_star.yaml
```

### Make shortcuts

```bash
make run CONFIG=configs/tic441420236.yaml            # local python
make run-container CONFIG=configs/tic441420236.yaml  # apptainer
make submit CONFIG=configs/tic441420236.yaml         # sbatch
make validate CONFIG=configs/tic441420236.yaml       # check config only
make app                                             # streamlit explorer
make test                                            # run the test suite
make shell                                           # shell in container
```

## DVC pipeline

The project uses [DVC](https://dvc.org) for reproducible pipelines and data
versioning. `params.yaml` selects the active config and the run directory
that DVC tracks:

```yaml
config: configs/tic441420236.yaml
run_dir: results/tic441420236
```

Then run the pipeline:

```bash
dvc repro                  # run the fit (skips if nothing changed)
dvc metrics show           # print metrics.json
dvc plots show             # view diagnostic plots
dvc push                   # push result.h5 to shared storage
```

The HDF5 result is a declared pipeline output, so `dvc push`/`dvc pull`
move it through the DVC remote. If your config reads a local data file,
add it to the `deps` list in `dvc.yaml` so edits trigger a re-run.

### Comparing experiments

Create a git branch per experiment, change the config, and compare:

```bash
git checkout -b exponential-envelope
# edit the config → change envelope type
dvc repro
git add -A && git commit -m "test exponential envelope"

dvc metrics diff main      # compare metrics against main branch
dvc plots diff main        # compare plots
```

### Configuring the DVC remote

By default, DVC uses a local storage directory inside the project
(`.dvc-storage/`, resolved relative to `.dvc/config` so clones keep
working). For a shared HPC filesystem:

```bash
dvc remote modify local_storage url /ourdisk/hpc/your-project/dvc-cache
```

## Experiment tracking

Enable experiment tracking by adding a `wandb` section to your config.

### wandb

1. Install wandb:

   ```bash
   pip install wandb
   ```

2. Create a free account at [wandb.ai](https://wandb.ai) if you don't have one.

3. Authenticate on your machine:

   ```bash
   wandb login
   ```

   This will prompt you to paste your API key from
   [wandb.ai/authorize](https://wandb.ai/authorize). The key is stored in
   `~/.netrc` so you only need to do this once per machine.

4. Enable wandb in your config YAML by adding a `wandb` section:

   ```yaml
   wandb:
     project: my-project
   ```

   The `project` field sets the wandb project name where runs are grouped
   (defaults to `star_name` if omitted).

The runner automatically logs the following to wandb:

- **Config**: envelope type, visibility type, all model parameters, bounds
- **Metrics**: negative log posterior (MAP), sampler statistics (log-evidence,
  divergences, step size), posterior parameter summaries
- **Plots**: ACF comparison, GP prediction, corner plot (uploaded as images)
- **Artifacts**: the HDF5 result file (versioned as a wandb Artifact)

View results at `wandb.ai/<your-username>/<project>`.

MLflow tracking is also supported — see `scripts/run_fit.py` for details.

If neither wandb nor MLflow is configured, the runner still saves metrics and
plots into the run directory for DVC tracking.

## Running multiple experiments

Create one config per experiment:

```bash
cp configs/tic441420236.yaml configs/tic441420236_exponential.yaml
# edit the envelope type, then:
python scripts/run_fit.py configs/tic441420236_exponential.yaml
```

Each config produces a unique run directory. The configs are tracked in git,
so the full history of what was tried is in the commit log.

## Tests

```bash
make test        # or: pytest -v
```

Config-schema and data-helper tests run with only numpy/pyyaml installed;
the end-to-end smoke test (a MAP fit on a tiny synthetic light curve) runs
when spotgp and jax are available and skips otherwise. The same suite runs
in CI via `.github/workflows/ci.yml`.
