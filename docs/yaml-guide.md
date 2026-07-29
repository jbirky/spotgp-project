# YAML Config Guide

Every fit is driven by a single YAML config file. This page documents
each section and option.

## Quick start

Copy the example config and edit it for your star:

```bash
cp configs/example.yaml configs/my_star.yaml
python scripts/run_fit.py configs/my_star.yaml
```

Validate a config without running:

```bash
python scripts/run_fit.py configs/my_star.yaml --validate
```

All paths in the config (data files, output directories) are resolved
relative to the current working directory. When running from a different
location, use `--project-dir`:

```bash
python scripts/run_fit.py configs/my_star.yaml --project-dir ~/projects/kepler-411
```

See [Project Setup](project-setup.md) for the full project directory
workflow.

## Config reference

### `star_name`

The target identifier. Used for downloading from MAST and naming the
output directory. The mission is inferred from the prefix:

| Prefix | Mission |
|--------|---------|
| `TIC`  | TESS (SPOC) |
| `KIC`  | Kepler |
| `EPIC` | K2 |

```yaml
star_name: TIC 441420236
```

### `seed`

Optional integer for reproducible MAP restarts and sampling.

```yaml
seed: 42
```

### `device`

Optional compute device. Set to `gpu` or `tpu` to run JAX on accelerators.
When set to `gpu`, environment variables for memory allocation are
configured automatically (`XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`). Falls back to the default device
if the requested one is unavailable.

```yaml
device: gpu
```

### `data`

Controls how the light curve is loaded and preprocessed.

**Download from MAST:**

```yaml
data:
  sectors: [1, 2]       # specific sectors/quarters (omit for all)
  normalize: true        # normalize flux to per-segment median
  zero_mean: false       # subtract mean after normalization
  downsample: 1          # keep every Nth point
```

**Local file** (CSV with columns time, flux, flux_err — or `.npz` with
matching keys):

```yaml
data:
  path: data/lightcurve.csv
  normalize: true
  zero_mean: false
```

**Cached light curve** — when both `star_name` and `data.path` are set,
the runner first checks for a cached `.npz` file at `path`. If missing, it
downloads from MAST using `star_name` and saves the result to `path` for
future runs. The DVC `download` stage and the GUI's "Fetch light curves"
button both populate this cache.

```yaml
star_name: TIC 441420236
data:
  path: data/lightcurves/example.npz
  sectors: [1, 2]
  normalize: true
```

**Time binning** — set `dt` to bin the light curve into non-overlapping
windows of the given width (in days) using inverse-variance weighted
averaging. Only applies when `dt` exceeds the native cadence by at least
1.5x.

```yaml
data:
  dt: 0.02              # bin to ~30 min cadence
  normalize: true
```

### `model`

Defines the spot evolution model: envelope, visibility, and optionally a
latitude distribution.

#### Single-component model

```yaml
model:
  envelope: TrapezoidSymmetricEnvelope
  envelope_params:
    lspot: 20.0
    tau_spot: 5.0
  visibility: VisibilityFunction
  visibility_params:
    peq: 5.0
    kappa: 0.0
    inc: 1.047
  sigma_k: 0.01
```

#### Composite model

Use a `components` list to combine multiple spot kernels with different
envelopes or amplitudes. The total kernel is the sum of all component
kernels. Each component shares the same visibility and latitude distribution.

```yaml
model:
  visibility: VisibilityFunction
  visibility_params:
    peq: 5.0
    kappa: 0.0
    inc: 1.047
  components:
    - label: active-regions
      envelope: TrapezoidSymmetricEnvelope
      envelope_params:
        lspot: 20.0
        tau_spot: 5.0
      sigma_k: 0.01
    - label: long-lived
      envelope: ExponentialEnvelope
      envelope_params:
        tau_spot: 30.0
      sigma_k: 0.005
```

#### SHO terms

Add Simple Harmonic Oscillator terms to model quasi-periodic or stochastic
variability not captured by the spot model (e.g. granulation, p-mode
oscillations). SHO terms are additive — they are summed with the spot
kernel(s).

```yaml
model:
  envelope: TrapezoidSymmetricEnvelope
  envelope_params:
    lspot: 20.0
    tau_spot: 5.0
  visibility: VisibilityFunction
  visibility_params:
    peq: 5.0
    kappa: 0.0
    inc: 1.047
  sigma_k: 0.01
  sho_terms:
    - label: granulation
      sigma: 0.005
      rho: 2.0
      tau: 10.0
```

Each SHO term has:

| Parameter | Description |
|-----------|-------------|
| `label` | Name for identification |
| `sigma` | Amplitude |
| `rho` | Undamped period (days) |
| `tau` | Damping time (days) — use either `tau` or `Q` |
| `Q` | Quality factor — use either `Q` or `tau` |

#### Envelope types

| Name | Parameters | Description |
|------|-----------|-------------|
| `TrapezoidSymmetricEnvelope` | `lspot`, `tau_spot` | Symmetric trapezoidal spot lifetime |
| `TrapezoidAsymmetricEnvelope` | `lspot`, `tau_em`, `tau_dec` | Asymmetric emergence/decay |
| `ExponentialEnvelope` | `tau_spot` | Exponential decay |
| `ExponentialAsymmetricEnvelope` | `tau_em`, `tau_dec` | Separate emergence/decay timescales |
| `SkewedGaussianEnvelope` | `sigma_sn`, `n_sn` | Skewed Gaussian profile |

#### Visibility types

| Name | Parameters | Description |
|------|-----------|-------------|
| `VisibilityFunction` | `peq`, `kappa`, `inc` | Full differential rotation + inclination |
| `EdgeOnVisibilityFunction` | `peq` | Edge-on approximation (inc = 90 deg) |
| `FullGeometryVisibilityFunction` | `peq`, `kappa`, `inc` | Full geometry with limb effects |

Parameter notes:

- `peq` — equatorial rotation period in days
- `kappa` — differential rotation shear (0 = solid body, positive = solar-like)
- `inc` — stellar inclination in **radians**

#### Latitude distributions (optional)

```yaml
model:
  latitude: ButterflyLatitude
  latitude_params:
    phi0_deg: 15.0
    sigma_deg: 7.0
```

| Name | Parameters | Description |
|------|-----------|-------------|
| `UniformDoubleHemisphereBand` | `min_lat_deg`, `max_lat_deg` | Uniform within a latitude band |
| `ButterflyLatitude` | `phi0_deg`, `sigma_deg` | Double-Gaussian solar butterfly pattern |

If omitted, spots are distributed uniformly over the full sphere.

### `bounds`

Parameter bounds for optimization and sampling. Each entry is a
`[min, max]` pair:

```yaml
bounds:
  peq: [0.1, 40.0]
  kappa: [-1.0, 1.0]
  log_sigma_k: [-4.0, -1.0]
```

Prefix a parameter name with `log_` to sample in log10 space. For example,
`log_sigma_k: [-4.0, -1.0]` samples sigma_k between 10^-4 and 10^-1.

For composite models, suffix envelope parameters and `log_sigma_k` with
the component label to set per-component bounds:

```yaml
bounds:
  peq: [0.1, 40.0]
  log_sigma_k_active-regions: [-4.0, -1.0]
  log_sigma_k_long-lived: [-5.0, -2.0]
  tau_spot_active-regions: [1.0, 30.0]
```

Parameters without explicit bounds use built-in defaults.

### `priors`

Optional Gaussian priors on top of the uniform-in-bounds prior. Keys use
sampling-space names (i.e. `log_sigma_k` when bounds put that parameter in
log space):

```yaml
priors:
  peq:
    mu: 5.0
    sigma: 0.5
```

Only Gaussian (`type: gaussian`) priors are supported.

### `solver`

Kernel and matrix solver settings:

```yaml
solver:
  kernel_type: analytic
  matrix_solver: cholesky_banded
  n_harmonics: 3
  n_lat: 64
```

| Option | Default | Description |
|--------|---------|-------------|
| `kernel_type` | `analytic` | Kernel implementation |
| `matrix_solver` | `cholesky_banded` | `cholesky_banded` or `cholesky_full` |
| `n_harmonics` | `3` | Number of Fourier harmonics |
| `n_lat` | `64` | Latitude quadrature points |

### `fitting`

Controls the fitting pipeline. All three stages are optional — include only
the ones you want to run.

#### ACF initialization

Cheap ACF-based fit to initialize MAP restarts:

```yaml
fitting:
  acf_init:
    nopt: 5
```

When both `acf_init` and `map` are present with `nopt > 1`, the MAP
multi-start trials are jittered around the ACF solution.

#### MAP optimization

```yaml
fitting:
  map:
    nopt: 10              # number of multi-start restarts
```

#### MCMC sampling

```yaml
fitting:
  sampling:
    sampler: dynesty       # or: blackjax
    nlive: 500             # dynesty: number of live points
    # n_warmup: 500        # blackjax: NUTS warmup steps
    # n_samples: 2000      # blackjax: NUTS samples
```

| Sampler | Options | Description |
|---------|---------|-------------|
| `dynesty` | `nlive` | Nested sampling |
| `blackjax` | `n_warmup`, `n_samples` | NUTS (Hamiltonian MC) |

### `output`

```yaml
output:
  save_dir: results
  # run_dir: results/my-run   # override auto-generated directory name
```

Paths are relative to the project directory (see
[Project Setup](project-setup.md)).

Each run writes to `{save_dir}/{star}_{envelope}_{hash}/` containing:

```
result.h5       # data, model, solver state, MAP result, samples
config.yaml     # copy of the resolved config
metrics.json    # scalar metrics (+ git rev, wandb URL)
plots/          # ACF comparison, GP prediction, corner plot
```

### Experiment tracking

#### wandb

```yaml
wandb:
  project: my-project      # wandb project name (default: star_name)
```

The runner logs config, metrics (MAP, evidence, posterior summaries),
plots, and the HDF5 result as a versioned artifact.

#### MLflow

```yaml
mlflow:
  experiment: my-experiment
  tracking_uri: http://lab-server:5000    # optional
```

## Full example

```yaml
star_name: TIC 441420236
seed: 42
device: gpu

data:
  sectors: [1, 2]
  path: data/lightcurves/example.npz
  normalize: true
  zero_mean: false

model:
  envelope: TrapezoidSymmetricEnvelope
  envelope_params:
    lspot: 20.0
    tau_spot: 5.0
  visibility: VisibilityFunction
  visibility_params:
    peq: 5.0
    kappa: 0.0
    inc: 1.047
  sigma_k: 0.01
  latitude: ButterflyLatitude
  latitude_params:
    phi0_deg: 15.0
    sigma_deg: 7.0

bounds:
  peq: [0.1, 40.0]
  kappa: [-1.0, 1.0]
  log_sigma_k: [-4.0, -1.0]
  phi0_deg: [0.0, 60.0]
  sigma_deg: [1.0, 30.0]

priors:
  peq:
    mu: 5.0
    sigma: 0.5

solver:
  kernel_type: analytic
  matrix_solver: cholesky_banded
  n_harmonics: 3
  n_lat: 64

fitting:
  acf_init:
    nopt: 5
  map:
    nopt: 10
  sampling:
    sampler: dynesty
    nlive: 500

output:
  save_dir: results

wandb:
  project: my-project
```

## DVC pipeline

Each project directory (see [Project Setup](project-setup.md)) has its own
`dvc.yaml` and `params.yaml`. The pipeline defines three stages that run
for every config registered in `params.yaml`:

| Stage | Description |
|-------|-------------|
| `download` | Fetch light curves into `data/lightcurves/<key>.npz` |
| `fit` | Run `run_fit.py` and produce `result.h5`, `metrics.json`, and plots |
| `index` | Aggregate all runs into `results/results_index.csv` |

### Registering targets

`params.yaml` maps short keys to config file paths:

```yaml
configs:
  example: configs/example.yaml
  KIC_7286309: configs/KIC_7286309.yaml
```

You can add entries manually, or use the GUI's **Add object to pipeline**
button to register the current config automatically.

### Running the pipeline

```bash
dvc repro               # run all stages (skips if nothing changed)
dvc metrics show         # print metrics.json
dvc plots show           # view diagnostic plots
dvc push                 # push result.h5 to shared storage
```

### Comparing experiments

```bash
git checkout -b exponential-envelope
# edit config, then:
dvc repro
git add -A && git commit -m "test exponential envelope"
dvc metrics diff main    # compare metrics
dvc plots diff main      # compare plots
```

## Batch fitting on HPC

For running many configs on a SLURM cluster, use the job array script:

```bash
bash scripts/batch_fit.sh configs/
```

This submits one SLURM array job with one task per YAML file in the
directory. Extra SBATCH flags are forwarded:

```bash
bash scripts/batch_fit.sh configs/ --partition=gpu --gres=gpu:1 --time=02:00:00
```

GPU environment variables are configured automatically when GPU resources
are requested.

## Bulk light curve download

Download all `params.yaml` targets in parallel before fitting:

```bash
python scripts/fetch_lightcurves.py --params params.yaml --workers 6
```

Options:

| Flag | Description |
|------|-------------|
| `--params` | Path to `params.yaml` with a `configs:` map (default) |
| `--config` | Single config file (single-target mode) |
| `--out` | Output directory (bulk) or `.npz` file path (single) |
| `--only` | Comma-separated subset of config keys to fetch |
| `--workers` | Number of parallel download threads (default: 6) |
| `--retries` | Retry count with exponential backoff (default: 3) |
| `--force` | Re-download even if the cache file exists |

A manifest CSV (`data/lightcurves/manifest.csv`) records every target's
status, point/sector counts, and any errors.

## Results index

Aggregate all pipeline runs into a single CSV:

```bash
python scripts/build_index.py results --out results/results_index.csv
```

This scans `results/*/config.yaml` and `results/*/metrics.json` pairs and
builds one row per run. The DVC `index` stage runs this automatically after
all fits complete.
