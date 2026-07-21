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

### `model`

Defines the spot evolution model: envelope, visibility, and optionally a
latitude distribution.

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

Each run writes to `{save_dir}/{star}_{envelope}_{hash}/` containing:

```
result.h5       # data, model, solver state, MAP result, samples
config.yaml     # copy of the resolved config
metrics.json    # scalar metrics
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

data:
  sectors: [1, 2]
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

The project uses [DVC](https://dvc.org) for reproducible pipelines.
`params.yaml` selects the active config:

```yaml
config: configs/my_star.yaml
run_dir: results/my_star
```

```bash
dvc repro               # run the fit (skips if nothing changed)
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
