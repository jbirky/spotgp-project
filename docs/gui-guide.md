# Interactive Explorer (GUI)

The Streamlit app provides an interactive interface for loading light curves,
tuning model parameters by hand, running MAP optimization, and exporting
configs for batch fitting.

## Launching the app

From a project directory (see [Project Setup](project-setup.md)):

```bash
make app
```

Or from anywhere, pointing at a specific project:

```bash
streamlit run /path/to/spotgp-project/scripts/app.py -- --project-dir ~/projects/kepler-411
```

The app opens at `http://localhost:8501`. On an HPC cluster, submit through
SLURM and tunnel the port:

```bash
sbatch scripts/run_app.slurm
```

## Top panel — Data & Fit Results

The collapsible panel at the top of the main area contains data loading
controls and a live fit-results table.

### Data source

Load a light curve from one of two sources:

- **TIC / KIC ID** — enter a target name (e.g. `TIC 441420236` or
  `KIC 7286309`) and optionally restrict to specific sectors or quarters.
  Select a **Pipeline** (Auto-detect, TESS/SPOC, TESS/QLP, TESS/TESS-SPOC,
  Kepler, K2) to prefer a specific data product when multiple are available
  for the same sector.
  Click **Download** to fetch the light curve from MAST via lightkurve.
- **Local file** — enter the path to a CSV or `.npz` file and click **Load**.

After loading, data processing options appear:

- **Bin width dt** — bin the light curve in time (0 = no binning)
- **Normalize** — normalize flux to the per-segment median
- **Zero mean** — subtract the mean flux

### Fit results table

Every object you download or load gets a row in the results table. Columns
fill in as you **Build model** and **Run MAP Fit**:

| Column | Source |
|--------|--------|
| Object ID | Star name or filename |
| Visibility function | From the model sidebar |
| Latitude function | From the model sidebar |
| Kernel components | Labels and types of all components |
| MAP parameter values | Fitted values after a MAP run |
| Neg. log posterior | MAP objective value |
| Run dir | Set when loading pipeline runs |
| Git rev | Commit hash from pipeline runs |
| W&B | Link to the wandb run (if tracked) |

The table can be downloaded as CSV, or cleared and rebuilt.

#### Pipeline integration buttons

Four buttons below the table connect the interactive session to the
DVC/W&B tracking pipeline:

| Button | Action |
|--------|--------|
| **Load from results/** | Merge tracked pipeline runs (`results/*/metrics.json`) into the table |
| **Add object to pipeline** | Write the current config and register it in `params.yaml` for `dvc exp run` |
| **Write index CSV** | Scan `results/` and write `results/results_index.csv` |
| **Push table to W&B** | Log the current table as a `wandb.Table` dashboard |

The **Fetch light curves** button downloads every target registered in
`params.yaml` into `data/lightcurves/` in parallel (configurable number of
workers), with retry and resume support. A progress bar tracks completion.

## Sidebar controls

The sidebar is organized into numbered sections. Work through them in order
for a typical workflow.

### Device

Select the compute device (CPU, GPU, or TPU) for JAX computations. The
available options depend on your hardware.

### Project directory

All file paths -- configs, data, results, `params.yaml` -- resolve relative
to this directory. Defaults to the current working directory (or the
`--project-dir` CLI argument).

Change it to switch between projects without restarting the app. See
[Project Setup](project-setup.md) for details on creating project
directories.

### 1 - Data (source queue)

The sidebar's data section provides a **batch queue** for stepping through
multiple targets without restarting the app.

Enter the path to an **Object list** file:

- **Plain text** — one star name per line
- **CSV** — first column is the ID; additional columns can set slider
  parameters (e.g. `peq`, `kappa`, `inc`, `lspot`, `tau_spot`,
  `log_sigma_k`) that are applied automatically when stepping to each target

When a queue is loaded, **Prev** / **Next** buttons step through the list,
and the counter shows progress (e.g. `3/12 (2 done)`). **Save & Next**
exports the current config to a directory and advances to the next target.

### 3 - Model

#### Visibility

Choose the visibility function and adjust its parameters:

| Visibility | Parameters |
|------------|-----------|
| `VisibilityFunction` | `P_eq`, `kappa`, `inc` |
| `EdgeOnVisibilityFunction` | `P_eq` |
| `FullGeometryVisibilityFunction` | `P_eq`, `kappa`, `inc` |

#### Latitude distribution

| Distribution | Parameters |
|-------------|-----------|
| Uniform | (none) |
| Uniform Band | `min_lat`, `max_lat` |
| Butterfly | `phi0`, `sigma_phi` |

#### Kernel components

The model is built from one or more **kernel components**. Each component is
either a **Spot** (spot evolution model) or an **SHO Term** (simple harmonic
oscillator).

Click **+ Add Component** to add additional components. Each component gets
its own collapsible section with a label, type selector, and parameter
sliders. Components can be removed individually.

**Spot components** have:

- **Envelope** — the spot evolution profile:

| Envelope | Parameters |
|----------|-----------|
| `TrapezoidSymmetricEnvelope` | `lspot`, `tau_spot` |
| `TrapezoidAsymmetricEnvelope` | `lspot`, `tau_em`, `tau_dec` |
| `ExponentialEnvelope` | `tau_spot` |
| `ExponentialAsymmetricEnvelope` | `tau_em`, `tau_dec` |
| `SkewedGaussianEnvelope` | `sigma_sn`, `n_sn` |

- **log sigma_k** — the log amplitude of this component's kernel

**SHO Term components** model quasi-periodic or stochastic variability not
captured by the spot model (e.g. granulation, p-mode oscillations):

| Parameter | Description |
|-----------|-------------|
| `sigma` | Amplitude |
| `rho` (days) | Undamped period |
| `tau` (days) | Damping time (alternative to Q) |
| `Q` | Quality factor (alternative to tau) |

The damping can be specified as either a damping time (`tau`) or a quality
factor (`Q`), selected via a radio button.

Every slider is paired with a number input box for precise values.

**Build model** computes the kernel from the current parameters. **Preview GP**
computes the GP prediction on the loaded data using the manual parameters.

When multiple components are present, the total kernel is the sum of all
component kernels (composite kernel).

### 4 - Fit (Optimize)

Run a MAP (maximum a posteriori) optimization on the loaded data:

1. Select the parameters to fit from the multiselect (defaults to all free
   parameters).
2. Adjust the bounds for each selected parameter using the min/max inputs.
3. Set the number of restarts.
4. Click **Run MAP Fit**.

After the fit completes, the MAP kernel and GP prediction are overlaid on
the manual model in the main panel plots.

!!! note
    SHO term parameters are not included in the MAP fit. Use their
    sliders to adjust them manually.

## Main panel

The main panel is organized into four tabs:

### Light curve tab

The light curve scatter plot. Sector boundaries are shaded and labeled
(e.g. S1, S2). When a GP prediction or MAP fit is available, the mean and
2-sigma envelope are overlaid in red (manual) or blue (MAP).

### Kernel & PSD tab

**Kernel / ACF** — The model autocovariance function compared to the data
ACF (when data is loaded). Controls above the plot:

- **Max lag** and **Number of lag points** — adjust the lag range
- **Show components** — when multiple kernel components are present, overlay
  each component's contribution as a dashed line
- **n=0, 1, 2, 3** checkboxes — toggle individual harmonic components.
  These filter both the manual model and MAP kernel curves.

**Power spectral density** — The model PSD with vertical dotted lines at
harmonics of the rotation frequency (1/P, 2/P, 3/P). The data PSD is shown
when data is loaded. When "Show components" is enabled and multiple
components exist, individual component PSDs are shown as dashed lines.

### Spot model tab

**Single-spot flux** — Simulated flux from a single spot with adjustable
longitude, latitude, contrast, and duration. Shows both the spot evolution
envelope and the rotational modulation. When a composite model is active,
a dropdown selects which spot component to simulate.

**Model equations** — Expandable section showing the symbolic (SymPy)
expressions for the envelope, visibility, latitude distribution, and any
SHO terms with the current parameter values substituted in.

### Export tab

**Config (YAML)** — Click **Generate config** to build a YAML config from
the current sidebar state. The config is displayed in an editable code
editor; click **Export config** to write it to disk.

For composite models, the exported config uses a `components` list and/or
`sho_terms` list in the `model` section.

**Fit results (HDF5)** — After running a MAP fit, save the data, model,
solver state, and fit results to an HDF5 file.

**Plots (HTML)** — Saves the currently displayed plots to a single static
HTML file that can be shared, archived, or committed alongside a project
write-up. The plots stay interactive (zoom, pan, legend toggles) — no
Streamlit server needed.

1. Choose which figures to include (defaults to all currently shown).
2. Leave **Self-contained** checked to embed `plotly.js` in the file (~5 MB,
   works offline). Uncheck it to load `plotly.js` from a CDN instead, which
   produces a file of a few hundred KB that needs internet access to render.
3. Click **Build HTML**, then **Download HTML**.

The exported page is titled with the star name and includes a YAML summary of
the model parameters used to make the plots. It is a snapshot: rebuild after
changing parameters or rerunning a fit.

Individual plots can also be saved as PNG using the camera icon in each
plot's toolbar.

## Typical workflow

1. **Load data** — enter a star ID and download, or load a local file
2. **Adjust model** — choose envelope/visibility types and tune sliders to
   roughly match the kernel to the data ACF
3. **Add components** — optionally add SHO terms or extra spot components
4. **Build model** — compute the model kernel to compare with the data
5. **Optimize** — select parameters, set bounds, run MAP fit
6. **Inspect** — check the MAP kernel/PSD overlays and parameter table
7. **Export** — save the config YAML for a full run with sampling, and/or
   download the plots as a standalone HTML file

### Batch workflow

For fitting many targets in sequence:

1. Prepare a CSV or text file listing your targets (optionally with initial
   parameter guesses as columns)
2. Load the object list in the sidebar
3. For each target: download, adjust model, run MAP fit
4. Click **Save & Next** to export the config and advance
5. When finished, use `dvc repro` or `batch_fit.sh` to run full sampling
   on all exported configs

![App overview](img/gui_overview.png)
