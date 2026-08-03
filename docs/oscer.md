# Running the explorer on OSCER

This page describes how to serve the interactive explorer to a group of users
on OU's [OSCER](https://www.ou.edu/oscer) Schooner cluster, so that each person
logs in through their browser instead of installing anything themselves.

## How it works

Each user gets **their own Streamlit process**, inside **their own SLURM job**,
reached through OSCER's [Open OnDemand](https://ondemand.oscer.ou.edu) reverse
proxy:

```
browser  ──►  ondemand.oscer.ou.edu  ──►  awesomestars node
 (OSCER login)     (OOD proxy)          SLURM job: Jupyter + Streamlit
                                        runs as <username>
                                        writes to /ourdisk/hpc/astrogroup/<username>/<project>/
```

One process per user is deliberate, not incidental. A single shared Streamlit
server would put every user in one Python process, where `st.cache_data` /
`st.cache_resource`, the JAX default device (`app.py` calls
`jax.config.update("jax_default_device", ...)`, which is process-global), the
wandb credentials in `~/.netrc`, and file ownership are all shared. Per-user
processes give isolation for free, at the cost of one SLURM job each.

## One-time setup (group admin)

Run once from a Schooner login node, as the owner of the allocation:

```bash
git clone https://github.com/jbirky/spotgp-project.git /tmp/spotgp-setup
bash /tmp/spotgp-setup/scripts/ood/install_shared.sh --users alice,bob,carol
```

This creates:

```
/ourdisk/hpc/astrogroup/
├── shared/
│   ├── spotgp-project/     read-only clone of this repo
│   └── containers/         shared Apptainer image
├── alice/                  per-user project roots
├── bob/
└── carol/
```

Users need read and execute on `shared/`, and write access only to their own
directory.

!!! warning "Don't put a conda environment on /ourdisk"

    [OSCER asks](https://www.ou.edu/oscer/applications/python/mamba) that
    Python environments not be created on `/ourdisk` — they scatter huge
    numbers of small files across Ceph. A *shared* runtime therefore has to be
    an Apptainer image (one file, which is fine), launched with
    `SPOTGP_SIF=/ourdisk/hpc/astrogroup/shared/containers/spotgp.sif`.
    Without an image, each user builds their own environment in their **home**
    directory, as below.

## Python environment (each user, once)

Skip this if your group has a shared container image. Otherwise, from a
Schooner login node:

```bash
module load Mamba
mamba create -n spotgp python=3.11
source activate spotgp
pip install -r ~/spotgp-project/requirements.txt
```

OSCER documents `source activate`, not `conda activate`, and advises against
`mamba init`. Installing packages on a login node is fine; *running* fits
there is not — that's what the SLURM job below is for.

## Launching (each user)

1. Go to [ondemand.oscer.ou.edu](https://ondemand.oscer.ou.edu) and log in with
   your OSCER username and password.
2. **Interactive Apps → Jupyter**. Set **partition** to `awesomestars`, pick
   cores, memory, and wall time, and launch. This SLURM job is where the app
   will run.
3. When the session starts, click **Connect to Jupyter**, then open a
   **Terminal** from the JupyterLab launcher.
4. Activate the environment and run the launcher:

```bash
module load Mamba && source activate spotgp     # or: export SPOTGP_SIF=...
~/spotgp-project/scripts/spotgp_app.sh my-project
```

(Once the group install exists, use
`/ourdisk/hpc/astrogroup/shared/spotgp-project/scripts/spotgp_app.sh` instead
of your own clone.)

The script creates `/ourdisk/hpc/astrogroup/$USER/my-project/` on first run
(via `init_project.py`, so it gets its own `configs/`, `data/`, `results/`,
`dvc.yaml`, and `params.yaml`), picks a free port, and prints a URL like:

```
https://ondemand.oscer.ou.edu/node/c653/8501/
```

Open that in the same browser you're logged into OnDemand with. Ctrl-C in the
terminal stops the app; anything already written to `results/` stays.

!!! note "The app is bound to the whole node"

    `--bind 0.0.0.0` is required for the OnDemand proxy to reach it, which
    also means anyone else on that node could reach your session while it
    runs. On a group node that is usually acceptable; if it isn't, launch
    through the JupyterLab tile below, which binds to localhost only.

### Optional: a launcher tile instead of a terminal command

If `jupyter-server-proxy` is installed in the environment that runs the
**Jupyter server** (not the notebook kernel), the app can appear as a tile in
the JupyterLab launcher. Enable it once:

```bash
mkdir -p ~/.jupyter
ln -s /ourdisk/hpc/astrogroup/shared/spotgp-project/scripts/ood/jupyter_server_config.py \
      ~/.jupyter/jupyter_server_config.py
```

Check whether it will work before bothering:

```bash
python -c "import jupyter_server_proxy; print('available')"
```

If that fails inside the OnDemand session, OSCER's Jupyter app doesn't ship it
— use the terminal route above, which needs no extra packages.

### Optional: skip Jupyter entirely

To run the explorer as its own SLURM job on `awesomestars`:

```bash
cd /ourdisk/hpc/astrogroup/$USER
sbatch /ourdisk/hpc/astrogroup/shared/spotgp-project/scripts/ood/spotgp_app.slurm my-project
```

The URL appears in `logs/spotgp-app_<jobid>.out`. The `/node/` link works
while you're logged into OnDemand in the same browser; the log also prints an
SSH tunnel command as a fallback.

## Where your data goes

Everything the app writes lands in your own project directory:

| Path | Contents |
|------|----------|
| `/ourdisk/hpc/astrogroup/<user>/<project>/configs/` | YAML configs saved from the GUI |
| `/ourdisk/hpc/astrogroup/<user>/<project>/data/` | uploaded and downloaded light curves |
| `/ourdisk/hpc/astrogroup/<user>/<project>/results/` | HDF5 fit results, plots, metrics |
| `/ourdisk/hpc/astrogroup/<user>/<project>/params.yaml` | configs registered for `dvc repro` |

When `SPOTGP_PROJECT_ROOT` is set — the launcher sets it — the sidebar's
free-form "Project directory" box is replaced by a dropdown of your own
projects plus a **New project** option, so the app cannot be pointed at
someone else's files.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPOTGP_ROOT` | the checkout the script lives in | shared clone |
| `SPOTGP_PROJECT_ROOT` | `/ourdisk/hpc/astrogroup/$USER` on OSCER, else `~/spotgp-projects` | your project root |
| `SPOTGP_PROJECT` | `default` | project name when none is given |
| `SPOTGP_ENV` | unset | conda env or venv to activate |
| `SPOTGP_SIF` | unset | Apptainer image, used if `SPOTGP_ENV` is unset |
| `SPOTGP_PROXY_HOST` | `hostname -s` | hostname in the proxy URL |
| `SPOTGP_DISABLE_XSRF` | `0` | set to `1` if uploads fail behind the proxy |

## Running it anywhere else

None of this is OSCER-specific except the defaults. `scripts/spotgp_app.sh`
detects its mode from `$SLURM_JOB_ID`: inside a SLURM job it binds all
interfaces and serves under the OnDemand proxy prefix, and everywhere else —
laptop, lab workstation, any server — it binds `127.0.0.1` and serves at the
plain root URL:

```bash
scripts/spotgp_app.sh my-project        # → http://localhost:8501
```

Projects go to `~/spotgp-projects/<name>/` when `/ourdisk/hpc/astrogroup`
doesn't exist. Force either behaviour with `--local` / `--cluster`, and point
the root anywhere with `SPOTGP_PROJECT_ROOT=/data/spotgp scripts/spotgp_app.sh`.

For a single-user run against whatever directory you're standing in, nothing
has changed:

```bash
make app                                  # cwd as the project
streamlit run scripts/app.py              # same thing, no Makefile
```

The sidebar shows the free-form "Project directory" box exactly as before —
the constrained project dropdown appears **only** when `SPOTGP_PROJECT_ROOT`
is set, which the launcher does and `make app` does not.

## Troubleshooting

**The URL 404s.** OnDemand's node proxy may expect the fully qualified
hostname, or the rewriting proxy. Try the `/rnode/` URL the script prints, or
relaunch with `SPOTGP_PROXY_HOST=$(hostname -f)`.

**The page loads but stays blank, or the websocket keeps reconnecting.**
Streamlit's base path has to match the proxy prefix exactly. Pass it
explicitly: `--base-path node/$(hostname -s)/8501`.

**File uploads fail.** The proxy is probably dropping the XSRF cookie. Relaunch
with `SPOTGP_DISABLE_XSRF=1`.

**Out of GPU memory with several users on one node.** The launcher already sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.
Lower the fraction further, or pick the CPU device in the sidebar for
interactive exploration and submit real fits with `sbatch`.

**The session disappeared.** OnDemand jobs end at their wall-time limit, which
takes the app with it. Fits already written to `results/` are safe; slider
state is not. For long runs, export a config from the GUI and submit it with
`make submit` or `batch_fit.sh`.
