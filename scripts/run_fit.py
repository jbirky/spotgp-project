"""Run a spotgp fit from a YAML config file."""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import download_lightcurve, load_local_file, process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spotgp.runner")


class ConfigError(Exception):
    """Raised when a config file fails validation."""


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    star = cfg.get("star_name", "unknown")
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, str):
                    section[k] = v.format(star_name=star)
    return cfg


def _spotgp_version():
    try:
        from importlib.metadata import version
        return version("spotgp")
    except Exception:
        return "unknown"


def validate_config(cfg):
    """Check config structure and class names, raising ConfigError with all
    problems found. Class-name checks are skipped if spotgp is not installed."""
    errors = []

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        errors.append("missing 'model' section")
        model_cfg = {}
    else:
        for key in ("envelope", "visibility"):
            if key not in model_cfg:
                errors.append(f"'model' section is missing '{key}'")

    data_cfg = cfg.get("data", {})
    if "path" not in data_cfg and "star_name" not in cfg:
        errors.append("set 'star_name' (MAST download) or 'data.path' "
                      "(local file)")
    if "path" in data_cfg and not os.path.exists(data_cfg["path"]):
        errors.append(f"data file not found: {data_cfg['path']}")

    for k, v in cfg.get("bounds", {}).items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            errors.append(f"bounds['{k}'] must be a [min, max] pair, "
                          f"got {v!r}")

    for k, spec in (cfg.get("priors") or {}).items():
        if not isinstance(spec, dict) or "mu" not in spec or "sigma" not in spec:
            errors.append(f"priors['{k}'] must be a dict with 'mu' and "
                          f"'sigma' keys, got {spec!r}")
        elif str(spec.get("type", "gaussian")).lower() not in ("gaussian",
                                                               "normal"):
            errors.append(f"priors['{k}']: only gaussian priors are "
                          f"supported, got type={spec['type']!r}")

    sampler = cfg.get("fitting", {}).get("sampling", {}).get("sampler")
    if sampler is not None and sampler not in ("dynesty", "blackjax"):
        errors.append(f"unknown sampler '{sampler}' "
                      "(valid: dynesty, blackjax)")

    try:
        import spotgp
    except ImportError:
        spotgp = None
        logger.warning("spotgp not installed; skipping class-name checks")

    if spotgp is not None:
        checks = [
            ("envelope", model_cfg.get("envelope"),
             sorted(n for n in dir(spotgp)
                    if n.endswith("Envelope"))),
            ("visibility", model_cfg.get("visibility"),
             sorted(n for n in dir(spotgp)
                    if n.endswith("VisibilityFunction"))),
            ("latitude", model_cfg.get("latitude"),
             sorted(n for n in dir(spotgp)
                    if "Latitude" in n or n == "UniformDoubleHemisphereBand")),
        ]
        for key, name, options in checks:
            if name is not None and not hasattr(spotgp, name):
                errors.append(f"unknown {key} '{name}' "
                              f"(valid: {', '.join(options)})")

    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))


def generate_run_name(cfg):
    """Generate a descriptive run name from config contents.

    Format: {star}_{envelope_short}_{config_hash}
    The hash is derived from the full config so different parameters
    always produce different names.
    """
    star = cfg.get("star_name", "unknown").replace(" ", "_")
    envelope = cfg.get("model", {}).get("envelope", "unknown")
    envelope_short = envelope.replace("Envelope", "").replace("Asymmetric", "Asym")

    config_bytes = json.dumps(cfg, sort_keys=True, default=str).encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()[:6]

    return f"{star}_{envelope_short}_{config_hash}"


def build_model(cfg):
    import spotgp

    model_cfg = cfg["model"]

    envelope_cls = getattr(spotgp, model_cfg["envelope"])
    envelope = envelope_cls(**model_cfg.get("envelope_params", {}))

    vis_cls = getattr(spotgp, model_cfg["visibility"])
    visibility = vis_cls(**model_cfg.get("visibility_params", {}))

    kwargs = {"sigma_k": model_cfg.get("sigma_k", 0.01)}

    if "latitude" in model_cfg:
        lat_cls = getattr(spotgp, model_cfg["latitude"])
        kwargs["latitude_distribution"] = lat_cls(
            **model_cfg.get("latitude_params", {}))

    return spotgp.SpotEvolutionModel(
        envelope=envelope, visibility=visibility, **kwargs)


def load_data(cfg):
    from spotgp import TimeSeriesData

    data_cfg = cfg["data"]
    normalize = data_cfg.get("normalize", True)
    zero_mean = data_cfg.get("zero_mean", False)
    downsample = int(data_cfg.get("downsample", 1))
    dt = data_cfg.get("dt")

    if "path" in data_cfg:
        segments = load_local_file(data_cfg["path"])
    else:
        star_name = cfg["star_name"]
        sectors = data_cfg.get("sectors")

        logger.info("Downloading light curve for %s", star_name)
        segments, _sector_nums, err = download_lightcurve(star_name, sectors)
        if err:
            raise ValueError(err)
        logger.info("Downloaded %d data points (%d segments)",
                    sum(len(s[0]) for s in segments), len(segments))

    t, y, yerr = process_data(segments, dt=dt, downsample=downsample,
                              normalize=normalize, zero_mean=zero_mean)
    return TimeSeriesData(t, y, yerr, normalize=False)


def build_log_prior(cfg, param_keys, bounds):
    """Build a custom log-prior from the config's ``priors`` section.

    Gaussian priors are added on top of the default soft uniform-in-bounds
    prior. Prior keys use sampling-space names (i.e. ``log_sigma_k`` when
    the bounds put sigma_k in log space).
    """
    priors_cfg = cfg.get("priors")
    if not priors_cfg:
        return None

    import jax.numpy as jnp
    from spotgp.gp_solver import _default_log_prior

    keys = list(param_keys)
    idx, mus, sigmas = [], [], []
    for name, spec in priors_cfg.items():
        if name not in keys:
            raise ConfigError(
                f"prior on unknown parameter '{name}' "
                f"(available: {', '.join(keys)})")
        idx.append(keys.index(name))
        mus.append(float(spec["mu"]))
        sigmas.append(float(spec["sigma"]))

    idx = jnp.array(idx)
    mu = jnp.array(mus)
    sigma = jnp.array(sigmas)

    def log_prior(theta_arr):
        lp = _default_log_prior(theta_arr, bounds)
        z = (theta_arr[idx] - mu) / sigma
        return lp + jnp.sum(-0.5 * z ** 2 - jnp.log(sigma)
                            - 0.5 * jnp.log(2 * jnp.pi))

    return log_prior


def build_solver(cfg, data, model):
    from spotgp import GPSolver

    solver_cfg = cfg.get("solver", {})
    bounds_cfg = cfg.get("bounds", {})

    kwargs = dict(
        bounds={k: tuple(v) for k, v in bounds_cfg.items()},
        kernel_type=solver_cfg.get("kernel_type", "analytic"),
        matrix_solver=solver_cfg.get("matrix_solver", "cholesky_banded"),
        n_harmonics=solver_cfg.get("n_harmonics", 3),
        n_lat=solver_cfg.get("n_lat", 64),
    )

    gp = GPSolver(data, model, **kwargs)

    # param_keys/bounds only exist after construction, so build the prior
    # from a first solver instance and reconstruct with it if needed.
    log_prior = build_log_prior(cfg, gp.param_keys, gp.bounds)
    if log_prior is not None:
        logger.info("Using Gaussian priors on: %s",
                    ", ".join(cfg["priors"]))
        gp = GPSolver(data, model, log_prior=log_prior, **kwargs)

    return gp.build_jax()


# ── Tracking setup ───────────────────────────────────────────────────


def setup_wandb(cfg, run_name):
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None:
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping experiment tracking")
        return None

    project = wandb_cfg.get("project", cfg.get("star_name", "spotgp"))
    wandb.init(
        project=project,
        name=run_name,
        config={
            "star_name": cfg.get("star_name"),
            "envelope": cfg["model"]["envelope"],
            "visibility": cfg["model"]["visibility"],
            **cfg.get("model", {}).get("envelope_params", {}),
            **cfg.get("model", {}).get("visibility_params", {}),
            **{f"bound_{k}": v for k, v in cfg.get("bounds", {}).items()},
        },
    )
    return wandb


def setup_mlflow(cfg, run_name):
    mlflow_cfg = cfg.get("mlflow")
    if mlflow_cfg is None:
        return None

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed, skipping experiment tracking")
        return None

    uri = mlflow_cfg.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)

    experiment = mlflow_cfg.get("experiment", cfg.get("star_name", "default"))
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    return mlflow


def save_gp(path, gp, data, cfg, run_name, theta_map=None, result=None):
    """Save GP state to HDF5 following the spotgp save/load format.

    Root attributes embed the full config, spotgp version, and seed so
    result files are self-describing and reproducible on their own.
    """
    import h5py

    with h5py.File(path, "a") as f:
        f.attrs["run_name"] = run_name
        f.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
        f.attrs["spotgp_version"] = _spotgp_version()
        f.attrs["created"] = datetime.datetime.now().isoformat(
            timespec="seconds")
        if cfg.get("seed") is not None:
            f.attrs["seed"] = int(cfg["seed"])

        if "data" not in f:
            dg = f.create_group("data")
            dg.create_dataset("time", data=np.asarray(data.x))
            dg.create_dataset("flux", data=np.asarray(data.y))
            dg.create_dataset("flux_err", data=np.asarray(data.yerr))

        if "model" not in f:
            mg = f.create_group("model")
            model_cfg = cfg["model"]
            mg.attrs["envelope"] = model_cfg["envelope"]
            mg.attrs["visibility"] = model_cfg["visibility"]
            mg.attrs["sigma_k"] = model_cfg.get("sigma_k", 0.01)
            for k, v in model_cfg.get("envelope_params", {}).items():
                mg.attrs[f"envelope_{k}"] = v
            for k, v in model_cfg.get("visibility_params", {}).items():
                mg.attrs[f"visibility_{k}"] = v

        if "solver" not in f:
            sg = f.create_group("solver")
            sg.attrs["kernel_type"] = gp.kernel_type
            sg.attrs["matrix_solver"] = gp.matrix_solver
            sg.create_dataset("param_keys", data=list(gp.param_keys))
            sg.create_dataset("bounds", data=gp.bounds)

        if result is not None:
            if "map" in f:
                del f["map"]
            rg = f.create_group("map")
            rg.attrs["neg_log_posterior"] = float(result.fun)
            if isinstance(theta_map, dict):
                for k, v in theta_map.items():
                    rg.attrs[k] = float(v)


def posterior_summary(samples, param_keys):
    """Median and 16/84-percentile uncertainties per parameter."""
    samples = np.asarray(samples)
    if samples.ndim == 3:  # (n_chains, n_samples, n_params)
        samples = samples.reshape(-1, samples.shape[-1])

    summary = {}
    for i, key in enumerate(param_keys):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        summary[f"{key}_median"] = float(med)
        summary[f"{key}_err_minus"] = float(med - lo)
        summary[f"{key}_err_plus"] = float(hi - med)
    return summary


class Tracker:
    """Unified interface for logging to wandb, MLflow, and/or DVC metrics."""

    def __init__(self, wandb=None, mlflow=None, plots_dir="plots"):
        self.wandb = wandb
        self.mlflow = mlflow
        self.plots_dir = plots_dir
        self._metrics = {}
        os.makedirs(plots_dir, exist_ok=True)

    def log_params(self, params):
        if self.wandb:
            self.wandb.config.update(params)
        if self.mlflow:
            self.mlflow.log_params(
                {k: f"{v:.6f}" if isinstance(v, float) else str(v)
                 for k, v in params.items()})

    def log_metrics(self, metrics):
        self._metrics.update(metrics)
        if self.wandb:
            self.wandb.log(metrics)
        if self.mlflow:
            self.mlflow.log_metrics(metrics)

    def log_figure(self, fig, name):
        if not hasattr(fig, "savefig"):  # spotgp plot methods return Axes
            fig = fig.figure
        path = os.path.join(self.plots_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if self.wandb:
            self.wandb.log({name.replace(".png", ""): self.wandb.Image(path)})
        if self.mlflow:
            self.mlflow.log_artifact(path)

        logger.info("Saved plot: %s", path)

    def log_artifact(self, path):
        if self.wandb:
            artifact = self.wandb.Artifact(
                os.path.basename(path).replace(".", "-"), type="model")
            artifact.add_file(path)
            self.wandb.log_artifact(artifact)
        if self.mlflow:
            self.mlflow.log_artifact(path)

    def save_metrics_json(self, path="metrics.json"):
        with open(path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=float)
        logger.info("Metrics saved to %s", path)

    def finish(self):
        if self.wandb:
            self.wandb.finish()
        if self.mlflow:
            self.mlflow.end_run()


# ── Diagnostic plots ─────────────────────────────────────────────────


def plot_diagnostics(gp, tracker, sampler=None):
    try:
        fig = gp.plot_acf()
        if fig is not None:
            tracker.log_figure(fig, "acf_comparison.png")
    except Exception as e:
        logger.warning("Could not generate ACF plot: %s", e)

    try:
        fig = gp.plot_prediction()
        if fig is not None:
            tracker.log_figure(fig, "prediction.png")
    except Exception as e:
        logger.warning("Could not generate prediction plot: %s", e)

    if sampler is not None and sampler.samples is not None:
        try:
            fig = sampler.plot_corner()
            if fig is not None:
                tracker.log_figure(fig, "corner_plot.png")
        except Exception as e:
            logger.warning("Could not generate corner plot: %s", e)


# ── Main pipeline ────────────────────────────────────────────────────


def run(cfg, output_dir=None):
    star = cfg.get("star_name", "unknown")
    fit_cfg = cfg.get("fitting", {})
    output_cfg = cfg.get("output", {})

    run_name = generate_run_name(cfg)
    save_dir = output_cfg.get("save_dir", "results")
    run_dir = (output_dir or output_cfg.get("run_dir")
               or os.path.join(save_dir, run_name))

    os.makedirs(run_dir, exist_ok=True)
    save_path = os.path.join(run_dir, "result.h5")
    metrics_path = os.path.join(run_dir, "metrics.json")

    # Copy of the resolved config so every run directory is self-contained.
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = cfg.get("device")
    if device:
        import jax
        if device == "gpu":
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
            os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
            os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        try:
            devs = jax.devices(device)
            jax.config.update("jax_default_device", devs[0])
            logger.info("Device: %s", devs[0])
        except RuntimeError:
            logger.warning("Requested device '%s' not available, "
                           "falling back to default", device)

    seed = cfg.get("seed")
    rng = np.random.default_rng(seed)

    logger.info("Star: %s", star)
    logger.info("Run:  %s", run_name)
    logger.info("Dir:  %s", run_dir)
    if seed is not None:
        logger.info("Seed: %d", seed)

    data = load_data(cfg)
    model = build_model(cfg)
    gp = build_solver(cfg, data, model)

    wb = setup_wandb(cfg, run_name)
    mf = setup_mlflow(cfg, run_name)
    tracker = Tracker(wandb=wb, mlflow=mf,
                      plots_dir=os.path.join(run_dir, "plots"))

    sampler = None
    theta_map = None
    result = None

    # Cheap ACF-based initialization for the MAP optimization
    theta_acf = None
    acf_cfg = fit_cfg.get("acf_init", {})
    if acf_cfg:
        nopt_acf = acf_cfg.get("nopt", 5)
        logger.info("Running ACF initialization (nopt=%d)", nopt_acf)
        theta_acf, acf_result = gp.fit_acf(nopt=nopt_acf, rng=rng)
        tracker.log_metrics({"acf_loss": float(acf_result.fun)})
        logger.info("ACF init complete: loss=%.4f", acf_result.fun)

    # MAP optimization
    map_cfg = fit_cfg.get("map", {})
    if map_cfg:
        nopt = map_cfg.get("nopt", 10)
        logger.info("Running MAP optimization (nopt=%d)", nopt)
        # With an ACF initialization and nopt > 1, the multi-start
        # trials are jittered around the ACF solution.
        theta_map, result = gp.fit_map(theta0=theta_acf, nopt=nopt, rng=rng)
        save_gp(save_path, gp, data, cfg, run_name, theta_map, result)

        tracker.log_metrics({"neg_log_posterior": float(result.fun)})
        if isinstance(theta_map, dict):
            tracker.log_params({f"map_{k}": v for k, v in theta_map.items()})

        logger.info("MAP complete: fun=%.4f", result.fun)

    # MCMC sampling
    sample_cfg = fit_cfg.get("sampling", {})
    if sample_cfg:
        sampler_type = sample_cfg.get("sampler", "dynesty")

        if sampler_type == "blackjax":
            from spotgp import BlackJAXSampler
            sampler = BlackJAXSampler(gp, save_dir=run_dir)

            n_warmup = sample_cfg.get("n_warmup", 500)
            n_samples = sample_cfg.get("n_samples", 2000)

            logger.info("Running NUTS warm-up (%d steps)", n_warmup)
            sampler.run_warmup(n_warmup=n_warmup)

            logger.info("Running NUTS sampling (%d samples)", n_samples)
            samples, info = sampler.run_sampling(n_samples=n_samples)
            sampler.save_checkpoint(save_path)

            tracker.log_metrics({
                "n_samples": info.get("n_samples", n_samples),
                "n_divergent": info.get("n_divergent", 0),
                "step_size": float(info.get("step_size", 0)),
            })
            tracker.log_metrics(posterior_summary(samples, gp.param_keys))

        elif sampler_type == "dynesty":
            from spotgp import DynestySampler
            sampler = DynestySampler(gp, save_dir=run_dir)

            nlive = sample_cfg.get("nlive", 500)

            logger.info("Running Dynesty (nlive=%d)", nlive)
            samples, info = sampler.run_sampling(nlive=nlive, rstate=rng)
            sampler.save_checkpoint(save_path)

            tracker.log_metrics({
                "logz": info["logz"],
                "logzerr": info["logzerr"],
                "n_samples": info["n_samples"],
            })
            tracker.log_metrics(posterior_summary(samples, gp.param_keys))

    # Save final state and diagnostics
    save_gp(save_path, gp, data, cfg, run_name, theta_map, result)
    plot_diagnostics(gp, tracker, sampler)
    tracker.log_artifact(save_path)
    tracker.save_metrics_json(metrics_path)
    tracker.finish()

    logger.info("Results saved to %s", run_dir)


def main():
    parser = argparse.ArgumentParser(description="Run a spotgp fit from config")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--output-dir", default=None,
                        help="Override the run output directory "
                             "(default: results/{run_name})")
    parser.add_argument("--validate", action="store_true",
                        help="Check the config and exit without running")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        validate_config(cfg)
    except ConfigError as e:
        logger.error("%s", e)
        sys.exit(1)

    if args.validate:
        logger.info("Config OK: %s", args.config)
        return

    run(cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
