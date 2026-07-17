"""End-to-end smoke test: MAP fit on a tiny synthetic light curve.

Skipped automatically when spotgp/jax are not installed.
"""

import json

import numpy as np
import pytest

pytest.importorskip("spotgp")
pytest.importorskip("jax")

from run_fit import run, validate_config


def _synthetic_npz(path, n=150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27, n)
    y = 1.0 + 0.01 * np.sin(2 * np.pi * t / 3.0) + rng.normal(0, 2e-3, n)
    yerr = np.full(n, 2e-3)
    np.savez(path, time=t, flux=y, flux_err=yerr)


def test_map_fit_end_to_end(tmp_path):
    data_path = tmp_path / "lc.npz"
    _synthetic_npz(data_path)
    run_dir = tmp_path / "run"

    cfg = {
        "star_name": "SYNTH 1",
        "seed": 42,
        "data": {"path": str(data_path), "normalize": True,
                 "zero_mean": True},
        "model": {
            "envelope": "ExponentialEnvelope",
            "envelope_params": {"tau_spot": 5.0},
            "visibility": "EdgeOnVisibilityFunction",
            "visibility_params": {"peq": 3.0},
            "sigma_k": 0.01,
        },
        "bounds": {"peq": [1.0, 10.0]},
        "solver": {"n_harmonics": 2, "n_lat": 16},
        "fitting": {"map": {"nopt": 1}},
        "output": {"run_dir": str(run_dir)},
    }

    validate_config(cfg)
    run(cfg)

    assert (run_dir / "result.h5").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "plots" / "prediction.png").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "neg_log_posterior" in metrics

    h5py = pytest.importorskip("h5py")
    with h5py.File(run_dir / "result.h5", "r") as f:
        assert "config_yaml" in f.attrs
        assert "spotgp_version" in f.attrs
        assert f.attrs["seed"] == 42
        assert "map" in f
        assert "data" in f
