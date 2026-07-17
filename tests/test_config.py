"""Fast tests for config handling and data helpers (no spotgp required)."""

import numpy as np
import pytest

from data_utils import detect_mission, process_data
from run_fit import ConfigError, generate_run_name, validate_config


def _minimal_cfg():
    return {
        "star_name": "TIC 441420236",
        "data": {},
        "model": {
            "envelope": "TrapezoidSymmetricEnvelope",
            "visibility": "VisibilityFunction",
        },
    }


def test_detect_mission():
    assert detect_mission("KIC 7286309") == ("Kepler", "Kepler")
    assert detect_mission("epic 201367065") == ("K2", "K2")
    assert detect_mission("TIC 441420236") == ("TESS", "SPOC")


def test_process_data_normalize_and_downsample():
    t = np.arange(10.0)
    y = np.full(10, 2.0)
    yerr = np.full(10, 0.2)
    tt, yy, ee = process_data([(t, y, yerr)], downsample=2, normalize=True)
    assert len(tt) == 5
    assert np.allclose(yy, 1.0)
    assert np.allclose(ee, 0.1)


def test_process_data_stitches_segments():
    seg = (np.arange(3.0), np.ones(3), np.ones(3))
    tt, yy, ee = process_data([seg, seg], normalize=False)
    assert len(tt) == 6


def test_run_name_is_stable_and_sanitized():
    cfg = _minimal_cfg()
    name = generate_run_name(cfg)
    assert name == generate_run_name(_minimal_cfg())
    assert " " not in name
    assert name.startswith("TIC_441420236_TrapezoidSymmetric_")


def test_run_name_changes_with_params():
    cfg = _minimal_cfg()
    other = _minimal_cfg()
    other["model"]["envelope_params"] = {"lspot": 10.0}
    assert generate_run_name(cfg) != generate_run_name(other)


def test_validate_missing_model():
    with pytest.raises(ConfigError, match="model"):
        validate_config({"star_name": "TIC 1"})


def test_validate_needs_star_or_path():
    cfg = _minimal_cfg()
    del cfg["star_name"]
    with pytest.raises(ConfigError, match="star_name"):
        validate_config(cfg)


def test_validate_bad_bounds():
    cfg = _minimal_cfg()
    cfg["bounds"] = {"peq": [1.0]}
    with pytest.raises(ConfigError, match="peq"):
        validate_config(cfg)


def test_validate_bad_sampler():
    cfg = _minimal_cfg()
    cfg["fitting"] = {"sampling": {"sampler": "emcee"}}
    with pytest.raises(ConfigError, match="emcee"):
        validate_config(cfg)


def test_validate_bad_prior():
    cfg = _minimal_cfg()
    cfg["priors"] = {"peq": {"mu": 3.0}}
    with pytest.raises(ConfigError, match="sigma"):
        validate_config(cfg)


def test_validate_good_config():
    validate_config(_minimal_cfg())


def test_validate_unknown_envelope():
    pytest.importorskip("spotgp")
    cfg = _minimal_cfg()
    cfg["model"]["envelope"] = "TrapezoidalEnvelope"
    with pytest.raises(ConfigError, match="TrapezoidalEnvelope"):
        validate_config(cfg)


def test_validate_unknown_visibility():
    pytest.importorskip("spotgp")
    cfg = _minimal_cfg()
    cfg["model"]["visibility"] = "VisFunction"
    with pytest.raises(ConfigError, match="VisFunction"):
        validate_config(cfg)
