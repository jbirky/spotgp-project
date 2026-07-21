"""Interactive spotgp kernel explorer and fitting tool."""

import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_utils
from data_utils import process_data

st.set_page_config(page_title="spotgp Explorer", layout="wide")


# ── Cached helpers ───────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _import_spotgp():
    import spotgp
    return spotgp


def _build_latitude_dist(name, params):
    spotgp = _import_spotgp()
    return getattr(spotgp, name)(**params)


@st.cache_data(show_spinner="Computing kernel...", ttl=300)
def cached_kernel(envelope_name, envelope_params_tuple, visibility_name,
                  visibility_params_tuple, sigma_k,
                  latitude_name, latitude_params_tuple):
    spotgp = _import_spotgp()
    envelope_params = dict(envelope_params_tuple)
    visibility_params = dict(visibility_params_tuple)

    envelope_cls = getattr(spotgp, envelope_name)
    envelope = envelope_cls(**envelope_params)
    vis_cls = getattr(spotgp, visibility_name)
    visibility = vis_cls(**visibility_params)
    lat_dist = _build_latitude_dist(latitude_name, dict(latitude_params_tuple))
    model = spotgp.SpotEvolutionModel(
        envelope=envelope, visibility=visibility, sigma_k=sigma_k,
        latitude_distribution=lat_dist)

    max_lag = model.envelope.kernel_support() * 1.5
    lag = np.linspace(0, max_lag, 500)
    ak = spotgp.AnalyticKernel(model)
    K = np.asarray(ak.kernel(lag))

    harmonics = _compute_harmonic_components(ak, lag)
    return lag, K, max_lag, harmonics


def _compute_harmonic_components(ak, lag):
    import jax.numpy as jnp

    lag_flat = jnp.asarray(lag, dtype=float).ravel()
    R = ak.R_Gamma(lag_flat)
    n_harmonics = ak.n_harmonics

    if isinstance(ak.visibility, _import_spotgp().EdgeOnVisibilityFunction):
        cn_sq = ak.cn_squared(0.0)
        w0 = ak.omega0(0.0)
        components = {}
        components[0] = np.asarray(ak.sigma_k ** 2 * R * cn_sq[0])
        for n in range(1, n_harmonics + 1):
            components[n] = np.asarray(
                ak.sigma_k ** 2 * R * 2 * cn_sq[n]
                * jnp.cos(n * w0 * lag_flat))
        return components

    lat_dist = ak.spot_model.latitude_distribution
    if ak.quadrature == "gauss-legendre":
        phi_grid = ak._quad_nodes
        quad_w = ak._quad_weights
        user_w = jnp.array([lat_dist(float(p)) for p in phi_grid])
        weights = user_w * quad_w
        norm = jnp.sum(weights)
    else:
        phi_min, phi_max = ak.lat_range
        phi_grid = jnp.linspace(phi_min, phi_max, ak.n_lat)
        dphi = phi_grid[1] - phi_grid[0]
        user_w = jnp.array([lat_dist(float(p)) for p in phi_grid])
        weights = user_w * dphi
        norm = jnp.trapezoid(user_w, phi_grid)

    accum = {n: jnp.zeros_like(lag_flat) for n in range(n_harmonics + 1)}
    for i, phi in enumerate(phi_grid):
        cn_sq = ak.cn_squared(float(phi))
        w0 = ak.omega0(float(phi))
        w = weights[i]
        accum[0] = accum[0] + w * cn_sq[0]
        for n in range(1, n_harmonics + 1):
            accum[n] = accum[n] + w * 2 * cn_sq[n] * jnp.cos(
                n * w0 * lag_flat)

    components = {}
    for n in range(n_harmonics + 1):
        components[n] = np.asarray(
            ak.sigma_k ** 2 * R * accum[n] / norm)
    return components


@st.cache_data(show_spinner="Downloading light curve...")
def download_lightcurve(star_name, sectors_tuple):
    return data_utils.download_lightcurve(star_name, sectors_tuple)


@st.cache_data(show_spinner="Loading file...")
def load_local_file(path):
    return data_utils.load_local_file(path)


_LAT_PARAM_MAP = {
    "ButterflyLatitude": {
        "phi0_deg": ("phi0", np.deg2rad, np.rad2deg),
        "sigma_deg": ("sigma_phi", np.deg2rad, np.rad2deg),
    },
    "UniformDoubleHemisphereBand": {
        "min_lat_deg": ("lat_min", np.deg2rad, np.rad2deg),
        "max_lat_deg": ("lat_max", np.deg2rad, np.rad2deg),
    },
}


def _default_fit_bounds(key, value):
    """Heuristic optimization bounds centered on the current slider value."""
    if key == "log_sigma_k":
        return value - 2.0, value + 2.0
    if key == "kappa":
        return 0.0, min(1.0, abs(value) * 3 + 0.1)
    if key == "inc":
        return 0.01, np.pi / 2
    if key == "peq":
        return max(0.5, value * 0.5), value * 2.0
    if value > 0:
        return value * 0.5, value * 2.0
    return value - 1.0, value + 1.0


def build_map_model_params(mp, theta_map):
    env_params = dict(mp["envelope_params"])
    vis_params = dict(mp["visibility_params"])
    lat_params = dict(mp.get("latitude_params", {}))
    sigma_k = mp["sigma_k"]
    log_sigma_k = mp["log_sigma_k"]

    lat_name = mp.get("latitude_name", "LatitudeDistributionFunction")
    internal_to_app = {}
    for app_key, (internal_key, _, to_app) in _LAT_PARAM_MAP.get(
            lat_name, {}).items():
        internal_to_app[internal_key] = (app_key, to_app)

    for key, val in theta_map.items():
        if key == "log_sigma_k":
            log_sigma_k = float(val)
            sigma_k = 10.0 ** log_sigma_k
        elif key in vis_params:
            vis_params[key] = float(val)
        elif key in env_params:
            env_params[key] = float(val)
        elif key in lat_params:
            lat_params[key] = float(val)
        elif key in internal_to_app:
            app_key, to_app = internal_to_app[key]
            lat_params[app_key] = float(to_app(val))
    return dict(
        envelope_name=mp["envelope_name"],
        envelope_params=env_params,
        visibility_name=mp["visibility_name"],
        visibility_params=vis_params,
        latitude_name=mp.get("latitude_name", "LatitudeDistributionFunction"),
        latitude_params=lat_params,
        sigma_k=sigma_k,
        log_sigma_k=log_sigma_k,
    )


def _synced_slider(container, label, min_value, max_value, value, step,
                   key, fmt=None):
    """Slider with a synced number input box."""
    skey = f"_sync_{key}"
    sk = f"{skey}_s"
    nk = f"{skey}_n"

    if skey not in st.session_state:
        st.session_state[skey] = value
    if sk not in st.session_state:
        st.session_state[sk] = st.session_state[skey]
    if nk not in st.session_state:
        st.session_state[nk] = st.session_state[skey]

    def _on_slider():
        st.session_state[skey] = st.session_state[sk]
        st.session_state[nk] = st.session_state[sk]

    def _on_number():
        v = max(min_value, min(max_value, st.session_state[nk]))
        st.session_state[skey] = v
        st.session_state[sk] = v

    slider_kw = {}
    number_kw = {}
    if fmt:
        slider_kw["format"] = fmt
        number_kw["format"] = fmt

    c1, c2 = container.columns([3, 1])
    c1.slider(label, min_value=min_value, max_value=max_value, step=step,
              key=sk, on_change=_on_slider, **slider_kw)
    c2.number_input(label, min_value=min_value, max_value=max_value,
                    step=step, key=nk, on_change=_on_number,
                    label_visibility="collapsed", **number_kw)
    return st.session_state[skey]


# ── Sidebar ──────────────────────────────────────────────────────────


st.sidebar.title("spotgp Explorer")

import jax
_available_devices = []
for _backend in ["cpu", "gpu", "tpu"]:
    try:
        _available_devices.extend(jax.devices(_backend))
    except RuntimeError:
        pass
_device_labels = [str(d) for d in _available_devices]
_device_idx = st.sidebar.selectbox(
    "Device", range(len(_device_labels)),
    format_func=lambda i: _device_labels[i])
jax.config.update("jax_default_device", _available_devices[_device_idx])

st.sidebar.markdown("---")
st.sidebar.header("Data")
data_source = st.sidebar.radio("Source", ["TIC / KIC ID", "Local file"])

if data_source == "TIC / KIC ID":
    star_name = st.sidebar.text_input("Star name", value="KIC 7286309")
    sectors_input = st.sidebar.text_input(
        "Sectors / Quarters (comma-separated, blank for all)", value="")
    sectors = tuple(int(s.strip()) for s in sectors_input.split(",")
                    if s.strip()) if sectors_input.strip() else None
    if st.sidebar.button("Download"):
        segments, err = download_lightcurve(star_name, sectors)
        if err:
            st.sidebar.error(err)
        else:
            st.session_state["raw_segments"] = segments
            st.session_state["star_name"] = star_name
else:
    file_path = st.sidebar.text_input("File path", value="data/lightcurve.csv")
    if st.sidebar.button("Load"):
        try:
            segments = load_local_file(file_path)
            st.session_state["raw_segments"] = segments
            st.session_state["star_name"] = file_path
        except Exception as e:
            st.sidebar.error(str(e))

has_raw = "raw_segments" in st.session_state

if has_raw:
    segments = st.session_state["raw_segments"]
    n_raw = sum(len(s[0]) for s in segments)
    st.sidebar.caption(
        f"{n_raw} raw data points across {len(segments)} segment(s)")
    bin_dt = st.sidebar.number_input(
        "Bin width dt (days, 0 = no binning)", min_value=0.0,
        value=0.0, step=0.01, format="%.3f")
    normalize = st.sidebar.checkbox("Normalize", value=True)
    zero_mean = st.sidebar.checkbox("Zero mean", value=True)
    st.session_state["data_arrays"] = process_data(
        segments, dt=bin_dt if bin_dt > 0 else None,
        normalize=normalize, zero_mean=zero_mean)

has_data = "data_arrays" in st.session_state

st.sidebar.markdown("---")
st.sidebar.header("Manual Fit")

envelope_options = [
    "TrapezoidSymmetricEnvelope",
    "TrapezoidAsymmetricEnvelope",
    "ExponentialEnvelope",
    "ExponentialAsymmetricEnvelope",
    "SkewedGaussianEnvelope",
]
envelope_name = st.sidebar.selectbox("Envelope", envelope_options)

envelope_params = {}
if envelope_name == "TrapezoidSymmetricEnvelope":
    envelope_params["lspot"] = _synced_slider(
        st.sidebar, "lspot (days)", 0.0, 100.0, 15.0, 0.5, key="lspot")
    envelope_params["tau_spot"] = _synced_slider(
        st.sidebar, "tau_spot (days)", 0.1, 50.0, 5.0, 0.1, key="tau_spot")
elif envelope_name == "TrapezoidAsymmetricEnvelope":
    envelope_params["lspot"] = _synced_slider(
        st.sidebar, "lspot (days)", 0.0, 100.0, 15.0, 0.5, key="lspot")
    envelope_params["tau_em"] = _synced_slider(
        st.sidebar, "tau_em (days)", 0.1, 50.0, 5.0, 0.1, key="tau_em")
    envelope_params["tau_dec"] = _synced_slider(
        st.sidebar, "tau_dec (days)", 0.1, 50.0, 5.0, 0.1, key="tau_dec")
elif envelope_name == "ExponentialEnvelope":
    envelope_params["tau_spot"] = _synced_slider(
        st.sidebar, "tau_spot (days)", 0.1, 50.0, 5.0, 0.1, key="tau_spot")
elif envelope_name == "ExponentialAsymmetricEnvelope":
    envelope_params["tau_em"] = _synced_slider(
        st.sidebar, "tau_em (days)", 0.1, 50.0, 5.0, 0.1, key="tau_em")
    envelope_params["tau_dec"] = _synced_slider(
        st.sidebar, "tau_dec (days)", 0.1, 50.0, 5.0, 0.1, key="tau_dec")
elif envelope_name == "SkewedGaussianEnvelope":
    envelope_params["sigma_sn"] = _synced_slider(
        st.sidebar, "sigma_sn (days)", 0.1, 50.0, 5.0, 0.1, key="sigma_sn")
    envelope_params["n_sn"] = _synced_slider(
        st.sidebar, "n_sn", 0.0, 10.0, 2.0, 0.1, key="n_sn")

st.sidebar.markdown("---")

visibility_options = [
    "VisibilityFunction",
    "EdgeOnVisibilityFunction",
    "FullGeometryVisibilityFunction",
]
visibility_name = st.sidebar.selectbox("Visibility", visibility_options)

visibility_params = {}
visibility_params["peq"] = _synced_slider(
    st.sidebar, "P_eq (days)", 0.01, 40.0, 10.0, 0.01, key="peq")
if visibility_name != "EdgeOnVisibilityFunction":
    visibility_params["kappa"] = _synced_slider(
        st.sidebar, "kappa", -1.0, 1.0, 0.3, 0.01, key="kappa")
    visibility_params["inc"] = _synced_slider(
        st.sidebar, "inc (deg)", 0.0, 90.0, 60.0, 1.0,
        key="inc") * np.pi / 180.0

st.sidebar.markdown("---")

_latitude_labels = {
    "Uniform": "LatitudeDistributionFunction",
    "Uniform Band": "UniformDoubleHemisphereBand",
    "Butterfly": "ButterflyLatitude",
}
_lat_label = st.sidebar.selectbox(
    "Latitude distribution", list(_latitude_labels.keys()))
latitude_name = _latitude_labels[_lat_label]

latitude_params = {}
if latitude_name == "UniformDoubleHemisphereBand":
    latitude_params["min_lat_deg"] = _synced_slider(
        st.sidebar, "Min latitude (deg)", 0.0, 89.0, 0.0, 1.0,
        key="min_lat_deg")
    latitude_params["max_lat_deg"] = _synced_slider(
        st.sidebar, "Max latitude (deg)", 1.0, 90.0, 40.0, 1.0,
        key="max_lat_deg")
elif latitude_name == "ButterflyLatitude":
    latitude_params["phi0_deg"] = _synced_slider(
        st.sidebar, "phi0 (deg)", 0.0, 60.0, 15.0, 1.0, key="phi0_deg")
    latitude_params["sigma_deg"] = _synced_slider(
        st.sidebar, "sigma_phi (deg)", 1.0, 30.0, 7.0, 0.5,
        key="sigma_deg")

st.sidebar.markdown("---")

log_sigma_k = _synced_slider(
    st.sidebar, "log sigma_k", -6.0, 0.0, -2.0, 0.1, key="log_sigma_k")
sigma_k = 10.0 ** log_sigma_k

col_gen, col_fit = st.sidebar.columns(2)
if col_gen.button("Generate", type="primary", use_container_width=True):
    st.session_state["model_params"] = dict(
        envelope_name=envelope_name,
        envelope_params=envelope_params,
        visibility_name=visibility_name,
        visibility_params=visibility_params,
        latitude_name=latitude_name,
        latitude_params=latitude_params,
        sigma_k=sigma_k,
        log_sigma_k=log_sigma_k,
    )
    st.session_state.pop("gp_prediction", None)
    for _key in ("fit_result", "theta_map", "gp",
                 "map_model_params", "map_kernel_data",
                 "map_psd_data", "map_gp_prediction"):
        st.session_state.pop(_key, None)

if col_fit.button("Predict", use_container_width=True):
    if "model_params" not in st.session_state:
        st.sidebar.error("Generate a model first.")
    elif "data_arrays" not in st.session_state:
        st.sidebar.error("Load data first.")
    else:
        mp_fit = st.session_state["model_params"]
        t_fit, y_fit, yerr_fit = st.session_state["data_arrays"]
        try:
            _sg = _import_spotgp()
            _data = _sg.TimeSeriesData(t_fit, y_fit, yerr_fit, normalize=False)
            _env = getattr(_sg, mp_fit["envelope_name"])(
                **mp_fit["envelope_params"])
            _vis = getattr(_sg, mp_fit["visibility_name"])(
                **mp_fit["visibility_params"])
            _lat = _build_latitude_dist(
                mp_fit.get("latitude_name", "LatitudeDistributionFunction"),
                mp_fit.get("latitude_params", {}))
            _mdl = _sg.SpotEvolutionModel(
                envelope=_env, visibility=_vis, sigma_k=mp_fit["sigma_k"],
                latitude_distribution=_lat)
            _pred_bounds = {}
            _pred_lat_map = _LAT_PARAM_MAP.get(
                mp_fit.get("latitude_name",
                           "LatitudeDistributionFunction"), {})
            _pred_lat_p = mp_fit.get("latitude_params", {})
            for _ak, (_ik, _conv, _) in _pred_lat_map.items():
                _av = _pred_lat_p.get(_ak)
                if _av is not None:
                    _iv = float(_conv(_av))
                    _pred_bounds[_ik] = (max(0.001, _iv * 0.2), _iv * 3.0)
            _gp = _sg.GPSolver(_data, _mdl,
                               bounds=_pred_bounds or None).build_jax()
            _mu, _var = _gp.predict(t_fit)
            _std = np.sqrt(np.clip(_var, 0, None))
            st.session_state["gp_prediction"] = (t_fit, _mu, _std)
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")


# ── Optimize Fit (sidebar) ─────────────────────────────────────────


st.sidebar.markdown("---")
st.sidebar.header("Optimize Fit")

_opt_ready = "model_params" in st.session_state and has_data
if not _opt_ready:
    st.sidebar.info("Load data and generate a model first.")
else:
    _opt_mp = st.session_state["model_params"]

    nopt = st.sidebar.number_input("Number of restarts", min_value=1,
                                   max_value=50, value=5)

    _candidates = {}
    _candidates.update(_opt_mp["visibility_params"])
    _candidates.update(_opt_mp["envelope_params"])
    _candidates.update(_opt_mp.get("latitude_params", {}))
    _candidates["log_sigma_k"] = _opt_mp["log_sigma_k"]

    fit_keys = st.sidebar.multiselect(
        "Parameters to fit", list(_candidates),
        default=list(_candidates))

    fit_bounds = {}
    for _key in fit_keys:
        _lo_def, _hi_def = _default_fit_bounds(_key, _candidates[_key])
        _c_lo, _c_hi = st.sidebar.columns(2)
        _lo = _c_lo.number_input(f"{_key} min", value=float(_lo_def),
                                 key=f"bound_lo_{_key}", format="%.4f")
        _hi = _c_hi.number_input(f"{_key} max", value=float(_hi_def),
                                 key=f"bound_hi_{_key}", format="%.4f")
        fit_bounds[_key] = (_lo, _hi)
    st.session_state["fit_bounds"] = fit_bounds

    _star_lbl = st.session_state.get("star_name", "")
    default_save = (f"results/{_star_lbl.replace(' ', '_')}.h5"
                    if _star_lbl else "results/fit.h5")
    save_path = st.sidebar.text_input("Save path", value=default_save)

    if st.sidebar.button("Run MAP Fit", type="primary",
                         use_container_width=True):
        spotgp = _import_spotgp()
        t, y, yerr = st.session_state["data_arrays"]
        data_obj = spotgp.TimeSeriesData(t, y, yerr, normalize=False)

        envelope_cls = getattr(spotgp, _opt_mp["envelope_name"])
        envelope = envelope_cls(**_opt_mp["envelope_params"])
        vis_cls = getattr(spotgp, _opt_mp["visibility_name"])
        visibility = vis_cls(**_opt_mp["visibility_params"])
        fit_lat = _build_latitude_dist(
            _opt_mp.get("latitude_name", "LatitudeDistributionFunction"),
            _opt_mp.get("latitude_params", {}))
        model = spotgp.SpotEvolutionModel(
            envelope=envelope, visibility=visibility,
            sigma_k=_opt_mp["sigma_k"], latitude_distribution=fit_lat)

        solver_bounds = {}
        _lat_map = _LAT_PARAM_MAP.get(
            _opt_mp.get("latitude_name", "LatitudeDistributionFunction"), {})
        for k, v in fit_bounds.items():
            if k in _lat_map:
                internal_key, to_internal, _ = _lat_map[k]
                solver_bounds[internal_key] = (
                    float(to_internal(v[0])), float(to_internal(v[1])))
            else:
                solver_bounds[k] = v
        _lat_params = _opt_mp.get("latitude_params", {})
        for app_key, (internal_key, to_internal, _) in _lat_map.items():
            if internal_key not in solver_bounds:
                app_val = _lat_params.get(app_key)
                if app_val is not None:
                    iv = float(to_internal(app_val))
                    solver_bounds[internal_key] = (
                        max(0.001, iv * 0.2), iv * 3.0)
                else:
                    solver_bounds[internal_key] = (0.0, np.pi / 2)

        _solver_fit_keys = []
        for k in fit_keys:
            if k in _lat_map:
                _solver_fit_keys.append(_lat_map[k][0])
            else:
                _solver_fit_keys.append(k)

        try:
            with st.sidebar.status("Running MAP fit..."):
                gp = spotgp.GPSolver(data_obj, model,
                                     bounds=solver_bounds).build_jax()
                theta_map, result = gp.fit_map(
                    nopt=nopt, keys=_solver_fit_keys or None)

            st.session_state["gp"] = gp
            st.session_state["theta_map"] = theta_map
            st.session_state["fit_result"] = result

            map_mp = build_map_model_params(_opt_mp, theta_map)
            st.session_state["map_model_params"] = map_mp

            _map_env_cls = getattr(spotgp, map_mp["envelope_name"])
            _map_env = _map_env_cls(**map_mp["envelope_params"])
            _map_vis_cls = getattr(spotgp, map_mp["visibility_name"])
            _map_vis = _map_vis_cls(**map_mp["visibility_params"])
            _map_lat = _build_latitude_dist(
                map_mp.get("latitude_name",
                           "LatitudeDistributionFunction"),
                map_mp.get("latitude_params", {}))
            _map_model = spotgp.SpotEvolutionModel(
                envelope=_map_env, visibility=_map_vis,
                sigma_k=map_mp["sigma_k"],
                latitude_distribution=_map_lat)
            _map_kernel_obj = spotgp.AnalyticKernel(_map_model)

            _map_max_lag = _map_model.envelope.kernel_support() * 1.5
            _map_lag = np.linspace(0, _map_max_lag, 500)
            _map_K = np.asarray(_map_kernel_obj.kernel(_map_lag))
            _map_harmonics = _compute_harmonic_components(
                _map_kernel_obj, _map_lag)
            st.session_state["map_kernel_data"] = (
                _map_lag, _map_K, _map_harmonics)

            _map_omega_max = (2 * np.pi
                              / map_mp["visibility_params"]["peq"] * 5)
            _map_omega = np.linspace(0.01, _map_omega_max, 1000)
            _map_psd_freq, _map_psd_power = _map_kernel_obj.compute_psd(
                _map_omega)
            st.session_state["map_psd_data"] = (
                np.asarray(_map_psd_freq), np.asarray(_map_psd_power))

            _map_mu, _map_var = gp.predict(t)
            _map_std = np.sqrt(np.clip(_map_var, 0, None))
            st.session_state["map_gp_prediction"] = (t, _map_mu, _map_std)

            st.sidebar.success("MAP fit complete.")
        except Exception as e:
            st.sidebar.error(f"Fit failed: {e}")

    if "fit_result" in st.session_state:
        if st.sidebar.button("Save results", use_container_width=True):
            import h5py
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            if not save_path.endswith(".h5"):
                save_path += ".h5"
            with h5py.File(save_path, "w") as f:
                t_s, y_s, yerr_s = st.session_state["data_arrays"]
                dg = f.create_group("data")
                dg.create_dataset("time", data=t_s)
                dg.create_dataset("flux", data=y_s)
                dg.create_dataset("flux_err", data=yerr_s)

                mg = f.create_group("model")
                mg.attrs["envelope"] = _opt_mp["envelope_name"]
                mg.attrs["visibility"] = _opt_mp["visibility_name"]
                mg.attrs["sigma_k"] = _opt_mp["sigma_k"]
                for k, v in _opt_mp["envelope_params"].items():
                    mg.attrs[f"envelope_{k}"] = v
                for k, v in _opt_mp["visibility_params"].items():
                    mg.attrs[f"visibility_{k}"] = v

                gp = st.session_state["gp"]
                sg = f.create_group("solver")
                sg.attrs["kernel_type"] = gp.kernel_type
                sg.attrs["matrix_solver"] = gp.matrix_solver
                sg.create_dataset("param_keys",
                                  data=list(gp.param_keys))
                sg.create_dataset("bounds", data=gp.bounds)

                rg = f.create_group("map")
                result = st.session_state["fit_result"]
                theta_map = st.session_state["theta_map"]
                rg.attrs["neg_log_posterior"] = float(result.fun)
                if isinstance(theta_map, dict):
                    for k, v in theta_map.items():
                        rg.attrs[k] = float(v)
            st.sidebar.success(f"Saved to {save_path}")


# ── Config export ────────────────────────────────────────────────────


st.sidebar.markdown("---")
st.sidebar.header("Export")

if data_source == "TIC / KIC ID":
    _export_star = star_name.strip()
    _export_stem = _export_star.replace(" ", "_")
else:
    _export_star = os.path.splitext(os.path.basename(file_path))[0]
    _export_stem = _export_star

export_path = st.sidebar.text_input(
    "Config path", value=f"configs/{_export_stem}.yaml")

if st.sidebar.button("Export config", use_container_width=True):
    _data_cfg = {}
    if data_source == "Local file":
        _data_cfg["path"] = file_path
    elif sectors:
        _data_cfg["sectors"] = list(sectors)
    if has_raw:
        _data_cfg["normalize"] = bool(normalize)
        _data_cfg["zero_mean"] = bool(zero_mean)
        if downsample > 1:
            _data_cfg["downsample"] = int(downsample)
    else:
        _data_cfg["normalize"] = True
        _data_cfg["zero_mean"] = True

    _model_cfg = {
        "envelope": envelope_name,
        "envelope_params": {k: float(v) for k, v in envelope_params.items()},
        "visibility": visibility_name,
        "visibility_params": {k: float(v)
                              for k, v in visibility_params.items()},
        "sigma_k": float(sigma_k),
    }
    if latitude_name != "LatitudeDistributionFunction":
        _model_cfg["latitude"] = latitude_name
        _model_cfg["latitude_params"] = {
            k: float(v) for k, v in latitude_params.items()}

    _export_bounds = st.session_state.get("fit_bounds") or {
        k: _default_fit_bounds(k, v)
        for k, v in {**visibility_params, "log_sigma_k": log_sigma_k}.items()
        if k != "inc"
    }

    _cfg = {
        "star_name": _export_star,
        "data": _data_cfg,
        "model": _model_cfg,
        "bounds": {k: [float(v[0]), float(v[1])]
                   for k, v in _export_bounds.items()},
        "solver": {"kernel_type": "analytic",
                   "matrix_solver": "cholesky_banded"},
        "fitting": {"map": {"nopt": 10},
                    "sampling": {"sampler": "dynesty", "nlive": 500}},
        "output": {"save_dir": "results"},
    }

    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
    with open(export_path, "w") as f:
        yaml.safe_dump(_cfg, f, sort_keys=False)
    st.sidebar.success(f"Wrote {export_path} — run with "
                       f"`python scripts/run_fit.py {export_path}`")
    with st.sidebar.expander("Config contents"):
        st.code(yaml.safe_dump(_cfg, sort_keys=False), language="yaml")


# ── Kernel computation (cached) ─────────────────────────────────────


model_ready = "model_params" in st.session_state
kernel_error = None
lag = K = max_lag = harmonics = None

if model_ready:
    mp = st.session_state["model_params"]
    envelope_params_tuple = tuple(sorted(mp["envelope_params"].items()))
    visibility_params_tuple = tuple(sorted(mp["visibility_params"].items()))
    latitude_params_tuple = tuple(sorted(mp.get("latitude_params", {}).items()))
    try:
        lag, K, max_lag, harmonics = cached_kernel(
            mp["envelope_name"], envelope_params_tuple,
            mp["visibility_name"], visibility_params_tuple, mp["sigma_k"],
            mp.get("latitude_name", "LatitudeDistributionFunction"),
            latitude_params_tuple)
    except Exception as e:
        kernel_error = str(e)


# ── Main panel ───────────────────────────────────────────────────────


star_label = st.session_state.get("star_name", "")


st.subheader("Data")
if has_data:
    t, y, yerr = st.session_state["data_arrays"]
    fig_data = go.Figure()
    fig_data.add_trace(go.Scattergl(
        x=t, y=y, mode="markers",
        marker=dict(size=2, color="black", opacity=0.4),
        name="Data",
        hovertemplate="Time: %{x:.2f}<br>Flux: %{y:.6f}<extra></extra>"))

    if "gp_prediction" in st.session_state:
        t_gp, mu_pred, std_pred = st.session_state["gp_prediction"]
        fig_data.add_trace(go.Scatter(
            x=np.concatenate([t_gp, t_gp[::-1]]),
            y=np.concatenate([mu_pred + 2 * std_pred,
                              (mu_pred - 2 * std_pred)[::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.15)",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip"))
        fig_data.add_trace(go.Scatter(
            x=t_gp, y=mu_pred, mode="lines",
            name="GP mean",
            line=dict(width=1.5, color="#d62728"),
            hovertemplate="Time: %{x:.2f}<br>Mean: %{y:.6f}"
                          "<extra></extra>"))

    if "map_gp_prediction" in st.session_state:
        t_map, mu_map, std_map = st.session_state["map_gp_prediction"]
        fig_data.add_trace(go.Scatter(
            x=np.concatenate([t_map, t_map[::-1]]),
            y=np.concatenate([mu_map + 2 * std_map,
                              (mu_map - 2 * std_map)[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip"))
        fig_data.add_trace(go.Scatter(
            x=t_map, y=mu_map, mode="lines",
            name="MAP GP mean",
            line=dict(width=1.5, color="#1f77b4"),
            hovertemplate="Time: %{x:.2f}<br>Mean: %{y:.6f}"
                          "<extra></extra>"))

    fig_data.update_layout(
        title=f"{len(t)} data points",
        xaxis_title="Time (days)", yaxis_title="Flux",
        height=300, margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig_data, use_container_width=True)
else:
    st.info("Load data using the sidebar to see the light curve.")

st.subheader("Kernel / ACF")
if not model_ready:
    st.info("Configure model parameters and click **Generate**.")
elif kernel_error is not None:
    st.error(f"Model error: {kernel_error}")
else:
    _harmonic_colors = ["#aaa", "#e377c2", "#2ca02c", "#ff7f0e"]
    _hcols = st.columns(len(harmonics))
    _selected_harmonics = []
    for n in range(len(harmonics)):
        with _hcols[n]:
            if st.checkbox(f"n={n}", value=(n != 0), key=f"harm_{n}"):
                _selected_harmonics.append(n)

    fig_kernel = go.Figure()
    _normalize_kernel = False
    _K0_manual = K[0] if K[0] != 0 else 1.0

    if has_data:
        try:
            spotgp_acf = _import_spotgp()
            t_acf, y_acf, yerr_acf = st.session_state["data_arrays"]
            data_obj = spotgp_acf.TimeSeriesData(
                t_acf, y_acf, yerr_acf, normalize=False)
            lag_centers, acf_data = data_obj.compute_acf()
            acf_mask = lag_centers <= lag[-1]
            _normalize_kernel = True
            fig_kernel.add_trace(go.Scatter(
                x=lag_centers[acf_mask], y=acf_data[acf_mask],
                mode="lines", name="Data ACF",
                line=dict(width=1, color="black"), opacity=0.7,
                hovertemplate="Lag: %{x:.2f}<br>ACF: %{y:.4f}"
                              "<extra></extra>"))
        except Exception:
            pass

    K_sum = (sum(harmonics[n] for n in _selected_harmonics)
             if _selected_harmonics else np.zeros_like(lag))
    if _normalize_kernel:
        K_sum = K_sum / _K0_manual
    fig_kernel.add_trace(go.Scatter(
        x=lag, y=K_sum, mode="lines",
        name="Model kernel",
        line=dict(width=2, color="#d62728"),
        hovertemplate="Lag: %{x:.2f}<br>K: %{y:.4f}<extra></extra>"))

    if "map_kernel_data" in st.session_state:
        map_lag, map_K, map_harmonics = st.session_state["map_kernel_data"]
        _K0_map = map_K[0] if map_K[0] != 0 else 1.0
        map_K_sum = (sum(map_harmonics[n] for n in _selected_harmonics)
                     if _selected_harmonics else np.zeros_like(map_lag))
        if _normalize_kernel:
            map_K_sum = map_K_sum / _K0_map
        fig_kernel.add_trace(go.Scatter(
            x=map_lag, y=map_K_sum, mode="lines",
            name="MAP kernel",
            line=dict(width=2, color="#1f77b4"),
            hovertemplate="Lag: %{x:.2f}<br>K: %{y:.4f}"
                          "<extra></extra>"))

    fig_kernel.add_hline(y=0, line_dash="dash", line_color="gray",
                         line_width=0.5)
    fig_kernel.update_layout(
        xaxis_title="Lag (days)", yaxis_title="Autocorrelation",
        height=300, margin=dict(l=50, r=20, t=20, b=40))
    st.plotly_chart(fig_kernel, use_container_width=True)


# ── Model PSD ────────────────────────────────────────────────────────


if model_ready and kernel_error is None:
    st.subheader("Power spectral density")
    spotgp_mod = _import_spotgp()

    envelope_cls = getattr(spotgp_mod, mp["envelope_name"])
    envelope_obj = envelope_cls(**mp["envelope_params"])
    vis_cls = getattr(spotgp_mod, mp["visibility_name"])
    visibility_obj = vis_cls(**mp["visibility_params"])
    psd_lat = _build_latitude_dist(
        mp.get("latitude_name", "LatitudeDistributionFunction"),
        mp.get("latitude_params", {}))
    psd_model = spotgp_mod.SpotEvolutionModel(
        envelope=envelope_obj, visibility=visibility_obj,
        sigma_k=mp["sigma_k"], latitude_distribution=psd_lat)
    psd_kernel = spotgp_mod.AnalyticKernel(psd_model)

    omega_max = 2 * np.pi / mp["visibility_params"]["peq"] * 5
    omega = np.linspace(0.01, omega_max, 1000)
    psd_freq, psd_power = psd_kernel.compute_psd(omega)
    psd_freq = np.asarray(psd_freq)
    psd_power = np.asarray(psd_power)

    p_rot = mp["visibility_params"]["peq"]

    fig_psd = go.Figure()

    if has_data:
        try:
            t_psd, y_psd, yerr_psd = st.session_state["data_arrays"]
            data_psd_obj = spotgp_mod.TimeSeriesData(
                t_psd, y_psd, yerr_psd, normalize=False)
            data_freq, data_power = data_psd_obj.compute_psd(
                freq_max=psd_freq[-1])
            fig_psd.add_trace(go.Scatter(
                x=np.asarray(data_freq), y=np.asarray(data_power),
                mode="lines", name="Data PSD",
                line=dict(width=1, color="black"), opacity=0.7,
                hovertemplate=("Freq: %{x:.4f} c/d<br>"
                               "Power: %{y:.4e}<extra></extra>")))
        except Exception:
            pass

    fig_psd.add_trace(go.Scatter(
        x=psd_freq, y=psd_power, mode="lines",
        name="Model PSD",
        line=dict(width=2, color="#d62728"),
        hovertemplate=("Freq: %{x:.4f} c/d<br>"
                       "Power: %{y:.4e}<extra></extra>")))

    if "map_psd_data" in st.session_state:
        map_psd_freq, map_psd_power = st.session_state["map_psd_data"]
        fig_psd.add_trace(go.Scatter(
            x=map_psd_freq, y=map_psd_power, mode="lines",
            name="MAP PSD",
            line=dict(width=2, color="#1f77b4"),
            hovertemplate=("Freq: %{x:.4f} c/d<br>"
                           "Power: %{y:.4e}<extra></extra>")))

    for n in range(1, 4):
        fig_psd.add_vline(
            x=n / p_rot, line_dash="dot", line_color="gray",
            line_width=0.8,
            annotation_text=f"{n}/P" if n == 1 else f"{n}/P",
            annotation_position="top")
    fig_psd.update_layout(
        xaxis_title="Frequency (cycles/day)",
        yaxis_title="Power",
        yaxis_type="log",
        height=320, margin=dict(l=50, r=20, t=20, b=40))
    st.plotly_chart(fig_psd, use_container_width=True)


# ── Model equations ──────────────────────────────────────────────────


if model_ready and kernel_error is None:
    import sympy
    spotgp = _import_spotgp()

    envelope_cls = getattr(spotgp, mp["envelope_name"])
    envelope_obj = envelope_cls(**mp["envelope_params"])
    vis_cls = getattr(spotgp, mp["visibility_name"])
    visibility_obj = vis_cls(**mp["visibility_params"])

    with st.expander("Model equations", expanded=False):
        env_exprs = envelope_obj.get_sympy()
        vis_exprs = visibility_obj.get_sympy()

        label_map = {
            "Gamma": r"\Gamma(t)",
            "Gamma_hat": r"\hat{\Gamma}(\omega)",
            "R_Gamma": r"R_\Gamma(\tau)",
            "omega0": r"\omega_0(\phi)",
            "a0": r"a_0",
            "a1": r"a_1",
            "theta_v": r"\theta_v",
            "c0": r"c_0",
            "c1": r"c_1",
            "cn": r"c_n",
            "pdf": r"p(\phi)",
        }

        st.markdown(f"**Envelope** — {mp['envelope_name']}")
        for name, expr in env_exprs.items():
            if expr is None:
                continue
            lhs = label_map.get(name, name)
            st.latex(f"{lhs} = {sympy.latex(expr)}")

        st.markdown(f"**Visibility** — {mp['visibility_name']}")
        for name, expr in vis_exprs.items():
            if expr is None:
                continue
            lhs = label_map.get(name, name)
            st.latex(f"{lhs} = {sympy.latex(expr)}")

        lat_name = mp.get("latitude_name", "LatitudeDistributionFunction")
        lat_obj = _build_latitude_dist(lat_name, mp.get("latitude_params", {}))
        lat_exprs = lat_obj.get_sympy(display=False)
        st.markdown(f"**Latitude distribution** — {lat_name}")
        for name, expr in lat_exprs.items():
            if expr is None:
                continue
            lhs = label_map.get(name, name)
            st.latex(f"{lhs} = {sympy.latex(expr)}")


# ── Single-spot flux ────────────────────────────────────────────────


if model_ready and kernel_error is None:
    st.subheader("Single-spot flux")
    spotgp = _import_spotgp()

    col_sp1, col_sp2, col_sp3, col_sp4 = st.columns(4)
    with col_sp1:
        spot_long = st.slider("Longitude (deg)", 0.0, 360.0, 180.0, 1.0)
    with col_sp2:
        spot_lat = st.slider("Latitude (deg)", -90.0, 90.0, 30.0, 1.0)
    with col_sp3:
        spot_alpha = st.slider("Spot contrast (alpha_max)", 0.001, 0.5, 0.1,
                               0.001, format="%.3f")
    with col_sp4:
        spot_tsim = st.slider("Duration (days)", 10.0, 200.0, 60.0, 5.0)

    envelope_cls = getattr(spotgp, mp["envelope_name"])
    envelope_obj = envelope_cls(**mp["envelope_params"])
    vis_cls = getattr(spotgp, mp["visibility_name"])
    visibility_obj = vis_cls(**mp["visibility_params"])
    spot_lat_dist = _build_latitude_dist(
        mp.get("latitude_name", "LatitudeDistributionFunction"),
        mp.get("latitude_params", {}))
    spot_model = spotgp.SpotEvolutionModel(
        envelope=envelope_obj, visibility=visibility_obj,
        sigma_k=mp["sigma_k"], latitude_distribution=spot_lat_dist)

    lc = spotgp.LightcurveModel.from_spot_model(
        spot_model, nspot=1,
        long=np.radians(spot_long),
        lat=np.radians(spot_lat),
        alpha_max=spot_alpha,
        tsim=spot_tsim, tsamp=0.02,
        tmax=spot_tsim / 2.0,
    )

    tmax_spot = spot_tsim / 2.0
    env_t = lc.t - tmax_spot
    env_vals = envelope_obj.Gamma(env_t)
    dspot_peak = lc.dspots[0].max()
    env_scaled = env_vals * dspot_peak if dspot_peak > 0 else env_vals

    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(
        x=lc.t, y=env_scaled, mode="lines",
        line=dict(width=1, color="gray", dash="dash"),
        name="Envelope",
        hovertemplate="Time: %{x:.2f} d<br>Env: %{y:.6f}<extra></extra>"))
    fig_spot.add_trace(go.Scatter(
        x=lc.t, y=lc.dspots[0], mode="lines",
        line=dict(width=1.5, color="#d62728"),
        name="Spot flux deficit",
        hovertemplate="Time: %{x:.2f} d<br>dF: %{y:.6f}<extra></extra>"))
    fig_spot.add_trace(go.Scatter(
        x=lc.t, y=lc.flux, mode="lines",
        line=dict(width=1.5, color="black"),
        name="Total flux",
        visible="legendonly",
        hovertemplate="Time: %{x:.2f} d<br>F: %{y:.6f}<extra></extra>"))

    if "map_model_params" in st.session_state:
        _mmp = st.session_state["map_model_params"]
        _me_cls = getattr(spotgp, _mmp["envelope_name"])
        _me = _me_cls(**_mmp["envelope_params"])
        _mv_cls = getattr(spotgp, _mmp["visibility_name"])
        _mv = _mv_cls(**_mmp["visibility_params"])
        _m_lat = _build_latitude_dist(
            _mmp.get("latitude_name", "LatitudeDistributionFunction"),
            _mmp.get("latitude_params", {}))
        _m_spot_model = spotgp.SpotEvolutionModel(
            envelope=_me, visibility=_mv, sigma_k=_mmp["sigma_k"],
            latitude_distribution=_m_lat)
        _m_lc = spotgp.LightcurveModel.from_spot_model(
            _m_spot_model, nspot=1,
            long=np.radians(spot_long),
            lat=np.radians(spot_lat),
            alpha_max=spot_alpha,
            tsim=spot_tsim, tsamp=0.02,
            tmax=spot_tsim / 2.0,
        )
        _m_env_t = _m_lc.t - spot_tsim / 2.0
        _m_env_vals = _me.Gamma(_m_env_t)
        _m_dspot_peak = _m_lc.dspots[0].max()
        _m_env_scaled = (_m_env_vals * _m_dspot_peak
                         if _m_dspot_peak > 0 else _m_env_vals)
        fig_spot.add_trace(go.Scatter(
            x=_m_lc.t, y=_m_env_scaled, mode="lines",
            line=dict(width=1, color="cornflowerblue", dash="dash"),
            name="MAP envelope",
            hovertemplate="Time: %{x:.2f} d<br>Env: %{y:.6f}"
                          "<extra></extra>"))
        fig_spot.add_trace(go.Scatter(
            x=_m_lc.t, y=_m_lc.dspots[0], mode="lines",
            line=dict(width=1.5, color="#1f77b4"),
            name="MAP spot flux",
            hovertemplate="Time: %{x:.2f} d<br>dF: %{y:.6f}"
                          "<extra></extra>"))
        fig_spot.add_trace(go.Scatter(
            x=_m_lc.t, y=_m_lc.flux, mode="lines",
            line=dict(width=1.5, color="steelblue"),
            name="MAP total flux",
            visible="legendonly",
            hovertemplate="Time: %{x:.2f} d<br>F: %{y:.6f}"
                          "<extra></extra>"))

    fig_spot.update_layout(
        xaxis_title="Time (days)", yaxis_title="Flux change",
        height=320, margin=dict(l=50, r=20, t=20, b=40))
    st.plotly_chart(fig_spot, use_container_width=True)


# ── MAP Results (main panel) ────────────────────────────────────────


st.markdown("---")
st.subheader("MAP Results")

if "fit_result" not in st.session_state:
    st.info("Configure and run a MAP fit from the sidebar.")
else:
    _fit_result = st.session_state["fit_result"]
    _fit_theta = st.session_state["theta_map"]
    _fit_gp = st.session_state["gp"]

    st.success(f"Optimization complete — neg log posterior: "
               f"{_fit_result.fun:.4f}")

    if isinstance(_fit_theta, dict):
        st.table({k: f"{v:.6f}" for k, v in _fit_theta.items()})
    else:
        st.write(_fit_theta)
