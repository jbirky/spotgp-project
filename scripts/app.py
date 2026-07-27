"""Interactive spotgp kernel explorer and fitting tool."""

import html
import os
import sys
from datetime import datetime

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
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
def cached_kernel(spot_components_tuple, sho_components_tuple,
                  visibility_name, visibility_params_tuple,
                  latitude_name, latitude_params_tuple,
                  max_lag_override=None, n_lag=500):
    """Compute kernel for spot models and optional SHO terms.

    spot_components_tuple: tuple of (label, envelope_name, env_params_tuple, sigma_k)
    sho_components_tuple: tuple of (label, sho_params_tuple)
    """
    spotgp = _import_spotgp()

    auto_max_lag = 0.0
    has_spot = len(spot_components_tuple) > 0
    has_sho = len(sho_components_tuple) > 0

    ak = None
    if has_spot:
        visibility_params = dict(visibility_params_tuple)
        vis_cls = getattr(spotgp, visibility_name)
        visibility = vis_cls(**visibility_params)
        lat_dist = _build_latitude_dist(
            latitude_name, dict(latitude_params_tuple))

        spot_models = []
        labels = []
        for label, env_name, env_params_t, sk in spot_components_tuple:
            env_cls = getattr(spotgp, env_name)
            envelope = env_cls(**dict(env_params_t))
            sm = spotgp.SpotEvolutionModel(
                envelope=envelope, visibility=visibility, sigma_k=sk,
                latitude_distribution=lat_dist)
            spot_models.append(sm)
            labels.append(label)

        if len(spot_models) == 1:
            model = spot_models[0]
            ak = spotgp.AnalyticKernel(model)
        else:
            model = spotgp.CompositeSpotModel(spot_models, labels=labels)
            ak = spotgp.CompositeAnalyticKernel(model)
        auto_max_lag = max(sm.envelope.kernel_support()
                          for sm in spot_models) * 1.5

    sho_terms = []
    if has_sho:
        for _label, sho_params_t in sho_components_tuple:
            sho_terms.append(spotgp.SHOTerm(**dict(sho_params_t)))
        sho_support = max(t.rho * 6 for t in sho_terms)
        auto_max_lag = max(auto_max_lag, sho_support)

    if auto_max_lag == 0:
        auto_max_lag = 50.0

    max_lag = max_lag_override if max_lag_override is not None else auto_max_lag
    lag = np.linspace(0, max_lag, n_lag)

    K_spot = np.asarray(ak.kernel(lag)) if ak else np.zeros_like(lag)
    K_sho = _eval_sho_kernel(sho_terms, lag) if sho_terms else np.zeros_like(lag)
    K = K_spot + K_sho

    harmonics = _compute_all_harmonics(ak, lag) if ak else {0: K_sho}
    if sho_terms and ak:
        harmonics[0] = harmonics.get(0, np.zeros_like(lag)) + K_sho

    component_kernels = {}
    if has_spot and len(spot_components_tuple) > 1:
        for i, (label, env_name, env_params_t, sk) in enumerate(
                spot_components_tuple):
            ck = ak._component_kernels[i]
            component_kernels[label] = np.asarray(ck.kernel(lag))
    elif has_spot:
        component_kernels[spot_components_tuple[0][0]] = K_spot
    for i, (label, sho_params_t) in enumerate(sho_components_tuple):
        component_kernels[label] = np.asarray(
            sho_terms[i].get_value(np.abs(lag)))

    return lag, K, max_lag, auto_max_lag, harmonics, component_kernels


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


def _compute_all_harmonics(kernel_obj, lag):
    """Compute harmonic components for single or composite kernels."""
    spotgp = _import_spotgp()
    if isinstance(kernel_obj, spotgp.CompositeAnalyticKernel):
        combined = None
        for ck in kernel_obj._component_kernels:
            h = _compute_harmonic_components(ck, lag)
            if combined is None:
                combined = {n: v.copy() for n, v in h.items()}
            else:
                for n in combined:
                    combined[n] = combined[n] + h[n]
        return combined
    return _compute_harmonic_components(kernel_obj, lag)


@st.cache_data(show_spinner="Downloading light curve...")
def download_lightcurve(star_name, sectors_tuple, pipeline=None):
    segments, sector_numbers, err = data_utils.download_lightcurve(
        star_name, sectors_tuple, pipeline)
    return segments, sector_numbers, err


@st.cache_data(show_spinner="Loading file...")
def load_local_file(path):
    return data_utils.load_local_file(path)


@st.cache_data(show_spinner="Computing ACF...")
def cached_acf(t, y, yerr, max_lag, n_bins=200):
    spotgp = _import_spotgp()
    data_obj = spotgp.TimeSeriesData(t, y, yerr, normalize=False)
    return data_obj.compute_acf(n_bins=n_bins, max_lag=max_lag)


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

ENVELOPE_OPTIONS = [
    "TrapezoidSymmetricEnvelope",
    "TrapezoidAsymmetricEnvelope",
    "ExponentialEnvelope",
    "ExponentialAsymmetricEnvelope",
    "SkewedGaussianEnvelope",
]


def _render_envelope_params(container, envelope_name, key_suffix):
    """Render envelope parameter sliders for a given envelope type."""
    params = {}
    if envelope_name == "TrapezoidSymmetricEnvelope":
        params["lspot"] = _synced_slider(
            container, "lspot (days)", 0.0, 100.0, 15.0, 0.5,
            key=f"lspot_{key_suffix}")
        params["tau_spot"] = _synced_slider(
            container, "tau_spot (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"tau_spot_{key_suffix}")
    elif envelope_name == "TrapezoidAsymmetricEnvelope":
        params["lspot"] = _synced_slider(
            container, "lspot (days)", 0.0, 100.0, 15.0, 0.5,
            key=f"lspot_{key_suffix}")
        params["tau_em"] = _synced_slider(
            container, "tau_em (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"tau_em_{key_suffix}")
        params["tau_dec"] = _synced_slider(
            container, "tau_dec (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"tau_dec_{key_suffix}")
    elif envelope_name == "ExponentialEnvelope":
        params["tau_spot"] = _synced_slider(
            container, "tau_spot (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"tau_spot_{key_suffix}")
    elif envelope_name == "ExponentialAsymmetricEnvelope":
        params["tau_em"] = _synced_slider(
            container, "tau_em (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"tau_em_{key_suffix}")
        params["tau_dec"] = _synced_slider(
            container, "tau_dec (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"tau_dec_{key_suffix}")
    elif envelope_name == "SkewedGaussianEnvelope":
        params["sigma_sn"] = _synced_slider(
            container, "sigma_sn (days)", 0.1, 50.0, 5.0, 0.1,
            key=f"sigma_sn_{key_suffix}")
        params["n_sn"] = _synced_slider(
            container, "n_sn", 0.0, 10.0, 2.0, 0.1,
            key=f"n_sn_{key_suffix}")
    return params


COMPONENT_TYPES = ["Spot", "SHO Term"]

SHO_DAMPING_OPTIONS = ["tau (damping time)", "Q (quality factor)"]


def _render_sho_params(container, key_suffix):
    """Render SHOTerm parameter sliders and return the params dict."""
    sigma = _synced_slider(
        container, "sigma", 0.0001, 0.1, 0.005, 0.0001,
        key=f"sho_sigma_{key_suffix}", fmt="%.4f")
    rho = _synced_slider(
        container, "rho (days)", 0.1, 50.0, 5.0, 0.1,
        key=f"sho_rho_{key_suffix}")
    damping = container.radio(
        "Damping", SHO_DAMPING_OPTIONS, key=f"sho_damp_{key_suffix}",
        horizontal=True)
    if damping == SHO_DAMPING_OPTIONS[0]:
        tau = _synced_slider(
            container, "tau (days)", 0.1, 100.0, 10.0, 0.1,
            key=f"sho_tau_{key_suffix}")
        return {"sigma": sigma, "rho": rho, "tau": tau}
    else:
        Q = _synced_slider(
            container, "Q", 0.1, 20.0, 2.0, 0.1,
            key=f"sho_Q_{key_suffix}")
        return {"sigma": sigma, "rho": rho, "Q": Q}


def _build_sho_terms(mp):
    """Build a list of SHOTerm objects from sho-type components."""
    spotgp = _import_spotgp()
    terms = []
    for comp in mp.get("components", []):
        if comp.get("type") != "sho":
            continue
        p = comp["sho_params"]
        terms.append(spotgp.SHOTerm(**p))
    return terms


def _eval_sho_kernel(terms, lag):
    """Evaluate summed SHOTerm kernel at given lag values."""
    K = np.zeros_like(lag)
    for term in terms:
        K = K + np.asarray(term.get_value(np.abs(lag)))
    return K


def _eval_sho_psd(terms, omega):
    """Evaluate summed SHOTerm PSD at given angular frequencies."""
    P = np.zeros_like(omega)
    for term in terms:
        P = P + np.asarray(term.get_psd(omega))
    return P


def _spot_components(mp):
    """Return only the spot-type components from model_params."""
    return [c for c in mp.get("components", [])
            if c.get("type", "spot") == "spot"]


def _sho_components(mp):
    """Return only the sho-type components from model_params."""
    return [c for c in mp.get("components", []) if c.get("type") == "sho"]


def _build_spotgp_model(mp):
    """Build a SpotEvolutionModel or CompositeSpotModel from spot components."""
    spotgp = _import_spotgp()
    vis_cls = getattr(spotgp, mp["visibility_name"])
    visibility = vis_cls(**mp["visibility_params"])
    lat_dist = _build_latitude_dist(
        mp.get("latitude_name", "LatitudeDistributionFunction"),
        mp.get("latitude_params", {}))

    spot_comps = _spot_components(mp)
    if not spot_comps:
        return None

    spot_models = []
    labels = []
    for comp in spot_comps:
        env_cls = getattr(spotgp, comp["envelope_name"])
        envelope = env_cls(**comp["envelope_params"])
        sm = spotgp.SpotEvolutionModel(
            envelope=envelope, visibility=visibility,
            sigma_k=comp["sigma_k"], latitude_distribution=lat_dist)
        spot_models.append(sm)
        labels.append(comp["label"])

    if len(spot_models) == 1:
        return spot_models[0]
    return spotgp.CompositeSpotModel(spot_models, labels=labels)


def _build_kernel_object(model):
    """Build an AnalyticKernel or CompositeAnalyticKernel."""
    spotgp = _import_spotgp()
    if isinstance(model, spotgp.CompositeSpotModel):
        return spotgp.CompositeAnalyticKernel(model)
    return spotgp.AnalyticKernel(model)


def _is_composite(mp):
    return len(_spot_components(mp)) > 1


def _default_fit_bounds(key, value):
    """Heuristic optimization bounds centered on the current slider value."""
    if key.startswith("log_sigma_k"):
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
    """Update model_params with MAP-fitted values from theta_map."""
    vis_params = dict(mp["visibility_params"])
    lat_params = dict(mp.get("latitude_params", {}))
    components = []
    for c in mp["components"]:
        if c.get("type") == "sho":
            components.append(dict(c))
            continue
        components.append({
            "label": c["label"],
            "type": c.get("type", "spot"),
            "envelope_name": c["envelope_name"],
            "envelope_params": dict(c["envelope_params"]),
            "sigma_k": c["sigma_k"],
            "log_sigma_k": c["log_sigma_k"],
        })

    lat_name = mp.get("latitude_name", "LatitudeDistributionFunction")
    internal_to_app = {}
    for app_key, (internal_key, _, to_app) in _LAT_PARAM_MAP.get(
            lat_name, {}).items():
        internal_to_app[internal_key] = (app_key, to_app)

    spot_comps = [c for c in components if c.get("type", "spot") == "spot"]
    is_multi = len(spot_comps) > 1

    for key, val in theta_map.items():
        if key in vis_params:
            vis_params[key] = float(val)
            continue
        if key in lat_params:
            lat_params[key] = float(val)
            continue
        if key in internal_to_app:
            app_key, to_app = internal_to_app[key]
            lat_params[app_key] = float(to_app(val))
            continue

        if is_multi:
            for comp in spot_comps:
                label = comp["label"]
                if key == f"log_sigma_k_{label}":
                    comp["log_sigma_k"] = float(val)
                    comp["sigma_k"] = 10.0 ** float(val)
                    break
                matched = False
                for ep_key in list(comp["envelope_params"]):
                    if key == f"{ep_key}_{label}":
                        comp["envelope_params"][ep_key] = float(val)
                        matched = True
                        break
                if matched:
                    break
        elif spot_comps:
            comp = spot_comps[0]
            if key == "log_sigma_k":
                comp["log_sigma_k"] = float(val)
                comp["sigma_k"] = 10.0 ** float(val)
            elif key in comp["envelope_params"]:
                comp["envelope_params"][key] = float(val)

    return dict(
        visibility_name=mp["visibility_name"],
        visibility_params=vis_params,
        latitude_name=lat_name,
        latitude_params=lat_params,
        components=components,
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


# ── Figure export helpers ────────────────────────────────────────────


EXPORT_FIGS = []


def show_fig(fig, name):
    """Render a Plotly figure and register it for the HTML export."""
    EXPORT_FIGS.append((name, fig))
    st.plotly_chart(fig, use_container_width=True)


_EXPORT_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
       Helvetica, Arial, sans-serif; margin: 0 auto; padding: 2rem 1.5rem;
       max-width: 1100px; color: #222; background: #fff; }
h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 2rem 0 0.5rem;
     padding-bottom: 0.3rem; border-bottom: 1px solid #e5e5e5; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
.params { background: #f7f7f7; border: 1px solid #e5e5e5; border-radius: 4px;
          padding: 0.75rem 1rem; font-size: 0.85rem; overflow-x: auto;
          white-space: pre; }
"""


def build_plots_html(figs, title, params_text=None, standalone=True):
    """Assemble selected Plotly figures into one static HTML document."""
    blocks = []
    for i, (name, fig) in enumerate(figs):
        if i == 0:
            include_js = True if standalone else "cdn"
        else:
            include_js = False
        blocks.append(
            f"<h2>{html.escape(name)}</h2>\n"
            + pio.to_html(fig, include_plotlyjs=include_js, full_html=False,
                          default_width="100%", div_id=f"fig-{i}"))

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [f"<h1>{html.escape(title)}</h1>",
              f'<div class="meta">spotgp Explorer — generated {stamp}</div>']
    if params_text:
        header.append(f'<div class="params">{html.escape(params_text)}</div>')

    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, "
            "initial-scale=1\">\n"
            f"<title>{html.escape(title)}</title>\n"
            f"<style>{_EXPORT_CSS}</style>\n</head>\n<body>\n"
            + "\n".join(header) + "\n" + "\n".join(blocks)
            + "\n</body>\n</html>\n")


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
if _device_idx is not None:
    jax.config.update("jax_default_device", _available_devices[_device_idx])

st.sidebar.markdown("---")
st.sidebar.header("Data")
data_source = st.sidebar.radio("Source", ["TIC / KIC ID", "Local file"])

if data_source == "TIC / KIC ID":
    star_name = st.sidebar.text_input("Star name", value="KIC 7286309")
    pipeline = st.sidebar.selectbox("Pipeline", list(data_utils.PIPELINES))
    sectors_input = st.sidebar.text_input(
        "Sectors / Quarters (comma-separated, blank for all)", value="")
    sectors = tuple(int(s.strip()) for s in sectors_input.split(",")
                    if s.strip()) if sectors_input.strip() else None
    if st.sidebar.button("Download"):
        segments, sector_numbers, err = download_lightcurve(
            star_name, sectors, pipeline)
        if err:
            st.sidebar.error(err)
        else:
            st.session_state["raw_segments"] = segments
            st.session_state["sector_numbers"] = sector_numbers
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
st.sidebar.header("Visibility")

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
        st.sidebar, "kappa", -1.0, 1.0, 0.0, 0.01, key="kappa")
    visibility_params["inc"] = _synced_slider(
        st.sidebar, "inc (deg)", 0.0, 90.0, 90.0, 1.0,
        key="inc") * np.pi / 180.0

st.sidebar.markdown("---")
st.sidebar.header("Latitude Distribution")

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
st.sidebar.header("Kernel Components")

if "components" not in st.session_state:
    st.session_state["components"] = [
        {"id": 0, "label": "default", "type": "spot"}]
    st.session_state["next_comp_id"] = 1

_comp_configs = []
for _ci, _comp in enumerate(st.session_state["components"]):
    _cid = _comp["id"]
    _ksuf = f"c{_cid}"
    _exp_label = _comp.get("label", f"comp{_cid}")
    _comp_type = _comp.get("type", "spot")
    _type_tag = "SHO" if _comp_type == "sho" else "Spot"
    _exp = st.sidebar.expander(
        f"{_exp_label} ({_type_tag})", expanded=True)
    with _exp:
        _new_label = st.text_input(
            "Label", value=_exp_label, key=f"label_{_ksuf}")
        _comp["label"] = _new_label

        _default_type_idx = COMPONENT_TYPES.index(
            "SHO Term" if _comp_type == "sho" else "Spot")
        _ctype = st.selectbox(
            "Type", COMPONENT_TYPES, index=_default_type_idx,
            key=f"type_{_ksuf}")
        _comp["type"] = "sho" if _ctype == "SHO Term" else "spot"

        if _comp["type"] == "spot":
            _env_name = st.selectbox(
                "Envelope", ENVELOPE_OPTIONS, key=f"env_{_ksuf}")
            _env_params = _render_envelope_params(_exp, _env_name, _ksuf)
            _log_sk = _synced_slider(
                _exp, "log sigma_k", -6.0, 0.0, -2.0, 0.1,
                key=f"log_sigma_k_{_ksuf}")
            _comp_configs.append({
                "type": "spot",
                "label": _new_label,
                "envelope_name": _env_name,
                "envelope_params": _env_params,
                "sigma_k": 10.0 ** _log_sk,
                "log_sigma_k": _log_sk,
            })
        else:
            _sho_params = _render_sho_params(_exp, _ksuf)
            _comp_configs.append({
                "type": "sho",
                "label": _new_label,
                "sho_params": _sho_params,
            })

        if len(st.session_state["components"]) > 1:
            if st.button("Remove", key=f"remove_{_ksuf}"):
                st.session_state["components"] = [
                    c for c in st.session_state["components"]
                    if c["id"] != _cid]
                st.rerun()

if st.sidebar.button("+ Add Component", use_container_width=True):
    _next_id = st.session_state["next_comp_id"]
    st.session_state["components"].append(
        {"id": _next_id, "label": f"comp{_next_id}", "type": "spot"})
    st.session_state["next_comp_id"] = _next_id + 1
    st.rerun()

st.sidebar.markdown("---")

col_gen, col_fit = st.sidebar.columns(2)
if col_gen.button("Generate", type="primary", use_container_width=True):
    st.session_state["model_params"] = dict(
        visibility_name=visibility_name,
        visibility_params=visibility_params,
        latitude_name=latitude_name,
        latitude_params=latitude_params,
        components=list(_comp_configs),
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
            _mdl = _build_spotgp_model(mp_fit)
            if _mdl is None:
                st.sidebar.error(
                    "Add at least one Spot component for GP prediction.")
                st.stop()
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
    _opt_is_composite = _is_composite(_opt_mp)

    nopt = st.sidebar.number_input("Number of restarts", min_value=1,
                                   max_value=50, value=5)

    _opt_spot = _spot_components(_opt_mp)
    _opt_sho = _sho_components(_opt_mp)
    _has_sho = len(_opt_sho) > 0

    _candidates = {}
    _candidates.update(_opt_mp["visibility_params"])
    _candidates.update(_opt_mp.get("latitude_params", {}))
    if _opt_is_composite:
        for _oc in _opt_spot:
            _ol = _oc["label"]
            for k, v in _oc["envelope_params"].items():
                _candidates[f"{k}_{_ol}"] = v
            _candidates[f"log_sigma_k_{_ol}"] = _oc["log_sigma_k"]
    elif _opt_spot:
        _oc0 = _opt_spot[0]
        _candidates.update(_oc0["envelope_params"])
        _candidates["log_sigma_k"] = _oc0["log_sigma_k"]

    if _has_sho:
        st.sidebar.caption(
            "SHO term parameters are not included in the MAP fit. "
            "Use their sliders to adjust manually.")

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
        model = _build_spotgp_model(_opt_mp)
        if model is None:
            st.sidebar.error(
                "Add at least one Spot component for MAP fitting.")
            st.stop()

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

            _map_model = _build_spotgp_model(map_mp)

            if _map_model is not None:
                _map_kernel_obj = _build_kernel_object(_map_model)
                if isinstance(_map_model, spotgp.CompositeSpotModel):
                    _map_max_lag = max(
                        c.envelope.kernel_support()
                        for c in _map_model.components) * 1.5
                else:
                    _map_max_lag = _map_model.envelope.kernel_support() * 1.5
                _map_lag = np.linspace(0, _map_max_lag, 500)
                _map_K = np.asarray(_map_kernel_obj.kernel(_map_lag))
                _map_harmonics = _compute_all_harmonics(
                    _map_kernel_obj, _map_lag)
            else:
                _map_max_lag = 50.0
                _map_lag = np.linspace(0, _map_max_lag, 500)
                _map_K = np.zeros_like(_map_lag)
                _map_harmonics = {0: np.zeros_like(_map_lag)}

            _map_sho_terms = _build_sho_terms(map_mp)
            if _map_sho_terms:
                _map_sho_K = _eval_sho_kernel(_map_sho_terms, _map_lag)
                _map_K = _map_K + _map_sho_K
                _map_harmonics[0] = _map_harmonics[0] + _map_sho_K

            st.session_state["map_kernel_data"] = (
                _map_lag, _map_K, _map_harmonics)

            _map_omega_max = (2 * np.pi
                              / map_mp["visibility_params"]["peq"] * 5)
            _map_omega = np.linspace(0.01, _map_omega_max, 1000)

            if _map_model is not None:
                try:
                    _map_psd_freq, _map_psd_power = \
                        _map_kernel_obj.compute_psd(_map_omega)
                except Exception:
                    with jax.default_device(jax.devices("cpu")[0]):
                        _map_psd_freq, _map_psd_power = \
                            _map_kernel_obj.compute_psd(_map_omega)
                _map_psd_freq = np.asarray(_map_psd_freq)
                _map_psd_power = np.asarray(_map_psd_power)
            else:
                _map_psd_freq = _map_omega / (2 * np.pi)
                _map_psd_power = np.zeros_like(_map_omega)

            if _map_sho_terms:
                _map_psd_power = (_map_psd_power
                                  + _eval_sho_psd(_map_sho_terms, _map_omega))

            st.session_state["map_psd_data"] = (
                _map_psd_freq, _map_psd_power)

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
                mg.attrs["visibility"] = _opt_mp["visibility_name"]
                for k, v in _opt_mp["visibility_params"].items():
                    mg.attrs[f"visibility_{k}"] = v
                _save_spot_idx = 0
                _save_sho_idx = 0
                for _oc in _opt_mp["components"]:
                    if _oc.get("type") == "sho":
                        _cg = mg.create_group(f"sho_{_save_sho_idx}")
                        _cg.attrs["label"] = _oc["label"]
                        for k, v in _oc["sho_params"].items():
                            _cg.attrs[k] = v
                        _save_sho_idx += 1
                    else:
                        _cg = mg.create_group(f"component_{_save_spot_idx}")
                        _cg.attrs["label"] = _oc["label"]
                        _cg.attrs["envelope"] = _oc["envelope_name"]
                        _cg.attrs["sigma_k"] = _oc["sigma_k"]
                        for k, v in _oc["envelope_params"].items():
                            _cg.attrs[f"envelope_{k}"] = v
                        _save_spot_idx += 1

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


def _build_config_dict():
    """Build the config dict from current sidebar state."""
    _data_cfg = {}
    if data_source == "Local file":
        _data_cfg["path"] = file_path
    elif sectors:
        _data_cfg["sectors"] = list(sectors)
    if has_raw:
        _data_cfg["normalize"] = bool(normalize)
        _data_cfg["zero_mean"] = bool(zero_mean)
    else:
        _data_cfg["normalize"] = True
        _data_cfg["zero_mean"] = True

    _exp_spot = [c for c in _comp_configs if c.get("type", "spot") == "spot"]
    _exp_sho = [c for c in _comp_configs if c.get("type") == "sho"]
    _model_cfg = {
        "visibility": visibility_name,
        "visibility_params": {k: float(v)
                              for k, v in visibility_params.items()},
    }
    if len(_exp_spot) == 1 and not _exp_sho:
        _ec = _exp_spot[0]
        _model_cfg["envelope"] = _ec["envelope_name"]
        _model_cfg["envelope_params"] = {
            k: float(v) for k, v in _ec["envelope_params"].items()}
        _model_cfg["sigma_k"] = float(_ec["sigma_k"])
    elif _exp_spot:
        _model_cfg["components"] = []
        for _ec in _exp_spot:
            _model_cfg["components"].append({
                "label": _ec["label"],
                "envelope": _ec["envelope_name"],
                "envelope_params": {
                    k: float(v) for k, v in _ec["envelope_params"].items()},
                "sigma_k": float(_ec["sigma_k"]),
            })
    if _exp_sho:
        _model_cfg["sho_terms"] = []
        for _ec in _exp_sho:
            _model_cfg["sho_terms"].append({
                "label": _ec["label"],
                **{k: float(v) for k, v in _ec["sho_params"].items()},
            })
    if latitude_name != "LatitudeDistributionFunction":
        _model_cfg["latitude"] = latitude_name
        _model_cfg["latitude_params"] = {
            k: float(v) for k, v in latitude_params.items()}

    _default_bounds_base = dict(visibility_params)
    if len(_exp_spot) == 1:
        _default_bounds_base["log_sigma_k"] = _exp_spot[0]["log_sigma_k"]
    elif _exp_spot:
        for _ec in _exp_spot:
            _default_bounds_base[
                f"log_sigma_k_{_ec['label']}"] = _ec["log_sigma_k"]
    _export_bounds = st.session_state.get("fit_bounds") or {
        k: _default_fit_bounds(k, v)
        for k, v in _default_bounds_base.items()
        if k != "inc"
    }

    return {
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


def _dump_config_yaml(cfg):
    """Dump config dict to YAML with short lists rendered inline."""
    class _FlowListDumper(yaml.SafeDumper):
        pass

    def _represent_list(dumper, data):
        if all(isinstance(v, (int, float)) for v in data) and len(data) <= 4:
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True)
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=False)

    _FlowListDumper.add_representer(list, _represent_list)
    return yaml.dump(cfg, Dumper=_FlowListDumper, sort_keys=False)


if st.sidebar.button("Generate Config", use_container_width=True):
    st.session_state["config_yaml"] = _dump_config_yaml(
        _build_config_dict())


# ── Kernel computation (cached) ─────────────────────────────────────


model_ready = "model_params" in st.session_state
kernel_error = None
lag = K = max_lag = auto_max_lag = harmonics = component_kernels = None

if model_ready:
    mp = st.session_state["model_params"]
    _spot_tuple = tuple(
        (c["label"], c["envelope_name"],
         tuple(sorted(c["envelope_params"].items())), c["sigma_k"])
        for c in _spot_components(mp)
    )
    _sho_tuple = tuple(
        (c["label"], tuple(sorted(c["sho_params"].items())))
        for c in _sho_components(mp)
    )
    visibility_params_tuple = tuple(sorted(mp["visibility_params"].items()))
    latitude_params_tuple = tuple(sorted(
        mp.get("latitude_params", {}).items()))
    _user_max_lag = st.session_state.get("kernel_max_lag")
    _user_n_lag = st.session_state.get("kernel_n_lag", 500)
    try:
        lag, K, max_lag, auto_max_lag, harmonics, component_kernels = \
            cached_kernel(
                _spot_tuple, _sho_tuple,
                mp["visibility_name"], visibility_params_tuple,
                mp.get("latitude_name", "LatitudeDistributionFunction"),
                latitude_params_tuple,
                max_lag_override=_user_max_lag,
                n_lag=_user_n_lag)
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

    _sector_nums = st.session_state.get("sector_numbers")
    if _sector_nums and has_raw:
        _raw_segs = st.session_state["raw_segments"]
        _sector_info = sorted(
            zip(_raw_segs, _sector_nums),
            key=lambda pair: pair[0][0].min())
        _shade_colors = ["rgba(200,200,200,0.15)", "rgba(160,160,160,0.15)"]
        for _si, (_seg, _sn) in enumerate(_sector_info):
            _t0, _t1 = float(_seg[0].min()), float(_seg[0].max())
            fig_data.add_vrect(
                x0=_t0, x1=_t1,
                fillcolor=_shade_colors[_si % 2],
                line_width=0, layer="below")
            fig_data.add_annotation(
                x=(_t0 + _t1) / 2, y=1.0, yref="paper",
                text=f"S{_sn}", showarrow=False,
                font=dict(size=10, color="gray"),
                yanchor="bottom")

    fig_data.update_layout(
        title=f"{len(t)} data points",
        xaxis_title="Time (days)", yaxis_title="Flux",
        height=300, margin=dict(l=50, r=20, t=40, b=40))
    show_fig(fig_data, "Data")
else:
    st.info("Load data using the sidebar to see the light curve.")

st.subheader("Kernel / ACF")
if not model_ready:
    st.info("Configure model parameters and click **Generate**.")
elif kernel_error is not None:
    st.error(f"Model error: {kernel_error}")
else:
    _n_total_comps = len(mp.get("components", []))

    _lag_col1, _lag_col2, _lag_col3 = st.columns([2, 2, 1])
    with _lag_col1:
        _kernel_max_lag = st.number_input(
            "Max lag (days)", min_value=1.0,
            value=st.session_state.get("kernel_max_lag", auto_max_lag),
            step=1.0, format="%.1f", key="kernel_max_lag")
    with _lag_col2:
        _kernel_n_lag = st.number_input(
            "Number of lag points", min_value=50, max_value=5000,
            value=st.session_state.get("kernel_n_lag", 500),
            step=50, key="kernel_n_lag")
    with _lag_col3:
        _show_comp_curves = st.checkbox(
            "Show components", value=False,
            key="show_comp_curves",
            disabled=_n_total_comps < 2)

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
            t_acf, y_acf, yerr_acf = st.session_state["data_arrays"]
            lag_centers, acf_data = cached_acf(
                t_acf, y_acf, yerr_acf, max_lag=float(lag[-1]))
            _normalize_kernel = True
            fig_kernel.add_trace(go.Scatter(
                x=lag_centers, y=acf_data,
                mode="lines", name="Data ACF",
                line=dict(width=1, color="black"), opacity=0.7,
                hovertemplate="Lag: %{x:.2f}<br>ACF: %{y:.4f}"
                              "<extra></extra>"))
        except Exception as e:
            st.warning(f"ACF computation failed: {e}")

    K_sum = (sum(harmonics[n] for n in _selected_harmonics)
             if _selected_harmonics else np.zeros_like(lag))
    if _normalize_kernel:
        K_sum = K_sum / _K0_manual
    fig_kernel.add_trace(go.Scatter(
        x=lag, y=K_sum, mode="lines",
        name="Model kernel",
        line=dict(width=2, color="#d62728"),
        hovertemplate="Lag: %{x:.2f}<br>K: %{y:.4f}<extra></extra>"))

    if _show_comp_curves and component_kernels:
        _comp_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
                        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for _ci, (_clabel, _cK) in enumerate(component_kernels.items()):
            _cK_plot = _cK / _K0_manual if _normalize_kernel else _cK
            fig_kernel.add_trace(go.Scatter(
                x=lag, y=_cK_plot, mode="lines",
                name=_clabel,
                line=dict(width=1.5, dash="dash",
                          color=_comp_colors[_ci % len(_comp_colors)]),
                hovertemplate=("Lag: %{x:.2f}<br>K: %{y:.4f}"
                               "<extra></extra>")))

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
    show_fig(fig_kernel, "Kernel / ACF")


# ── Model PSD ────────────────────────────────────────────────────────


if model_ready and kernel_error is None:
    st.subheader("Power spectral density")
    spotgp_mod = _import_spotgp()

    psd_model = _build_spotgp_model(mp)
    omega_max = 2 * np.pi / mp["visibility_params"]["peq"] * 5
    omega = np.linspace(0.01, omega_max, 1000)

    if psd_model is not None:
        psd_kernel = _build_kernel_object(psd_model)
        try:
            psd_freq, psd_power = psd_kernel.compute_psd(omega)
        except Exception:
            with jax.default_device(jax.devices("cpu")[0]):
                psd_freq, psd_power = psd_kernel.compute_psd(omega)
        psd_freq = np.asarray(psd_freq)
        psd_power = np.asarray(psd_power)
    else:
        psd_freq = omega / (2 * np.pi)
        psd_power = np.zeros_like(omega)

    _sho_terms = _build_sho_terms(mp)
    if _sho_terms:
        psd_power = psd_power + _eval_sho_psd(_sho_terms, omega)

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

    if _show_comp_curves and _n_total_comps > 1:
        _comp_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
                        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        _psd_spot_comps = _spot_components(mp)
        _psd_sho_comps = _sho_components(mp)
        _psd_ci = 0
        if psd_model is not None and len(_psd_spot_comps) > 1:
            for _sc, _ck in zip(_psd_spot_comps,
                                psd_kernel._component_kernels):
                try:
                    _cf, _cp = _ck.compute_psd(omega)
                except Exception:
                    with jax.default_device(jax.devices("cpu")[0]):
                        _cf, _cp = _ck.compute_psd(omega)
                fig_psd.add_trace(go.Scatter(
                    x=np.asarray(_cf), y=np.asarray(_cp), mode="lines",
                    name=_sc["label"],
                    line=dict(width=1.5, dash="dash",
                              color=_comp_colors[_psd_ci % len(
                                  _comp_colors)]),
                    hovertemplate=("Freq: %{x:.4f} c/d<br>"
                                   "Power: %{y:.4e}<extra></extra>")))
                _psd_ci += 1
        elif psd_model is not None:
            fig_psd.add_trace(go.Scatter(
                x=psd_freq, y=psd_power - _eval_sho_psd(
                    _sho_terms, omega) if _sho_terms else psd_power,
                mode="lines",
                name=_psd_spot_comps[0]["label"] if _psd_spot_comps else "Spot",
                line=dict(width=1.5, dash="dash",
                          color=_comp_colors[0]),
                hovertemplate=("Freq: %{x:.4f} c/d<br>"
                               "Power: %{y:.4e}<extra></extra>")))
            _psd_ci += 1
        for _sc in _psd_sho_comps:
            _sp = _sc["sho_params"]
            _term = spotgp_mod.SHOTerm(**_sp)
            _cp = np.asarray(_term.get_psd(omega))
            fig_psd.add_trace(go.Scatter(
                x=psd_freq, y=_cp, mode="lines",
                name=_sc["label"],
                line=dict(width=1.5, dash="dash",
                          color=_comp_colors[_psd_ci % len(_comp_colors)]),
                hovertemplate=("Freq: %{x:.4f} c/d<br>"
                               "Power: %{y:.4e}<extra></extra>")))
            _psd_ci += 1

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
    show_fig(fig_psd, "Power spectral density")


# ── Model equations ──────────────────────────────────────────────────


if model_ready and kernel_error is None:
    import sympy
    spotgp = _import_spotgp()

    vis_cls = getattr(spotgp, mp["visibility_name"])
    visibility_obj = vis_cls(**mp["visibility_params"])

    with st.expander("Model equations", expanded=False):
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

        _eq_spot_comps = _spot_components(mp)
        _eq_sho_comps = _sho_components(mp)
        _eq_multi = len(mp["components"]) > 1

        for _eq_comp in _eq_spot_comps:
            _eq_label = _eq_comp["label"]
            _eq_suffix = f" [{_eq_label}]" if _eq_multi else ""
            _eq_env_cls = getattr(spotgp, _eq_comp["envelope_name"])
            _eq_env_obj = _eq_env_cls(**_eq_comp["envelope_params"])
            _eq_env_exprs = _eq_env_obj.get_sympy()
            st.markdown(
                f"**Envelope{_eq_suffix}** — {_eq_comp['envelope_name']}")
            for name, expr in _eq_env_exprs.items():
                if expr is None:
                    continue
                lhs = label_map.get(name, name)
                st.latex(f"{lhs} = {sympy.latex(expr)}")

        for _eq_sho in _eq_sho_comps:
            _sp = _eq_sho["sho_params"]
            _eq_suffix = f" [{_eq_sho['label']}]" if _eq_multi else ""
            st.markdown(f"**SHO Term{_eq_suffix}**")
            _sigma_s = sympy.Symbol(r"\sigma")
            _rho_s = sympy.Symbol(r"\rho")
            _S0_s = sympy.Symbol("S_0")
            _w0_s = sympy.Symbol(r"\omega_0")
            _Q_s = sympy.Symbol("Q")
            st.latex(
                r"k(\tau) = S_0\,\omega_0\,Q\,"
                r"e^{-\omega_0\tau/(2Q)}"
                r"\left[\cos\eta\omega_0\tau "
                r"+ \frac{\sin\eta\omega_0\tau}{2Q\eta}\right]")
            _param_strs = [
                f"\\sigma={_sp['sigma']:.4f}",
                f"\\rho={_sp['rho']:.2f}"]
            if "Q" in _sp:
                _param_strs.append(f"Q={_sp['Q']:.2f}")
            else:
                _param_strs.append(f"\\tau={_sp['tau']:.2f}")
            st.latex(r",\quad ".join(_param_strs))

        vis_exprs = visibility_obj.get_sympy()
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


if model_ready and kernel_error is None and _spot_components(mp):
    st.subheader("Single-spot flux")
    spotgp = _import_spotgp()

    _spot_comps_list = _spot_components(mp)
    _spot_comp_labels = [c["label"] for c in _spot_comps_list]
    if len(_spot_comp_labels) > 1:
        _spot_comp_idx = st.selectbox(
            "Component", range(len(_spot_comp_labels)),
            format_func=lambda i: _spot_comp_labels[i],
            key="spot_comp_select")
    else:
        _spot_comp_idx = 0
    _spot_comp = _spot_comps_list[_spot_comp_idx]

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

    _spot_env_cls = getattr(spotgp, _spot_comp["envelope_name"])
    _spot_env_obj = _spot_env_cls(**_spot_comp["envelope_params"])
    vis_cls = getattr(spotgp, mp["visibility_name"])
    visibility_obj = vis_cls(**mp["visibility_params"])
    spot_lat_dist = _build_latitude_dist(
        mp.get("latitude_name", "LatitudeDistributionFunction"),
        mp.get("latitude_params", {}))
    spot_model = spotgp.SpotEvolutionModel(
        envelope=_spot_env_obj, visibility=visibility_obj,
        sigma_k=_spot_comp["sigma_k"], latitude_distribution=spot_lat_dist)

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
    env_vals = _spot_env_obj.Gamma(env_t)
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
        _map_spot_comps = _spot_components(_mmp)
        _map_spot_comp = _map_spot_comps[min(
            _spot_comp_idx, len(_map_spot_comps) - 1)]
        _me_cls = getattr(spotgp, _map_spot_comp["envelope_name"])
        _me = _me_cls(**_map_spot_comp["envelope_params"])
        _mv_cls = getattr(spotgp, _mmp["visibility_name"])
        _mv = _mv_cls(**_mmp["visibility_params"])
        _m_lat = _build_latitude_dist(
            _mmp.get("latitude_name", "LatitudeDistributionFunction"),
            _mmp.get("latitude_params", {}))
        _m_spot_model = spotgp.SpotEvolutionModel(
            envelope=_me, visibility=_mv,
            sigma_k=_map_spot_comp["sigma_k"],
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
    show_fig(fig_spot, "Single-spot flux")


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


# ── Config editor (main panel) ─────────────────────────────────────


st.markdown("---")
st.subheader("Config")

if "config_yaml" not in st.session_state:
    st.info('Click **Generate Config** in the sidebar to build a '
            'config file from the current settings.')
else:
    from streamlit_ace import st_ace
    _edited_yaml = st_ace(
        value=st.session_state["config_yaml"],
        language="yaml", theme="tomorrow", height=400,
        key="config_editor")

    _cfg_c1, _cfg_c2 = st.columns([3, 1])
    _default_export_path = f"configs/{_export_stem}.yaml"
    _export_path = _cfg_c1.text_input(
        "Config file path", value=_default_export_path)
    if _cfg_c2.button("Export Config", type="primary",
                      use_container_width=True):
        try:
            yaml.safe_load(_edited_yaml)
        except yaml.YAMLError as exc:
            st.error(f"Invalid YAML: {exc}")
        else:
            os.makedirs(os.path.dirname(_export_path) or ".", exist_ok=True)
            with open(_export_path, "w") as f:
                f.write(_edited_yaml)
            st.success(
                f"Wrote `{_export_path}` — run with "
                f"`python scripts/run_fit.py {_export_path}`")


# ── Plot export (main panel) ────────────────────────────────────────


st.markdown("---")
st.subheader("Export plots")

if not EXPORT_FIGS:
    st.info("Load data or generate a model to export plots.")
else:
    _all_names = [name for name, _ in EXPORT_FIGS]
    _e1, _e2 = st.columns([3, 1])
    _picked = _e1.multiselect("Figures to include", _all_names,
                              default=_all_names)
    _standalone = _e2.checkbox(
        "Self-contained", value=True,
        help="Embed plotly.js in the file (~4 MB) so it works offline. "
             "Uncheck to load plotly.js from a CDN instead (small file, "
             "needs internet).")

    _html_name = f"{_export_stem or 'spotgp'}_plots.html"
    _selected = [(name, fig) for name, fig in EXPORT_FIGS
                 if name in _picked]

    if not _selected:
        st.caption("Select at least one figure.")
    elif st.button("Build HTML", use_container_width=False):
        _params_text = None
        if model_ready:
            _mp = st.session_state["model_params"]
            _summary = {
                "envelope": _mp["envelope_name"],
                "envelope_params": {k: float(v) for k, v
                                    in _mp["envelope_params"].items()},
                "visibility": _mp["visibility_name"],
                "visibility_params": {k: float(v) for k, v
                                      in _mp["visibility_params"].items()},
                "sigma_k": float(_mp["sigma_k"]),
            }
            if _mp.get("latitude_name",
                       "LatitudeDistributionFunction") != \
                    "LatitudeDistributionFunction":
                _summary["latitude"] = _mp["latitude_name"]
                _summary["latitude_params"] = {
                    k: float(v)
                    for k, v in _mp.get("latitude_params", {}).items()}
            _params_text = yaml.safe_dump(_summary, sort_keys=False).strip()

        st.session_state["plots_html"] = build_plots_html(
            _selected,
            title=star_label or _export_star or "spotgp Explorer",
            params_text=_params_text,
            standalone=_standalone)
        st.session_state["plots_html_name"] = _html_name
        st.session_state["plots_html_figs"] = list(_picked)

    if "plots_html" in st.session_state:
        _doc = st.session_state["plots_html"]
        st.download_button(
            "Download HTML", data=_doc,
            file_name=st.session_state["plots_html_name"],
            mime="text/html", type="primary")
        st.caption(
            f"{', '.join(st.session_state['plots_html_figs'])} — "
            f"{len(_doc.encode('utf-8')) / 1e6:.1f} MB, interactive (zoom, "
            "pan, legend toggles) in any browser. Rebuild after changing "
            "parameters.")
