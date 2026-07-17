"""Shared light-curve loading helpers for run_fit.py and app.py."""

import re

import numpy as np


def detect_mission(star_name):
    """Map a star identifier to a (mission, author) pair for lightkurve.

    KIC → Kepler, EPIC → K2, anything else (TIC, TOI, plain names) → TESS/SPOC.
    """
    name = star_name.strip().upper()
    if name.startswith("KIC"):
        return "Kepler", "Kepler"
    if name.startswith("EPIC"):
        return "K2", "K2"
    return "TESS", "SPOC"


def download_lightcurve(star_name, sectors=None):
    """Download light curve segments from MAST via lightkurve.

    Parameters
    ----------
    star_name : str
        Star identifier (TIC, KIC, or EPIC ID). The mission and pipeline
        author are inferred with :func:`detect_mission`.
    sectors : sequence of int, optional
        TESS sectors / Kepler quarters / K2 campaigns to keep. None keeps
        all available.

    Returns
    -------
    segments : list of (time, flux, flux_err) arrays or None
        One entry per downloaded light curve, non-finite points removed.
    error : str or None
        Error message when nothing was found, else None.
    """
    import lightkurve as lk

    mission, author = detect_mission(star_name)
    search = lk.search_lightcurve(star_name, mission=mission, author=author)
    if sectors:
        seq = []
        for m in search.table["mission"]:
            match = re.search(r"(\d+)", str(m))
            seq.append(int(match.group(1)) if match else -1)
        search = search[np.isin(seq, list(sectors))]
    if len(search) == 0:
        return None, f"No {mission}/{author} data found for {star_name}"

    lcc = search.download_all()
    segments = []
    for lc in lcc:
        t = np.asarray(lc.time.value, dtype=float)
        y = np.asarray(lc.flux.value, dtype=float)
        yerr = np.asarray(lc.flux_err.value, dtype=float)
        mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr)
        if mask.any():
            segments.append((t[mask], y[mask], yerr[mask]))
    return segments, None


def load_local_file(path):
    """Load a local light curve file as a single segment.

    CSV files are read positionally (columns: time, flux, flux_err);
    ``.npz`` files must contain ``time``, ``flux``, ``flux_err`` keys.
    """
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        t = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        yerr = df.iloc[:, 2].values
    else:
        arr = np.load(path)
        t, y, yerr = arr["time"], arr["flux"], arr["flux_err"]
    return [(t, y, yerr)]


def process_data(segments, downsample=1, normalize=True, zero_mean=False):
    """Normalize per segment, optionally zero-mean and downsample, then stitch.

    Normalization divides each segment by its median flux, so multi-sector
    data with different instrumental offsets stitch cleanly.

    Returns concatenated (time, flux, flux_err) arrays.
    """
    processed = []
    for t, y, yerr in segments:
        if normalize:
            med = np.median(y)
            y = y / med
            yerr = yerr / med
        if zero_mean:
            y = y - np.mean(y)
        if downsample > 1:
            t = t[::downsample]
            y = y[::downsample]
            yerr = yerr[::downsample]
        processed.append((t, y, yerr))
    t = np.concatenate([s[0] for s in processed])
    y = np.concatenate([s[1] for s in processed])
    yerr = np.concatenate([s[2] for s in processed])
    return t, y, yerr
