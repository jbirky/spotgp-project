"""Shared light-curve loading helpers for run_fit.py and app.py."""

import os
import re

import numpy as np


def cache_stem(star_name):
    """Filesystem-safe stem for a target's cached light curve."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(star_name).strip())


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


PIPELINES = {
    "Auto-detect": None,
    "TESS / SPOC": ("TESS", "SPOC"),
    "TESS / QLP": ("TESS", "QLP"),
    "TESS / TESS-SPOC": ("TESS", "TESS-SPOC"),
    "Kepler": ("Kepler", "Kepler"),
    "K2": ("K2", "K2"),
}


def _parse_sector(mission_str):
    """Extract the integer sector/quarter/campaign number from a mission string."""
    match = re.search(r"(\d+)", str(mission_str))
    return int(match.group(1)) if match else -1


def download_lightcurve(star_name, sectors=None, pipeline=None):
    """Download light curve segments from MAST via lightkurve.

    At most one light curve is downloaded per sector.  The user's
    preferred *pipeline* is used when available; otherwise the highest
    cadence (shortest exposure time) result is chosen.

    Parameters
    ----------
    star_name : str
        Star identifier (TIC, KIC, or EPIC ID).
    sectors : sequence of int, optional
        TESS sectors / Kepler quarters / K2 campaigns to keep. None keeps
        all available.
    pipeline : str, optional
        Key from :data:`PIPELINES`.  The corresponding author is preferred
        for every sector where it is available.  When ``None`` or
        ``"Auto-detect"``, the highest-cadence result per sector is used.

    Returns
    -------
    segments : list of (time, flux, flux_err) arrays or None
        One entry per sector, non-finite points removed.
    error : str or None
        Error message when nothing was found, else None.
    """
    import lightkurve as lk

    if pipeline and pipeline in PIPELINES and PIPELINES[pipeline] is not None:
        mission, preferred_author = PIPELINES[pipeline]
    else:
        mission, _ = detect_mission(star_name)
        preferred_author = None

    search = lk.search_lightcurve(star_name, mission=mission)
    if len(search) == 0:
        return None, f"No {mission} data found for {star_name}"

    seq = np.array([_parse_sector(m) for m in search.table["mission"]])
    if sectors:
        mask = np.isin(seq, list(sectors))
        search = search[mask]
        seq = seq[mask]
    if len(search) == 0:
        return None, f"No {mission} data found for {star_name}"

    authors = np.array(search.table["author"])
    exptimes = np.array(search.table["exptime"], dtype=float)

    selected = []
    for sector in sorted(np.unique(seq)):
        idxs = np.where(seq == sector)[0]
        if preferred_author is not None:
            pref = idxs[authors[idxs] == preferred_author]
            if len(pref):
                idxs = pref
        selected.append(idxs[np.argmin(exptimes[idxs])])

    selected = sorted(selected)
    sector_numbers = [int(seq[i]) for i in selected]
    search = search[selected]
    lcc = search.download_all()
    segments = []
    kept_sectors = []
    for lc, sn in zip(lcc, sector_numbers):
        t = np.asarray(lc.time.value, dtype=float)
        y = np.asarray(lc.flux.value, dtype=float)
        yerr = np.asarray(lc.flux_err.value, dtype=float)
        mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr)
        if mask.any():
            segments.append((t[mask], y[mask], yerr[mask]))
            kept_sectors.append(sn)
    return segments, kept_sectors, None


def load_local_file(path):
    """Load a local light curve file into a list of segments.

    CSV files are read positionally (columns: time, flux, flux_err) as a
    single segment. ``.npz`` files must contain ``time``, ``flux``,
    ``flux_err``; a ``seg_lengths`` array (written by :func:`save_segments`)
    restores the original per-sector segment boundaries.
    """
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        return [(df.iloc[:, 0].values, df.iloc[:, 1].values,
                 df.iloc[:, 2].values)]

    arr = np.load(path)
    t, y, yerr = arr["time"], arr["flux"], arr["flux_err"]
    if "seg_lengths" in getattr(arr, "files", []):
        segments = []
        i = 0
        for n in arr["seg_lengths"]:
            n = int(n)
            segments.append((t[i:i + n], y[i:i + n], yerr[i:i + n]))
            i += n
        return segments
    return [(t, y, yerr)]


def save_segments(path, segments, sectors=None):
    """Save light-curve segments to an ``.npz`` cache file.

    Stores the raw (unprocessed) segments concatenated, plus a
    ``seg_lengths`` array so :func:`load_local_file` can restore the
    per-sector boundaries that :func:`process_data` normalizes over.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if segments:
        t = np.concatenate([s[0] for s in segments])
        y = np.concatenate([s[1] for s in segments])
        yerr = np.concatenate([s[2] for s in segments])
    else:
        t = y = yerr = np.array([], dtype=float)
    payload = dict(time=t, flux=y, flux_err=yerr,
                   seg_lengths=np.array([len(s[0]) for s in segments],
                                        dtype=int))
    if sectors is not None:
        payload["sectors"] = np.asarray(list(sectors), dtype=int)
    np.savez(path, **payload)


def process_data(segments, dt=None, downsample=1, normalize=True,
                 zero_mean=False):
    """Normalize per segment, optionally zero-mean and bin, then stitch.

    Normalization divides each segment by its median flux, so multi-sector
    data with different instrumental offsets stitch cleanly.

    When *downsample* > 1, each segment is decimated by keeping every
    Nth point (no spotgp dependency).  When *dt* is given and larger
    than the native cadence, each segment is instead binned into
    non-overlapping windows of width *dt* using spotgp's
    inverse-variance weighted binning (``TimeSeriesData.downsample``).

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
        if dt is not None and dt > 0 and len(t) > 1:
            native_dt = np.median(np.diff(t))
            if dt > native_dt * 1.5:
                from spotgp import TimeSeriesData
                ts = TimeSeriesData(t, y, yerr, normalize=False)
                ts.downsample(dt)
                t, y, yerr = ts.x, ts.y, ts.yerr
        elif downsample > 1:
            t = t[::downsample]
            y = y[::downsample]
            yerr = yerr[::downsample]
        processed.append((t, y, yerr))
    t = np.concatenate([s[0] for s in processed])
    y = np.concatenate([s[1] for s in processed])
    yerr = np.concatenate([s[2] for s in processed])
    return t, y, yerr
