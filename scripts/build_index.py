"""Aggregate spotgp pipeline runs into a single results index.

Scans a results root for ``<run>/config.yaml`` + ``<run>/metrics.json``
pairs (as produced by ``run_fit.py`` / the DVC ``fit`` stage) and builds
one row per run. Used both by the DVC ``index`` stage (CLI below) and by
the Streamlit app's "Load from results/" action, so the interactive table
and the versioned CSV stay in sync.

Kept dependency-light (no streamlit / spotgp import) so it can run as a
plain pipeline step.
"""

import csv
import glob
import json
import math
import os

import yaml

# Column order shared by the CSV index and the app's results table.
COLUMNS = ["object_id", "visibility", "latitude", "kernel", "map_params",
           "neg_log_post", "run_dir", "git_rev", "wandb_url"]

_VIS_FRIENDLY = {
    "VisibilityFunction": "Standard",
    "EdgeOnVisibilityFunction": "Edge-on",
    "FullGeometryVisibilityFunction": "Full geometry",
}
_LAT_FRIENDLY = {
    "LatitudeDistributionFunction": "Uniform",
    "UniformDoubleHemisphereBand": "Uniform band",
    "ButterflyLatitude": "Butterfly",
}


def _fmt_visibility(model):
    name = model.get("visibility", "")
    friendly = _VIS_FRIENDLY.get(name, name)
    vp = model.get("visibility_params", {}) or {}
    parts = []
    if "peq" in vp:
        parts.append(f"P_eq={vp['peq']:.2f}")
    if "inc" in vp:  # stored in radians in the config
        parts.append(f"inc={math.degrees(vp['inc']):.0f}°")
    if "kappa" in vp:
        parts.append(f"κ={vp['kappa']:.2f}")
    return f"{friendly} ({', '.join(parts)})" if parts else friendly


def _fmt_latitude(model):
    name = model.get("latitude", "LatitudeDistributionFunction")
    friendly = _LAT_FRIENDLY.get(name, name)
    lp = model.get("latitude_params", {}) or {}
    if lp:
        parts = [f"{k}={v:.1f}" for k, v in lp.items()]
        return f"{friendly} ({', '.join(parts)})"
    return friendly


def _fmt_kernel(model):
    parts = []
    if isinstance(model.get("components"), list):
        for c in model["components"]:
            env = str(c.get("envelope", "")).replace("Envelope", "")
            parts.append(f"{c.get('label', '?')} ({env})")
    elif model.get("envelope"):
        parts.append(str(model["envelope"]).replace("Envelope", ""))
    for s in model.get("sho_terms", []) or []:
        parts.append(f"{s.get('label', 'sho')} (SHO)")
    return ", ".join(parts)


def _row_from_run(cfg, metrics, run_dir):
    """Build one results-table row from a run's config + metrics."""
    model = cfg.get("model", {}) or {}
    map_items = {k[4:]: v for k, v in metrics.items() if k.startswith("map_")}
    map_str = ", ".join(f"{k}={float(v):.3g}" for k, v in map_items.items())
    nlp = metrics.get("neg_log_posterior")
    return {
        "object_id": cfg.get("star_name") or os.path.basename(run_dir),
        "visibility": _fmt_visibility(model),
        "latitude": _fmt_latitude(model),
        "kernel": _fmt_kernel(model),
        "map_params": map_str,
        "neg_log_post": (f"{float(nlp):.4f}" if nlp is not None else ""),
        "run_dir": run_dir,
        "git_rev": str(metrics.get("git_rev", "")),
        "wandb_url": str(metrics.get("wandb_url", "")),
    }


def scan_results(root="results"):
    """Return one row dict per ``<root>/*/config.yaml`` run directory."""
    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "config.yaml"))):
        run_dir = os.path.dirname(cfg_path)
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue
        metrics = {}
        mpath = os.path.join(run_dir, "metrics.json")
        if os.path.isfile(mpath):
            try:
                with open(mpath) as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}
        rows.append(_row_from_run(cfg, metrics, run_dir))
    return rows


def write_index(root="results", out="results/results_index.csv"):
    """Scan ``root`` and write the results index CSV. Returns (out, n_rows)."""
    rows = scan_results(root)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})
    return out, len(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Aggregate spotgp run results into an index CSV")
    ap.add_argument("root", nargs="?", default="results",
                    help="Results root to scan (default: results)")
    ap.add_argument("--out", default="results/results_index.csv",
                    help="Output CSV path")
    args = ap.parse_args()
    out, n = write_index(args.root, args.out)
    print(f"Wrote {out} ({n} runs)")


if __name__ == "__main__":
    main()
