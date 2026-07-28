"""Batch-download light curves into a local, DVC-trackable cache.

Downloads every target registered under ``configs:`` in params.yaml (or a
single config) into ``data/lightcurves/<key>.npz`` using the shared
``download_lightcurve`` helper. Designed for pulling many targets at once:

  * parallel (thread pool — downloads are I/O bound),
  * retrying with exponential backoff (MAST is flaky),
  * resumable — already-cached targets are skipped unless ``--force``,
  * a manifest (``data/lightcurves/manifest.csv``) recording every target's
    status, point/sector counts, and any error.

The cache files preserve per-sector segment boundaries (see
``data_utils.save_segments``) so downstream normalization is unchanged.

Usage
-----
    # Bulk: all params.yaml targets, 6 workers, skip cached
    python scripts/fetch_lightcurves.py --params params.yaml --workers 6

    # Force re-download of a subset
    python scripts/fetch_lightcurves.py --only KIC_7286309,example --force

    # Single config -> single file (used by the DVC `download` stage)
    python scripts/fetch_lightcurves.py --config configs/example.yaml \
        --out data/lightcurves/example.npz
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_utils
from data_utils import save_segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spotgp.fetch")

MANIFEST_COLUMNS = ["key", "star_name", "status", "n_points", "n_sectors",
                    "sectors", "path", "error"]


def _read_config_target(cfg_path):
    """Return (star_name, sectors) for a config, or (None, None) if the
    file is missing / has no star_name."""
    if not os.path.isfile(cfg_path):
        return None, None
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None, None
    star = cfg.get("star_name")
    sectors = (cfg.get("data") or {}).get("sectors")
    return star, sectors


def targets_from_params(params_path="params.yaml"):
    """Build a target list from the ``configs:`` map in params.yaml.

    Each target is ``{key, star_name, sectors, config}``. star_name may be
    None (missing config or local-file config); such targets are skipped.
    Config paths are resolved relative to the params.yaml directory.
    """
    with open(params_path) as f:
        params = yaml.safe_load(f) or {}
    configs = params.get("configs") or {}
    params_dir = os.path.dirname(os.path.abspath(params_path))
    targets = []
    for key, cfg_path in configs.items():
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(params_dir, cfg_path)
        star, sectors = _read_config_target(cfg_path)
        targets.append({"key": key, "star_name": star, "sectors": sectors,
                        "config": cfg_path})
    return targets


def _download_with_retry(star_name, sectors, retries):
    """Download one target, retrying transient failures with backoff."""
    last_err = None
    for attempt in range(max(1, retries)):
        try:
            segments, kept_sectors, err = data_utils.download_lightcurve(
                star_name, sectors)
            if err:
                last_err = err
            else:
                return segments, kept_sectors, None
        except Exception as e:  # network/MAST errors
            last_err = str(e)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, ...
    return None, None, last_err or "unknown error"


def _fetch_one(target, out_dir, retries, force):
    """Fetch a single target dict into ``out_dir/<key>.npz``; return a
    manifest row."""
    key = target["key"]
    star = target.get("star_name")
    out_path = os.path.join(out_dir, f"{key}.npz")
    base = {"key": key, "star_name": star or "", "path": out_path,
            "n_points": 0, "n_sectors": 0, "sectors": "", "error": ""}

    if not star:
        return {**base, "status": "skipped", "error": "no star_name"}
    if os.path.exists(out_path) and not force:
        return {**base, "status": "cached"}

    segments, sectors, err = _download_with_retry(
        star, tuple(target["sectors"]) if target.get("sectors") else None,
        retries)
    if err or not segments:
        return {**base, "status": "failed",
                "error": err or "no data returned"}

    save_segments(out_path, segments, sectors)
    n_points = sum(len(s[0]) for s in segments)
    return {**base, "status": "ok", "n_points": n_points,
            "n_sectors": len(segments),
            "sectors": " ".join(str(s) for s in (sectors or []))}


def write_manifest(out_dir, rows):
    path = os.path.join(out_dir, "manifest.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLUMNS})
    return path


def fetch_targets(targets, out_dir="data/lightcurves", workers=6, retries=3,
                  force=False, progress=None, write_manifest_file=True):
    """Download a list of targets in parallel. Returns the manifest rows.

    ``progress`` (optional) is called as ``progress(done, total, row)`` in
    the calling thread after each target completes.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    total = len(targets)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_fetch_one, t, out_dir, retries, force): t
                for t in targets}
        for done, fut in enumerate(as_completed(futs), start=1):
            row = fut.result()
            rows.append(row)
            logger.info("[%d/%d] %s: %s%s", done, total, row["key"],
                        row["status"],
                        f" ({row['error']})" if row["error"] else "")
            if progress is not None:
                progress(done, total, row)
    # Keep a stable order (by key) in the manifest.
    rows.sort(key=lambda r: r["key"])
    if write_manifest_file:
        write_manifest(out_dir, rows)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Batch-download light curves into a local cache")
    ap.add_argument("--params", default="params.yaml",
                    help="params.yaml with a `configs:` map (bulk mode)")
    ap.add_argument("--config", default=None,
                    help="Single config file (single-target mode)")
    ap.add_argument("--out", default=None,
                    help="Output dir (bulk) or .npz file (with --config)")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of config keys to fetch")
    ap.add_argument("--project-dir", default=None,
                    help="Project directory — relative paths resolve "
                         "against this (default: cwd)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the cache file exists")
    args = ap.parse_args()

    if args.project_dir:
        os.chdir(args.project_dir)

    if args.config:
        # Single-target mode (used by the DVC `download` stage).
        star, sectors = _read_config_target(args.config)
        key = os.path.splitext(os.path.basename(args.config))[0]
        out = args.out or os.path.join("data/lightcurves", f"{key}.npz")
        out_dir = os.path.dirname(out) or "."
        # Name the file exactly as requested by using key = filename stem.
        target = {"key": os.path.splitext(os.path.basename(out))[0],
                  "star_name": star, "sectors": sectors, "config": args.config}
        rows = fetch_targets([target], out_dir=out_dir, workers=1,
                             retries=args.retries, force=args.force,
                             write_manifest_file=False)
        row = rows[0]
        if row["status"] == "failed":
            logger.error("Failed to fetch %s: %s", star, row["error"])
            sys.exit(1)
        logger.info("%s -> %s (%s)", star, out, row["status"])
        return

    targets = targets_from_params(args.params)
    if args.only:
        wanted = {k.strip() for k in args.only.split(",") if k.strip()}
        targets = [t for t in targets if t["key"] in wanted]

    out_dir = args.out or "data/lightcurves"
    rows = fetch_targets(targets, out_dir=out_dir, workers=args.workers,
                         retries=args.retries, force=args.force)

    n_ok = sum(1 for r in rows if r["status"] in ("ok", "cached"))
    n_fail = [r for r in rows if r["status"] == "failed"]
    n_skip = [r for r in rows if r["status"] == "skipped"]
    logger.info("Done: %d/%d available in %s", n_ok, len(rows), out_dir)
    if n_skip:
        logger.warning("Skipped (no star_name): %s",
                       ", ".join(r["key"] for r in n_skip))
    if n_fail:
        logger.warning("Failed: %s",
                       ", ".join(f"{r['key']} ({r['error']})" for r in n_fail))
        sys.exit(1)


if __name__ == "__main__":
    main()
