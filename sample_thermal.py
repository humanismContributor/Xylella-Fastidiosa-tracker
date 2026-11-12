#!/usr/bin/env python3
"""
Landsat thermal anomaly at every sampled tree (calibration data, prototype).

Phase 1 of the thermal test. For every summer it takes the clear Landsat 8/9
surface-temperature scenes over a region, subtracts from each scene the median
temperature of the cells the scanner may judge (so the weather of the day drops
out and what is left is how much hotter a cell runs than the groves around it),
and keeps the seasonal median of that anomaly. A stressed canopy transpires
less and runs hotter, so the question for replay_thermal.py is whether this
anomaly tells declining Xylella trees from the scanner's other flags.

The sampled trees and their cells are read from the OSAVI tables written by
sample_osavi.py, so both tables line up row for row and no sample download is
needed. Nothing here touches PARAMS, state.npz or the scan outputs.

Usage (needs the conda packages from main.yml)
    python sample_thermal.py --region puglia-monopoli-polignano
    python sample_thermal.py --all          # every region with an OSAVI table

Outputs, in .cache/thermal/ (not committed: like the OSAVI tables, each row is a sampled tree)
    <region-id>.csv.gz   one row per sample, same order as the OSAVI table
    <region-id>.json     scenes used and skipped per season, rules, columns

What it can't see: Landsat's thermal band is 100 m (delivered resampled to
30 m), so a value mixes many crowns with soil and grass. It can mark a hot
patch, not a hot tree.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import sample_osavi as so
import scene_inventory as si
import xylella_scanner as xs

OUT_DIR = so.OUT_DIR.parent / "thermal"
COLLECTION = "landsat-c2-l2"
PLATFORMS = ("landsat-8", "landsat-9")      # Landsat 7's thermal band has scan-line gaps since 2003
FIRST_YEAR = 2018                            # one summer before the first judged season
# Surface temperature scaling from the collection's raster:bands metadata
# (https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2).
LST_SCALE, LST_OFFSET = 0.00341802, 149.0
# QA_PIXEL bits rejected: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow,
# 5 snow, 7 water (https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands).
QA_REJECT = 0b1011_1111
MIN_SCENE_SHARE = 0.5        # share of judged cells that must be clear for a scene to count
MIN_DATES = xs.MIN_DATES     # clear scenes a cell needs in a season, as for OSAVI

COLUMNS = {
    "id": "sample ID (as in the OSAVI table)", "row": "row of the tree's cell", "col": "column of the tree's cell",
    "dt_<year>": "seasonal median of the cell's temperature minus the scene median of judged cells (K); "
                 "empty: fewer than MIN_DATES clear scenes",
    "dt3_<year>": "median of the 3 x 3 window's seasonal values (K)",
    "n_<year>": "clear scenes at the cell that summer",
}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def lst_kelvin(dn, qa):
    """Surface temperature in K, NaN where the pixel is fill, cloud, shadow, snow or water."""
    ok = (dn > 0) & ((qa & QA_REJECT) == 0)
    return np.where(ok, dn.astype("float32") * LST_SCALE + LST_OFFSET, np.nan).astype("float32")


def scene_anomaly(lst, landmask, min_share: float = MIN_SCENE_SHARE):
    """Temperature minus the median of the clear judged cells, or None when too
    few judged cells are clear for that median to describe the region."""
    ref = lst[landmask]
    clear = np.isfinite(ref)
    if landmask.sum() == 0 or clear.mean() < min_share:
        return None
    return (lst - np.median(ref[clear])).astype("float32")


def season_anomaly(anoms: list, shape, min_dates: int = MIN_DATES):
    """Median anomaly per cell over the season's scenes (NaN where fewer than
    min_dates clear) and the number of clear scenes per cell."""
    if not anoms:
        return np.full(shape, np.nan, "float32"), np.zeros(shape, np.int16)
    stack = np.stack(anoms)
    n = np.isfinite(stack).sum(axis=0).astype(np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN cells
        med = np.nanmedian(stack, axis=0)
    return np.where(n >= min_dates, med, np.nan).astype("float32"), n


def read_cells(path: Path):
    """Sample ids, rows and columns of an OSAVI table, in table order."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return ([r["id"] for r in rows], np.array([int(r["row"]) for r in rows]),
            np.array([int(r["col"]) for r in rows]))


def select_scenes(items, bbox, cloud_max: float = si.CLOUD_MAX) -> tuple[list, list]:
    """Landsat 8/9 scenes clear enough at scene level and taken around midday,
    plus (id, reason) for every scene left out."""
    lon = (bbox[0] + bbox[2]) / 2
    hours = si.SENSORS["landsat"]["local_hours"]
    keep, skipped = [], []
    for it in items:
        p = it.properties if hasattr(it, "properties") else it["properties"]
        iid = it.id if hasattr(it, "id") else it["id"]
        cloud = p.get("eo:cloud_cover")
        h = si.local_solar_hour(si.parse_time(p["datetime"]), lon)
        if p.get("platform") not in PLATFORMS:
            skipped.append((iid, "platform"))
        elif cloud is not None and cloud > cloud_max:
            skipped.append((iid, f"scene cloud {cloud:.0f} %"))
        elif not hours[0] <= h <= hours[1]:
            skipped.append((iid, f"local time {h:.1f} h"))
        else:
            keep.append(it)
    return keep, skipped


def table(ids, rows, cols, own: dict, win: dict, count: dict):
    years = sorted(own)
    header = (["id", "row", "col"] + [f"dt_{y}" for y in years] + [f"dt3_{y}" for y in years]
              + [f"n_{y}" for y in years])
    fmt = lambda v: "" if not np.isfinite(v) else f"{v:.3f}"
    out = [[ids[k], int(rows[k]), int(cols[k])] + [fmt(own[y][k]) for y in years]
           + [fmt(win[y][k]) for y in years] + [int(count[y][k]) for y in years] for k in range(len(ids))]
    return header, out


# --------------------------------------------------------------------------
# Satellite side (conda environment)
# --------------------------------------------------------------------------
def load_year(catalog, gbox, bbox, year: int, landmask, window=xs.SEASON):
    """Season anomaly and clear-scene count for one summer, plus a log of the scenes."""
    import odc.stac
    items = catalog.search(collections=[COLLECTION], bbox=bbox,
                           datetime=f"{year}-{window[0]}/{year}-{window[1]}").item_collection()
    keep, skipped = select_scenes(items, bbox)
    log = {"used": [], "skipped": [{"id": i, "why": w} for i, w in skipped if w != "platform"]}
    anoms = []
    if keep:
        ds = odc.stac.load(keep, bands=["lwir11", "qa_pixel"], geobox=gbox,
                           resampling="nearest", groupby="solar_day")
        for k in range(ds.sizes["time"]):
            when = str(ds["time"].values[k])[:10]
            a = scene_anomaly(lst_kelvin(ds["lwir11"].values[k], ds["qa_pixel"].values[k]), landmask)
            if a is None:
                log["skipped"].append({"id": when, "why": f"under {MIN_SCENE_SHARE:.0%} of judged cells clear"})
            else:
                anoms.append(a)
                log["used"].append(when)
    med, n = season_anomaly(anoms, tuple(gbox.shape))
    return med, n, log


LANDSAT = {
    "imagery": f"Landsat 8/9 Collection 2 Level-2 surface temperature (lwir11), {COLLECTION}, "
               "Microsoft Planetary Computer",
    "rules": {"platforms": PLATFORMS, "scene_cloud_max": si.CLOUD_MAX, "qa_reject_bits": bin(QA_REJECT),
              "local_solar_hours": si.SENSORS["landsat"]["local_hours"],
              "min_scene_share": MIN_SCENE_SHARE, "min_dates": MIN_DATES},
    "first_year": FIRST_YEAR,
}


def extract_region(region: dict, today: dt.date, out_dir: Path = OUT_DIR, loader=None,
                   source: dict = LANDSAT) -> dict | None:
    """Thermal table for one region. `loader(catalog, gbox, bbox, year, landmask, window)`
    returns (season anomaly, clear-scene count, scene log); Landsat's load_year by
    default, sample_ecostress.py passes its own with `source` describing it."""
    loader = loader or load_year
    osavi_csv = so.OUT_DIR / f"{region['id']}.csv.gz"
    if not osavi_csv.exists():
        print(f"{region['id']}: no OSAVI table; run sample_osavi.py --region {region['id']} first")
        return None
    rules = xs.profile_rules(region.get("profile", "olive"))
    window = tuple(rules["season"])
    latest, complete = xs.latest_season(today, window)
    last_year = latest if complete else latest - 1
    gbox = xs.region_geobox(region["bbox"])
    ids, rows, cols = read_cells(osavi_csv)
    meta_osavi = json.loads((so.OUT_DIR / f"{region['id']}.json").read_text())
    if meta_osavi["grid"]["shape"] != list(gbox.shape) or \
            meta_osavi["grid"]["transform"] != list(gbox.transform)[:6]:
        raise RuntimeError(f"{region['id']}: the OSAVI table was made on another grid; re-run sample_osavi.py")

    catalog = xs.open_catalog()
    landmask, mask_info = xs.build_landmask(catalog, gbox, region["bbox"], region["id"],
                                            keep_classes=rules["keep_worldcover"])
    own, win, count, scenes = {}, {}, {}, {}
    for y in range(source["first_year"], last_year + 1):
        print(f"{region['id']}: thermal season {y} ...", flush=True)
        try:
            med, n, scenes[y] = loader(catalog, gbox, region["bbox"], y, landmask, window)
        except Exception as exc:
            print(f"  failed ({exc.__class__.__name__}: {exc}); season left empty")
            med, n = season_anomaly([], tuple(gbox.shape))
            scenes[y] = {"used": [], "skipped": [], "error": f"{exc.__class__.__name__}: {exc}"[:300]}
        own[y], win[y] = so.cell_values(med, rows, cols)
        count[y] = n[rows, cols]
        print(f"  {len(scenes[y]['used'])} scenes used, "
              f"{np.isfinite(own[y]).mean():.0%} of samples with a value")

    header, out = table(ids, rows, cols, own, win, count)
    so.write_table(out_dir / f"{region['id']}.csv.gz", header, out)
    meta = {
        "region": region["id"], "generated": today.isoformat(),
        "imagery": source["imagery"],
        "grid": meta_osavi["grid"], "osavi_table": osavi_csv.name, "season_window": f"{window[0]} to {window[1]}",
        "rules": source["rules"],
        "masks": mask_info, "samples": len(out),
        "scenes": {str(y): s for y, s in scenes.items()},
        "columns": COLUMNS,
    }
    xs.write_json(out_dir / f"{region['id']}.json", meta)
    print(f"{region['id']}: wrote {len(out)} rows")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--region", help="region id from regions.json")
    g.add_argument("--all", action="store_true", help="every enabled region with an OSAVI table")
    ap.add_argument("--today", help="pretend date (YYYY-MM-DD), for testing")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    regions = [r for r in xs.load_regions() if (args.all and r.get("enabled", True)) or r["id"] == args.region]
    if not regions:
        print(f"Unknown region {args.region!r}", file=sys.stderr)
        return 2
    for region in regions:
        extract_region(region, today, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
