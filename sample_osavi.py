#!/usr/bin/env python3
"""
Summer OSAVI at every sampled tree (calibration data, prototype).

calibrate_scanner.py can only say whether a sampled tree's cell was flagged,
because state.npz keeps flag bits, not greenness. This script loads the same
seasonal composites the scanner uses and saves, for every Puglia sample inside
a scan region, the summer OSAVI of the tree's cell and the median of the 3 x 3
window around it for every season, whether the scanner may judge that cell
(land mask) and whether it burned. With that table the decline rules can be
replayed offline with other thresholds: nothing here touches PARAMS, state.npz
or the scan outputs, so no region is rebuilt.

Usage (needs the conda packages from main.yml)
    python calibrate_scanner.py --download
    python sample_osavi.py --region puglia-monopoli-polignano
    python sample_osavi.py --all

Outputs, in data/calibration/osavi/
    <region-id>.csv.gz   one row per sample (columns described in <region-id>.json)
    <region-id>.json     source, grid, rules, seasons and scene counts
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import sys
import warnings
from pathlib import Path

import numpy as np

import calibrate_scanner as cal
import xylella_scanner as xs

OUT_DIR = cal.OUT_DIR / "osavi"
FIRST_YEAR = 2017            # the scanner's earliest baseline summer
RADIUS_PX = 1

COLUMNS = {
    "id": "sample ID in the regional data set",
    "date": "sampling date",
    "season": "scanner season the sample is compared with (calibrate_scanner.sample_season)",
    "positive": "1 if the lab result was positive",
    "symptomatic": "1 if the inspector recorded symptoms",
    "lat": "latitude as recorded", "lon": "longitude as recorded",
    "row": "row of the tree's cell on the region grid", "col": "column of the tree's cell",
    "judged": "1 if the scanner's land mask lets it judge this cell",
    "judged3": "share of the 3 x 3 window the land mask lets it judge",
    "flag_bits": "effective scanner flags of the cell at extraction time (bit k = season 2010 + k)",
    "flag_bits3": "the same, OR-ed over the 3 x 3 window",
    "osavi_<year>": "median July-August OSAVI of the tree's cell (empty: too few clear dates)",
    "osavi3_<year>": "median of the 3 x 3 window's seasonal values",
    "burned_<season>": "1 if MODIS mapped the cell as burned in the scanner's fire window, empty if unknown",
}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def on_grid(samples: dict, crs: str, transform, shape, radius: int = RADIUS_PX):
    """Indices, rows and columns of the samples whose whole window lies on the grid."""
    rows, cols = cal.grid_cells(samples["lat"], samples["lon"], crs, transform)
    h, w = shape
    ok = (rows - radius >= 0) & (rows + radius < h) & (cols - radius >= 0) & (cols + radius < w)
    idx = np.nonzero(ok)[0]
    return idx, rows[idx], cols[idx]


def cell_values(arr, rows, cols, radius: int = RADIUS_PX):
    """Value of each cell and the NaN-aware median of the window around it."""
    own = arr[rows, cols].astype("float32")
    stack = np.stack([arr[rows + dr, cols + dc] for dr in range(-radius, radius + 1)
                      for dc in range(-radius, radius + 1)]).astype("float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN windows
        win = np.nanmedian(stack, axis=0)
    return own, win


def window_or(bits, rows, cols, radius: int = RADIUS_PX):
    acc = np.zeros(rows.shape, bits.dtype)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            acc |= bits[rows + dr, cols + dc]
    return acc


def window_mean(mask, rows, cols, radius: int = RADIUS_PX):
    acc = np.zeros(rows.shape, float)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            acc += mask[rows + dr, cols + dc]
    return acc / (2 * radius + 1) ** 2


def collect(load_year, years, rows, cols, radius: int = RADIUS_PX):
    """Seasonal values at the sample cells. `load_year(year)` returns (composite
    or None, scene count); only the sampled cells are kept, so memory stays small."""
    own, win, scenes = {}, {}, {}
    for y in years:
        comp, n = load_year(y)
        scenes[y] = n
        if comp is None:
            own[y] = win[y] = np.full(rows.shape, np.nan, "float32")
        else:
            own[y], win[y] = cell_values(comp, rows, cols, radius)
    return own, win, scenes


def _fmt(v, nd=4):
    return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def table(samples: dict, idx, rows, cols, own: dict, win: dict, judged, judged3,
          bits, bits3, burned: dict):
    """Header and rows of the output table."""
    years, seasons = sorted(own), sorted(burned)
    header = (["id", "date", "season", "positive", "symptomatic", "lat", "lon", "row", "col",
               "judged", "judged3", "flag_bits", "flag_bits3"]
              + [f"osavi_{y}" for y in years] + [f"osavi3_{y}" for y in years]
              + [f"burned_{s}" for s in seasons])
    out = []
    for k, i in enumerate(idx):
        out.append([samples["id"][i], samples["date"][i], int(samples["season"][i]),
                    int(samples["positive"][i]), int(samples["symptomatic"][i]),
                    f"{samples['lat'][i]:.6f}", f"{samples['lon'][i]:.6f}", int(rows[k]), int(cols[k]),
                    int(judged[k]), f"{judged3[k]:.2f}", int(bits[k]), int(bits3[k])]
                   + [_fmt(own[y][k]) for y in years] + [_fmt(win[y][k]) for y in years]
                   + ["" if burned[s] is None else int(burned[s][k]) for s in seasons])
    return header, out


def write_table(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the file byte-identical when the content doesn't change
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
        w = csv.writer(text, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------
# Satellite side (conda environment)
# --------------------------------------------------------------------------
def extract_region(region: dict, samples: dict, today: dt.date, radius: int = RADIUS_PX,
                   out_dir: Path = OUT_DIR) -> dict | None:
    profile = region.get("profile", "olive")
    if profile != "olive":
        print(f"{region['id']}: skipped, the regional samples are matched to olive regions only")
        return None
    rules = xs.profile_rules(profile)
    window = tuple(rules["season"])
    latest, complete = xs.latest_season(today, window)
    last_year = latest if complete else latest - 1          # only finished summers
    gbox = xs.region_geobox(region["bbox"])
    crs, transform, shape = str(gbox.crs), list(gbox.transform)[:6], tuple(gbox.shape)

    idx, rows, cols = on_grid(samples, crs, transform, shape, radius)
    if len(idx) == 0:
        print(f"{region['id']}: no sampled trees on the grid")
        return None
    print(f"{region['id']}: {len(idx)} samples on the grid", flush=True)

    catalog = xs.open_catalog()
    landmask, mask_info = xs.build_landmask(catalog, gbox, region["bbox"], region["id"],
                                            keep_classes=rules["keep_worldcover"])

    def load_year(y):
        print(f"{region['id']}: loading season {y} ...", flush=True)
        try:
            return xs.load_season(catalog, gbox, region["bbox"], y, window=window)
        except Exception as exc:
            print(f"  failed ({exc.__class__.__name__}: {exc}); season left empty")
            return None, 0

    years = list(range(FIRST_YEAR, last_year + 1))
    own, win, scenes = collect(load_year, years, rows, cols, radius)

    seasons = [s for s in range(xs.FIRST_SEASON, last_year + 1)]
    burned_maps, mask_info["burned_area"] = xs.load_burned(catalog, gbox, region["bbox"], seasons,
                                                           baseline_years=rules["baseline_years"], window=window)
    burned = {s: (None if m is None else m[rows, cols]) for s, m in burned_maps.items()}

    state = xs.load_state(xs.DATA_DIR / region["id"] / "state.npz")
    if state is not None and tuple(state["shape"]) == shape:
        eff = state["flagged"] & ~state["recovered"]
        bits, bits3 = eff[rows, cols], window_or(eff, rows, cols, radius)
        state_hash = state.get("params_hash")
    else:
        bits = bits3 = np.zeros(rows.shape, np.uint32)
        state_hash = None

    header, out = table(samples, idx, rows, cols, own, win, landmask[rows, cols],
                        window_mean(landmask, rows, cols, radius), bits, bits3, burned)
    write_table(out_dir / f"{region['id']}.csv.gz", header, out)
    meta = {
        "region": region["id"],
        "name": region["name"],
        "generated": today.isoformat(),
        "source": cal.SOURCE,
        "imagery": "Sentinel-2 L2A, Microsoft Planetary Computer; same composites as xylella_scanner.load_season",
        "grid": {"crs": crs, "transform": transform, "shape": list(shape), "pixel_m": xs.PIXEL_M},
        "window_px": 2 * radius + 1,
        "profile": profile,
        "rules": {k: v for k, v in rules.items() if k != "keep_worldcover"},
        "scanner_params": xs.PARAMS,
        "scanner_params_hash": xs.params_hash(region),
        "state_params_hash": state_hash,
        "years": years,
        "scenes": {str(y): scenes[y] for y in years},
        "masks": mask_info,
        "samples": len(out),
        "positives": int(samples["positive"][idx].sum()),
        "columns": COLUMNS,
    }
    xs.write_json(out_dir / f"{region['id']}.json", meta)
    print(f"{region['id']}: wrote {len(out)} rows")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--region", help="region id from regions.json")
    g.add_argument("--all", action="store_true", help="every enabled region")
    ap.add_argument("--samples", type=Path, default=cal.CACHE_FILE, help="path to the regional CSV")
    ap.add_argument("--today", help="pretend date (YYYY-MM-DD), for testing")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"output folder (default {OUT_DIR})")
    args = ap.parse_args(argv)

    if not args.samples.exists():
        print(f"No sample file at {args.samples}; run calibrate_scanner.py --download first.", file=sys.stderr)
        return 2
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    samples = cal.load_samples(args.samples)
    regions = [r for r in xs.load_regions()
               if (args.all and r.get("enabled", True)) or r["id"] == args.region]
    if not regions:
        print(f"Unknown region {args.region!r}", file=sys.stderr)
        return 2
    for region in regions:
        extract_region(region, samples, today, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
