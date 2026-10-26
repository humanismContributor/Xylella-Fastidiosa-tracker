#!/usr/bin/env python3
"""
Scanner check against Regione Puglia's field sampling (prototype).

The weekly scanner's thresholds come from the literature, not from field data.
Regione Puglia publishes every tree its inspectors sampled, with coordinates,
date, symptoms and the lab result, positives and negatives alike. This script
puts each sampled olive tree on the scanner's grid and asks whether trees in
flagged cells tested positive more often than trees elsewhere.

It only reads data/<region>/state.npz, so it never changes the scan and needs
no satellite libraries. It measures the current rules; tuning them needs the
OSAVI values at each tree, which state.npz does not keep.

Usage
    python calibrate_scanner.py --download            # fetch the CSV into .cache/
    python calibrate_scanner.py                       # every region with samples
    python calibrate_scanner.py --region puglia-monopoli-polignano

Outputs, in data/calibration/
    <region-id>.json   counts and rates per lag, all trees and symptomatic only
    summary.json       one line per region

Source: Regione Puglia, "Dati monitoraggio Xylella fastidiosa" (CC BY 4.0),
https://dati.puglia.it/ckan/dataset/dati-monitoraggio-xylella-fastidiosa
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

import xylella_scanner as xs

ROOT = Path(__file__).resolve().parent
OUT_DIR = xs.DATA_DIR / "calibration"
CACHE_FILE = ROOT / ".cache" / "puglia_samples.csv"

SOURCE = {
    "title": "Regione Puglia, Dati monitoraggio Xylella fastidiosa (campaigns 2020-2023)",
    "url": "https://dati.puglia.it/ckan/dataset/dati-monitoraggio-xylella-fastidiosa",
    "license": "CC BY 4.0",
}
SAMPLES_URL = ("https://dati.puglia.it/ckan/dataset/6d0fb7c5-e197-4c16-bec9-4986011c05b6/resource/"
               "fe41c298-e6b4-4728-ba19-50d17e411372/download/camp_2020_2022.csv")

# Matching rules. These only affect this check, not the scan or its hash.
# GPS on a phone or field unit is good to roughly 5-10 m, so a tree is matched
# to its own cell and the eight around it.
RADIUS_PX = 1
# Quick decline shows in the canopy months to years after infection, so a tree
# is also checked against the summer after its sampling season.
LAGS = (0, 1)
SPECIES_PREFIX = "Olivo"     # the Puglia regions use the olive profile

CAVEATS = [
    "Sampling is targeted, not random: inspectors favour symptomatic trees and the buffer and "
    "containment zones, and stop sampling once an area is declared infected.",
    "Cells the scanner never judges (built-up land, non-WorldCover classes) count as not flagged, "
    "because state.npz does not store the land mask.",
    "Outside the infected zone, positive trees must be removed, and in eradication areas also the "
    "host plants within 50 m (Reg. (EU) 2020/1201, https://eur-lex.europa.eu/eli/reg_impl/2020/1201/oj). "
    "Removal looks like decline to the scanner. A flag in the season "
    "after a positive (lag 1) may show the eradication rather than the disease.",
    "About a third of positives had no symptoms; no canopy-based method can see those.",
    "Coordinates are those recorded by the inspectors; a tree near a cell edge may be matched "
    "to the wrong cell, which the 3 x 3 window only partly absorbs.",
]


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------
def _num(s: str) -> float:
    return float(s.strip().replace(",", "."))


def read_samples(text: io.TextIOBase, species_prefix: str | None = SPECIES_PREFIX) -> dict:
    """Parse the regional CSV (semicolon-separated, decimal commas) into arrays.
    Rows without coordinates or a clear result are skipped."""
    header = text.readline().strip().split(";")
    col = {name: i for i, name in enumerate(header)}
    lat, lon, season, pos, sym, skipped = [], [], [], [], [], 0
    for line in text:
        p = line.rstrip("\r\n").split(";")
        if len(p) != len(header):
            skipped += 1
            continue
        if species_prefix and not p[col["SPECIE"]].startswith(species_prefix):
            continue
        result = p[col["RISULTATO"]].strip().lower()
        if result not in ("positivo", "negativo"):
            skipped += 1
            continue
        try:
            la, lo = _num(p[col["LATITUDINE"]]), _num(p[col["LONGITUDINE"]])
            date = dt.datetime.strptime(p[col["DATA_RILEVAMENTO"]].strip(), "%d/%m/%Y").date()
        except ValueError:
            skipped += 1
            continue
        lat.append(la)
        lon.append(lo)
        season.append(sample_season(date))
        pos.append(result == "positivo")
        sym.append(p[col["SINTOMO"]].strip().lower() == "presente")
    return {"lat": np.array(lat), "lon": np.array(lon), "season": np.array(season, int),
            "positive": np.array(pos, bool), "symptomatic": np.array(sym, bool), "skipped": skipped}


def sample_season(date: dt.date) -> int:
    """The scanner season a sample is compared with: the latest summer composite
    available when the tree was sampled (July-December: this year's summer)."""
    return date.year if date.month >= 7 else date.year - 1


def load_samples(path: Path = CACHE_FILE) -> dict:
    with open(path, encoding="latin-1", newline="") as f:
        return read_samples(f)


def download(path: Path = CACHE_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SAMPLES_URL} ...", flush=True)
    urllib.request.urlretrieve(SAMPLES_URL, path)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# --------------------------------------------------------------------------
# Grid lookup (pure numpy, so the tests need no pyproj)
# --------------------------------------------------------------------------
def utm_forward(lat, lon, epsg: int):
    """WGS84 latitude/longitude to UTM easting/northing (Krüger series, mm accuracy
    within a zone). `epsg` is 326zz (north) or 327zz (south)."""
    zone, south = epsg % 100, epsg // 100 == 327
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    n = f / (2 - f)
    big_a = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    phi = np.radians(np.asarray(lat, float))
    dlam = np.radians(np.asarray(lon, float)) - math.radians((zone - 1) * 6 - 180 + 3)
    c = 2 * math.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(dlam))
    eta = np.arctanh(np.sin(dlam) / np.sqrt(1 + t ** 2))
    e, nn = eta.copy(), xi.copy()
    for j, aj in enumerate(alpha, start=1):
        e += aj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        nn += aj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000 + k0 * big_a * e, (10000000 if south else 0) + k0 * big_a * nn


def grid_cells(lat, lon, crs: str, transform):
    """Row and column of each point on a scanner grid (north-up affine transform)."""
    a, _, c, _, e, f = transform
    x, y = utm_forward(lat, lon, int(crs.split(":")[1]))
    return np.floor((y - f) / e).astype(int), np.floor((x - c) / a).astype(int)


def window_bits(flagged, recovered, rows, cols, radius: int = RADIUS_PX):
    """Effective flag bits (flagged and not recovered) OR-ed over the window around
    each cell, and whether the whole window lies on the grid."""
    eff = flagged & ~recovered
    h, w = eff.shape
    inside = (rows - radius >= 0) & (rows + radius < h) & (cols - radius >= 0) & (cols + radius < w)
    bits = np.zeros(rows.shape, np.uint32)
    r, c = rows[inside], cols[inside]
    acc = np.zeros(r.shape, np.uint32)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            acc |= eff[r + dr, c + dc]
    bits[inside] = acc
    return bits, inside


def flagged_by(bits, seasons, lag: int):
    """True where any season up to and including season + lag is flagged."""
    upto = np.array([(1 << (s + lag - xs.BIT_BASE_YEAR + 1)) - 1 for s in seasons], np.uint64)
    return (bits.astype(np.uint64) & upto) != 0


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson interval for a proportion, or None when n is 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [round(mid - half, 4), round(mid + half, 4)]


def score(flag, positive) -> dict:
    """Two-by-two table of scanner flag against lab result, with the rates that
    matter for a lead-for-inspection tool."""
    tp = int((flag & positive).sum())
    fp = int((flag & ~positive).sum())
    fn = int((~flag & positive).sum())
    tn = int((~flag & ~positive).sum())
    rate_in = tp / (tp + fp) if tp + fp else None
    rate_out = fn / (fn + tn) if fn + tn else None
    rr = rate_in / rate_out if rate_in is not None and rate_out else None
    return {
        "trees": tp + fp + fn + tn, "positive": tp + fn,
        # share of all sampled trees in a flagged cell: sensitivity means little without it
        "flagged_share": round((tp + fp) / (tp + fp + fn + tn), 4) if tp + fp + fn + tn else None,
        "flagged_positive": tp, "flagged_negative": fp, "unflagged_positive": fn, "unflagged_negative": tn,
        # share of positive trees that sat in a flagged cell
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        # share of trees in flagged cells that tested positive
        "positive_rate_flagged": round(rate_in, 4) if rate_in is not None else None,
        "positive_rate_flagged_ci95": wilson(tp, tp + fp),
        "positive_rate_unflagged": round(rate_out, 4) if rate_out is not None else None,
        "positive_rate_unflagged_ci95": wilson(fn, fn + tn),
        "relative_risk": round(rr, 2) if rr is not None else None,
    }


def evaluate_region(region: dict, state: dict, samples: dict, radius: int = RADIUS_PX, lags=LAGS) -> dict | None:
    """Scores for one region, or None when no sampled tree falls on its grid."""
    rows, cols = grid_cells(samples["lat"], samples["lon"], state["crs"], state["transform"])
    bits, inside = window_bits(state["flagged"], state["recovered"], rows, cols, radius)
    judged = samples["season"] >= xs.FIRST_SEASON
    keep = inside & judged
    if not keep.any():
        return None
    bits, seasons = bits[keep], samples["season"][keep]
    pos, sym = samples["positive"][keep], samples["symptomatic"][keep]

    out_lags = []
    for lag in lags:
        ok = seasons + lag <= state["complete_through"]
        flag = flagged_by(bits[ok], seasons[ok], lag)
        p, s, se = pos[ok], sym[ok], seasons[ok]
        out_lags.append({
            "lag": lag,
            "rule": f"cell flagged in any season up to the sampling season + {lag}",
            "all_trees": score(flag, p),
            "symptomatic_trees": score(flag[s], p[s]),
            "by_season": {str(y): score(flag[se == y], p[se == y]) for y in sorted(set(se.tolist()))},
        })
    return {
        "region": region["id"],
        "name": region["name"],
        "generated": dt.date.today().isoformat(),
        "source": SOURCE,
        "scanner_params_hash": state.get("params_hash"),
        "scanner_complete_through": state["complete_through"],
        "match": {"radius_px": radius, "window_m": (2 * radius + 1) * xs.PIXEL_M, "species": SPECIES_PREFIX},
        "samples_on_grid": int(keep.sum()),
        "positives_on_grid": int(pos.sum()),
        "results": out_lags,
        "caveats": CAVEATS,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--region", help="region id from regions.json (default: all with samples)")
    ap.add_argument("--samples", type=Path, default=CACHE_FILE, help="path to the regional CSV")
    ap.add_argument("--download", action="store_true", help="download the CSV first")
    ap.add_argument("--radius", type=int, default=RADIUS_PX, help="match window radius in cells (0 = own cell)")
    args = ap.parse_args(argv)

    if args.download:
        download(args.samples)
    if not args.samples.exists():
        print(f"No sample file at {args.samples}; run with --download first.", file=sys.stderr)
        return 2
    samples = load_samples(args.samples)
    print(f"{len(samples['lat'])} {SPECIES_PREFIX} samples read ({samples['skipped']} rows skipped)")

    regions = [r for r in xs.load_regions() if not args.region or r["id"] == args.region]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for region in regions:
        state = xs.load_state(xs.DATA_DIR / region["id"] / "state.npz")
        if state is None:
            continue
        res = evaluate_region(region, state, samples, radius=args.radius)
        if res is None:
            print(f"{region['id']}: no sampled trees on the grid")
            continue
        xs.write_json(OUT_DIR / f"{region['id']}.json", res)
        for r in res["results"]:
            a = r["all_trees"]
            print(f"{region['id']} lag {r['lag']}: {a['trees']} trees, {a['positive']} positive, "
                  f"sensitivity {a['sensitivity']}, positive rate flagged {a['positive_rate_flagged']} "
                  f"vs unflagged {a['positive_rate_unflagged']} (RR {a['relative_risk']})")
        summary.append({"region": region["id"], "samples_on_grid": res["samples_on_grid"],
                        "positives_on_grid": res["positives_on_grid"],
                        **{f"lag{r['lag']}": {k: r["all_trees"][k] for k in
                                              ("flagged_share", "sensitivity", "positive_rate_flagged",
                                               "positive_rate_unflagged", "relative_risk")}
                           for r in res["results"]}})
    if summary:
        xs.write_json(OUT_DIR / "summary.json", {"generated": dt.date.today().isoformat(),
                                                 "source": SOURCE, "regions": summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
