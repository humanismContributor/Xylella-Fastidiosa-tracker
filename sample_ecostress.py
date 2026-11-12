#!/usr/bin/env python3
"""
ECOSTRESS thermal anomaly at every sampled tree (calibration data, prototype).

The same measure as sample_thermal.py (each summer's median of a cell's
temperature minus the median of the region's judged cells in the same scene),
from ECOSTRESS instead of Landsat: 70 m pixels instead of 100 m, and an
overpass time that drifts through the day, so more midday looks per summer.
Its tables go through replay_thermal.py --sensor ecostress.

Needs a NASA Earthdata login in ~/.netrc or ~/_netrc (Windows: %USERPROFILE%):
    machine urs.earthdata.nasa.gov login <user> password <password>
Downloads the LST and cloud layers of the midday tiles covering the region into
.cache/ecostress/ (not committed; both Monopoli regions share tile 33TXF; about
3 MB per tile, some 300 MB for 2018-2026). On Windows, if the checkout path is
long, pass a short --cache-dir: the tile names alone are over 70 characters.

Usage (conda packages from main.yml plus earthaccess)
    python sample_ecostress.py --region puglia-monopoli-polignano
    python sample_ecostress.py --all

Outputs, in .cache/thermal_ecostress/ (same columns as sample_thermal.py; not committed)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

import sample_thermal as st
import scene_inventory as si
import xylella_scanner as xs

ROOT = Path(__file__).resolve().parent
OUT_DIR = st.OUT_DIR.parent / "thermal_ecostress"
CACHE_DIR = ROOT / ".cache" / "ecostress"
SHORT_NAME, VERSION = "ECO_L2T_LSTE", "002"   # tiled LST, 70 m (https://doi.org/10.5067/ECOSTRESS/ECO_L2T_LSTE.002)
LAYERS = ("LST", "cloud")
FIRST_YEAR = 2018                              # ECOSTRESS began science operations in July 2018
MIN_KELVIN = 200.0                             # below this an LST value is fill, not a surface

ECOSTRESS = {
    "imagery": f"ECOSTRESS {SHORT_NAME} V{VERSION} (LST and cloud layers), NASA LP DAAC",
    "rules": {"local_solar_hours": si.SENSORS["ecostress"]["local_hours"], "cloud_layer": "cloud == 0",
              "min_kelvin": MIN_KELVIN, "min_scene_share": st.MIN_SCENE_SHARE, "min_dates": st.MIN_DATES,
              "overpass_grouping_minutes": si.GROUP_MINUTES},
    "first_year": FIRST_YEAR,
}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def layer_urls(umm: dict, layers=LAYERS) -> dict | None:
    """HTTPS link of each layer's GeoTIFF, or None if one is missing."""
    links = [u["URL"] for u in umm.get("RelatedUrls", []) if u.get("URL", "").startswith("https")]
    out = {l: next((u for u in links if u.endswith(f"_{l}.tif")), None) for l in layers}
    return None if None in out.values() else out


def pick_overpasses(items: list, bbox, hours=si.SENSORS["ecostress"]["local_hours"]) -> tuple[list, list]:
    """Midday overpasses whose tile holds the whole region. One overpass can
    split the tile across consecutive scene granules, so each overpass keeps
    every granule of that tile; they are pieced together when read."""
    lon = (bbox[0] + bbox[2]) / 2
    grans, skipped = [], []
    for it in items:
        sc = si.cmr_scene(it)
        if sc is None:                                     # night pass
            continue
        gid = sc["id"]
        h = si.local_solar_hour(sc["time"], lon)
        if not hours[0] <= h <= hours[1]:
            skipped.append((gid, f"local time {h:.1f} h"))
        elif si.coverage(bbox, sc["polys"]) < 1.0:
            continue                                       # a neighbouring tile; the full one is listed too
        elif (urls := layer_urls(it["umm"])) is None:
            skipped.append((gid, "LST or cloud layer missing"))
        else:
            grans.append({"id": gid, "time": sc["time"], "urls": urls})
    out = []
    for g in sorted(grans, key=lambda g: g["time"]):
        if out and (g["time"] - out[-1]["time_last"]).total_seconds() <= si.GROUP_MINUTES * 60:
            out[-1]["granules"].append(g)
            out[-1]["time_last"] = g["time"]
        else:
            out.append({"time": g["time"], "time_last": g["time"], "granules": [g]})
    for o in out:
        o.pop("time_last")
    return out, skipped


def mask_lst(lst, cloud):
    """LST in K, NaN where cloudy, unflagged or fill."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(lst) & (lst >= MIN_KELVIN) & (cloud == 0)
    return np.where(ok, lst, np.nan).astype("float32")


def piece_together(parts: list):
    """First finite value per cell across the granules of one overpass."""
    out = np.full(parts[0].shape, np.nan, "float32")
    for p in parts:
        out = np.where(np.isfinite(out), out, p)
    return out


def local_name(url: str) -> str:
    return url.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------
# Download and read (earthaccess, rasterio)
# --------------------------------------------------------------------------
def read_on_grid(path: Path, gbox, nodata_fill, dtype):
    """One tile layer resampled (nearest) onto the region grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    dst = np.full(tuple(gbox.shape), nodata_fill, dtype)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst, src_nodata=src.nodata,
                  dst_transform=gbox.transform, dst_crs=str(gbox.crs), dst_nodata=nodata_fill,
                  resampling=Resampling.nearest)
    return dst


def make_loader(cache: Path = CACHE_DIR):
    """A sample_thermal.extract_region loader that logs in once and downloads what it needs."""
    import earthaccess
    if not os.environ.get("NETRC"):
        # earthaccess looks for ~/_netrc on Windows and ~/.netrc elsewhere; accept either name
        found = [p for p in (Path.home() / ".netrc", Path.home() / "_netrc") if p.exists()]
        if found:
            os.environ["NETRC"] = str(found[0])
    auth = earthaccess.login(strategy="netrc")
    if not getattr(auth, "authenticated", False):
        raise RuntimeError("Earthdata login failed: put 'machine urs.earthdata.nasa.gov login <user> "
                           "password <password>' in ~/.netrc (see the module docstring)")
    cache.mkdir(parents=True, exist_ok=True)

    def loader(catalog, gbox, bbox, year, landmask, window):
        items = si.cmr_search(SHORT_NAME, VERSION, bbox, f"{year}-{window[0]}", f"{year}-{window[1]}")
        overpasses, skipped = pick_overpasses(items, bbox)
        missing = [u for o in overpasses for g in o["granules"] for u in g["urls"].values()
                   if not (cache / local_name(u)).exists()]
        if missing:
            print(f"  downloading {len(missing)} files", flush=True)
            earthaccess.download(missing, local_path=str(cache))
        log = {"used": [], "skipped": [{"id": i, "why": w} for i, w in skipped]}
        anoms = []
        for o in overpasses:
            parts = []
            for g in o["granules"]:
                lst = read_on_grid(cache / local_name(g["urls"]["LST"]), gbox, np.nan, "float32")
                cloud = read_on_grid(cache / local_name(g["urls"]["cloud"]), gbox, 255, "uint8")
                parts.append(mask_lst(lst, cloud))
            when = o["time"].strftime("%Y-%m-%d %H:%MZ")
            a = st.scene_anomaly(piece_together(parts), landmask)
            if a is None:
                log["skipped"].append({"id": when, "why": f"under {st.MIN_SCENE_SHARE:.0%} of judged cells clear"})
            else:
                anoms.append(a)
                log["used"].append(when)
        med, n = st.season_anomaly(anoms, tuple(gbox.shape))
        return med, n, log

    return loader


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--region", help="region id from regions.json")
    g.add_argument("--all", action="store_true", help="every enabled region with an OSAVI table")
    ap.add_argument("--today", help="pretend date (YYYY-MM-DD), for testing")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                    help="where tiles are downloaded (a short path avoids Windows' 260-character limit)")
    args = ap.parse_args(argv)
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    regions = [r for r in xs.load_regions() if (args.all and r.get("enabled", True)) or r["id"] == args.region]
    if not regions:
        print(f"Unknown region {args.region!r}", file=sys.stderr)
        return 2
    loader = make_loader(args.cache_dir)
    for region in regions:
        st.extract_region(region, today, out_dir=args.out_dir, loader=loader, source=ECOSTRESS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
