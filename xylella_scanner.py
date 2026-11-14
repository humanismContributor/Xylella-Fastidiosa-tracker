#!/usr/bin/env python3
"""
Xylella decline scanner: Sentinel-2 L2A from Microsoft Planetary Computer.

For every region in regions.json it builds one July-August OSAVI composite per
year, flags olive-grove cells whose summer OSAVI fell sharply compared with the
previous summer, and keeps a per-cell history. Each scan is therefore compared
with all earlier seasons, and every flagged area is labelled:

    new         flagged for the first time in the latest season
    continuing  flagged in the latest season and in an earlier one
    earlier     flagged in an earlier season, did not recover, not flagged now

Usage
    python xylella_scanner.py --region puglia-ostuni
    python xylella_scanner.py --all
    python xylella_scanner.py --region puglia-ostuni --today 2026-09-15   # testing

Outputs per region, in data/<region-id>/
    zones.geojson   polygons with status, first season, area, shape
    history.json    flagged area per season
    status.json     scan date, seasons, scene counts, mask and parameter info
    state.npz       per-cell history (needed for the next run; keep it in git)

What this can and cannot see
    At 10 m a pixel holds roughly one olive crown plus soil and grass, so only
    advanced decline is visible, not early or symptomless infection
    (Zarco-Tejada et al. 2018; Hornero et al. 2020). A flagged area is a lead
    for a field check, never a diagnosis: drought, fire, abandonment and
    removal of trees for other reasons look the same from orbit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REGIONS_FILE = ROOT / "regions.json"
DATA_DIR = ROOT / "data"
MASK_DIR = ROOT / "masks"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# --------------------------------------------------------------------------
# Detection parameters. Changing any of them makes the next run rebuild the
# whole history, so all seasons are judged by the same rules.
# --------------------------------------------------------------------------
PIXEL_M = 10
FIRST_SEASON = 2019          # first season judged; 2017-2018 serve as baselines
SEASON = ("07-01", "08-31")  # dry-season window used by Hornero 2020 and Blonda 2023
SCENE_CLOUD_MAX = 30         # scene-level filter; pixels are masked with SCL anyway
MIN_DATES = 3                # valid observations per pixel per season
VALID_SCL = (4, 5)           # vegetation, bare soil
REFLECTANCE_OFFSET = 1000    # added to L2A digital numbers from processing baseline 04.00

# Decline rule (unchanged from the previous calibration):
# symptomatic olive crowns at 10 m sit roughly 45-55 % below healthy ones (Scala et
# al. 2024), so a 30 % drop plus at least 0.04 OSAVI marks clear decline.
BASELINE_OSAVI_MIN = 0.12    # healthy olive pixels sit near 0.12-0.16 OSAVI in summer
MIN_REL_DROP = 0.30
MIN_ABS_DROP = 0.04

# New rules, added after comparing scan output with the official picture.
# 1. Summer-green fields are not olive groves. Irrigated vegetables and fodder
#    peak far above an olive crown at 10 m, and their year-to-year swings were a
#    likely source of the patches along the coastal road.
BASELINE_OSAVI_MAX = 0.50
# 2. Olive trees are evergreen perennials with steady summer greenness. A cell
#    whose baseline summer was much greener than the summer before is a rotating
#    crop or a one-off green-up, not a grove that later declined.
MAX_BASELINE_RISE = 0.25     # baseline may be at most 25 % above the summer before it
# 3. Xylella decline is progressive: infected olives do not green up again.
#    Cells that win back at least half of the drop the next summer were pruned,
#    drought-stressed or cut for hay, and are removed from the record.
RECOVERY_SHARE = 0.50
# 4. Built-up land and its edges caused the large patches at Rosa Marina and
#    Villanova. Only tree, shrub, grass and cropland cells (ESA WorldCover 2021)
#    are judged, and two pixels around built-up land are excluded.
KEEP_WORLDCOVER = (10, 20, 30, 40)
BUILTUP_CLASS = 50
BUILTUP_BUFFER_PX = 2
# 5. Patch size. An infected-zone delimitation starts at 50 m around a tree; three
#    pixels (300 m2) is the smallest patch kept.
MIN_AREA_M2 = 300
# 6. Shape. Whole groves die and get uprooted in the infected zone, so rectangular
#    patches are no longer dropped. Large compact ones are labelled
#    'parcel-shaped' so the app can show them as possible tree removal.
PARCEL_EXTENT = 0.75
PARCEL_MIN_M2 = 2000
# 7. Wildfire. A burned grove loses its greenness in one summer just like a dying
#    one. Cells mapped as burned (MODIS burned area, 500 m) from the start of the
#    baseline summer to the end of the summer judged are not judged for that season.
BURNED_COLLECTION = "modis-64A1-061"

PARAMS = {
    "first_season": FIRST_SEASON, "season": SEASON, "scene_cloud_max": SCENE_CLOUD_MAX,
    "min_dates": MIN_DATES, "valid_scl": VALID_SCL, "reflectance_offset": REFLECTANCE_OFFSET,
    "baseline_osavi_min": BASELINE_OSAVI_MIN, "baseline_osavi_max": BASELINE_OSAVI_MAX,
    "min_rel_drop": MIN_REL_DROP, "min_abs_drop": MIN_ABS_DROP,
    "max_baseline_rise": MAX_BASELINE_RISE, "recovery_share": RECOVERY_SHARE,
    "keep_worldcover": KEEP_WORLDCOVER, "builtup_buffer_px": BUILTUP_BUFFER_PX,
    "min_area_m2": MIN_AREA_M2, "parcel_extent": PARCEL_EXTENT, "parcel_min_m2": PARCEL_MIN_M2,
    "burned_area_filter": BURNED_COLLECTION,
    "burned_window": "from the start of the baseline summer",
}

# --------------------------------------------------------------------------
# Crop profiles. The rules above are the olive profile. Other hosts break its
# assumptions, so a region in regions.json can name a profile ("profile") that
# overrides some of them, and should then also name a crop mask ("mask", see
# crop_masks.py), because WorldCover can't tell one orchard crop from another.
# Changing a profile rebuilds only the regions that use it.
# --------------------------------------------------------------------------
OLIVE_RULES = {
    "season": SEASON, "baseline_osavi_min": BASELINE_OSAVI_MIN, "baseline_osavi_max": BASELINE_OSAVI_MAX,
    "min_rel_drop": MIN_REL_DROP, "min_abs_drop": MIN_ABS_DROP, "baseline_years": 1,
    "max_baseline_spread": None, "recovery_share": RECOVERY_SHARE, "keep_worldcover": KEEP_WORLDCOVER,
}
PROFILES = {
    "olive": {
        "label": "Olive",
        "reliability": "literature-based",
        "overrides": {},
        "notes": "Evergreen, steady summer greenness; olive quick decline is progressive and trees don't "
                 "recover. Drop thresholds from Scala et al. 2024 and Hornero et al. 2020.",
    },
    "almond": {
        "label": "Almond",
        "reliability": "inferred",
        "overrides": {
            # Leaf scorch begins as early as June and keeps developing through the summer
            # (UC IPM), so the window is shifted to late summer when scorch is fullest.
            "season": ("08-01", "09-30"),
            # Rainfed almond canopies are sparse; summer greenness at 10 m is lower than olive.
            "baseline_osavi_min": 0.10,
            # Decline is gradual: infected trees often survive many years while scorch spreads
            # through the canopy (UC IPM), so the yearly drop is smaller than in olive quick decline.
            "min_rel_drop": 0.25,
            "min_abs_drop": 0.03,
            # Rainfed almonds react strongly to dry years: compare with the median of three
            # summers and skip cells whose greenness swings a lot from year to year.
            "baseline_years": 3,
            "max_baseline_spread": 0.35,
        },
        "notes": "Thresholds inferred from symptom biology (UC IPM almond leaf scorch guidelines) and "
                 "airborne studies (Camino et al. 2021, Alicante); no Sentinel-2 calibration exists. "
                 "Late-summer drought defoliation and harvest can look similar.",
    },
    "grapevine": {
        "label": "Grapevine",
        "reliability": "experimental",
        "overrides": {
            # Before harvest and leaf removal, which change the canopy more than early disease.
            "season": ("07-01", "08-15"),
            "baseline_osavi_min": 0.10,
            "baseline_years": 3,
            "max_baseline_spread": 0.35,
        },
        "notes": "Experimental. At 10 m a pixel mixes several vine rows with bare soil, and hedging, "
                 "leaf removal and harvest dominate the signal. Treat any result as unvalidated.",
    },
}


def profile_rules(name: str | None) -> dict:
    """Full rule set for a profile name (olive when missing)."""
    prof = PROFILES.get(name or "olive")
    if prof is None:
        raise ValueError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}")
    return {**OLIVE_RULES, **prof["overrides"]}


BIT_BASE_YEAR = 2010         # bit k of the history masks = season 2010 + k
STATUS_NAMES = {1: "new", 2: "continuing", 3: "earlier"}
PX_HA = PIXEL_M * PIXEL_M / 10_000


# --------------------------------------------------------------------------
# Pure array logic (no network, unit-tested)
# --------------------------------------------------------------------------
def latest_season(today: dt.date, window=SEASON) -> tuple[int, bool]:
    """Most recent season with usable imagery and whether that season is over.
    A season counts from two weeks after its window opens."""
    start = dt.date(today.year, *map(int, window[0].split("-")))
    end = dt.date(today.year, *map(int, window[1].split("-")))
    if today >= start + dt.timedelta(days=14):
        return today.year, today > end
    return today.year - 1, True


def season_bit(season: int) -> np.uint32:
    return np.uint32(1 << (season - BIT_BASE_YEAR))


def decline_mask(cur, base, prev=None, base_max: float = BASELINE_OSAVI_MAX, rules: dict | None = None):
    """Cells whose summer OSAVI fell clearly from the baseline summer `base`.
    `rules` (a profile_rules() dict) overrides the olive thresholds."""
    r = rules or OLIVE_RULES
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = base - cur
        ok = np.isfinite(cur) & np.isfinite(base)
        ok &= (base >= r["baseline_osavi_min"]) & (base <= base_max)
        ok &= drop >= r["min_abs_drop"]
        ok &= drop >= r["min_rel_drop"] * base
        if prev is not None:
            greened = np.isfinite(prev) & (base > prev * (1 + MAX_BASELINE_RISE))
            ok &= ~greened
    return ok


def recovered_mask(base, cur, nxt, share: float = RECOVERY_SHARE):
    """Cells that won back at least `share` of their drop one summer later."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(nxt) & np.isfinite(cur) & ((nxt - cur) >= share * (base - cur))


def remove_small(mask, min_px: int):
    """Drop 8-connected patches smaller than min_px cells."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def baseline(comps: dict, s: int, years: int = 1, max_spread: float | None = None):
    """Baseline greenness for season s and a mask of cells it may be used for.

    years=1: the previous summer; cells that greened up sharply compared with
    the summer before that are excluded (rotating crops).
    years>1: the median of the previous `years` summers (at least two needed per
    cell). With max_spread, cells whose summers varied by more than that share
    of the median are excluded: olive groves are steady from year to year,
    grassland, cereals and fodder are not."""
    if years <= 1:
        base, prev = comps.get(s - 1), comps.get(s - 2)
        if base is None:
            return None, None
        ok = np.ones(base.shape, bool)
        if prev is not None:
            with np.errstate(invalid="ignore"):
                ok &= ~(np.isfinite(prev) & (base > prev * (1 + MAX_BASELINE_RISE)))
        return base, ok
    stack = [comps[s - k] for k in range(1, years + 1) if comps.get(s - k) is not None]
    if len(stack) < 2:
        return None, None
    arr = np.stack(stack)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
        med = np.nanmedian(arr, axis=0)
        n = np.isfinite(arr).sum(axis=0)
        ok = n >= 2
        if max_spread is not None:
            spread = (np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)) / med
            ok &= np.isfinite(spread) & (spread <= max_spread)
    return med, ok


def update_history(comps: dict, seasons, landmask, flagged, recovered,
                   min_px: int | None = None, burned: dict | None = None,
                   baseline_years: int = 1, max_spread: float | None = None,
                   base_max: float = BASELINE_OSAVI_MAX, rules: dict | None = None) -> None:
    """Recompute the flag bits for `seasons` and the recovery bits wherever the
    following summer is available. Arrays are updated in place.
    `burned` maps a season to cells that burned between the two summers compared.
    With `rules` (a crop profile), its thresholds, baseline and recovery settings
    replace the keyword defaults."""
    if min_px is None:
        min_px = max(1, int(np.ceil(MIN_AREA_M2 / PIXEL_M ** 2)))
    if rules is not None:
        baseline_years, max_spread = rules["baseline_years"], rules["max_baseline_spread"]
        base_max = rules["baseline_osavi_max"]
    share = (rules or OLIVE_RULES)["recovery_share"]
    burned = burned or {}
    bases: dict = {}  # each season's baseline is needed twice; a multi-year one is a full nanmedian

    def base_of(s):
        if s not in bases:
            bases[s] = baseline(comps, s, baseline_years, max_spread)
        return bases[s]

    for s in seasons:
        b = season_bit(s)
        flagged &= ~b
        cur = comps.get(s)
        base, ok = base_of(s)
        if cur is None or base is None:
            continue
        m = decline_mask(cur, base, None, base_max, rules) & ok & landmask
        if burned.get(s) is not None:
            m &= ~burned[s]
        flagged[remove_small(m, min_px)] |= b
    for s in sorted(comps):
        if comps.get(s) is None or comps.get(s + 1) is None:
            continue
        base, _ = base_of(s)
        if base is None:
            continue
        b = season_bit(s)
        recovered &= ~b
        rec = recovered_mask(base, comps[s], comps[s + 1], share) & ((flagged & b) != 0)
        recovered[rec] |= b


def classify(flagged, recovered, latest: int):
    """Status raster (0 none, 1 new, 2 continuing, 3 earlier) and first flagged season."""
    eff = flagged & ~recovered
    lb = season_bit(latest)
    in_latest = (eff & lb) != 0
    before = (eff & np.uint32(lb - 1)) != 0
    status = np.zeros(eff.shape, np.uint8)
    status[in_latest & ~before] = 1
    status[in_latest & before] = 2
    status[~in_latest & before] = 3
    first = np.zeros(eff.shape, np.uint16)
    for s in range(latest, BIT_BASE_YEAR - 1, -1):
        first[(eff & season_bit(s)) != 0] = s
    return status, first


def season_stats(flagged, recovered, seasons, px_ha: float = PX_HA) -> dict:
    eff = flagged & ~recovered
    out = {}
    for s in seasons:
        b = season_bit(s)
        now = (eff & b) != 0
        seen_before = (eff & np.uint32(b - 1)) != 0
        out[s] = {
            "new_ha": round(float((now & ~seen_before).sum()) * px_ha, 2),
            "again_ha": round(float((now & seen_before).sum()) * px_ha, 2),
            "recovered_ha": round(float(((flagged & recovered & b) != 0).sum()) * px_ha, 2),
        }
    return out


def ring_area(ring) -> float:
    xs = np.asarray([p[0] for p in ring])
    ys = np.asarray([p[1] for p in ring])
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def polygon_area(geom) -> float:
    rings = geom["coordinates"]
    return ring_area(rings[0]) - sum(ring_area(r) for r in rings[1:])


def round_coords(obj, nd=6):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), nd) for v in obj]
        return [round_coords(o, nd) for o in obj]
    return obj


def params_hash(region: dict) -> str:
    """Changes when anything that affects a region's results changes. Olive regions
    without a crop mask keep the hash they had before profiles existed."""
    blob = {"params": PARAMS, "bbox": region["bbox"]}
    name = region.get("profile", "olive")
    if name != "olive" or region.get("mask"):
        blob["profile"] = {"name": name, "rules": profile_rules(name), "mask": region.get("mask")}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=list).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Geospatial I/O (needs the conda environment from the workflow)
# --------------------------------------------------------------------------
def region_geobox(bbox, resolution: float = PIXEL_M):
    """Grid in the area's UTM zone, identical for every season."""
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import box
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    bb = box(*bbox, "EPSG:4326").to_crs(f"EPSG:{epsg}").boundingbox
    return GeoBox.from_bbox(bb, resolution=resolution)


def open_catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _processing_baseline(item) -> float:
    v = item.properties.get("s2:processing_baseline")
    try:
        return float(v)
    except (TypeError, ValueError):
        when = item.datetime.date() if item.datetime else dt.date(2000, 1, 1)
        return 4.0 if when >= dt.date(2022, 1, 25) else 2.0


def load_season(catalog, gbox, bbox, year: int, resampling=None, window=SEASON):
    """Median OSAVI over the season window of one year (NaN where too few clear
    dates). `resampling` is passed to odc.stac.load (the basin screen uses
    'average' for the bands so a 100 m cell is the mean of what it covers)."""
    import odc.stac
    import xarray as xr

    items = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=f"{year}-{window[0]}/{year}-{window[1]}",
        query={"eo:cloud_cover": {"lt": SCENE_CLOUD_MAX}},
    ).item_collection()
    if len(items) == 0:
        return None, 0

    groups: dict[int, list] = {}
    for it in items:
        off = REFLECTANCE_OFFSET if _processing_baseline(it) >= 4.0 else 0
        groups.setdefault(off, []).append(it)

    parts = []
    for off, its in groups.items():
        extra = {"resampling": resampling} if resampling else {}
        ds = odc.stac.load(its, bands=["B04", "B08", "SCL"], geobox=gbox,
                           groupby="solar_day", chunks={"x": 2048, "y": 2048}, **extra)
        red = ((ds["B04"].astype("float32") - off) / 10000).clip(min=0)
        nir = ((ds["B08"].astype("float32") - off) / 10000).clip(min=0)
        valid = ds["SCL"].isin(list(VALID_SCL)) & (ds["B04"] > 0) & (ds["B08"] > 0)
        parts.append((1.16 * (nir - red) / (nir + red + 0.16)).where(valid))
    osavi = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")
    n_valid = osavi.notnull().sum("time")
    med = osavi.median("time", skipna=True).where(n_valid >= MIN_DATES)
    return med.compute().values.astype("float32"), int(osavi.sizes["time"])


def burned_mask(catalog, gbox, bbox, season: int, first_year: int | None = None, window=SEASON):
    """Cells mapped as burned from the start of the baseline summer to the end of
    this one (MODIS MCD64A1, 500 m). Wildfire scars look like sudden decline, so
    they are excluded. The window starts with the baseline summer itself: a fire
    in mid-August barely moves that summer's median but shows fully a year later.
    With a multi-year baseline, `first_year` is the earliest baseline summer."""
    import odc.stac
    first_year = season - 1 if first_year is None else first_year
    items = catalog.search(collections=[BURNED_COLLECTION], bbox=bbox,
                           datetime=f"{first_year}-{window[0]}/{season}-{window[1]}").item_collection()
    if len(items) == 0:
        return np.zeros(tuple(gbox.shape), bool)
    ds = odc.stac.load(items, bands=["Burn_Date"], geobox=gbox, resampling="nearest")
    return (ds["Burn_Date"] > 0).any("time").values


def load_burned(catalog, gbox, bbox, seasons, baseline_years: int = 1, window=SEASON):
    """Burned-cell masks for each season, plus a one-line description."""
    out, failed = {}, []
    for s in seasons:
        try:
            out[s] = burned_mask(catalog, gbox, bbox, s, first_year=s - baseline_years, window=window)
        except Exception as exc:
            out[s] = None
            failed.append(f"{s} ({exc.__class__.__name__})")
    if failed:
        return out, "MODIS burned area unavailable for " + ", ".join(failed)
    burnt = sum(int(m.sum()) for m in out.values() if m is not None)
    return out, f"MODIS burned area (MCD64A1): {burnt} cell-seasons excluded"


def build_landmask(catalog, gbox, bbox, region_id: str, builtup_buffer_px: int = BUILTUP_BUFFER_PX,
                   resampling: str = "nearest", keep_classes=KEEP_WORLDCOVER):
    """Cells allowed to be judged, plus a description for status.json."""
    from scipy import ndimage
    shape = tuple(gbox.shape)
    mask = np.ones(shape, bool)
    info = {"worldcover": "not used", "crop_mask": None}
    try:
        import odc.stac
        items = catalog.search(collections=["esa-worldcover"], bbox=bbox,
                               datetime="2021-01-01/2021-12-31").item_collection()
        if len(items):
            wc = odc.stac.load(items, bands=["map"], geobox=gbox,
                               resampling=resampling)["map"].max("time").values
            built = wc == BUILTUP_CLASS
            if builtup_buffer_px > 0:
                built = ndimage.binary_dilation(built, iterations=builtup_buffer_px)
            mask &= np.isin(wc, keep_classes) & ~built
            info["worldcover"] = "ESA WorldCover 2021"
            info["worldcover_share_kept"] = round(float(mask.mean()), 3)
        else:
            info["worldcover"] = "no WorldCover tile found"
    except Exception as exc:  # the scan still runs, just less filtered
        info["worldcover"] = f"failed: {exc.__class__.__name__}"

    olive_file = MASK_DIR / f"{region_id}.geojson"
    if olive_file.exists():
        from rasterio import features
        from rasterio.warp import transform_geom
        fc = json.loads(olive_file.read_text())
        crs = str(gbox.crs)
        shapes = [(transform_geom("EPSG:4326", crs, f["geometry"]), 1)
                  for f in fc.get("features", []) if f.get("geometry")]
        if shapes:
            inside = features.rasterize(shapes, out_shape=shape, transform=gbox.transform,
                                        fill=0, dtype="uint8") == 1
            mask &= inside
            info["crop_mask"] = olive_file.name
    return mask, info


def vectorize(status, first, latest: int, transform, crs: str) -> list:
    """Polygons in EPSG:4326 with status, first season, area and shape."""
    from rasterio import features
    from rasterio.warp import transform_geom
    from scipy import ndimage

    any_ = status > 0
    comp, ncomp = ndimage.label(any_, structure=np.ones((3, 3), bool))
    if ncomp == 0:
        return []
    comp_px = np.bincount(comp.ravel(), minlength=ncomp + 1)
    slices = ndimage.find_objects(comp)
    key = np.where(any_, comp.astype(np.int64) * 4 + status, 0).astype(np.int32)
    keys = np.unique(key[key > 0])
    first_min = ndimage.minimum(np.where(any_, first, 9999), labels=key, index=keys)
    first_by_key = {int(k): int(v) for k, v in zip(keys, np.atleast_1d(first_min))}

    feats = []
    for geom, val in features.shapes(key, mask=key > 0, transform=transform):
        k = int(val)
        cid, code = divmod(k, 4)
        rows, cols = slices[cid - 1]
        bbox_px = (rows.stop - rows.start) * (cols.stop - cols.start)
        patch_m2 = int(comp_px[cid]) * PIXEL_M ** 2
        extent = comp_px[cid] / bbox_px
        shape = "parcel-shaped" if (patch_m2 >= PARCEL_MIN_M2 and extent >= PARCEL_EXTENT) else "irregular"
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": round_coords(transform_geom(crs, "EPSG:4326", geom)["coordinates"])},
            "properties": {
                "status": STATUS_NAMES[code],
                "first_season": first_by_key[k],
                "latest_season": latest,
                "area_m2": int(round(polygon_area(geom))),
                "patch_area_m2": patch_m2,
                "extent": round(float(extent), 2),
                "shape": shape,
            },
        })
    return feats


# --------------------------------------------------------------------------
# State and outputs
# --------------------------------------------------------------------------
def load_state(path: Path):
    if not path.exists():
        return None
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        return {"flagged": z["flagged"].copy(), "recovered": z["recovered"].copy(), **meta}


def load_meta(path: Path):
    """Only the metadata of a state file (the history arrays are not read)."""
    if not path.exists():
        return None
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def is_final(region: dict, meta: dict | None, today: dt.date) -> bool:
    """True when the region's latest season is already final under the current
    parameters, so a scan would only update last_checked. Needs numpy only, so
    the weekly workflow can decide without installing the geospatial packages.
    The grid is not rebuilt here: bbox is in the hash, and the pixel size is checked."""
    if not meta:
        return False
    latest, _ = latest_season(today, tuple(profile_rules(region.get("profile"))["season"]))
    return (meta.get("params_hash") == params_hash(region)
            and abs(float((meta.get("transform") or [0])[0])) == PIXEL_M
            and meta.get("complete_through", 0) >= latest
            and (DATA_DIR / region["id"] / "status.json").exists())


def mark_checked(region: dict, today: dt.date) -> dict:
    out = DATA_DIR / region["id"]
    status = json.loads((out / "status.json").read_text())
    status["last_checked"] = today.isoformat()
    write_json(out / "status.json", status)
    return status


def save_state(path: Path, flagged, recovered, meta: dict) -> None:
    np.savez_compressed(path, flagged=flagged, recovered=recovered, meta=np.array(json.dumps(meta)))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def scan_region(region: dict, today: dt.date) -> dict:
    out = DATA_DIR / region["id"]
    out.mkdir(parents=True, exist_ok=True)
    profile = region.get("profile", "olive")
    rules = profile_rules(profile)
    window = tuple(rules["season"])
    latest, complete = latest_season(today, window)
    gbox = region_geobox(region["bbox"])
    shape = list(gbox.shape)
    phash = params_hash(region)

    state = load_state(out / "state.npz")
    reuse = bool(state) and state.get("params_hash") == phash and state.get("shape") == shape
    if state and not reuse:
        print("Parameters or area changed: rebuilding the full history.")

    if reuse and is_final(region, state, today):
        print(f"{region['id']}: season {latest} already final, nothing new to scan.")
        return mark_checked(region, today)

    if reuse:
        flagged, recovered = state["flagged"], state["recovered"]
        history = {int(k): v for k, v in state.get("history", {}).items()}
        evaluate = [s for s in (latest - 1, latest) if s >= FIRST_SEASON]
    else:
        flagged = np.zeros(shape, np.uint32)
        recovered = np.zeros(shape, np.uint32)
        history = {}
        evaluate = list(range(FIRST_SEASON, latest + 1))

    # A crop-specific profile must only see fields of that crop: build the parcel
    # mask first, and stop (keeping the previous results) if that isn't possible.
    import crop_masks
    mask_meta = crop_masks.ensure_mask(region)

    catalog = open_catalog()
    landmask, mask_info = build_landmask(catalog, gbox, region["bbox"], region["id"],
                                         keep_classes=rules["keep_worldcover"])
    if mask_meta:
        mask_info.update(mask_meta)

    comps, dates = {}, {}
    for year in range(min(evaluate) - max(2, rules["baseline_years"]), latest + 1):
        print(f"{region['id']}: loading season {year} ({window[0]} to {window[1]}) ...", flush=True)
        try:
            comps[year], dates[year] = load_season(catalog, gbox, region["bbox"], year, window=window)
        except Exception as exc:
            print(f"  failed ({exc.__class__.__name__}: {exc}); season skipped")
            comps[year], dates[year] = None, 0
        if comps[year] is None:
            print("  not enough clear imagery")

    burned, mask_info["burned_area"] = load_burned(catalog, gbox, region["bbox"], evaluate,
                                                   baseline_years=rules["baseline_years"], window=window)
    update_history(comps, evaluate, landmask, flagged, recovered, burned=burned, rules=rules)

    all_seasons = list(range(FIRST_SEASON, latest + 1))
    stats = season_stats(flagged, recovered, all_seasons)
    for s in all_seasons:
        entry = history.get(s, {})
        if s in dates:
            entry["dates"] = dates[s]
            entry["judged"] = comps.get(s) is not None and comps.get(s - 1) is not None
        entry.update(stats[s])
        entry["complete"] = s < latest or complete
        history[s] = entry

    status_r, first = classify(flagged, recovered, latest)
    features = vectorize(status_r, first, latest, gbox.transform, str(gbox.crs))
    write_json(out / "zones.geojson", {"type": "FeatureCollection", "features": features})
    write_json(out / "history.json", [{"season": s, **history[s]} for s in all_seasons])

    base = comps.get(latest - 1)
    share = None
    if base is not None and landmask.any():
        b = base[landmask]
        share = round(float(((b >= rules["baseline_osavi_min"]) & (b <= rules["baseline_osavi_max"])).mean()), 3)

    totals = {name: round(float((status_r == code).sum()) * PX_HA, 2) for code, name in STATUS_NAMES.items()}
    have_latest = comps.get(latest) is not None and comps.get(latest - 1) is not None
    status = {
        "region": region["id"],
        "name": region["name"],
        "scan_date": today.isoformat(),
        "last_checked": today.isoformat(),
        "latest_season": latest,
        "season_complete": complete,
        "season_window": f"{window[0]} to {window[1]}",
        "profile": profile,
        "profile_label": PROFILES[profile]["label"],
        "profile_reliability": PROFILES[profile]["reliability"],
        "profile_notes": PROFILES[profile]["notes"],
        "compared_with": (latest - 1 if rules["baseline_years"] == 1
                          else f"median of {latest - rules['baseline_years']}–{latest - 1}"),
        "result": ("ok" if have_latest else "no usable imagery for the latest comparison"),
        "totals_ha": totals,
        "polygons": len(features),
        "baseline_share_in_range": share,
        "masks": mask_info,
        "parameters": {**PARAMS, **{k: v for k, v in rules.items() if k != "keep_worldcover"},
                       "keep_worldcover": rules["keep_worldcover"]},
    }
    write_json(out / "status.json", status)

    meta = {"params_hash": phash, "shape": shape, "crs": str(gbox.crs),
            "transform": list(gbox.transform)[:6],
            "complete_through": latest if complete else latest - 1,
            "history": {str(k): v for k, v in history.items()}}
    save_state(out / "state.npz", flagged, recovered, meta)
    print(f"{region['id']}: {totals} (ha), {len(features)} polygons")
    return status


def load_regions() -> list:
    return json.loads(REGIONS_FILE.read_text())["regions"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--region", help="region id from regions.json")
    g.add_argument("--all", action="store_true", help="scan every enabled region")
    g.add_argument("--plan", action="store_true",
                   help="print the enabled regions that need a scan as a JSON list, and update "
                        "last_checked of the others (numpy only; used by the weekly workflow)")
    ap.add_argument("--today", help="pretend date (YYYY-MM-DD), for testing")
    args = ap.parse_args(argv)

    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    regions = load_regions()
    if args.plan:
        pending = []
        for r in regions:
            if not r.get("enabled", True):
                continue
            if is_final(r, load_meta(DATA_DIR / r["id"] / "state.npz"), today):
                mark_checked(r, today)
                print(f"{r['id']}: already final, skipped", file=sys.stderr)
            else:
                pending.append(r["id"])
        print(json.dumps(pending))
        return 0
    if args.all:
        todo = [r for r in regions if r.get("enabled", True)]
    else:
        todo = [r for r in regions if r["id"] == args.region]
        if not todo:
            print(f"Unknown region {args.region!r}", file=sys.stderr)
            return 2
    for region in todo:
        scan_region(region, today)
    return 0


if __name__ == "__main__":
    sys.exit(main())
