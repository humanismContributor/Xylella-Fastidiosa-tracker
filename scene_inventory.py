#!/usr/bin/env python3
"""
Scene inventory for the hyperspectral and thermal calibration test (phase 0, prototype).

Before any hyperspectral or thermal imagery is extracted at the sampled trees,
this script checks whether there is enough of it: it counts, per region, sensor
and summer, the overpasses that cover the area, are clear enough and (for
thermal) fall around midday. It reads catalogue metadata only, downloads no
imagery and needs no account.

    sensor       pixel  catalogue                                   role
    PRISMA       30 m   none public: a list exported from prisma.asi.it  hyperspectral
    EnMAP        30 m   DLR EOC Geoservice STAC (from 2022)         hyperspectral
    ECOSTRESS    70 m   NASA CMR (tiled LST, V002)                  thermal
    Landsat 8/9  100 m  Microsoft Planetary Computer (C2 L2)        thermal
    Sentinel-2   10 m   Microsoft Planetary Computer                reference only

The stop rule from the experiment scope: a sensor goes ahead only if at least
MIN_SEASONS of the sample seasons each have enough usable overpasses (one for
hyperspectral, MIN_DATES for thermal, since a thermal anomaly is a seasonal median).

Usage (standard library only)
    python scene_inventory.py
    python scene_inventory.py --region puglia-monopoli-fasano --years 2019-2023
    python scene_inventory.py --prisma-csv prisma_search.csv

PRISMA list: a CSV with one row per scene and the columns
    datetime      acquisition start, ISO 8601 in UTC (e.g. 2021-07-14T09:52:31Z)
    cloud_cover   scene cloud cover in percent (empty if unknown)
    west,south,east,north   scene bounding box in degrees (optional; empty = assumed to cover)
Without --prisma-csv, PRISMA is reported as not checked.

Output: data/calibration/scene_inventory.json

What metadata can't tell: cloud cover is for the whole scene, not the region,
and ECOSTRESS records carry no cloud cover at all and list the whole 110 km
tile rather than the swath, so its counts are upper bounds. Phase 1 reads the
cloud masks; this only decides whether phase 1 is worth doing.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import xylella_scanner as xs

OUT_FILE = xs.DATA_DIR / "calibration" / "scene_inventory.json"
PC_STAC = xs.STAC_URL
DLR_STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

REGIONS = ("puglia-monopoli-polignano", "puglia-monopoli-fasano")   # the two front regions with field samples
SAMPLE_SEASONS = (2019, 2023)  # Puglia samples 2020 to mid-2023, mapped by calibrate_scanner.sample_season
CLOUD_MAX = xs.SCENE_CLOUD_MAX
MIN_COVER = 0.5              # share of the region's box an overpass must cover
MIN_SEASONS = 2
GROUP_MINUTES = 10           # scenes of one overpass (adjacent rows or tiles) are minutes apart
GRID_N = 11                  # coverage is measured on GRID_N x GRID_N points in the box

SENSORS = {
    "prisma": {"label": "PRISMA", "kind": "hyperspectral", "pixel_m": 30, "min_per_season": 1,
               "local_hours": None, "source": "list exported from prisma.asi.it (--prisma-csv)"},
    "enmap": {"label": "EnMAP", "kind": "hyperspectral", "pixel_m": 30, "min_per_season": 1,
              "local_hours": None, "source": f"{DLR_STAC} collection ENMAP_HSI_L2A",
              # check against {DLR_STAC}/collections if the search stops matching
              "stac": DLR_STAC, "collection": "ENMAP_HSI_L2A"},
    "ecostress": {"label": "ECOSTRESS", "kind": "thermal", "pixel_m": 70, "min_per_season": xs.MIN_DATES,
                  # around midday, when canopy temperature reflects water stress best
                  "local_hours": (10.0, 15.0), "source": "NASA CMR, ECO_L2T_LSTE V002",
                  "cmr": {"short_name": "ECO_L2T_LSTE", "version": "002"}},
    "landsat": {"label": "Landsat 8/9", "kind": "thermal", "pixel_m": 100, "min_per_season": xs.MIN_DATES,
                "local_hours": (10.0, 15.0), "source": "Planetary Computer, landsat-c2-l2",
                "stac": PC_STAC, "collection": "landsat-c2-l2",
                "filter": lambda p: p.get("platform") in ("landsat-8", "landsat-9")},
    "sentinel2": {"label": "Sentinel-2 (reference)", "kind": "reference", "pixel_m": 10,
                  "min_per_season": xs.MIN_DATES, "local_hours": None,
                  "source": "Planetary Computer, sentinel-2-l2a", "stac": PC_STAC, "collection": "sentinel-2-l2a"},
}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def parse_time(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.strip().replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def local_solar_hour(t: dt.datetime, lon: float) -> float:
    return (t.hour + t.minute / 60 + t.second / 3600 + lon / 15) % 24


def bbox_polygon(w, s, e, n) -> list:
    return [[[w, s], [e, s], [e, n], [w, n], [w, s]]]


def polygons_of(geom: dict | None) -> list:
    """List of polygons (each a list of rings) of a GeoJSON Polygon or MultiPolygon."""
    if not geom:
        return []
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return list(geom["coordinates"])
    raise ValueError(f"unsupported geometry {geom['type']}")


def _in_ring(x, y, ring) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def in_polygons(x, y, polys) -> bool:
    return any(_in_ring(x, y, p[0]) and not any(_in_ring(x, y, h) for h in p[1:]) for p in polys)


def coverage(bbox, polys, n: int = GRID_N) -> float:
    """Share of the box (lon/lat) inside the union of the polygons, on an n x n point grid."""
    w, s, e, nn = bbox
    pts = [(w + (e - w) * (i + 0.5) / n, s + (nn - s) * (j + 0.5) / n) for i in range(n) for j in range(n)]
    return round(sum(in_polygons(x, y, polys) for x, y in pts) / len(pts), 3)


def in_season(t: dt.datetime, window=xs.SEASON) -> bool:
    start, end = (dt.date(t.year, *map(int, d.split("-"))) for d in window)
    return start <= t.date() <= end


def group_overpasses(scenes: list, bbox, minutes: int = GROUP_MINUTES) -> list:
    """Merge scenes taken within `minutes` of each other (adjacent rows or tiles
    of one overpass). Coverage is that of the union; cloud cover the worst member's,
    or None when no member reports it."""
    out = []
    for sc in sorted(scenes, key=lambda s: s["time"]):
        if out and (sc["time"] - out[-1]["members"][-1]["time"]).total_seconds() <= minutes * 60:
            out[-1]["members"].append(sc)
        else:
            out.append({"members": [sc]})
    for g in out:
        m = g.pop("members")
        clouds = [s["cloud"] for s in m if s["cloud"] is not None]
        polys = [p for s in m for p in s["polys"]]
        g.update({
            "time": m[0]["time"],
            "ids": [s["id"] for s in m],
            "cloud": max(clouds) if clouds else None,
            "cover": coverage(bbox, polys) if all(s["polys"] for s in m) else 1.0,
        })
    return out


def usable(o: dict, sensor: dict, lon: float, cloud_max: float = CLOUD_MAX) -> bool:
    """Covers enough of the box, clear enough (unknown cloud counts as clear,
    which is why such counts are upper bounds) and, for thermal, near midday."""
    if o["cover"] < MIN_COVER or (o["cloud"] is not None and o["cloud"] > cloud_max):
        return False
    hours = sensor["local_hours"]
    if hours:
        h = local_solar_hour(o["time"], lon)
        return hours[0] <= h <= hours[1]
    return True


def summarise(overpasses: list, sensor: dict, bbox, seasons, window=xs.SEASON) -> dict:
    """Per-season counts and the go / no-go verdict for one sensor and region."""
    lon = (bbox[0] + bbox[2]) / 2
    per = {s: {"overpasses": 0, "covering": 0, "usable": 0, "dates": []} for s in seasons}
    for o in overpasses:
        s = o["time"].year
        if s not in per or not in_season(o["time"], window):
            continue
        per[s]["overpasses"] += 1
        per[s]["covering"] += o["cover"] >= MIN_COVER
        if usable(o, sensor, lon):
            per[s]["usable"] += 1
            per[s]["dates"].append(o["time"].strftime("%Y-%m-%d %H:%MZ"))
    need = sensor["min_per_season"]
    ok = [s for s in seasons if per[s]["usable"] >= need]
    pairs = [s for s in ok if s - 1 in ok]
    return {
        "seasons": {str(s): v for s, v in per.items()},
        "seasons_with_enough": ok,
        "consecutive_pairs": [[s - 1, s] for s in pairs],
        "verdict": "go" if len(ok) >= MIN_SEASONS else "no-go",
    }


def read_prisma_csv(path: Path) -> list:
    """Scenes from a PRISMA catalogue export (columns in the module docstring)."""
    scenes = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            box = [row.get(k, "").strip() for k in ("west", "south", "east", "north")]
            cloud = (row.get("cloud_cover") or "").strip()
            scenes.append({
                "id": row.get("id") or f"prisma-{i}",
                "time": parse_time(row["datetime"]),
                "cloud": float(cloud) if cloud else None,
                "polys": [bbox_polygon(*map(float, box))] if all(box) else [],
            })
    return scenes


def stac_scene(feature: dict) -> dict:
    p = feature.get("properties", {})
    return {"id": feature.get("id"), "time": parse_time(p.get("datetime") or p["start_datetime"]),
            "cloud": p.get("eo:cloud_cover"), "polys": polygons_of(feature.get("geometry"))}


def cmr_scene(item: dict) -> dict | None:
    """One ECOSTRESS granule; None for night passes."""
    u = item["umm"]
    if (u.get("DataGranule", {}).get("DayNightFlag") or "").lower() == "night":
        return None
    rects = u.get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {}) \
             .get("BoundingRectangles", [])
    polys = [bbox_polygon(r["WestBoundingCoordinate"], r["SouthBoundingCoordinate"],
                          r["EastBoundingCoordinate"], r["NorthBoundingCoordinate"]) for r in rects]
    cc = u.get("CloudCover")
    return {"id": u["GranuleUR"], "time": parse_time(u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]),
            "cloud": float(cc) if cc is not None else None, "polys": polys}


# --------------------------------------------------------------------------
# Catalogues (network)
# --------------------------------------------------------------------------
def http_json(url: str, body: dict | None = None) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "xylella-watch scene inventory"}
    data = None
    if body is not None:             # CMR answers 400 to a GET that declares a JSON body
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def stac_search(root: str, collection: str, bbox, start: str, end: str, max_pages: int = 50) -> list:
    """All features of a STAC API search, following 'next' links."""
    url, body = f"{root}/search", {"collections": [collection], "bbox": list(bbox),
                                   "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 250}
    feats = []
    for _ in range(max_pages):
        page = http_json(url, body)
        feats += page.get("features", [])
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            return feats
        url = nxt["href"]
        if nxt.get("method", "GET").upper() == "POST":
            body = {**body, **nxt["body"]} if nxt.get("merge") else nxt.get("body", body)
        else:
            body = None
    raise RuntimeError(f"more than {max_pages} result pages from {root}")


def cmr_search(short_name: str, version: str, bbox, start: str, end: str) -> list:
    q = urllib.parse.urlencode({"short_name": short_name, "version": version, "page_size": 2000,
                                "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
                                "bounding_box": ",".join(map(str, bbox))})
    d = http_json(f"{CMR_GRANULES}?{q}")
    if d.get("hits", 0) > len(d.get("items", [])):
        raise RuntimeError(f"{d['hits']} granules, more than one page; narrow the search")
    return d.get("items", [])


def fetch_scenes(key: str, sensor: dict, bbox, years, window=xs.SEASON, prisma_csv: Path | None = None) -> list:
    if key == "prisma":
        if prisma_csv is None:
            raise LookupError("no catalogue export given (--prisma-csv)")
        return read_prisma_csv(prisma_csv)
    scenes = []
    for y in years:
        start, end = f"{y}-{window[0]}", f"{y}-{window[1]}"
        if "cmr" in sensor:
            items = cmr_search(sensor["cmr"]["short_name"], sensor["cmr"]["version"], bbox, start, end)
            scenes += [s for s in map(cmr_scene, items) if s is not None]
        else:
            feats = stac_search(sensor["stac"], sensor["collection"], bbox, start, end)
            keep = sensor.get("filter") or (lambda p: True)
            scenes += [stac_scene(f) for f in feats if keep(f.get("properties", {}))]
    return scenes


def inventory_region(region: dict, years, prisma_csv: Path | None = None) -> dict:
    bbox = region["bbox"]
    out = {}
    for key, sensor in SENSORS.items():
        print(f"{region['id']}: {sensor['label']} ...", flush=True)
        entry = {"label": sensor["label"], "kind": sensor["kind"], "pixel_m": sensor["pixel_m"],
                 "source": sensor["source"], "needed_per_season": sensor["min_per_season"]}
        try:
            scenes = fetch_scenes(key, sensor, bbox, years, prisma_csv=prisma_csv)
        except LookupError as exc:
            out[key] = {**entry, "verdict": "not checked", "error": str(exc)}
            continue
        except Exception as exc:  # one catalogue down must not stop the others
            out[key] = {**entry, "verdict": "unknown", "error": f"{exc.__class__.__name__}: {exc}"[:300]}
            continue
        entry["scenes"] = len(scenes)
        entry.update(summarise(group_overpasses(scenes, bbox), sensor, bbox, years))
        if sensor["kind"] == "reference":
            entry.pop("verdict")
        out[key] = entry
    return out


def parse_years(text: str) -> list:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--region", action="append", help=f"region id (repeatable; default {', '.join(REGIONS)})")
    ap.add_argument("--years", default=f"{SAMPLE_SEASONS[0]}-{SAMPLE_SEASONS[1]}", help="e.g. 2019-2023")
    ap.add_argument("--prisma-csv", type=Path, help="PRISMA catalogue export (see the module docstring)")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args(argv)

    wanted = args.region or list(REGIONS)
    regions = {r["id"]: r for r in xs.load_regions()}
    unknown = [r for r in wanted if r not in regions]
    if unknown:
        print(f"Unknown region(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    years = parse_years(args.years)
    report = {
        "generated": dt.date.today().isoformat(),
        "season_window": f"{xs.SEASON[0]} to {xs.SEASON[1]}",
        "years": years,
        "rules": {"cloud_max_percent": CLOUD_MAX, "min_cover": MIN_COVER, "min_seasons": MIN_SEASONS,
                  "thermal_local_solar_hours": SENSORS["ecostress"]["local_hours"],
                  "overpass_grouping_minutes": GROUP_MINUTES},
        "caveats": [
            "Cloud cover is per scene, not per region; scenes without cloud metadata count as clear.",
            "ECOSTRESS granules list the whole tile, not the swath, and carry no cloud cover: its counts are upper bounds.",
            "EnMAP launched in April 2022, so it can only cover the 2022 and 2023 sample seasons.",
            "A 'go' only means phase 1 is worth trying; phase 1 reads the cloud masks per pixel.",
        ],
        "regions": {rid: inventory_region(regions[rid], years, args.prisma_csv) for rid in wanted},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    xs.write_json(args.out, report)

    for rid, sensors in report["regions"].items():
        print(f"\n{rid}  (usable overpasses per summer {years[0]}-{years[-1]})")
        for e in sensors.values():
            if "seasons" in e:
                counts = " ".join(f"{e['seasons'][str(y)]['usable']:>3}" for y in years)
                print(f"  {e['label']:24s} {counts}   need {e['needed_per_season']}/season  "
                      f"{e.get('verdict', '')}")
            else:
                print(f"  {e['label']:24s} {e['verdict']}: {e['error']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
