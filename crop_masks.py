#!/usr/bin/env python3
"""
Crop masks from public farm-parcel registers, so a crop profile is applied only
to fields of that crop.

ESA WorldCover can't tell olive groves from almond orchards or vineyards. The
parcel registers that EU farm subsidies are based on can: every parcel carries a
land-use code. This module downloads the parcels of the requested uses inside a
region's bounding box and stores them as masks/<region-id>.geojson, which
xylella_scanner.py then rasterises.

Supported sources (add more in SOURCES):
    sigpac-ib   Spanish SIGPAC, Balearic Islands (IDEIB ArcGIS service, open data)
                Uses: FS dry-fruit orchards (mostly almond on Mallorca), FL dry fruit
                with olive, FV dry fruit with vineyard, OV olive, VI vineyard ...

Usage
    python crop_masks.py --region mallorca-pla          # build if missing
    python crop_masks.py --region mallorca-pla --force  # rebuild

Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MASK_DIR = ROOT / "masks"
USER_AGENT = "xylella-watch/1.0 (crop masks; github.com)"

SOURCES = {
    "sigpac-ib": {
        "url": "https://ideib.caib.es/geoserveis/rest/services/public/GOIB_SIGPAC_IB/MapServer/1/query",
        # the layer is a joined view, so field names carry the table prefix
        "use_field": "SIGPAC_FOGAIBA.USO_SIGPAC",
        "page_size": 2000,
        "title": "SIGPAC Illes Balears (Govern de les Illes Balears, IDEIB)",
        "source_url": "https://ideib.caib.es/geoserveis/rest/services/public/GOIB_SIGPAC_IB/MapServer",
    },
}


class MaskError(RuntimeError):
    pass


def _get(url: str, params: dict, tries: int = 3) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict) and "error" in data:
                raise MaskError(f"service error: {data['error']}")
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(5 * (i + 1))
    raise MaskError(f"could not reach {url}: {last}")


def _exceeded(page: dict) -> bool:
    return bool(page.get("exceededTransferLimit") or (page.get("properties") or {}).get("exceededTransferLimit"))


def fetch_parcels(source: str, bbox, uses, fetch=_get) -> list:
    """GeoJSON features (EPSG:4326) of parcels with one of `uses` that touch bbox.
    Asks the server to filter by use; if it refuses that query, downloads every
    parcel in the box and filters locally."""
    cfg = SOURCES[source]
    field = cfg["use_field"]
    uses = [u.upper() for u in uses]
    base = {
        "geometry": ",".join(str(v) for v in bbox), "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outSR": 4326,
        "outFields": field, "returnGeometry": "true", "geometryPrecision": 6, "f": "geojson",
        "resultRecordCount": cfg["page_size"],
    }
    quoted = ",".join(f"'{u}'" for u in uses)
    for where in (f"{field} IN ({quoted})", "1=1"):
        try:
            feats, offset = [], 0
            while True:
                page = fetch(cfg["url"], {**base, "where": where, "resultOffset": offset})
                got = page.get("features", [])
                feats.extend(got)
                if not got or not (_exceeded(page) or len(got) >= cfg["page_size"]):
                    break
                offset += len(got)
                if offset > 500_000:
                    raise MaskError("too many parcels; make the region smaller")
            break
        except MaskError:
            if where == "1=1":
                raise
            continue                      # server refused the filter: filter locally
    keep = []
    for f in feats:
        props = f.get("properties") or {}
        use = str(props.get(field) or props.get(field.lower()) or "").upper()
        if use in uses and f.get("geometry"):
            keep.append({"type": "Feature", "geometry": f["geometry"], "properties": {"use": use}})
    return keep


def mask_path(region: dict) -> Path:
    return MASK_DIR / f"{region['id']}.geojson"


def ensure_mask(region: dict, force: bool = False, fetch=_get) -> dict | None:
    """Make sure the region's crop mask exists. Returns a short description for
    status.json, or None if the region has no mask spec. Raises MaskError if a
    mask is required but can't be built: scanning the wrong fields with a
    crop-specific profile would produce misleading results."""
    spec = region.get("mask")
    if not spec:
        return None
    path = mask_path(region)
    if path.exists() and not force:
        fc = json.loads(path.read_text())
        return {"crop_mask": path.name, "parcels": len(fc.get("features", [])),
                "built": fc.get("built"), "source": fc.get("source_title")}
    if spec.get("type") not in SOURCES:
        raise MaskError(f"unknown mask source {spec.get('type')!r}")
    feats = fetch_parcels(spec["type"], region["bbox"], spec["uses"], fetch=fetch)
    if not feats:
        raise MaskError(f"no parcels with uses {spec['uses']} found in {region['id']}")
    cfg = SOURCES[spec["type"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "source_title": cfg["title"], "source_url": cfg["source_url"],
          "uses": spec["uses"], "built": dt.date.today().isoformat(), "features": feats}
    path.write_text(json.dumps(fc, separators=(",", ":")) + "\n")
    return {"crop_mask": path.name, "parcels": len(feats), "built": fc["built"], "source": cfg["title"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build crop masks from parcel registers")
    ap.add_argument("--region", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    regions = json.loads((ROOT / "regions.json").read_text())["regions"]
    region = next((r for r in regions if r["id"] == a.region), None)
    if region is None:
        print(f"Unknown region {a.region!r}", file=sys.stderr)
        return 2
    info = ensure_mask(region, force=a.force)
    print(json.dumps(info or {"crop_mask": None}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
