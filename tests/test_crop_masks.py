"""Crop-mask download logic with a fake parcel service."""
import json

import pytest

import crop_masks as cm

FIELD = cm.SOURCES["sigpac-ib"]["use_field"]


def parcel(i, use):
    x = 3.0 + i * 0.001
    return {"type": "Feature", "properties": {FIELD: use},
            "geometry": {"type": "Polygon", "coordinates": [[[x, 39.5], [x + 0.0005, 39.5],
                                                             [x + 0.0005, 39.5005], [x, 39.5]]]}}


class FakeService:
    """Pages of `page_size`, optional refusal of the server-side use filter."""

    def __init__(self, parcels, refuse_filter=False, page_size=2000):
        self.parcels, self.refuse, self.page = parcels, refuse_filter, page_size
        self.calls = []

    def __call__(self, url, params):
        self.calls.append(params)
        where = params["where"]
        if where != "1=1" and self.refuse:
            raise cm.MaskError("service error: invalid where")
        rows = self.parcels if where == "1=1" else [
            p for p in self.parcels if f"'{p['properties'][FIELD]}'" in where]
        off = params["resultOffset"]
        chunk = rows[off:off + self.page]
        return {"type": "FeatureCollection", "features": chunk,
                "properties": {"exceededTransferLimit": off + self.page < len(rows)}}


@pytest.fixture
def parcels():
    uses = ["FS", "TA", "OV", "FS", "FL", "PS", "FV", "FS"] * 600     # 4,800 parcels
    return [parcel(i, u) for i, u in enumerate(uses)]


def test_server_filter_with_pagination(parcels, monkeypatch):
    monkeypatch.setattr(cm, "SOURCES", {**cm.SOURCES, "sigpac-ib": {**cm.SOURCES["sigpac-ib"], "page_size": 1000}})
    svc = FakeService(parcels, page_size=1000)
    got = cm.fetch_parcels("sigpac-ib", [2.9, 39.4, 3.1, 39.6], ["FS", "FL", "FV"], fetch=svc)
    assert len(got) == 600 * 5
    assert {f["properties"]["use"] for f in got} == {"FS", "FL", "FV"}
    # three full pages, plus at most one empty check when a page comes back exactly full
    assert len(svc.calls) in (3, 4) and all(c["where"].startswith(f"{FIELD} IN") for c in svc.calls)
    assert [c["resultOffset"] for c in svc.calls][:3] == [0, 1000, 2000]


def test_falls_back_to_local_filter(parcels):
    svc = FakeService(parcels, refuse_filter=True)
    got = cm.fetch_parcels("sigpac-ib", [2.9, 39.4, 3.1, 39.6], ["FS"], fetch=svc)
    assert len(got) == 600 * 3
    assert svc.calls[-1]["where"] == "1=1"


def test_ensure_mask_writes_and_reuses(parcels, tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MASK_DIR", tmp_path)
    region = {"id": "test", "bbox": [2.9, 39.4, 3.1, 39.6], "mask": {"type": "sigpac-ib", "uses": ["FS"]}}
    svc = FakeService(parcels)
    info = cm.ensure_mask(region, fetch=svc)
    assert info["parcels"] == 1800 and (tmp_path / "test.geojson").exists()
    fc = json.loads((tmp_path / "test.geojson").read_text())
    assert fc["uses"] == ["FS"] and fc["built"]
    n = len(svc.calls)
    assert cm.ensure_mask(region, fetch=svc)["parcels"] == 1800 and len(svc.calls) == n   # cached


def test_no_mask_spec_is_fine():
    assert cm.ensure_mask({"id": "olive", "bbox": [0, 0, 1, 1]}) is None


def test_empty_or_unreachable_mask_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MASK_DIR", tmp_path)
    region = {"id": "t", "bbox": [0, 0, 1, 1], "mask": {"type": "sigpac-ib", "uses": ["FS"]}}
    with pytest.raises(cm.MaskError):
        cm.ensure_mask(region, fetch=FakeService([parcel(0, "TA")]))

    def down(url, params):
        raise cm.MaskError("could not reach")
    with pytest.raises(cm.MaskError):
        cm.ensure_mask(region, fetch=down)
    with pytest.raises(cm.MaskError):
        cm.ensure_mask({**region, "mask": {"type": "nowhere", "uses": ["FS"]}})
