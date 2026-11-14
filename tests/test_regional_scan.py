"""scan_region end to end, with the satellite catalogue replaced by synthetic composites."""
import datetime as dt
import json
import types

import numpy as np
import pytest

import crop_masks
import xylella_scanner as xs

H, W = 40, 50


@pytest.fixture
def env(tmp_path, monkeypatch, rng):
    world = {y: (0.25 + rng.normal(0, 0.004, (H, W))).astype("float32") for y in range(2014, 2029)}
    for y in range(2022, 2029):
        world[y][10:15, 10:16] = 0.10               # decline from 2022
    world[2026][25:30, 30:36] = 0.11                # new in 2026
    loads = []
    gbox = types.SimpleNamespace(shape=(H, W), crs="EPSG:32633",
                                 transform=(10.0, 0.0, 700000.0, 0.0, -10.0, 4520000.0))
    monkeypatch.setattr(xs, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(xs, "region_geobox", lambda bbox, resolution=10: gbox)
    monkeypatch.setattr(xs, "open_catalog", lambda: None)
    monkeypatch.setattr(xs, "build_landmask",
                        lambda c, g, b, rid, **kw: (np.ones((H, W), bool), {"worldcover": "stub", **kw}))
    monkeypatch.setattr(xs, "load_burned",
                        lambda c, g, b, seasons, **kw: ({s: None for s in seasons}, "stub"))

    def load(c, g, b, year, resampling=None, window=xs.SEASON):
        loads.append((year, tuple(window)))
        return world[year].copy(), 12

    monkeypatch.setattr(xs, "load_season", load)
    monkeypatch.setattr(crop_masks, "ensure_mask", lambda region, **kw: (
        {"crop_mask": "x.geojson", "parcels": 42, "source": "stub"} if region.get("mask") else None))
    return types.SimpleNamespace(world=world, loads=loads, out=tmp_path / "data")


REGION = {"id": "test-region", "name": "Test", "bbox": [17.55, 40.75, 17.65, 40.80]}


def test_full_build_then_reuse(env):
    st = xs.scan_region(REGION, dt.date(2026, 10, 21))
    assert sorted({y for y, _ in env.loads}) == list(range(2017, 2027))
    assert st["totals_ha"] == {"new": 0.3, "continuing": 0.0, "earlier": 0.3}
    assert st["profile"] == "olive" and st["compared_with"] == 2025
    zones = json.loads((env.out / "test-region/zones.geojson").read_text())
    by = {}
    for f in zones["features"]:
        by[f["properties"]["status"]] = by.get(f["properties"]["status"], 0) + f["properties"]["area_m2"]
    assert by == {"earlier": 3000, "new": 3000}
    env.loads.clear()
    xs.scan_region(REGION, dt.date(2026, 10, 28))
    assert env.loads == []                           # season final, nothing reloaded


def test_mid_season_update_and_rebuild_on_change(env, monkeypatch):
    xs.scan_region(REGION, dt.date(2026, 10, 21))
    env.world[2027] = env.world[2026].copy()
    env.world[2027][2:7, 40:46] = 0.10
    env.loads.clear()
    st = xs.scan_region(REGION, dt.date(2027, 8, 1))
    assert sorted({y for y, _ in env.loads}) == [2024, 2025, 2026, 2027]
    assert not st["season_complete"] and st["totals_ha"]["new"] == 0.3
    monkeypatch.setitem(xs.PARAMS, "min_rel_drop", 0.31)
    env.loads.clear()
    xs.scan_region(REGION, dt.date(2027, 8, 8))
    assert min(y for y, _ in env.loads) == 2017      # parameter change: full rebuild


def test_almond_region_uses_profile_and_mask(env):
    region = {**REGION, "id": "almond", "profile": "almond", "mask": {"type": "sigpac-ib", "uses": ["FS"]}}
    st = xs.scan_region(region, dt.date(2026, 10, 21))
    assert {w for _, w in env.loads} == {("08-01", "09-30")}
    assert min(y for y, _ in env.loads) == 2016      # three-summer baseline
    assert st["profile_reliability"] == "inferred"
    assert st["masks"]["parcels"] == 42 and st["masks"]["keep_classes"] == (10, 20, 30, 40)
    assert st["compared_with"] == "median of 2023–2025"


def test_missing_mask_stops_scan_and_keeps_results(env, monkeypatch):
    region = {**REGION, "id": "almond", "profile": "almond", "mask": {"type": "sigpac-ib", "uses": ["FS"]}}

    def fail(region, **kw):
        raise crop_masks.MaskError("could not reach")

    monkeypatch.setattr(crop_masks, "ensure_mask", fail)
    with pytest.raises(crop_masks.MaskError):
        xs.scan_region(region, dt.date(2026, 10, 21))
    assert not (env.out / "almond" / "zones.geojson").exists()


def test_plan_skips_final_regions_without_loading(env, monkeypatch, capsys):
    xs.scan_region(REGION, dt.date(2026, 10, 21))
    meta = xs.load_meta(env.out / "test-region" / "state.npz")
    assert xs.is_final(REGION, meta, dt.date(2026, 11, 9))
    assert not xs.is_final(REGION, meta, dt.date(2027, 7, 20))     # next season is under way
    assert not xs.is_final({**REGION, "bbox": [17.5, 40.7, 17.6, 40.8]}, meta, dt.date(2026, 11, 9))
    assert not xs.is_final(REGION, None, dt.date(2026, 11, 9))

    monkeypatch.setattr(xs, "load_regions", lambda: [REGION, {**REGION, "id": "unscanned"},
                                                     {**REGION, "id": "off", "enabled": False}])
    env.loads.clear()
    capsys.readouterr()                              # drop the scan's own log lines
    assert xs.main(["--plan", "--today", "2026-11-09"]) == 0
    assert json.loads(capsys.readouterr().out) == ["unscanned"]
    assert env.loads == []
    status = json.loads((env.out / "test-region" / "status.json").read_text())
    assert status["last_checked"] == "2026-11-09" and status["scan_date"] == "2026-10-21"
