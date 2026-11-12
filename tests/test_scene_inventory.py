"""Scene inventory on synthetic catalogue answers (no network)."""
import datetime as dt
import json

import pytest

import scene_inventory as si

BBOX = [17.18, 40.93, 17.30, 41.00]
UTC = dt.timezone.utc


def scene(when, cloud=5.0, box=None, sid="s"):
    return {"id": sid, "time": dt.datetime.fromisoformat(when).replace(tzinfo=UTC), "cloud": cloud,
            "polys": [si.bbox_polygon(*(box or (17.0, 40.8, 17.5, 41.1)))]}


def test_coverage_full_half_none_and_hole():
    assert si.coverage(BBOX, [si.bbox_polygon(17.0, 40.8, 17.5, 41.1)]) == 1.0
    assert si.coverage(BBOX, [si.bbox_polygon(17.0, 40.8, 17.245, 41.1)]) == pytest.approx(6 / 11, abs=1e-3)
    assert si.coverage(BBOX, [si.bbox_polygon(18.0, 40.8, 18.5, 41.1)]) == 0.0
    holed = [si.bbox_polygon(17.0, 40.8, 17.5, 41.1)[0], si.bbox_polygon(17.0, 40.8, 17.245, 41.1)[0][::-1]]
    assert si.coverage(BBOX, [holed]) == pytest.approx(5 / 11, abs=1e-3)


def test_polygons_of_multipolygon():
    mp = {"type": "MultiPolygon", "coordinates": [si.bbox_polygon(17.0, 40.8, 17.245, 41.1),
                                                   si.bbox_polygon(17.245, 40.8, 17.5, 41.1)]}
    assert si.coverage(BBOX, si.polygons_of(mp)) == 1.0


def test_local_solar_hour():
    t = dt.datetime(2021, 7, 1, 9, 30, tzinfo=UTC)
    assert si.local_solar_hour(t, 15.0) == pytest.approx(10.5)


def test_adjacent_tiles_form_one_overpass_with_union_cover_and_worst_cloud():
    west = scene("2021-07-14T09:40:00", cloud=10, box=(17.0, 40.8, 17.245, 41.1), sid="a")
    east = scene("2021-07-14T09:40:30", cloud=20, box=(17.245, 40.8, 17.5, 41.1), sid="b")
    later = scene("2021-07-30T09:40:00", sid="c")
    groups = si.group_overpasses([later, east, west], BBOX)
    assert len(groups) == 2
    assert groups[0]["ids"] == ["a", "b"] and groups[0]["cover"] == 1.0 and groups[0]["cloud"] == 20


def test_usable_cloud_cover_and_midday():
    thermal, hyper = si.SENSORS["landsat"], si.SENSORS["enmap"]
    lon = 17.24
    noon = {"time": dt.datetime(2021, 7, 14, 10, 0, tzinfo=UTC), "cover": 1.0, "cloud": 5.0}
    assert si.usable(noon, thermal, lon)
    assert not si.usable({**noon, "cloud": 60.0}, thermal, lon)
    assert si.usable({**noon, "cloud": None}, thermal, lon)          # unknown counts as clear
    assert not si.usable({**noon, "cover": 0.3}, hyper, lon)
    dawn = {**noon, "time": dt.datetime(2021, 7, 14, 4, 0, tzinfo=UTC)}
    assert not si.usable(dawn, thermal, lon) and si.usable(dawn, hyper, lon)


def test_summarise_verdict_seasons_and_pairs():
    sensor = si.SENSORS["enmap"]
    ov = si.group_overpasses([scene("2022-07-10T10:00:00"), scene("2023-08-02T10:00:00", sid="b"),
                              scene("2023-09-15T10:00:00", sid="out of window"),
                              scene("2021-07-10T10:00:00", cloud=80, sid="cloudy")], BBOX)
    res = si.summarise(ov, sensor, BBOX, [2020, 2021, 2022, 2023])
    assert res["seasons"]["2021"] == {"overpasses": 1, "covering": 1, "usable": 0, "dates": []}
    assert res["seasons_with_enough"] == [2022, 2023] and res["consecutive_pairs"] == [[2022, 2023]]
    assert res["verdict"] == "go"
    thermal = si.summarise(ov, si.SENSORS["ecostress"], BBOX, [2020, 2021, 2022, 2023])
    assert thermal["verdict"] == "no-go"                              # needs MIN_DATES per season


def test_read_prisma_csv(tmp_path):
    f = tmp_path / "prisma.csv"
    f.write_text("id,datetime,cloud_cover,west,south,east,north\n"
                 "p1,2021-07-14T09:52:31Z,3.5,17.0,40.8,17.5,41.1\n"
                 "p2,2022-08-01T09:50:00Z,,,,,\n", encoding="utf-8")
    a, b = si.read_prisma_csv(f)
    assert a["id"] == "p1" and a["cloud"] == 3.5 and a["time"].hour == 9 and a["polys"]
    assert b["cloud"] is None and b["polys"] == []
    assert si.group_overpasses([b], BBOX)[0]["cover"] == 1.0          # no footprint: assumed to cover


def cmr_item(when, flag="Day"):
    return {"umm": {"GranuleUR": f"ECOv002_L2T_LSTE_{when}", "DataGranule": {"DayNightFlag": flag},
                    "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": when}},
                    "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [
                        {"WestBoundingCoordinate": 16.18, "SouthBoundingCoordinate": 40.54,
                         "EastBoundingCoordinate": 17.51, "NorthBoundingCoordinate": 41.55}]}}}}}


def test_cmr_scene_skips_night_and_reads_tile_box():
    assert si.cmr_scene(cmr_item("2021-07-01T21:13:25.108Z", "Night")) is None
    s = si.cmr_scene(cmr_item("2021-07-01T07:13:25.108Z"))
    assert s["cloud"] is None and si.coverage(BBOX, s["polys"]) == 1.0


def test_stac_search_follows_post_next_links(monkeypatch):
    calls = []
    pages = [{"features": [{"id": "1"}], "links": [{"rel": "next", "href": "https://x/search", "method": "POST",
                                                    "body": {"token": "next:1"}, "merge": True}]},
             {"features": [{"id": "2"}], "links": []}]
    monkeypatch.setattr(si, "http_json", lambda url, body=None: calls.append(body) or pages[len(calls) - 1])
    feats = si.stac_search("https://x", "c", BBOX, "2021-07-01", "2021-08-31")
    assert [f["id"] for f in feats] == ["1", "2"]
    assert calls[1]["token"] == "next:1" and calls[1]["collections"] == ["c"]


def test_cmr_search_refuses_partial_answer(monkeypatch):
    monkeypatch.setattr(si, "http_json", lambda url, body=None: {"hits": 3, "items": [{}]})
    with pytest.raises(RuntimeError):
        si.cmr_search("ECO_L2T_LSTE", "002", BBOX, "2021-07-01", "2021-08-31")


def test_main_writes_report_and_survives_a_failing_catalogue(tmp_path, monkeypatch):
    def fake_fetch(key, sensor, bbox, years, window=None, prisma_csv=None):
        if key == "prisma":
            raise LookupError("not checked")
        if key == "enmap":
            raise OSError("HTTP Error 502")
        return [scene(f"{y}-07-{d:02d}T09:40:00", sid=f"{key}{y}{d}") for y in years for d in (5, 15, 25)]
    monkeypatch.setattr(si, "fetch_scenes", fake_fetch)
    out = tmp_path / "inv.json"
    assert si.main(["--region", "puglia-monopoli-fasano", "--years", "2020-2023", "--out", str(out)]) == 0
    r = json.loads(out.read_text())["regions"]["puglia-monopoli-fasano"]
    assert r["prisma"]["verdict"] == "not checked"
    assert r["enmap"]["verdict"] == "unknown" and "502" in r["enmap"]["error"]
    assert r["landsat"]["verdict"] == "go" and r["landsat"]["seasons"]["2021"]["usable"] == 3
    assert "verdict" not in r["sentinel2"]


def test_main_rejects_unknown_region(tmp_path):
    assert si.main(["--region", "nowhere", "--out", str(tmp_path / "x.json")]) == 2
