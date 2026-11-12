"""Landsat thermal anomaly helpers, with synthetic scenes instead of imagery."""
import gzip

import numpy as np
import pytest

import sample_osavi as so
import sample_thermal as st

H, W = 6, 8
CLEAR, CLOUD, WATER = 21824, 21824 | 0b1000, 21952


def dn_for(kelvin):
    return np.round((np.asarray(kelvin, float) - st.LST_OFFSET) / st.LST_SCALE).astype(np.uint16)


def test_lst_kelvin_scales_and_masks():
    dn = np.array([[dn_for(300.0), dn_for(310.0), 0, dn_for(305.0)]], np.uint16)
    qa = np.array([[CLEAR, CLOUD, CLEAR, WATER]], np.uint16)
    k = st.lst_kelvin(dn, qa)
    assert k[0, 0] == pytest.approx(300.0, abs=0.01)
    assert np.isnan(k[0, 1:]).all()                       # cloud, fill, water


def test_scene_anomaly_is_relative_to_judged_cells():
    lst = np.full((H, W), 310.0, "float32")
    lst[:, :2] = 330.0                                    # built-up strip, not judged
    lst[2, 4] = 313.0
    land = np.ones((H, W), bool)
    land[:, :2] = False
    a = st.scene_anomaly(lst, land)
    assert a[2, 4] == pytest.approx(3.0) and a[0, 5] == pytest.approx(0.0)


def test_scene_anomaly_rejects_mostly_cloudy_scene():
    lst = np.full((H, W), np.nan, "float32")
    lst[0, :] = 300.0                                     # 1 row of 6 clear
    assert st.scene_anomaly(lst, np.ones((H, W), bool)) is None
    assert st.scene_anomaly(lst, np.zeros((H, W), bool)) is None


def test_season_anomaly_needs_min_dates():
    a = [np.full((H, W), v, "float32") for v in (1.0, 2.0, 4.0)]
    a[2][0, 0] = np.nan
    med, n = st.season_anomaly(a, (H, W), min_dates=3)
    assert med[1, 1] == pytest.approx(2.0) and n[1, 1] == 3
    assert np.isnan(med[0, 0]) and n[0, 0] == 2
    empty, n0 = st.season_anomaly([], (H, W))
    assert np.isnan(empty).all() and (n0 == 0).all()


def item(iid, platform, cloud, when):
    return {"id": iid, "properties": {"platform": platform, "eo:cloud_cover": cloud, "datetime": when}}


def test_select_scenes():
    bbox = [17.18, 40.86, 17.4, 41.0]
    keep, skipped = st.select_scenes([item("a", "landsat-9", 3.0, "2022-07-16T09:28:00Z"),
                                      item("b", "landsat-7", 0.0, "2022-07-19T07:56:00Z"),
                                      item("c", "landsat-8", 55.0, "2022-07-08T09:29:00Z"),
                                      item("d", "landsat-8", 1.0, "2022-07-08T03:00:00Z")], bbox)
    assert [k["id"] for k in keep] == ["a"]
    assert [s[0] for s in skipped] == ["b", "c", "d"]


def test_table_round_trip_lines_up_with_osavi_rows(tmp_path):
    src = tmp_path / "osavi.csv.gz"
    so.write_table(src, ["id", "row", "col", "osavi_2021"], [["s1", 2, 3, "0.2"], ["s2", 4, 5, ""]])
    ids, rows, cols = st.read_cells(src)
    assert ids == ["s1", "s2"] and list(rows) == [2, 4] and list(cols) == [3, 5]
    med = np.arange(H * W, dtype="float32").reshape(H, W) / 10
    med[4, 5] = np.nan
    own, win = so.cell_values(med, rows, cols)
    header, out = st.table(ids, rows, cols, {2021: own}, {2021: win}, {2021: np.array([5, 0])})
    path = tmp_path / "t.csv.gz"
    so.write_table(path, header, out)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0] == "id,row,col,dt_2021,dt3_2021,n_2021"
    assert lines[1].startswith("s1,2,3,1.900,") and lines[2].split(",")[3] == ""
