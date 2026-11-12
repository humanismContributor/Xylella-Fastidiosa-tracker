"""ECOSTRESS granule selection and masking, with synthetic CMR records instead of downloads."""
import numpy as np
import pytest

import sample_ecostress as se

BBOX = [17.18, 40.86, 17.40, 41.00]
TXF = (16.18, 40.54, 17.51, 41.55)                       # holds the whole box
TYF = (17.36, 40.51, 18.71, 41.53)                       # only its eastern edge


def granule(orbit_scene, tile, when, box, flag="Day", layers=("LST", "cloud", "QC")):
    gid = f"ECOv002_L2T_LSTE_{orbit_scene}_{tile}_{when}_0712_01"
    base = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ECO_L2T_LSTE.002/{gid}/{gid}"
    w, s, e, n = box
    return {"umm": {
        "GranuleUR": gid, "DataGranule": {"DayNightFlag": flag},
        "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": f"{when[:4]}-{when[4:6]}-{when[6:8]}T"
                                                                  f"{when[9:11]}:{when[11:13]}:{when[13:15]}Z"}},
        "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [
            {"WestBoundingCoordinate": w, "SouthBoundingCoordinate": s,
             "EastBoundingCoordinate": e, "NorthBoundingCoordinate": n}]}}},
        "RelatedUrls": [{"URL": f"{base}_{l}.tif"} for l in layers]
                       + [{"URL": f"s3://lp-prod-protected/{gid}_LST.tif"}]}}


def test_layer_urls_https_only_and_missing_layer():
    u = se.layer_urls(granule("17504_003", "33TXF", "20210807T105602", TXF)["umm"])
    assert u["LST"].startswith("https") and u["LST"].endswith("_LST.tif") and u["cloud"].endswith("_cloud.tif")
    assert se.layer_urls(granule("1_1", "33TXF", "20210807T105602", TXF, layers=("LST",))["umm"]) is None


def test_pick_overpasses_groups_scenes_and_drops_partial_tiles_night_and_early():
    items = [
        granule("16927_013", "33TXF", "20210807T105500", TXF),   # one overpass split over two scenes
        granule("16927_014", "33TXF", "20210807T105602", TXF),
        granule("16927_014", "33TYF", "20210807T105602", TYF),   # neighbouring tile: not used
        granule("17000_001", "33TXF", "20210810T061000", TXF),   # 07:19 local: too early
        granule("17001_001", "33TXF", "20210811T211000", TXF, flag="Night"),
        granule("17002_001", "33TXF", "20210812T120000", TXF, layers=("LST",)),
        granule("17003_001", "33TXF", "20210815T113000", TXF),
    ]
    ov, skipped = se.pick_overpasses(items, BBOX)
    assert [len(o["granules"]) for o in ov] == [2, 1]
    assert all("33TXF" in g["id"] for o in ov for g in o["granules"])
    assert [why.split()[0] for _, why in skipped] == ["local", "LST"]


def test_mask_lst():
    lst = np.array([[300.0, 310.0, 0.0, np.nan]], "float32")
    cloud = np.array([[0, 1, 0, 0]], np.uint8)
    out = se.mask_lst(lst, cloud)
    assert out[0, 0] == pytest.approx(300.0) and np.isnan(out[0, 1:]).all()


def test_piece_together_keeps_first_finite():
    a = np.array([[1.0, np.nan, np.nan]], "float32")
    b = np.array([[9.0, 2.0, np.nan]], "float32")
    out = se.piece_together([a, b])
    assert out[0, 0] == 1.0 and out[0, 1] == 2.0 and np.isnan(out[0, 2])


def test_local_name():
    assert se.local_name("https://x/y/ECOv002_L2T_LSTE_1_LST.tif") == "ECOv002_L2T_LSTE_1_LST.tif"
