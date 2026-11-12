"""Thermal filter replay on synthetic OSAVI and thermal tables."""
import numpy as np
import pytest

import replay_rules as rr
import replay_thermal as rt
import sample_osavi as so

YEARS = list(range(2017, 2025))


def make_table(rng, n=400, positives=40, false_pos=40):
    """Even samples belong to season 2020, odd ones to 2022. The first `positives`
    are positive, the next `false_pos` negative, and all of them decline in their
    season; positives run 2 K hot from then on, the false positives 1 K cool."""
    season = np.where(np.arange(n) % 2 == 0, 2020, 2022)
    osavi = {y: (0.25 + rng.normal(0, 0.003, n)).astype("float32") for y in YEARS}
    dt = {y: rng.normal(0, 0.3, n).astype("float32") for y in YEARS}
    hit = np.arange(n) < positives + false_pos
    for y in YEARS:
        down = hit & (y >= season)
        osavi[y][down] = 0.125
        dt[y][down & (np.arange(n) < positives)] = 2.0
        dt[y][down & (np.arange(n) >= positives)] = -1.0
    pos = np.arange(n) < positives
    return {"years": YEARS, "osavi": osavi, "burned": {s: np.zeros(n, bool) for s in range(2019, 2025)},
            "season": season, "positive": pos, "judged": np.ones(n, bool), "block": np.arange(n) // 4, "dt": dt,
            "row": (np.arange(n) // 20) * 10, "col": (np.arange(n) % 20) * 10}


def test_site_checks_see_heat_that_was_there_before_the_outbreak(rng):
    t = make_table(rng)
    first = min(t["dt"])
    t["dt"][first][:40] = 3.0                            # later positives already hot in the first summer
    sc = rt.site_checks(t)
    assert sc["first_summer"] == first and sc["positives"] == 40
    assert sc["auc_first_summer"] == 1.0 and sc["auc_within_1km_blocks"] == 1.0
    assert sc["auc_within_greenness_quintiles"] == 1.0
    t["dt"][first][:40] = 0.0                            # not hot before: no separation
    assert rt.site_checks(t)["auc_first_summer"] < 0.75
    t["dt"][first][:] = np.nan                           # a summer without data is passed over
    assert rt.site_checks(t)["first_summer"] == first + 1


def test_stratified_auc_weights_groups_by_positives():
    score = np.array([1.0, 0.0, 0.0, 1.0, 0.5, 0.5])
    pos = np.array([True, False, True, False, True, False])
    groups = np.array([0, 0, 1, 1, 2, 2])
    assert rt.stratified_auc(score, pos, groups) == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_filter_removes_cool_flags_and_keeps_unknown():
    t = {"season": np.full(3, 2021), "dt": {2020: np.array([0, 0, 0], "float32"),
                                             2021: np.array([2.0, -1.0, np.nan], "float32")}}
    base = {2021: np.array([True, True, True])}
    level = rt.filtered_flags(t, base, {"feature": "level", "min_k": 0.5})
    assert list(level[2021]) == [True, False, True]
    change = rt.feature(t, 2021, "change")
    assert change[0] == pytest.approx(2.0) and np.isnan(change[2])
    assert rt.filtered_flags(t, base, rt.candidates()[0]) is base


def test_auc():
    pos = np.array([True, True, False, False, False])
    assert rt.auc(np.array([3.0, 2.0, 1.0, 0.0, np.nan]), pos) == 1.0
    assert rt.auc(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), pos) == 0.5
    assert rt.auc(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.zeros(5, bool)) is None


def test_candidates_are_fixed_in_advance():
    names = [c["name"] for c in rt.candidates()]
    assert names[0] == "current" and len(names) == 1 + len(rt.FEATURES) * len(rt.THRESHOLDS)


def test_thermal_filter_wins_when_false_positives_run_cool(rng):
    tables = {"a": make_table(rng), "b": make_table(np.random.default_rng(1))}
    res = rt.evaluate(tables, lags=(0,))
    for sp in res["lags"][0]["splits"]:
        cur, best = sp["test"]["current"], sp["test"]["best_thermal"]
        assert best["rule"] != "current"
        assert best["false_positive_rate"] < cur["false_positive_rate"]
        assert best["sensitivity"] == cur["sensitivity"] == 1.0
        assert best["delta_j_vs_current_ci95"][0] > 0
    sep = res["separation"]["a"]["level"]["flagged_by_current"]
    assert sep["auc"] == 1.0 and sep["positives_with_value"] == 40


def test_parts_with_too_few_positives_are_not_scored(rng):
    t = make_table(rng)
    t["season"][:] = 2020                                # nothing after TRAIN_UNTIL
    assert rt.evaluate({"a": t}, lags=(0,))["lags"][0]["splits"] == []
    assert "skipped" in rt.run_split("x", rt.candidates()[:2], [([np.zeros(3, bool)] * 2, np.zeros(3, bool),
                                                                  np.zeros(3, bool), np.arange(3))], [])


def test_load_region_refuses_misaligned_tables(tmp_path):
    osavi = ["id", "date", "season", "positive", "symptomatic", "lat", "lon", "row", "col", "judged",
             "judged3", "flag_bits", "flag_bits3", "osavi_2020", "osavi3_2020"]
    so.write_table(tmp_path / "o" / "r.csv.gz", osavi,
                   [["s1", "2020-09-01", 2020, 1, 0, "40.9", "17.3", 3, 4, 1, "1.00", 0, 0, "0.2", "0.2"]])
    so.write_table(tmp_path / "t" / "r.csv.gz", ["id", "row", "col", "dt_2020", "dt3_2020", "n_2020"],
                   [["s1", 3, 4, "1.5", "1.4", 4]])
    t = rt.load_region(tmp_path / "o", tmp_path / "t", "r")
    assert t["dt"][2020][0] == pytest.approx(1.5) and t["block"][0] == 0
    so.write_table(tmp_path / "t" / "r.csv.gz", ["id", "row", "col", "dt_2020", "dt3_2020", "n_2020"],
                   [["other", 3, 4, "1.5", "1.4", 4]])
    with pytest.raises(ValueError):
        rt.load_region(tmp_path / "o", tmp_path / "t", "r")
