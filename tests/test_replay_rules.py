"""Offline rule replay on synthetic OSAVI tables."""
import numpy as np
import pytest

import replay_rules as rr
import sample_osavi as so
import xylella_scanner as xs

YEARS = list(range(2017, 2025))


def make_table(rng, n=400, positives=40, drop=0.5, base=0.25):
    """n samples in season 2020; the first `positives` are positive and their
    cells lose `drop` of their greenness in 2021 (and stay low)."""
    osavi = {y: (base + rng.normal(0, 0.003, n)).astype("float32") for y in YEARS}
    for y in YEARS:
        if y >= 2021:
            osavi[y][:positives] = base * (1 - drop)
    pos = np.zeros(n, bool)
    pos[:positives] = True
    return {"years": YEARS, "osavi": osavi, "burned": {s: np.zeros(n, bool) for s in range(2019, 2025)},
            "season": np.full(n, 2020), "positive": pos, "judged": np.ones(n, bool),
            "block": np.arange(n) // 4}


def test_current_rule_matches_the_scanner(rng):
    """The replayed current rule gives the same answer as xs.baseline + xs.decline_mask."""
    cur = (0.25 + rng.normal(0, 0.05, 2000)).astype("float32")
    base = (0.25 + rng.normal(0, 0.05, 2000)).astype("float32")
    prev = (0.25 + rng.normal(0, 0.05, 2000)).astype("float32")
    comps = {2019: prev, 2020: base, 2021: cur}
    b, ok = xs.baseline(comps, 2021)
    expected = xs.decline_mask(cur, b) & ok
    assert (rr.decline(cur, base, prev, rr.CURRENT) == expected).all()


def test_split_rule_uses_the_low_threshold_for_sparse_groves():
    base = np.array([0.15, 0.30], "float32")
    cur = base * 0.8                                   # a 20 % drop in both
    rule = {"split": 0.20, "rel_low": 0.15, "rel_high": 0.30, "abs_min": 0.02}
    assert list(rr.decline(cur, base, None, rule)) == [True, False]


def test_recovery_and_fire_filters(rng):
    t = make_table(rng, n=20, positives=10)
    t["osavi"][2022][:5] = t["osavi"][2020][:5]        # half the declines recover the next summer
    t["burned"][2021][5:7] = True                      # two burned
    flags = rr.season_flags(t, rr.CURRENT)
    assert flags[2021][:10].sum() == 3
    assert not flags[2021][10:].any()


def test_predict_lag_and_usable(rng):
    t = make_table(rng, n=20, positives=10)
    flags = rr.season_flags(t, rr.CURRENT)
    p0, u0 = rr.predict(t, flags, 0)                   # decline is in 2021, sampling season 2020
    p1, u1 = rr.predict(t, flags, 1)
    assert not p0.any() and p1[:10].all() and not p1[10:].any()
    t["judged"][0] = False
    _, u = rr.predict(t, flags, 1)
    assert not u[0] and u[1:].all()


def test_metrics_youden_and_relative_risk():
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0], bool)
    pos = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    m = rr.metrics(pred, pos)
    assert m["sensitivity"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["false_positive_rate"] == pytest.approx(0.2)
    assert m["youden_j"] == pytest.approx(2 / 3 - 0.2, abs=1e-4)
    assert m["relative_risk"] == pytest.approx((2 / 3) / (1 / 5), abs=0.01)


def test_bootstrap_interval_brackets_a_clear_difference(rng):
    pos = np.zeros(400, bool); pos[:40] = True
    good = pos.copy()
    bad = np.zeros(400, bool)
    ci = rr.block_bootstrap_dj(good, bad, pos, np.arange(400) // 4, n=200)
    assert ci[0] > 0.8 and ci[1] <= 1.0


def test_evaluate_finds_a_lower_threshold_when_declines_are_small(rng):
    """Positives drop 20 %: below the current 30 %, so a chosen rule should beat it."""
    tables = {"a": make_table(rng, drop=0.20), "b": make_table(np.random.default_rng(1), drop=0.20)}
    for t in tables.values():
        t["season"][200:] = 2021                       # half the samples in the test period
        t["positive"][200:240] = True
        for y in YEARS:
            if y >= 2022:
                t["osavi"][y][200:240] = t["osavi"][2017][200:240] * 0.8
    res = rr.evaluate(tables, lags=(1,))
    spatial = [s for s in res[0]["splits"] if s["split"].startswith("spatial")][0]
    assert spatial["test"]["current"]["youden_j"] == pytest.approx(0, abs=0.02)
    assert spatial["test"]["best_fixed"]["youden_j"] > 0.9


def test_load_table_round_trip(tmp_path):
    header = ["id", "date", "season", "positive", "symptomatic", "lat", "lon", "row", "col", "judged", "judged3",
              "flag_bits", "flag_bits3", "osavi_2019", "osavi_2020", "osavi3_2019", "osavi3_2020", "burned_2020"]
    rows = [["1", "2020-09-01", 2020, 1, 0, "40.9", "17.3", 12, 34, 1, "1.00", 0, 0, "0.2500", "", "0.2", "", 0]]
    so.write_table(tmp_path / "r.csv.gz", header, rows)
    t = rr.load_table(tmp_path / "r.csv.gz")
    assert t["years"] == [2019, 2020] and np.isnan(t["osavi"][2020][0])
    assert t["positive"][0] and t["judged"][0] and t["block"][0] == 1 * 100000 + 3
