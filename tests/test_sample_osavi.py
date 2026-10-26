"""OSAVI extraction at sampled trees, with synthetic composites instead of imagery."""
import csv
import gzip
import io

import numpy as np
import pytest

import calibrate_scanner as cal
import sample_osavi as so
import xylella_scanner as xs

H, W = 30, 30


def test_cell_values_own_and_window_median():
    arr = np.arange(H * W, dtype="float32").reshape(H, W)
    arr[4, 4] = np.nan
    own, win = so.cell_values(arr, np.array([5, 10]), np.array([5, 10]))
    assert own[1] == arr[10, 10]
    assert win[1] == pytest.approx(arr[10, 10])          # symmetric window: median is the centre
    assert np.isfinite(win[0])                            # one NaN neighbour is ignored
    allnan = np.full((H, W), np.nan, "float32")
    _, win_nan = so.cell_values(allnan, np.array([5]), np.array([5]))
    assert np.isnan(win_nan[0])


def test_window_helpers():
    bits = np.zeros((H, W), np.uint32)
    bits[6, 6] = 4
    mask = np.zeros((H, W), bool)
    mask[4:7, 4:6] = True
    rows, cols = np.array([5]), np.array([5])
    assert so.window_or(bits, rows, cols)[0] == 4
    assert so.window_mean(mask, rows, cols)[0] == pytest.approx(6 / 9)


def test_collect_keeps_missing_seasons_empty():
    comps = {2019: np.full((H, W), 0.2, "float32"), 2020: None}
    own, win, scenes = so.collect(lambda y: (comps[y], 0 if comps[y] is None else 7),
                                  [2019, 2020], np.array([3]), np.array([3]))
    assert own[2019][0] == pytest.approx(0.2) and np.isnan(own[2020][0])
    assert scenes == {2019: 7, 2020: 0}


def samples_at(points):
    """Sample dict for (lat, lon, positive) triples, as read_samples returns it."""
    n = len(points)
    return {"id": np.array([str(i) for i in range(n)]), "date": np.array(["2021-09-01"] * n),
            "lat": np.array([p[0] for p in points]), "lon": np.array([p[1] for p in points]),
            "season": np.full(n, 2021), "positive": np.array([p[2] for p in points]),
            "symptomatic": np.zeros(n, bool)}


@pytest.fixture
def grid():
    """Grid whose cell (10, 10) holds 40.95 N, 17.25 E (as in test_calibration)."""
    x, y = cal.utm_forward(40.95, 17.25, 32633)
    x0, y0 = float(np.floor(x)) - 105.0, float(np.floor(y)) + 105.0
    return "EPSG:32633", [10.0, 0.0, x0, 0.0, -10.0, y0]


def test_on_grid_drops_samples_near_the_edge(grid):
    crs, transform = grid
    s = samples_at([(40.95, 17.25, True), (41.5, 17.25, False)])
    idx, rows, cols = so.on_grid(s, crs, transform, (H, W))
    assert list(idx) == [0] and (rows[0], cols[0]) == (10, 10)


def test_table_round_trip(tmp_path, grid):
    crs, transform = grid
    s = samples_at([(40.95, 17.25, True)])
    idx, rows, cols = so.on_grid(s, crs, transform, (H, W))
    own = {2019: np.array([0.2], "float32"), 2020: np.array([np.nan], "float32")}
    header, rows_out = so.table(s, idx, rows, cols, own, own, np.array([True]), np.array([1.0]),
                                np.array([2048], np.uint32), np.array([2048], np.uint32),
                                {2020: np.array([False]), 2021: None})
    path = tmp_path / "r.csv.gz"
    so.write_table(path, header, rows_out)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert got[0]["positive"] == "1" and got[0]["osavi_2019"] == "0.2000"
    assert got[0]["osavi_2020"] == "" and got[0]["burned_2020"] == "0" and got[0]["burned_2021"] == ""
    first = path.read_bytes()
    so.write_table(path, header, rows_out)
    assert path.read_bytes() == first                     # no timestamp churn in git


def test_extracted_values_replay_the_scanner_rules(rng):
    """The point of the table: running the decline rules on the extracted values
    gives the same answer as the scanner on the full grid (for patches above the
    minimum size, inside the land mask)."""
    comps = {y: (0.25 + rng.normal(0, 0.003, (H, W))).astype("float32") for y in range(2017, 2024)}
    for y in range(2021, 2024):
        comps[y][8:14, 8:14] = 0.13 if y == 2021 else 0.10      # decline from 2021, no recovery
    comps[2022][20:25, 20:25] = 0.14                             # pruned in 2022, recovers in 2023
    flagged = np.zeros((H, W), np.uint32)
    recovered = np.zeros((H, W), np.uint32)
    xs.update_history(comps, list(range(2019, 2024)), np.ones((H, W), bool), flagged, recovered)

    rows, cols = np.array([10, 22, 3]), np.array([10, 22, 3])
    own, _, _ = so.collect(lambda y: (comps[y], 5), range(2017, 2024), rows, cols)
    for s in range(2019, 2024):
        base, ok = xs.baseline(own, s)
        replay = xs.decline_mask(own[s], base) & ok
        grid_flag = (flagged[rows, cols] & xs.season_bit(s)) != 0
        assert list(replay) == list(grid_flag), s
    nxt = xs.recovered_mask(own[2021], own[2022], own[2023])
    assert bool(nxt[1]) and not bool(nxt[0])


def test_read_samples_keeps_id_and_date():
    text = io.StringIO("ID;DATA_RILEVAMENTO;TIPOLOGIA;SPECIE;CULTIVAR;LATITUDINE;LONGITUDINE;COMUNE;RISULTATO;SINTOMO;PROVINCIA\n"
                       "730508;07/02/2020;Campione;Olivo (Olea europaea);;40,7;17,7;Carovigno;Negativo;Assente;Brindisi\n")
    s = cal.read_samples(text)
    assert s["id"][0] == "730508" and s["date"][0] == "2020-02-07"
