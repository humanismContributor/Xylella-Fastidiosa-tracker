"""Scanner check against field samples, on a synthetic grid and a few CSV rows."""
import io

import numpy as np
import pytest

import calibrate_scanner as cal
import xylella_scanner as xs

CSV_HEADER = "ID;DATA_RILEVAMENTO;TIPOLOGIA;SPECIE;CULTIVAR;LATITUDINE;LONGITUDINE;COMUNE;RISULTATO;SINTOMO;PROVINCIA\n"


def row(date, lat, lon, result="Negativo", species="Olivo (Olea europaea)", symptom="Assente"):
    la, lo = str(lat).replace(".", ","), str(lon).replace(".", ",")
    return f"1;{date};Campione;{species};Ogliarola;{la};{lo};Monopoli;{result};{symptom};Bari\n"


def test_utm_matches_reference_points():
    # Central meridian of zone 33 (15° E): easting 500 km, northing = k0 x meridian arc.
    x, y = cal.utm_forward(45.0, 15.0, 32633)
    assert x == pytest.approx(500000.0, abs=1e-6)
    assert y == pytest.approx(4982950.400, abs=0.01)
    x, y = cal.utm_forward(0.0, 15.0, 32633)
    assert (x, y) == pytest.approx((500000.0, 0.0), abs=1e-6)
    x, _ = cal.utm_forward(41.0, 17.0, 32633)
    assert x > 500000                      # east of the central meridian


def test_sample_season_uses_latest_summer():
    import datetime as dt
    assert cal.sample_season(dt.date(2021, 8, 3)) == 2021
    assert cal.sample_season(dt.date(2022, 2, 10)) == 2021
    assert cal.sample_season(dt.date(2022, 7, 1)) == 2022


def test_read_samples_filters_species_and_bad_rows():
    text = io.StringIO(CSV_HEADER
                       + row("07/02/2021", 40.95, 17.25)
                       + row("10/09/2021", 40.96, 17.26, result="Positivo", symptom="PRESENTE")
                       + row("10/09/2021", 40.96, 17.26, species="Mandorlo (Prunus dulcis)")
                       + row("10/09/2021", "", 17.26)
                       + row("10/09/2021", 40.96, 17.26, result="In corso")
                       + "broken;row\n")
    s = cal.read_samples(text)
    assert len(s["lat"]) == 2
    assert s["lat"][0] == pytest.approx(40.95)
    assert list(s["season"]) == [2020, 2021]
    assert list(s["positive"]) == [False, True]
    assert list(s["symptomatic"]) == [False, True]
    assert s["skipped"] == 3


@pytest.fixture
def grid():
    """A 20 x 20 grid of 10 m cells whose cell (10, 10) holds the point (40.95 N, 17.25 E)."""
    x, y = cal.utm_forward(40.95, 17.25, 32633)
    x0, y0 = float(np.floor(x)) - 105.0, float(np.floor(y)) + 105.0
    transform = [10.0, 0.0, x0, 0.0, -10.0, y0]
    flagged = np.zeros((20, 20), np.uint32)
    recovered = np.zeros((20, 20), np.uint32)
    return {"crs": "EPSG:32633", "transform": transform, "flagged": flagged,
            "recovered": recovered, "complete_through": 2023, "params_hash": "test"}


def test_point_lands_in_expected_cell(grid):
    r, c = cal.grid_cells(np.array([40.95]), np.array([17.25]), grid["crs"], grid["transform"])
    assert (int(r[0]), int(c[0])) == (10, 10)


def test_window_includes_neighbours_and_skips_recovered(grid):
    f, rec = grid["flagged"], grid["recovered"]
    f[11, 11] = xs.season_bit(2021)                 # diagonal neighbour
    f[10, 12] = xs.season_bit(2020)                 # two cells away
    f[9, 10] = xs.season_bit(2022)
    rec[9, 10] = xs.season_bit(2022)                # recovered: not a lead
    rows, cols = np.array([10, 0]), np.array([10, 5])
    bits, inside = cal.window_bits(f, rec, rows, cols, radius=1)
    assert list(inside) == [True, False]           # a window off the grid is dropped
    assert bits[0] == xs.season_bit(2021)
    bits0, _ = cal.window_bits(f, rec, rows, cols, radius=0)
    assert bits0[0] == 0
    bits2, _ = cal.window_bits(f, rec, rows, cols, radius=2)
    assert bits2[0] == xs.season_bit(2021) | xs.season_bit(2020)


def test_flagged_by_respects_lag():
    bits = np.array([xs.season_bit(2021)] * 3, np.uint32)
    seasons = np.array([2020, 2021, 2022])
    assert list(cal.flagged_by(bits, seasons, 0)) == [False, True, True]
    assert list(cal.flagged_by(bits, seasons, 1)) == [True, True, True]


def test_score_and_wilson():
    flag = np.array([True, True, True, True, False, False, False, False, False, False])
    pos = np.array([True, True, False, False, True, False, False, False, False, False])
    s = cal.score(flag, pos)
    assert (s["flagged_positive"], s["flagged_negative"], s["unflagged_positive"], s["unflagged_negative"]) == (2, 2, 1, 5)
    assert s["sensitivity"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["flagged_share"] == pytest.approx(0.4)
    assert s["relative_risk"] == pytest.approx(3.0)
    lo, hi = s["positive_rate_flagged_ci95"]
    assert lo < 0.5 < hi
    assert cal.wilson(0, 0) is None
    assert cal.score(np.zeros(3, bool), np.zeros(3, bool))["relative_risk"] is None


def test_evaluate_region_end_to_end(grid):
    grid["flagged"][10, 10] = xs.season_bit(2022)
    text = io.StringIO(CSV_HEADER
                       + row("01/09/2021", 40.95, 17.25, result="Positivo")       # flagged a year later
                       + row("01/09/2022", 40.95, 17.25)                          # flagged that summer
                       + row("01/09/2020", 41.50, 17.25)                          # off the grid
                       + row("01/09/2023", 40.95, 17.25))                         # lag 1 not complete yet
    samples = cal.read_samples(text)
    res = cal.evaluate_region({"id": "t", "name": "test"}, grid, samples)
    assert res["samples_on_grid"] == 3
    lag0, lag1 = res["results"]
    assert lag0["all_trees"]["trees"] == 3 and lag0["all_trees"]["flagged_positive"] == 0
    assert lag1["all_trees"]["trees"] == 2 and lag1["all_trees"]["flagged_positive"] == 1
    assert res["source"]["url"].startswith("https://dati.puglia.it/")


def test_evaluate_region_without_samples_on_grid(grid):
    samples = cal.read_samples(io.StringIO(CSV_HEADER + row("01/09/2021", 45.0, 12.0)))
    assert cal.evaluate_region({"id": "t", "name": "test"}, grid, samples) is None
