#!/usr/bin/env python3
"""
Offline test of a Landsat thermal filter on the decline rule (prototype).

Phase 2 of the thermal test. The scanner's current olive rule flags a cell; the
filter keeps the flag only if the cell also ran hotter than the region's judged
cells that summer (level) or got hotter than the summer before (change), by at
least a threshold. Cells without a thermal value keep their flag: the filter
can only remove flags it has evidence against. The question: does removing the
cooler flags cut false positives without losing the positives?

The candidate set is fixed before the first run (THRESHOLDS x FEATURES), the
filter is chosen on one part of the data and scored on the other with the same
splits as replay_rules.py, and the difference to the current rule gets a block
bootstrap interval with blocks three times the thermal pixel. It also reports
how well the anomaly alone separates positive from negative trees (AUC), over
all judged trees and over the trees the current rule flags.

    python sample_osavi.py --all               # or the sample_osavi.yml artifact in .cache/osavi/
    python sample_thermal.py --all
    python replay_thermal.py                   # Landsat
    python sample_ecostress.py --all           # optional, needs an Earthdata login
    python replay_thermal.py --sensor ecostress

Every region with both an OSAVI and a thermal table in .cache/ is used. It also
checks whether the heat at positive trees is a property of the site (site_checks:
the first summer's heat, within few-km blocks and greenness classes).

Output: data/calibration/thermal_replay.json (thermal_replay_ecostress.json for
ECOSTRESS), aggregates only. Nothing here changes PARAMS.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import calibrate_scanner as cal
import replay_rules as rr
import sample_osavi as so
import sample_thermal as st

SENSORS = {   # thermal tables and report per sensor
    "landsat": {"dir": st.OUT_DIR, "out": cal.OUT_DIR / "thermal_replay.json", "script": "sample_thermal.py",
                "label": "Landsat 8/9 surface temperature anomaly vs judged cells"},
    "ecostress": {"dir": st.OUT_DIR.parent / "thermal_ecostress", "out": cal.OUT_DIR / "thermal_replay_ecostress.json",
                  "script": "sample_ecostress.py", "label": "ECOSTRESS LST anomaly vs judged cells"},
}
FEATURES = ("level", "change")
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0)   # K above the judged cells' median (level) or last summer (change)
BLOCK_PX = 30                            # 300 m bootstrap blocks: three times the 100 m thermal pixel


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def load_thermal(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    years = sorted(int(k[3:]) for k in rows[0] if k.startswith("dt_"))
    num = lambda v: float(v) if v != "" else np.nan
    return {"ids": [r["id"] for r in rows],
            "row": np.array([int(r["row"]) for r in rows]), "col": np.array([int(r["col"]) for r in rows]),
            "dt": {y: np.array([num(r[f"dt_{y}"]) for r in rows], "float32") for y in years}}


def load_region(osavi_dir: Path, thermal_dir: Path, rid: str) -> dict:
    """The OSAVI table (as replay_rules reads it) with the thermal columns and
    300 m blocks added; refuses tables that don't line up row for row."""
    osavi_csv = osavi_dir / f"{rid}.csv.gz"
    table = rr.load_table(osavi_csv)
    th = load_thermal(thermal_dir / f"{rid}.csv.gz")
    ids, rows, cols = st.read_cells(osavi_csv)
    if ids != th["ids"] or not (np.array_equal(rows, th["row"]) and np.array_equal(cols, th["col"])):
        raise ValueError(f"{rid}: thermal and OSAVI tables don't line up; re-run sample_thermal.py")
    table["dt"] = th["dt"]
    table["row"], table["col"] = th["row"], th["col"]
    table["block"] = (th["row"] // BLOCK_PX) * 100000 + th["col"] // BLOCK_PX
    return table


# --------------------------------------------------------------------------
# Rules and scores
# --------------------------------------------------------------------------
def feature(table: dict, t: int, kind: str):
    d = table["dt"]
    n = len(table["season"])
    cur = d.get(t, np.full(n, np.nan, "float32"))
    if kind == "level":
        return cur
    return cur - d.get(t - 1, np.full(n, np.nan, "float32"))


def candidates() -> list:
    return [{"name": "current", "feature": None, "min_k": None}] + [
        {"name": f"current + {k} >= {thr:.1f} K", "feature": k, "min_k": thr}
        for k in FEATURES for thr in THRESHOLDS]


def filtered_flags(table: dict, base: dict, cand: dict) -> dict:
    """The current rule's flags, minus cells whose thermal feature is known and below the threshold."""
    if cand["feature"] is None:
        return base
    out = {}
    for t, m in base.items():
        with np.errstate(invalid="ignore"):
            cool = feature(table, t, cand["feature"]) < cand["min_k"]   # NaN compares False: flag kept
        out[t] = m & ~cool
    return out


def auc(score, pos):
    """Probability that a positive tree scores higher than a negative one (ties count half); None if undefined."""
    from scipy.stats import rankdata
    ok = np.isfinite(score)
    s, p = score[ok], pos[ok]
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return round(float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def separation(table: dict, base: dict) -> dict:
    """AUC of each feature in the sample's season, over all judged trees and over
    the trees the current rule flags that season."""
    s, pos = table["season"], table["positive"]
    out = {}
    for kind in FEATURES:
        score = np.full(len(s), np.nan, "float32")
        flagged = np.zeros(len(s), bool)
        for t in np.unique(s):
            m = s == t
            score[m] = feature(table, int(t), kind)[m]
            if int(t) in base:
                flagged[m] = base[int(t)][m]
        judged = table["judged"]
        out[kind] = {
            "all_judged": {"auc": auc(score[judged], pos[judged]),
                           "trees_with_value": int(np.isfinite(score[judged]).sum()),
                           "positives_with_value": int((np.isfinite(score) & judged & pos).sum())},
            "flagged_by_current": {"auc": auc(score[judged & flagged], pos[judged & flagged]),
                                   "trees_with_value": int(np.isfinite(score[judged & flagged]).sum()),
                                   "positives_with_value": int((np.isfinite(score) & judged & flagged & pos).sum())},
        }
    return out


def stratified_auc(score, pos, groups):
    """AUC within each group that holds positives, averaged with the positives as weights."""
    num = den = 0.0
    for g in np.unique(groups[pos]):
        m = groups == g
        a = auc(score[m], pos[m])
        if a is not None:
            w = int((pos[m] & np.isfinite(score[m])).sum())
            num, den = num + a * w, den + w
    return round(num / den, 4) if den else None


def site_checks(table: dict, block_km=(1, 2)) -> dict:
    """Is the heat at positive trees a property of the site rather than of the
    infection? The anomaly in the first summer with data (before the outbreak
    reached the Monopoli front) is scored against the later lab results, over
    all judged trees, within blocks of a few km (so a regional gradient can't
    explain it) and within greenness quintiles of that summer (so sparse canopy
    can't either)."""
    with_data = [y for y in sorted(table["dt"]) if np.isfinite(table["dt"][y][table["judged"]]).mean() >= 0.5]
    if not with_data:
        return {"first_summer": None}
    first = with_data[0]                     # first summer with values for most judged trees
    s, pos = table["season"], table["positive"]
    m = table["judged"] & (s > first)
    heat = table["dt"][first][m]
    out = {"first_summer": first, "trees": int(m.sum()), "positives": int(pos[m].sum()),
           "auc_first_summer": auc(heat, pos[m])}
    for km in block_km:
        px = km * 1000 // 10
        blocks = (table["row"][m] // px) * 100000 + table["col"][m] // px
        out[f"auc_within_{km}km_blocks"] = stratified_auc(heat, pos[m], blocks)
    green = table["osavi"].get(first)
    if green is not None:
        g = green[m]
        ok = np.isfinite(g)
        quint = np.digitize(g, np.quantile(g[ok], [0.2, 0.4, 0.6, 0.8])) if ok.any() else np.zeros(len(g))
        out["auc_within_greenness_quintiles"] = stratified_auc(np.where(ok, heat, np.nan), pos[m], quint)
        out["auc_low_greenness_first_summer"] = auc(-g, pos[m])
    return out


def run_split(name: str, cands: list, train: list, test: list) -> dict:
    """`train` / `test`: lists of (preds per candidate, positive, mask, blocks) per region."""
    def pooled(parts, i):
        return (np.concatenate([p[0][i][p[2]] for p in parts]), np.concatenate([p[1][p[2]] for p in parts]),
                np.concatenate([p[3][p[2]] for p in parts]))
    if not all(p[2].any() for p in train + test):
        return {"split": name, "skipped": "no scorable trees in one part"}
    train_j = []
    for i in range(len(cands)):
        pr, y, _ = pooled(train, i)
        train_j.append(rr.metrics(pr, y)["youden_j"])
    best = max(range(1, len(cands)), key=lambda i: -np.inf if train_j[i] is None else train_j[i])
    cur_te, y_te, blocks = pooled(test, 0)
    best_te, _, _ = pooled(test, best)
    return {
        "split": name,
        "train_j": {c["name"]: j for c, j in zip(cands, train_j)},
        "chosen_on_train": cands[best]["name"],
        "test": {
            "current": rr.metrics(cur_te, y_te),
            "best_thermal": {"rule": cands[best]["name"], **rr.metrics(best_te, y_te),
                             "delta_j_vs_current_ci95": rr.block_bootstrap_dj(best_te, cur_te, y_te, blocks)},
        },
    }


def evaluate(tables: dict, lags=(0, 1)) -> dict:
    cands = candidates()
    base = {rid: rr.season_flags(t, rr.CURRENT) for rid, t in tables.items()}
    out = {"separation": {rid: separation(t, base[rid]) for rid, t in tables.items()},
           "site_checks": {rid: site_checks(t) for rid, t in tables.items()}, "lags": []}
    for lag in lags:
        prep = {}
        for rid, t in tables.items():
            preds, usable = [], None
            for c in cands:
                p, usable = rr.predict(t, filtered_flags(t, base[rid], c), lag)
                preds.append(p)
            prep[rid] = (preds, usable)
        part = lambda rid, m: (prep[rid][0], tables[rid]["positive"], prep[rid][1] & m, tables[rid]["block"])
        # the same splits as replay_rules.evaluate, skipping parts with too few positives
        enough = lambda rid, m: int((tables[rid]["positive"] & prep[rid][1] & m).sum()) >= rr.MIN_POSITIVES
        splits = []
        for rid, t in tables.items():
            early = t["season"] <= rr.TRAIN_UNTIL
            if enough(rid, early) and enough(rid, ~early):
                splits.append(run_split(f"temporal {rid}: <= {rr.TRAIN_UNTIL} -> > {rr.TRAIN_UNTIL}", cands,
                                        [part(rid, early)], [part(rid, ~early)]))
        every = lambda rid: np.ones(len(tables[rid]["season"]), bool)
        ids = [rid for rid in tables if enough(rid, every(rid))]
        if len(ids) >= 2:
            for b in ids:                                  # each region held out, chosen on the others
                others = [a for a in ids if a != b]
                splits.append(run_split(f"spatial {' + '.join(others)} -> {b}", cands,
                                        [part(a, every(a)) for a in others], [part(b, every(b))]))
        out["lags"].append({"lag": lag, "splits": splits})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sensor", choices=list(SENSORS), default="landsat")
    ap.add_argument("--osavi-dir", type=Path, default=so.OUT_DIR)
    ap.add_argument("--thermal-dir", type=Path, help="default: the sensor's table folder")
    ap.add_argument("--out", type=Path, help="default: the sensor's report file")
    args = ap.parse_args(argv)
    sensor = SENSORS[args.sensor]
    args.thermal_dir = args.thermal_dir or sensor["dir"]
    args.out = args.out or sensor["out"]
    tables, sources = {}, {}
    for path in sorted(args.thermal_dir.glob("*.csv.gz")):
        rid = path.name[:-len(".csv.gz")]
        if not (args.osavi_dir / path.name).exists():
            print(f"{rid}: no OSAVI table in {args.osavi_dir}; skipped", file=sys.stderr)
            continue
        tables[rid] = load_region(args.osavi_dir, args.thermal_dir, rid)
        # the sampling data the OSAVI table was built from, which may be an older export than cal.SOURCE
        sources[rid] = json.loads((args.osavi_dir / f"{rid}.json").read_text()).get("source")
    if not tables:
        print(f"No thermal tables with a matching OSAVI table; run sample_osavi.py and {sensor['script']} first.",
              file=sys.stderr)
        return 2
    res = evaluate(tables)
    report = {
        "generated": dt.date.today().isoformat(),
        "source": sources,
        "thermal": f"{sensor['label']} ({sensor['script']})",
        "candidates": [c["name"] for c in candidates()],
        "selection": "highest Youden's J on the training part; cells without a thermal value keep their flag",
        "bootstrap": f"{rr.BOOTSTRAP} resamples of {BLOCK_PX * 10} m blocks",
        **res,
        "caveats": cal.CAVEATS + ["The thermal value is a 70-100 m pixel: it describes the patch, not the tree."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for rid, sep in res["separation"].items():
        print(f"\n{rid} AUC (0.5 = no separation)")
        for k, v in sep.items():
            print(f"  {k:6s} all judged {v['all_judged']['auc']} (n+={v['all_judged']['positives_with_value']})"
                  f"   flagged by current {v['flagged_by_current']['auc']}"
                  f" (n+={v['flagged_by_current']['positives_with_value']})")
        sc = res["site_checks"][rid]
        print(f"  site check, heat in {sc['first_summer']}: AUC {sc['auc_first_summer']}, within 1 km blocks "
              f"{sc['auc_within_1km_blocks']}, within greenness quintiles {sc.get('auc_within_greenness_quintiles')}")
    for lag in res["lags"]:
        print(f"\n=== lag {lag['lag']} ===")
        for sp in lag["splits"]:
            c, b = sp["test"]["current"], sp["test"]["best_thermal"]
            print(f"{sp['split']}\n  current  J={c['youden_j']} sens={c['sensitivity']} fpr={c['false_positive_rate']}"
                  f"\n  thermal  J={b['youden_j']} sens={b['sensitivity']} fpr={b['false_positive_rate']}"
                  f"  dJ {b['delta_j_vs_current_ci95']}  [{b['rule']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
