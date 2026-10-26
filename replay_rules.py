#!/usr/bin/env python3
"""
Offline test of decline rules on the per-tree OSAVI tables (prototype).

Replays the scanner's olive decline rule on the tables written by
sample_osavi.py, with the current thresholds and with alternatives, and scores
each against the lab results. The question it answers: would a different rule,
in particular one whose drop threshold depends on how green the grove was the
summer before, separate positive from negative trees better than the current one?

A rule is chosen on one part of the data and scored on another, never on the
same trees:
    temporal   choose on seasons up to 2020, score on 2021 onward (each region)
    spatial    choose on one region, score on the other

Nothing here changes PARAMS or any scan output. Re-run it whenever the tables
are refreshed (for example after new sampling campaigns are published):

    python sample_osavi.py --region puglia-monopoli-polignano
    python sample_osavi.py --region puglia-monopoli-fasano
    python replay_rules.py

Output: data/calibration/rule_replay.json

What it can't do: the scanner also drops patches under 3 pixels, which a
per-tree table can't see, and every caveat in calibrate_scanner.CAVEATS still
applies (targeted sampling, tree removal, symptomless infections).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import calibrate_scanner as cal
import sample_osavi as so
import xylella_scanner as xs

OUT_FILE = cal.OUT_DIR / "rule_replay.json"
REGIONS = ("puglia-monopoli-polignano", "puglia-monopoli-fasano")
TRAIN_UNTIL = 2020           # temporal split: choose on seasons <= this, score on later ones
BOOTSTRAP = 1000
BLOCK_PX = 10                # bootstrap blocks of 100 m: nearby trees are sampled together

# The scanner's olive rule as it stands (xylella_scanner.OLIVE_RULES).
CURRENT = {"name": "current", "split": None, "rel_low": xs.MIN_REL_DROP, "rel_high": xs.MIN_REL_DROP,
           "abs_min": xs.MIN_ABS_DROP}


def candidate_rules() -> list[dict]:
    """The current rule, fixed thresholds, and thresholds that are lower for
    groves that were less green the summer before (base < split)."""
    rules = [CURRENT]
    for rel, a in itertools.product((0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40), (0.02, 0.03, 0.04)):
        rules.append({"name": f"fixed rel>={rel:.2f} abs>={a:.2f}", "split": None,
                      "rel_low": rel, "rel_high": rel, "abs_min": a})
    for split, lo, hi, a in itertools.product((0.18, 0.20, 0.22, 0.25), (0.10, 0.15, 0.20),
                                              (0.20, 0.25, 0.30, 0.35), (0.02, 0.03, 0.04)):
        if lo < hi:
            rules.append({"name": f"split@{split:.2f} rel>={lo:.2f}/{hi:.2f} abs>={a:.2f}", "split": split,
                          "rel_low": lo, "rel_high": hi, "abs_min": a})
    return rules


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def load_table(path: Path) -> dict:
    """One OSAVI table as arrays: per-year OSAVI, burned flags and sample fields."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    keys = rows[0].keys()
    years = sorted(int(k[6:]) for k in keys if k.startswith("osavi_"))
    seasons = sorted(int(k[7:]) for k in keys if k.startswith("burned_"))
    num = lambda v: float(v) if v != "" else np.nan
    return {
        "years": years,
        "osavi": {y: np.array([num(r[f"osavi_{y}"]) for r in rows], "float32") for y in years},
        "burned": {s: np.array([r[f"burned_{s}"] == "1" for r in rows]) for s in seasons},
        "season": np.array([int(r["season"]) for r in rows]),
        "positive": np.array([r["positive"] == "1" for r in rows]),
        "judged": np.array([r["judged"] == "1" for r in rows]),
        "block": np.array([(int(r["row"]) // BLOCK_PX) * 100000 + int(r["col"]) // BLOCK_PX for r in rows]),
    }


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def decline(cur, base, prev, rule: dict):
    """The scanner's decline test (xs.baseline + xs.decline_mask for the olive
    profile), with the relative-drop threshold optionally split by baseline greenness."""
    with np.errstate(invalid="ignore"):
        rel = rule["rel_high"] if rule["split"] is None else np.where(base < rule["split"],
                                                                      rule["rel_low"], rule["rel_high"])
        drop = base - cur
        ok = np.isfinite(cur) & np.isfinite(base)
        ok &= (base >= xs.BASELINE_OSAVI_MIN) & (base <= xs.BASELINE_OSAVI_MAX)
        ok &= drop >= rule["abs_min"]
        ok &= drop >= rel * base
        if prev is not None:
            ok &= ~(np.isfinite(prev) & (base > prev * (1 + xs.MAX_BASELINE_RISE)))
    return ok


def season_flags(table: dict, rule: dict) -> dict:
    """Per season, which samples' cells the rule flags (after the fire and
    recovery filters, as the scanner applies them)."""
    o = table["osavi"]
    out = {}
    for t in table["years"]:
        if t < xs.FIRST_SEASON or t - 1 not in o:
            continue
        m = decline(o[t], o[t - 1], o.get(t - 2), rule)
        if t in table["burned"]:
            m &= ~table["burned"][t]
        if t + 1 in o:
            m &= ~xs.recovered_mask(o[t - 1], o[t], o[t + 1])
        out[t] = m
    return out


def predict(table: dict, flags: dict, lag: int):
    """Whether each sample's cell is flagged in its season or up to `lag` seasons
    later, and which samples can be scored (judged cells, all needed seasons present)."""
    s = table["season"]
    n = len(s)
    pred = np.zeros(n, bool)
    usable = table["judged"] & (s >= xs.FIRST_SEASON)
    for k in range(lag + 1):
        have = np.array([t in flags for t in (s + k)])
        usable &= have
        for t in np.unique(s + k):
            if t in flags:
                m = (s + k) == t
                pred[m] |= flags[t][m]
    return pred, usable


def metrics(pred, pos) -> dict:
    tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum()); tn = int((~pred & ~pos).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    rate_in = tp / (tp + fp) if tp + fp else np.nan
    rate_out = fn / (fn + tn) if fn + tn else np.nan
    r = lambda v, nd=4: None if not np.isfinite(v) else round(float(v), nd)
    return {"trees": tp + fp + fn + tn, "positive": tp + fn, "flagged_share": r((tp + fp) / max(1, tp + fp + fn + tn)),
            "sensitivity": r(sens), "false_positive_rate": r(fpr),
            # Youden's J: sensitivity minus false-positive rate; 0 = no better than chance
            "youden_j": r(sens - fpr),
            "relative_risk": r(rate_in / rate_out, 2) if rate_out else None}


def block_bootstrap_dj(pred_a, pred_b, pos, blocks, n: int = BOOTSTRAP, seed: int = 0):
    """95 % interval of J(a) - J(b), resampling 100 m blocks of trees."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(blocks, return_inverse=True)
    nb = inv.max() + 1
    count = lambda m: np.bincount(inv[m], minlength=nb).astype(float)
    # per block: positives, negatives, and how many of each each rule flags
    per = np.stack([count(pos), count(~pos), count(pred_a & pos), count(pred_a & ~pos),
                    count(pred_b & pos), count(pred_b & ~pos)])
    w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(n)]).astype(float)
    p, q, ap, an, bp, bn = (w @ per.T).T      # totals per resample
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ap / p - an / q) - (bp / p - bn / q)
    out = out[np.isfinite(out)]
    if len(out) == 0:
        return None
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def prepare(tables: dict, rules: list, lag: int) -> dict:
    """Predictions of every rule for every sample, computed once per region."""
    out = {}
    for rid, t in tables.items():
        preds, usable = [], None
        for rule in rules:
            p, u = predict(t, season_flags(t, rule), lag)
            preds.append(p)
            usable = u
        out[rid] = {"preds": preds, "usable": usable}
    return out


def run_split(name, rules, train: list, test: list) -> dict:
    """`train` and `test` are lists of (preds, positive, mask, blocks) per region."""
    def pooled(parts, i):
        return (np.concatenate([p[0][i][p[2]] for p in parts]),
                np.concatenate([p[1][p[2]] for p in parts]),
                np.concatenate([p[3][p[2]] for p in parts]))
    n_rules = len(rules)
    train_j = []
    for i in range(n_rules):
        pr, y, _ = pooled(train, i)
        train_j.append(metrics(pr, y)["youden_j"] or -np.inf)
    fixed_idx = [i for i, r in enumerate(rules) if r["split"] is None and r["name"] != "current"]
    split_idx = [i for i, r in enumerate(rules) if r["split"] is not None]
    pick = {"current": 0,
            "best_fixed": max(fixed_idx, key=lambda i: train_j[i]),
            "best_split": max(split_idx, key=lambda i: train_j[i])}
    res = {"split": name, "chosen_on_train": {}, "test": {}}
    cur_pred, y_test, blocks = pooled(test, 0)
    for label, i in pick.items():
        pr_tr, y_tr, _ = pooled(train, i)
        pr_te, _, _ = pooled(test, i)
        res["chosen_on_train"][label] = {"rule": rules[i]["name"], **metrics(pr_tr, y_tr)}
        res["test"][label] = {"rule": rules[i]["name"], **metrics(pr_te, y_test)}
        if label != "current":
            res["test"][label]["delta_j_vs_current_ci95"] = block_bootstrap_dj(pr_te, cur_pred, y_test, blocks)
    return res


def evaluate(tables: dict, rules: list | None = None, lags=(0, 1)) -> list:
    rules = rules or candidate_rules()
    results = []
    for lag in lags:
        prep = prepare(tables, rules, lag)
        part = lambda rid, m: (prep[rid]["preds"], tables[rid]["positive"], prep[rid]["usable"] & m,
                               tables[rid]["block"])
        out = {"lag": lag, "splits": []}
        for rid, t in tables.items():
            early = t["season"] <= TRAIN_UNTIL
            out["splits"].append(run_split(f"temporal {rid}: <= {TRAIN_UNTIL} -> > {TRAIN_UNTIL}", rules,
                                           [part(rid, early)], [part(rid, ~early)]))
        ids = list(tables)
        if len(ids) == 2:
            every = lambda rid: np.ones(len(tables[rid]["season"]), bool)
            for a, b in (ids, ids[::-1]):
                out["splits"].append(run_split(f"spatial {a} -> {b}", rules,
                                               [part(a, every(a))], [part(b, every(b))]))
        results.append(out)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tables-dir", type=Path, default=so.OUT_DIR, help="folder with the OSAVI tables")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args(argv)

    tables, meta = {}, {}
    for rid in REGIONS:
        path = args.tables_dir / f"{rid}.csv.gz"
        if not path.exists():
            print(f"Missing {path}; run sample_osavi.py --region {rid} first.", file=sys.stderr)
            return 2
        tables[rid] = load_table(path)
        m = json.loads((args.tables_dir / f"{rid}.json").read_text())
        meta[rid] = {"table_generated": m.get("generated"), "samples": m.get("samples"),
                     "positives": m.get("positives"), "years": m.get("years")}
    rules = candidate_rules()
    results = evaluate(tables, rules)
    report = {
        "generated": dt.date.today().isoformat(),
        "source": cal.SOURCE,
        "tables": meta,
        "rules_tested": len(rules),
        "selection": "highest Youden's J (sensitivity - false-positive rate) on the training part",
        "scored": "judged cells only; patch-size filter not applied",
        "bootstrap": f"{BOOTSTRAP} resamples of {BLOCK_PX * xs.PIXEL_M} m blocks",
        "results": results,
        "caveats": cal.CAVEATS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for res in results:
        print(f"\n=== lag {res['lag']} ===")
        for sp in res["splits"]:
            print(sp["split"])
            for label, m in sp["test"].items():
                ci = m.get("delta_j_vs_current_ci95")
                print(f"  {label:10s} J={m['youden_j']} sens={m['sensitivity']} fpr={m['false_positive_rate']} "
                      f"RR={m['relative_risk']} n+={m['positive']}  [{m['rule']}]"
                      + (f"  dJ vs current {ci}" if ci else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
