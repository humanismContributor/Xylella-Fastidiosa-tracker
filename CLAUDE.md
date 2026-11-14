# Xylella Watch

Streamlit app (`app.py`) plus GitHub Actions jobs that watch for olive and almond decline that could
point to *Xylella fastidiosa*. Deployed on Streamlit Community Cloud from `main`.

## How the pieces fit
- `xylella_scanner.py`: weekly 10 m Sentinel-2 scan of the areas in `regions.json`
  (workflow `main.yml`). Per-cell history in `data/<region>/state.npz`. The workflow's plan job
  (`--plan`, numpy only) starts scan jobs only for regions whose latest season isn't final yet;
  the others just get `last_checked` updated. `state.npz` also keeps the land mask (cells the scan
  may judge); a state file without one gets a single job that adds it without re-judging any season.
- `basin_scanner.py`: yearly 100 m screen of the olive belt in `basin.json`, in 1° cells
  (workflow `basin.yml`, Sept–Nov, at most 16 parallel jobs so the weekly scan always gets a runner). Output in `data/basin/`.
- `crop_masks.py`: parcel masks from farm registers (SIGPAC Balearics so far) for non-olive profiles.
- `puglia_campaigns.py` (needs openpyxl): downloads the Osservatorio Fitosanitario's campaign
  workbooks (2013–2025, about 1.5 million samples, from the emergenzaxylella.it download page) and
  writes them as one CSV, `.cache/puglia_campaigns.csv`. Layouts differ by year; sample IDs are reused
  across 2017–2020, so only exact repeats are dropped. No licence is stated, so the raw data is never
  committed.
- `calibrate_scanner.py` (prototype, run by hand): matches those sampled trees (positives and
  negatives) to the scanner's flags in `state.npz` and writes `data/calibration/`; trees in cells the
  land mask excludes are left out when the state has one. Reads state only,
  so it never triggers a rebuild. `--portal` uses the older dati.puglia.it export (CC BY 4.0, 2020–2023).
- `sample_osavi.py` (workflow `sample_osavi.yml`, manual): the summer OSAVI of every sampled tree's
  cell for every season since 2017, plus land mask and fire flags, in `.cache/osavi/`. Not committed:
  each row is a tree's exact position and lab result. The workflow uploads them as an artifact.
  Enough to replay the decline rules offline with other thresholds; tune there, not by editing
  `PARAMS` and rebuilding.
- `replay_rules.py` (run by hand): replays the current decline rule and ~140 alternatives on every
  table with enough positives, chooses on one part (seasons up to 2020, or all other regions) and
  scores on the rest, with a 100 m block bootstrap. Writes `data/calibration/rule_replay.json`.
  Re-run after new campaigns before changing `PARAMS`.
- `scene_inventory.py` (run by hand, standard library only): phase 0 of the hyperspectral/thermal
  test. Counts usable summer overpasses of PRISMA (from a portal export, `--prisma-csv`), EnMAP,
  ECOSTRESS and Landsat 8/9 over the two Monopoli regions from catalogue metadata, with a go/no-go
  per sensor, in `data/calibration/scene_inventory.json`. Downloads no imagery.
- `sample_thermal.py` (run by hand, conda packages): Landsat 8/9 surface temperature at every tree
  in the OSAVI tables, as each summer's median anomaly against the region's judged cells, in
  `.cache/thermal/` (same row order as the OSAVI tables). Not committed, for the same reason as them.
- `sample_ecostress.py` (run by hand, conda packages plus `earthaccess`, Earthdata login in
  `~/.netrc`): the same anomaly from ECOSTRESS (70 m, midday overpasses), in `.cache/thermal_ecostress/`.
  Tiles are cached in `.cache/ecostress/` (~170 MB).
- `replay_thermal.py` (run by hand, `--sensor landsat|ecostress`): adds a thermal filter (anomaly
  level or change, 0–2 K) to the current rule on every region with both tables, chooses and scores
  as `replay_rules.py` does, with 300 m blocks, plus site checks (heat in the first summer with data,
  within 1–2 km blocks and greenness quintiles). Writes `data/calibration/thermal_replay.json` /
  `thermal_replay_ecostress.json` (aggregates only). Both sensors, re-run Nov 2026 on the 2013–2025
  campaign tables (five regions; Bisceglie has one positive and is skipped): as a filter it cuts
  false positives but loses more true ones where the scanner works (Polignano, lag 1: ΔJ CI −0.31
  to −0.06 Landsat, −0.34 to −0.15 ECOSTRESS) and adds nothing at Cagnano Varano (−0.03 to 0.00),
  so not adopted. Later positives sit on cells that were already hot before (heat in the first summer
  with data, Landsat / ECOSTRESS): Fasano 0.88 / 0.87 (0.75 / 0.73 within 1 km blocks), Cagnano Varano
  0.76 / 0.63 (0.76 / 0.78 within blocks; outbreak found in 2025), not in Polignano (0.56 / 0.43) or
  Ostuni (0.54 / 0.54; infected zone for years). A site property in some places only (cause unknown:
  irrigation, soil, sampling), at most a local risk layer, untested out of sample.
- `puglia_samples_check.py`: weekly check of the sampling dataset on dati.puglia.it (CKAN API,
  workflow `puglia_samples.yml`, state in `puglia_samples_seen.json`). Opens one issue when a file
  is added, gets a download link or changes date, with the steps to re-run the calibration; it
  never downloads the data or changes `SAMPLES_URL` itself.
- `eppo_check.py`: weekly EPPO check that opens GitHub issues (workflow `eppo.yml`). New articles
  are queued in `auto_leads.json`; `country_status.json` is still updated by hand.
- `official_check.py`: weekly check of the other sources in `official_sources.json` (workflow
  `official.yml`); one issue per changed source, and new documents queued in `auto_leads.json`.
- `auto_outbreaks.py`: runs after both checks. Claude (`claude-opus-5-5`, repository secret
  `ANTHROPIC_API_KEY`) lists the positive findings in each queued document; they are geocoded to the
  town with Nominatim and appended to `official_outbreaks.json` with `"auto": true, "reviewed": false`,
  plus an `auto-added` issue. Guards: high confidence only, countries in `country_status.json`, not
  within 5 km of a record on the map, at most 10 findings per document. Without the secret the queue
  waits. Try one document with `python auto_outbreaks.py --url <url> --dry-run`.
- `literature_check.py`: weekly OpenAlex search for new Xylella papers, one digest issue per week and
  the newest 100 in `data/literature.json` (workflow `literature.yml`). Papers are leads, not records.

## Rules that are easy to break
- Detection parameters are hashed. Changing `PARAMS` in `xylella_scanner.py` makes every region and
  every basin cell rebuild its whole history (hours of runner time). Crop profiles live in `PROFILES`;
  changing one rebuilds only the regions using it. Olive regions without a mask must keep their hash.
- A non-olive region needs a `mask` in `regions.json`; if the mask can't be built the scan stops and
  keeps the previous results, by design.
- Thresholds are inferred from the literature, not calibrated against field data. Say so in any
  user-facing text; a flagged area is a lead for inspection, never a Xylella finding.
- An automatic record stays drawn as unreviewed (dashed, no spread rings) until a person sets
  `"reviewed": true`. Eradication, absence and survey notices must never become map points.
- The app must stay light: don't draw full-size data on the map (basin layer is capped at 2,500
  squares, large files are cached with `st.cache_resource` and treated as read-only).

## Commands
- Tests (synthetic data, no network): `pip install numpy scipy pytest && python -m pytest -q`
- Run one region locally (needs the conda packages from `main.yml`): `python xylella_scanner.py --region puglia-ostuni`
- Build a crop mask: `python crop_masks.py --region mallorca-pla`
- Check the scanner against field samples: `python calibrate_scanner.py --download` (needs openpyxl
  for the download; the check itself is numpy only)
- After new sampling data: `python calibrate_scanner.py --download`, run the `sample_osavi.yml`
  workflow with `all`, then `python calibrate_scanner.py` and `python replay_rules.py`. If a new
  campaign workbook appears, add it to `CAMPAIGNS` in `puglia_campaigns.py`.

## Working conventions
- Work on a branch and open a pull request; `main` is what the public app shows.
- Add or update a test in `tests/` with every rule change.
- Cite the source (URL) for every official record and every threshold.
