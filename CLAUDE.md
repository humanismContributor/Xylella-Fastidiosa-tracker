# Xylella Watch

Streamlit app (`app.py`) plus GitHub Actions jobs that watch for olive and almond decline that could
point to *Xylella fastidiosa*. Deployed on Streamlit Community Cloud from `main`.

## How the pieces fit
- `xylella_scanner.py`: weekly 10 m Sentinel-2 scan of the areas in `regions.json`
  (workflow `main.yml`). Per-cell history in `data/<region>/state.npz`.
- `basin_scanner.py`: yearly 100 m screen of the olive belt in `basin.json`, in 1° cells
  (workflow `basin.yml`, Sept–Nov, at most 16 parallel jobs so the weekly scan always gets a runner). Output in `data/basin/`.
- `crop_masks.py`: parcel masks from farm registers (SIGPAC Balearics so far) for non-olive profiles.
- `calibrate_scanner.py` (prototype, run by hand): matches Regione Puglia's sampled trees (open data,
  CC BY 4.0, positives and negatives, 2020–2023) to the scanner's flags in `state.npz` and writes
  `data/calibration/`. Reads state only, so it never triggers a rebuild. The 82 MB CSV is cached in
  `.cache/` and not committed.
- `sample_osavi.py` (workflow `sample_osavi.yml`, manual): the summer OSAVI of every sampled tree's
  cell for every season since 2017, plus land mask and fire flags, in `data/calibration/osavi/`.
  Enough to replay the decline rules offline with other thresholds; tune there, not by editing
  `PARAMS` and rebuilding.
- `replay_rules.py` (run by hand): replays the current decline rule and ~140 alternatives on those
  tables, chooses on one part (seasons up to 2020, or one region) and scores on the other, with a
  100 m block bootstrap. Writes `data/calibration/rule_replay.json`. First run (Oct 2026, 2020–23
  samples): no rule beats the current one in the sampling season; greenness-dependent thresholds
  add nothing over fixed ones. Re-run after new campaigns before changing `PARAMS`.
- `puglia_samples_check.py`: weekly check of the sampling dataset on dati.puglia.it (CKAN API,
  workflow `puglia_samples.yml`, state in `puglia_samples_seen.json`). Opens one issue when a file
  is added, gets a download link or changes date, with the steps to re-run the calibration; it
  never downloads the data or changes `SAMPLES_URL` itself.
- `eppo_check.py`: weekly EPPO check that opens GitHub issues (workflow `eppo.yml`). Records are
  added to `official_outbreaks.json` / `country_status.json` by hand after review, never automatically.
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
- The app must stay light: don't draw full-size data on the map (basin layer is capped at 2,500
  squares, large files are cached with `st.cache_resource` and treated as read-only).

## Commands
- Tests (synthetic data, no network): `pip install numpy scipy pytest && python -m pytest -q`
- Run one region locally (needs the conda packages from `main.yml`): `python xylella_scanner.py --region puglia-ostuni`
- Build a crop mask: `python crop_masks.py --region mallorca-pla`
- Check the scanner against field samples (numpy only): `python calibrate_scanner.py --download`
- After new sampling data: `python calibrate_scanner.py --download`, run the `sample_osavi.yml`
  workflow with `all`, then `python calibrate_scanner.py` and `python replay_rules.py`

## Working conventions
- Work on a branch and open a pull request; `main` is what the public app shows.
- Add or update a test in `tests/` with every rule change.
- Cite the source (URL) for every official record and every threshold.
