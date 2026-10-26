#!/usr/bin/env python3
"""
Weekly check of Regione Puglia's open Xylella sampling data for new campaigns.

calibrate_scanner.py checks the scanner against the sampled trees in the
dataset "Dati monitoraggio Xylella fastidiosa" on dati.puglia.it (CC BY 4.0).
This script reads the dataset's CKAN record and compares each file (resource)
with puglia_samples_seen.json. When a file is added, gets a download URL (some
are listed before they can be downloaded, e.g. CAMP_2020_2023.csv), changes
its date or is removed, it opens one GitHub issue listing what changed and the
steps to re-run the calibration. A change of the dataset's own
metadata_modified alone is recorded but not reported: it also moves when only
the description is edited. The first run only records what already exists.

Standard library only. Usage:
    python puglia_samples_check.py            # check and open an issue (needs GITHUB_TOKEN, GITHUB_REPOSITORY)
    python puglia_samples_check.py --dry-run  # print the current state and what would be reported
    python puglia_samples_check.py --from-file package_show.json --dry-run   # offline test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from eppo_check import open_issue
from official_check import fetch_text

ROOT = Path(__file__).resolve().parent
SEEN_FILE = ROOT / "puglia_samples_seen.json"
DATASET = "dati-monitoraggio-xylella-fastidiosa"
API_URL = f"https://dati.puglia.it/ckan/api/3/action/package_show?id={DATASET}"
DATASET_URL = f"https://dati.puglia.it/ckan/dataset/{DATASET}"


def parse_package(data: dict) -> dict:
    """The state to keep: the dataset's metadata_modified and, per resource id,
    its name, url, last_modified, created and size. Raises on anything that
    doesn't look like a package_show answer for this dataset."""
    if not data.get("success") or not isinstance(data.get("result"), dict):
        raise ValueError(f"CKAN answered without a result: {str(data.get('error') or data)[:200]}")
    pkg = data["result"]
    if not isinstance(pkg.get("resources"), list) or not pkg["resources"]:
        raise ValueError("the dataset lists no files; the API or the dataset changed")
    resources = {}
    for r in pkg["resources"]:
        resources[r.get("id") or r.get("name", "")] = {
            "name": r.get("name") or "",
            "url": r.get("url") or "",
            "last_modified": r.get("last_modified") or "",
            "created": r.get("created") or "",
            "size": r.get("size"),
        }
    return {"metadata_modified": pkg.get("metadata_modified") or "", "resources": resources}


def compare(old: dict, new: dict) -> list[str]:
    """What changed between two states, one Markdown line per file."""
    changes = []
    before, now = old.get("resources", {}), new.get("resources", {})
    for rid, r in now.items():
        o = before.get(rid)
        label = f"`{r['name'] or rid}`"
        if o is None:
            where = f"download: {r['url']}" if r["url"] else "listed without a download link yet"
            changes.append(f"New file {label} (dated {date_of(r)}, {size_text(r['size'])}; {where})")
            continue
        if r["url"] and not o.get("url"):
            changes.append(f"{label} can now be downloaded: {r['url']}")
        elif r["url"] != o.get("url"):
            changes.append(f"{label} has a new download URL: {r['url'] or '(none)'} (was {o.get('url') or '(none)'})")
        for key in ("last_modified", "created"):
            if r[key] != o.get(key, ""):
                changes.append(f"{label} {key} changed: {o.get(key) or '(none)'} → {r[key] or '(none)'}"
                               + (f" ({size_text(o.get('size'))} → {size_text(r['size'])})"
                                  if r["size"] != o.get("size") else ""))
                break
    for rid, o in before.items():
        if rid not in now:
            changes.append(f"`{o.get('name') or rid}` was removed from the dataset (was {o.get('url') or 'not downloadable'})")
    return changes


def date_of(r: dict) -> str:
    return (r.get("last_modified") or r.get("created") or "no date")[:10]


def size_text(size) -> str:
    return f"{size / 1e6:.1f} MB" if isinstance(size, (int, float)) else "size unknown"


def describe(state: dict) -> str:
    lines = [f"dataset metadata_modified {state['metadata_modified'] or '(none)'}"]
    for r in state["resources"].values():
        lines.append(f"  {r['name']}: dated {date_of(r)}, {size_text(r['size'])}, "
                     + (r["url"] or "no download link"))
    return "\n".join(lines)


def build_issue(changes: list[str], state: dict) -> tuple[str, str, list[str]]:
    rows = "\n".join(f"| {r['name']} | {date_of(r)} | {size_text(r['size'])} | "
                     f"{'[download](' + r['url'] + ')' if r['url'] else 'no link yet'} |"
                     for r in state["resources"].values())
    body = ("Regione Puglia's Xylella sampling data changed on dati.puglia.it:\n\n"
            + "\n".join(f"- {c}" for c in changes)
            + f"\n\nFiles now listed ({DATASET_URL}):\n\n| File | Date | Size | Link |\n|---|---|---|---|\n{rows}\n\n"
            "**To do** (see \"After new sampling data\" in CLAUDE.md)\n"
            "- [ ] If the new file has a different URL, update `SAMPLES_URL` in `calibrate_scanner.py`"
            " (and check that its columns still match)\n"
            "- [ ] `python calibrate_scanner.py --download`\n"
            "- [ ] Run the `sample_osavi.yml` workflow with `all`\n"
            "- [ ] `python calibrate_scanner.py`\n"
            "- [ ] `python replay_rules.py`\n"
            "- [ ] Close this issue\n")
    return "New Puglia sampling data: " + "; ".join(changes)[:200], body, ["sampling-data"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-file", metavar="PACKAGE_SHOW_JSON", help="read a saved API answer instead")
    a = ap.parse_args(argv)

    try:
        raw = Path(a.from_file).read_text(encoding="utf-8") if a.from_file else fetch_text(API_URL)
        state = parse_package(json.loads(raw))
    except (ValueError, RuntimeError) as exc:
        # The API changed or served an error page: fail loudly (GitHub emails the
        # repository owner) and keep the last state.
        print(f"Could not read the dataset: {exc}")
        return 1
    print("Now: " + describe(state))

    seen = json.loads(SEEN_FILE.read_text(encoding="utf-8")) if SEEN_FILE.exists() else None
    if seen is None:
        print(f"First run: recording {len(state['resources'])} files.")
        changes = []
    else:
        changes = compare(seen, state)
        if changes:
            t, b, labels = build_issue(changes, state)
            print(f"CHANGED: {t}\n" + "\n".join(f"  {c}" for c in changes))
            if not a.dry_run:
                open_issue(t, b, labels)
        elif seen.get("metadata_modified") != state["metadata_modified"]:
            print("Only the dataset metadata changed; no new data.")
        else:
            print("No new sampling data.")

    if not a.dry_run:
        SEEN_FILE.write_text(json.dumps({"checked": dt.date.today().isoformat(), **state},
                                        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
