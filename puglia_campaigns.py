#!/usr/bin/env python3
"""
Regione Puglia's Xylella monitoring campaigns, 2013-2025, as one CSV.

The Osservatorio Fitosanitario publishes one Excel workbook per campaign on
its portal (link below). The layouts differ: campaigns up to 2016-17 use older
column names and have no sample ID, later ones reorder the columns, 2021 adds
the official zone and 2024 onward the subspecies. This module downloads the
workbooks and writes them as one semicolon-separated file in the column
layout of the dati.puglia.it export, which calibrate_scanner.read_samples reads.

Usage (needs openpyxl)
    python puglia_campaigns.py              # download what's missing, then build the CSV
    python puglia_campaigns.py --force      # download every workbook again

Output: .cache/puglia_campaigns.csv (not committed; the licence is not stated
on the page, so the raw data stays out of the repository).

Notes from the page that matter for any analysis:
- Monitoring targets the free, buffer and containment zones; it does not
  measure how common the bacterium is in the infected zone.
- "Sintomo presente" includes small dry branches, which healthy trees also
  have; it does not mean olive quick decline.
- The 2015-2016 campaign is not published.
- In 2024 some positive IDs end in "*"; one carries the note "Pianta non
  abbattuta in seguito all'aggiornamento dell'area delimitata" (tree not felled
  after the demarcated area was updated). The suffix is kept in NOTA.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "camp"
OUT_FILE = ROOT / ".cache" / "puglia_campaigns.csv"

PAGE_URL = "http://www.emergenzaxylella.it/portal/portale_gestione_agricoltura/Download/mon_xf"
BASE_URL = "http://cartografia.sit.puglia.it/doc/xylella/download_monit_pauca/"
CAMPAIGNS = ("2013_2014", "2014_2015", "2016_2017", "2017_2018", "2018_2019", "2019_2020",
             "2020", "2021", "2022", "2023", "2024", "2025")
SOURCE = {
    "title": "Regione Puglia, Sezione Osservatorio Fitosanitario: Download dati Monitoraggio "
             "Xylella f. (campaigns 2013-2025)",
    "url": PAGE_URL,
    "license": "not stated on the page; reuse permission requested",
}

COLUMNS = ("ID", "DATA_RILEVAMENTO", "TIPOLOGIA", "SPECIE", "CULTIVAR", "LATITUDINE", "LONGITUDINE",
           "COMUNE", "RISULTATO", "SINTOMO", "PROVINCIA", "SUBSPECIE", "ZONA", "NOTA", "CAMPAGNA")
# Older names for the same fields
ALIASES = {"DATA_CAMPIONE": "DATA_RILEVAMENTO", "DATA_PRELIVEO": "DATA_RILEVAMENTO"}
OLIVE = "Olivo (Olea europaea)"


# --------------------------------------------------------------------------
# Normalising rows (pure, unit-tested)
# --------------------------------------------------------------------------
def _text(v) -> str:
    """Cell value as one clean field: no separators or line breaks inside."""
    if v is None:
        return ""
    return " ".join(str(v).replace(";", ",").split())


def _date(v) -> dt.date | None:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(_text(v), fmt).date()
        except ValueError:
            pass
    return None


def _coord(v) -> str:
    try:
        return f"{float(str(v).strip().replace(',', '.')):.8f}"
    except (TypeError, ValueError):
        return ""


def species(v) -> str:
    """One spelling for olive ('OLIVO', 'Olea Europaea (olivo)' and the like)."""
    s = _text(v)
    return OLIVE if ("olivo" in s.lower() or "olea europaea" in s.lower()) else s


def result(v) -> str:
    s = _text(v).lower()
    return {"positivo": "Positivo", "negativo": "Negativo"}.get(s, _text(v))


def split_id(v) -> tuple[str, str]:
    """Numeric part of a sample ID and anything after it ('1720108*' -> '1720108', '*')."""
    s = _text(v)
    digits = ""
    for ch in s:
        if not ch.isdigit():
            break
        digits += ch
    return digits, s[len(digits):].strip()


def normalize(header, rows, campaign: str):
    """Yield one dict per sample in COLUMNS layout. Rows without a date are
    skipped (counted by the caller through the 'skipped' key of the last item)."""
    names = [ALIASES.get(_text(h).upper(), _text(h).upper()) for h in header]
    col = {n: i for i, n in enumerate(names)}
    get = lambda r, k: r[col[k]] if k in col and col[k] < len(r) else None
    for k, r in enumerate(rows, start=2):
        if r is None or all(v is None for v in r):
            continue
        date = _date(get(r, "DATA_RILEVAMENTO"))
        if date is None:
            yield {"skipped": True}
            continue
        sid, note = split_id(get(r, "ID")) if "ID" in col else ("", "")
        if not sid:                                   # campaigns without IDs
            obj = _text(get(r, "OBJECTID"))
            sid = f"{campaign}-{obj or 'r' + str(k)}"
        yield {
            "ID": sid, "DATA_RILEVAMENTO": date.strftime("%d/%m/%Y"), "TIPOLOGIA": _text(get(r, "TIPOLOGIA")),
            "SPECIE": species(get(r, "SPECIE")), "CULTIVAR": _text(get(r, "CULTIVAR")),
            "LATITUDINE": _coord(get(r, "LATITUDINE")), "LONGITUDINE": _coord(get(r, "LONGITUDINE")),
            "COMUNE": _text(get(r, "COMUNE")), "RISULTATO": result(get(r, "RISULTATO")),
            "SINTOMO": _text(get(r, "SINTOMO")).capitalize(), "PROVINCIA": _text(get(r, "PROVINCIA")),
            "SUBSPECIE": _text(get(r, "SUBSPECIE")).lower(), "ZONA": _text(get(r, "ZONA")),
            "NOTA": note, "CAMPAGNA": campaign,
        }


RECORD_KEY = ("DATA_RILEVAMENTO", "LATITUDINE", "LONGITUDINE", "SPECIE", "RISULTATO")


def write_csv(out: Path, sources) -> dict:
    """Write samples from `sources` (an iterable of (campaign, header, rows)) as one
    CSV. A record repeated exactly (same ID, date, place, species and result) is
    kept once. IDs are not unique across 2017-2020: low numbers were reused for
    other samples, so a reused ID with a different record gets a suffix
    ('10' -> '10.2') and both samples are kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, stats = {}, {}
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(";".join(COLUMNS) + "\n")
        for campaign, header, rows in sources:
            s = stats.setdefault(campaign, {"rows": 0, "positive": 0, "duplicates": 0, "id_reused": 0, "skipped": 0})
            for rec in normalize(header, rows, campaign):
                if rec.get("skipped"):
                    s["skipped"] += 1
                    continue
                key = tuple(rec[k] for k in RECORD_KEY)
                records = seen.setdefault(rec["ID"], [])
                if key in records:
                    s["duplicates"] += 1
                    continue
                records.append(key)
                if len(records) > 1:
                    rec["ID"] = f"{rec['ID']}.{len(records)}"
                    s["id_reused"] += 1
                s["rows"] += 1
                s["positive"] += rec["RISULTATO"] == "Positivo"
                f.write(";".join(rec[c] for c in COLUMNS) + "\n")
    tmp.replace(out)
    return stats


# --------------------------------------------------------------------------
# Workbooks and download (network, openpyxl)
# --------------------------------------------------------------------------
def workbook_rows(path: Path):
    """(header, rows) of a campaign workbook's first sheet."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = wb.worksheets[0].iter_rows(values_only=True)
    header = next(rows)
    return header, rows


def workbook_sources(cache: Path = CACHE_DIR):
    for c in CAMPAIGNS:
        path = cache / f"CAMP_{c}.xlsx"
        if not path.exists():
            print(f"missing {path.name}; skipped", file=sys.stderr)
            continue
        print(f"reading {path.name} ...", flush=True)
        header, rows = workbook_rows(path)
        yield c, header, rows


def download(cache: Path = CACHE_DIR, force: bool = False, tries: int = 4, timeout: int = 600) -> None:
    """Fetch every campaign workbook. The server is slow, so a broken transfer
    resumes where it stopped; files are renamed into place only when complete."""
    cache.mkdir(parents=True, exist_ok=True)
    for c in CAMPAIGNS:
        dest = cache / f"CAMP_{c}.xlsx"
        if dest.exists() and not force:
            continue
        part = dest.with_suffix(".part")
        url = BASE_URL + dest.name
        for attempt in range(1, tries + 1):
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "xylella-tracker",
                                                       **({"Range": f"bytes={have}-"} if have else {})})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "ab" if have else "wb") as f:
                    if have and resp.status != 206:       # server ignored the range: start over
                        f.seek(0)
                        f.truncate()
                    total = int(resp.headers.get("Content-Length", 0)) + (have if resp.status == 206 else 0)
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                if total and part.stat().st_size < total:
                    raise OSError(f"incomplete: {part.stat().st_size} of {total} bytes")
                part.replace(dest)
                print(f"{dest.name}: {dest.stat().st_size / 1e6:.1f} MB")
                break
            except OSError as exc:
                print(f"{dest.name}: attempt {attempt} failed ({exc})", flush=True)
                if attempt == tries:
                    raise
                time.sleep(10 * attempt)


def build(out: Path = OUT_FILE, cache: Path = CACHE_DIR, force: bool = False) -> dict:
    download(cache, force=force)
    stats = write_csv(out, workbook_sources(cache))
    for c, s in stats.items():
        print(f"CAMP_{c}: {s['rows']} samples, {s['positive']} positive"
              + (f", {s['duplicates']} exact repeats dropped" if s["duplicates"] else "")
              + (f", {s['id_reused']} reused IDs kept with a suffix" if s["id_reused"] else "")
              + (f", {s['skipped']} without date" if s["skipped"] else ""))
    print(f"Wrote {out}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--force", action="store_true", help="download every workbook again")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args(argv)
    build(args.out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
