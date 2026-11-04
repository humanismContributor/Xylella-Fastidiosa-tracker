#!/usr/bin/env python3
"""
Adds new Xylella fastidiosa outbreaks to the map from official sources, marked as
not yet reviewed.

eppo_check.py and official_check.py queue every new EPPO article and every new
official document in auto_leads.json. This script reads each queued document,
asks Claude to list the positive findings in it (place, year, subspecies, host,
with a quote as evidence), places them at the municipality centre with
OpenStreetMap's Nominatim, and appends them to official_outbreaks.json with
"auto": true and "reviewed": false. The app draws those records differently until
a person sets "reviewed": true (or deletes the row). Each addition opens a GitHub
issue so it gets that review.

Guards, because a wrong point on a public map is worse than a late one:
  * only positive findings: eradication, absence, survey and regulation notices
    and interceptions in imported plants are left out by the prompt,
  * only findings Claude marks as high confidence, in a country of
    country_status.json, with a plausible year,
  * not within DUPLICATE_KM of a record already on the map,
  * a document listing more than MAX_PER_DOCUMENT findings (the EU's full list of
    demarcated areas, say) adds nothing and gets an issue for review by hand.
A lead that fails (source down, API error) stays queued and is retried on the next
run, up to MAX_ATTEMPTS times. Without ANTHROPIC_API_KEY the queue is kept and
nothing is added, so the issues from the checks remain the only notice.

Usage
    python auto_outbreaks.py            # process the queue (needs ANTHROPIC_API_KEY; GITHUB_TOKEN for issues)
    python auto_outbreaks.py --dry-run  # print what would be added, change nothing
    python auto_outbreaks.py --url https://gd.eppo.int/reporting/article-8512 --dry-run   # try one document
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEADS_FILE = ROOT / "auto_leads.json"
RECORDS_FILE = ROOT / "official_outbreaks.json"
COUNTRIES_FILE = ROOT / "country_status.json"
MODEL = "claude-opus-5-5"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
DUPLICATE_KM = 5.0       # a finding this close to a record on the map is the same place
FRONT_KM = 30.0          # farther than this from every record: a jump, drawn with spread rings once reviewed
MAX_PER_DOCUMENT = 10
MAX_ATTEMPTS = 3
MAX_BYTES = 30_000_000   # the API takes requests up to 32 MB
FIRST_YEAR = 2010

SUBSPECIES = ["pauca", "multiplex", "fastidiosa", "sandyi", "morus", "undetermined"]
SCHEMA = {
    "type": "object",
    "properties": {
        "document_kind": {"type": "string", "enum": [
            "new_findings", "eradication_or_lifted", "absence_or_survey", "regulation_or_procedure", "other"]},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "place": {"type": "string"},
                "municipality": {"type": "string"},
                "region": {"type": "string"},
                "country_code": {"type": "string"},
                "year": {"type": "integer"},
                "subspecies": {"type": "string", "enum": SUBSPECIES},
                "host": {"type": "string"},
                "what_happened": {"type": "string"},
                "evidence": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["place", "municipality", "region", "country_code", "year", "subspecies", "host",
                         "what_happened", "evidence", "confidence"],
            "additionalProperties": False,
        }},
    },
    "required": ["document_kind", "findings"],
    "additionalProperties": False,
}
SYSTEM = """You read official plant-health documents (EPPO Reporting Service articles, decrees and \
notices of national and regional plant-health services) and list positive findings of Xylella \
fastidiosa, for a public map of outbreaks around the Mediterranean.

List a finding only when the document reports Xylella fastidiosa confirmed in plants growing in a \
named place: a new outbreak, newly infected plants, or a new infected or demarcated area set up \
because of a new detection. Leave out eradication or lifting of a demarcated area, absence or \
negative surveys, findings only in imported consignments or intercepted plants, insect vectors \
without the bacterium, research results, and changes to rules or buffer zones without a new \
detection. One finding per municipality.

Fields:
- place: the place as it should read on the map, with the province in brackets when the document \
gives it, e.g. "Bisceglie (BT)".
- municipality: the smallest named place that a map search can find, as written locally, e.g. \
"Bisceglie". Never a whole country.
- region: the region or province.
- country_code: ISO 3166-1 alpha-2, e.g. IT, ES, FR.
- year: the year of detection (the document's year if no other is given).
- subspecies and host: as reported; "undetermined" when not given.
- what_happened: one short sentence for the map, e.g. "Four infected olive trees found (June 2026)".
- evidence: a quote of at most 200 characters from the document that supports the finding.
- confidence: "high" only when the document itself states the finding plainly; "medium" or "low" \
when you inferred the place, the date or that the finding is new.

If the document reports no positive finding, return an empty list."""


# ---------------------------------------------------------------------------
# The queue, filled by eppo_check.py and official_check.py
# ---------------------------------------------------------------------------
def load_leads() -> list[dict]:
    return json.loads(LEADS_FILE.read_text(encoding="utf-8")).get("leads", []) if LEADS_FILE.exists() else []


def save_leads(leads: list[dict]) -> None:
    LEADS_FILE.write_text(json.dumps({"leads": leads}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def queue_leads(new: list[dict]) -> None:
    """Adds {url, title, source, verify_tls} leads, skipping URLs already queued."""
    leads = load_leads()
    queued = {x["url"] for x in leads}
    leads += [{**x, "attempts": 0} for x in new if x["url"] not in queued]
    save_leads(leads)


# ---------------------------------------------------------------------------
# Reading a document and extracting findings
# ---------------------------------------------------------------------------
def document_block(data: bytes) -> dict:
    """A PDF as a document block, anything else as the page's text."""
    if data[:5] == b"%PDF-":
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": base64.standard_b64encode(data).decode("ascii")}}
    from eppo_check import strip_tags
    page = data.decode("utf-8", "replace")
    page = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    return {"type": "text", "text": strip_tags(page)}


def extract_with_claude(lead: dict, data: bytes) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        # Re-runs a request declined by a safety classifier on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            document_block(data),
            {"type": "text", "text": f"Source: {lead['source']}\nTitle: {lead['title']}\nURL: {lead['url']}"},
        ]}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"Claude stopped with '{resp.stop_reason}'")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


# ---------------------------------------------------------------------------
# Placing a finding on the map
# ---------------------------------------------------------------------------
SETTLEMENTS = {"city", "town", "village", "municipality", "hamlet", "suburb", "quarter", "neighbourhood",
               "locality", "borough", "city_district", "isolated_dwelling"}


def nominatim(municipality: str, region: str, country_code: str) -> tuple[float, float] | None:
    """Centre of the town or village, or None. A plain search can return the province of the same
    name first (Matera: 26 km off), so settlements are asked for first and only they are accepted.
    Nominatim allows one request a second."""
    import urllib.request
    from eppo_check import USER_AGENT

    for q in (f"{municipality}, {region}", municipality):
        for extra in ({"featureType": "city"}, {}):
            url = NOMINATIM + "?" + urllib.parse.urlencode(
                {"q": q, "countrycodes": country_code.lower(), "format": "jsonv2", "limit": 5, **extra})
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                hits = json.loads(r.read().decode("utf-8"))
            time.sleep(1.1)
            hit = next((h for h in hits if h.get("addresstype") in SETTLEMENTS), None)
            if hit:
                return round(float(hit["lat"]), 3), round(float(hit["lon"]), 3)
    return None


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def to_records(lead: dict, result: dict, existing: list[dict], countries: dict, today: dt.date,
               geocode=nominatim) -> tuple[list[dict], list[str]]:
    """Records to add, and a note on each finding that was left out."""
    added, skipped = [], []
    for f in result.get("findings", []):
        name = f"{f['place']} ({f['country_code']})"
        country = countries.get(f["country_code"].upper())
        if f["confidence"] != "high":
            skipped.append(f"{name}: {f['confidence']} confidence")
            continue
        if not country:
            skipped.append(f"{name}: outside the map's countries")
            continue
        if not FIRST_YEAR <= f["year"] <= today.year:
            skipped.append(f"{name}: implausible year {f['year']}")
            continue
        at = geocode(f["municipality"], f["region"], f["country_code"])
        if not at:
            skipped.append(f"{name}: '{f['municipality']}' not found on OpenStreetMap")
            continue
        on_map = existing + added
        near = min(((km(at, (r["lat"], r["lon"])), r) for r in on_map), default=(math.inf, None), key=lambda x: x[0])
        if near[0] <= DUPLICATE_KM:
            skipped.append(f"{name}: already on the map as {near[1]['place']}")
            continue
        added.append({
            "id": f"auto-{f['country_code'].lower()}-{slug(f['municipality'])}-{f['year']}",
            "place": f["place"], "region": f["region"], "country": country,
            "lat": at[0], "lon": at[1], "subspecies": f["subspecies"], "host": f["host"],
            "year": f["year"], "kind": f["what_happened"], "front": near[0] > FRONT_KM,
            "source_title": lead["title"][:200], "source_url": lead["url"], "source_type": "official",
            "auto": True, "reviewed": False, "added": today.isoformat(), "evidence": f["evidence"][:300],
        })
    return added, skipped


def append_records(new: list[dict], today: dt.date) -> None:
    """Appends rows to official_outbreaks.json, one per line like the rows written by hand."""
    text = RECORDS_FILE.read_text(encoding="utf-8")
    end = text.rstrip().rstrip("}").rstrip().rstrip("]").rstrip()
    rows = ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in new)
    text = end + ("\n" if end.endswith("[") else ",\n") + rows + "\n  ]\n}\n"
    text = re.sub(r'"updated": "[^"]*"', f'"updated": "{today.isoformat()}"', text, count=1)
    json.loads(text)                            # never write a file the app can't read
    RECORDS_FILE.write_text(text, encoding="utf-8")


def added_issue(lead: dict, recs: list[dict], skipped: list[str]) -> tuple[str, str, list[str]]:
    places = ", ".join(f"{r['place']} ({r['country']})" for r in recs)
    rows = "\n".join(f"- **{r['place']}**, {r['region']}, {r['country']} ({r['lat']}, {r['lon']}): {r['kind']} "
                     f"Subspecies {r['subspecies']}, host {r['host']}, {r['year']}.\n  > {r['evidence']}\n"
                     f"  `id`: `{r['id']}`" for r in recs)
    body = (f"Added to the map automatically from **{lead['source']}**: [{lead['title']}]({lead['url']})\n\n"
            f"{rows}\n\n"
            + (f"Left out from the same document:\n" + "\n".join(f"- {s}" for s in skipped) + "\n\n" if skipped else "")
            + "The app shows these as *not yet reviewed* until confirmed.\n\n**To do**\n"
              "- [ ] Check each finding against the source (place, year, subspecies, host)\n"
              "- [ ] Check the point sits on the right municipality\n"
              "- [ ] Confirm: set `\"reviewed\": true` on the row in `official_outbreaks.json` "
              "(and `front` if it extends the known range); or delete the row if it is wrong\n"
              "- [ ] Close this issue\n")
    return f"Added to the map (unreviewed): {places}"[:240], body, ["auto-added"]


def too_many_issue(lead: dict, n: int) -> tuple[str, str, list[str]]:
    body = (f"**{lead['source']}**: [{lead['title']}]({lead['url']}) lists {n} findings, more than the "
            f"{MAX_PER_DOCUMENT} a document may add automatically, so nothing was added. A long list is usually a "
            "full register (such as the EU list of demarcated areas) that repeats places already on the map.\n\n"
            "**To do**\n- [ ] Compare the document with `official_outbreaks.json` and add new places by hand\n"
            "- [ ] Close this issue\n")
    return f"Review by hand: {n} findings in {lead['title']}"[:240], body, ["official-update"]


# ---------------------------------------------------------------------------
def main(argv=None, extract=extract_with_claude, geocode=nominatim, fetch=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--url", help="read this one document instead of the queue (try with --dry-run)")
    a = ap.parse_args(argv)
    from eppo_check import open_issue
    if fetch is None:
        from official_check import fetch

    queued = load_leads()
    leads = [{"url": a.url, "title": a.url, "source": "added by hand", "verify_tls": True}] if a.url else queued
    if not leads:
        print("No queued documents.")
        return 0
    if extract is extract_with_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{len(leads)} document(s) queued, but ANTHROPIC_API_KEY is not set; nothing added. "
              "Add it under Settings > Secrets and variables > Actions.")
        return 0

    today = dt.date.today()
    countries = {c["eppo"]: c["name"] for c in json.loads(COUNTRIES_FILE.read_text(encoding="utf-8"))["countries"]}
    existing = json.loads(RECORDS_FILE.read_text(encoding="utf-8"))["records"]
    keep, all_new = [], []
    for lead in leads:
        try:
            data = fetch(lead["url"], lead.get("verify_tls", True))
            if len(data) > MAX_BYTES:
                raise RuntimeError(f"document is {len(data) / 1e6:.0f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit")
            result = extract(lead, data)
        except Exception as exc:  # one bad document must not stop the others
            lead["attempts"] = lead.get("attempts", 0) + 1
            print(f"FAILED ({lead['attempts']}/{MAX_ATTEMPTS}) {lead['url']}: {exc}")
            if lead["attempts"] < MAX_ATTEMPTS:
                keep.append(lead)
            continue

        findings = result.get("findings", [])
        print(f"{lead['title'][:80]}: {result.get('document_kind')}, {len(findings)} finding(s)")
        if len(findings) > MAX_PER_DOCUMENT:
            issue = too_many_issue(lead, len(findings))
            new, skipped = [], []
        else:
            new, skipped = to_records(lead, result, existing + all_new, countries, today, geocode)
            issue = added_issue(lead, new, skipped) if new else None
        for r in new:
            print(f"  ADD {r['place']} ({r['country']}, {r['year']}) at {r['lat']}, {r['lon']}")
        for s in skipped:
            print(f"  skip {s}")
        all_new += new
        if issue and not a.dry_run:
            open_issue(*issue)

    if not a.dry_run:
        if all_new:
            append_records(all_new, today)
        save_leads(queued if a.url else keep)
    print(f"Added {len(all_new)} record(s); {len(keep)} document(s) left in the queue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
