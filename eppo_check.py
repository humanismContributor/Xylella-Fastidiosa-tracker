#!/usr/bin/env python3
"""
Weekly check of the EPPO Global Database for new Xylella fastidiosa reports.

It reads two public EPPO pages:
  * the Reporting Service list for X. fastidiosa (new articles), and
  * the distribution table (country status: present / absent / transient ...).
Anything new is compared with eppo_seen.json and, when running in GitHub
Actions, posted as a GitHub issue. New articles are also queued in
auto_leads.json for auto_outbreaks.py, which adds the positive findings to
official_outbreaks.json, marked as not yet reviewed. Country-status changes are
still updated in country_status.json by hand.

Standard library only. Usage:
    python eppo_check.py            # check and open issues (needs GITHUB_TOKEN, GITHUB_REPOSITORY)
    python eppo_check.py --dry-run  # print what would be reported
    python eppo_check.py --from-files reporting.html distribution.html --dry-run   # offline test
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEN_FILE = ROOT / "eppo_seen.json"
COUNTRIES_FILE = ROOT / "country_status.json"
BASE = "https://gd.eppo.int"
REPORTING_URL = f"{BASE}/taxon/XYLEFA/reporting"
DISTRIBUTION_URL = f"{BASE}/taxon/XYLEFA/distribution"
USER_AGENT = "xylella-watch/1.0 (weekly check; github.com)"

# Words that make an article relevant to the Mediterranean map
MED_WORDS = [
    "Mediterranean", "Europe", "EU ", "Italy", "Puglia", "Apulia", "Calabria", "Basilicata", "Toscana",
    "Tuscany", "Lazio", "Sicil", "Sardegna", "Sardinia", "Spain", "Baleares", "Balearic", "Mallorca",
    "Alicante", "Valencia", "Extremadura", "Andaluc", "France", "Corse", "Corsica", "Occitanie",
    "Provence", "Alpes-Maritimes", "Portugal", "Greece", "Crete", "Malta", "Cyprus", "Croatia",
    "Slovenia", "Bosnia", "Montenegro", "Albania", "Türkiye", "Turkey", "Syria", "Lebanon", "Israel",
    "Palestin", "Jordan", "Egypt", "Libya", "Tunisia", "Algeria", "Morocco", "Monaco",
]


def fetch(url: str, tries: int = 3) -> str:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_reporting(page: str) -> list[dict]:
    """Articles as {id, num, month, title, url}, newest first. Works on table rows
    when present and falls back to bare links."""
    out, seen = [], set()
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I) or [page]
    for row in rows:
        for m in re.finditer(r'href="((?:https?://gd\.eppo\.int)?/reporting/article-(\d+))"[^>]*>(.*?)</a>',
                             row, flags=re.S | re.I):
            aid = m.group(2)
            if aid in seen:
                continue
            seen.add(aid)
            text = strip_tags(row)
            num = re.search(r"\b(\d{4}/\d{2,3})\b", text)
            month = re.search(r"\b(\d{4}-\d{2})\b", text)
            url = m.group(1) if m.group(1).startswith("http") else BASE + m.group(1)
            out.append({"id": aid, "num": num.group(1) if num else "", "month": month.group(1) if month else "",
                        "title": strip_tags(m.group(3)), "url": url})
    return out


STATUS_RE = re.compile(r"^(Present|Absent|Transient)\b.*", re.I)


def parse_distribution(page: str) -> dict:
    """{code: status}, e.g. {'IT': 'Present, restricted distribution', 'ES_ib': ...}."""
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        link = re.search(r"/taxon/XYLEFA/distribution/([A-Za-z]{2}(?:_[A-Za-z]+)?)", row)
        if not link:
            continue
        cells = [strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S | re.I)]
        status = next((c for c in cells if STATUS_RE.match(c)), None)
        if status:
            out[link.group(1)] = status
    return out


def is_mediterranean(title: str) -> bool:
    return any(w.lower() in title.lower() for w in MED_WORDS)


def open_issue(title: str, body: str, labels: list[str]) -> None:
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not (repo and token):
        print(f"[no GitHub credentials] would open issue: {title}")
        return
    data = json.dumps({"title": title, "body": body, "labels": labels}).encode()
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/issues", data=data, method="POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Accept": "application/vnd.github+json",
                                          "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"Opened issue: {json.loads(r.read())['html_url']}")


def article_issue(a: dict) -> tuple[str, str, list[str]]:
    med = is_mediterranean(a["title"])
    title = f"EPPO {a['num'] or 'report'}: {a['title']}"[:240]
    body = (f"EPPO published a new Xylella fastidiosa article"
            f"{' (' + a['month'] + ')' if a['month'] else ''}:\n\n**{a['title']}**\n{a['url']}\n\n"
            + ("It looks relevant to the Mediterranean map.\n\n" if med else
               "It doesn't name a Mediterranean country in the title; check it anyway, "
               "since general articles ('New data on quarantine pests', 'New EU regulations') can include findings.\n\n")
            + "**To do**\n- [ ] Read the article\n"
              "- [ ] New outbreak? It is added to the map automatically (look for an 'Added to the map' issue); "
              "if it was missed, add a row to `official_outbreaks.json` by hand\n"
              "- [ ] Status change for a country? Update `country_status.json`\n"
              "- [ ] Close this issue\n")
    return title, body, ["eppo-report"] + (["mediterranean"] if med else [])


def status_issue(changes: list[tuple[str, str, str, str]]) -> tuple[str, str, list[str]]:
    lines = "\n".join(f"| {name} ({code}) | {old or '(not listed)'} | {new or '(removed)'} |"
                      for code, name, old, new in changes)
    body = ("The EPPO distribution table for Xylella fastidiosa changed for countries on the map:\n\n"
            "| Country | Before | Now |\n|---|---|---|\n" + lines +
            f"\n\nSource: {DISTRIBUTION_URL}\n\n**To do**\n- [ ] Check the linked country pages\n"
            "- [ ] Update `category`, `eppo_status` and `note` in `country_status.json`\n- [ ] Close this issue\n")
    return "EPPO status change: " + ", ".join(n for _, n, _, _ in changes), body, ["eppo-report", "mediterranean"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-files", nargs=2, metavar=("REPORTING_HTML", "DISTRIBUTION_HTML"))
    a = ap.parse_args(argv)

    if a.from_files:
        rep_html, dist_html = (Path(p).read_text() for p in a.from_files)
    else:
        rep_html, dist_html = fetch(REPORTING_URL), fetch(DISTRIBUTION_URL)
    articles = parse_reporting(rep_html)
    dist = parse_distribution(dist_html)
    if len(articles) < 10 or len(dist) < 10:
        # The page layout changed or EPPO served an error page: fail loudly
        # (GitHub emails the repository owner) instead of reporting nonsense.
        print(f"Parsed only {len(articles)} articles and {len(dist)} countries; page format may have changed.")
        return 1

    countries = json.loads(COUNTRIES_FILE.read_text())["countries"]
    watched = {c["eppo"]: c["name"] for c in countries}
    med_dist = {code: st for code, st in dist.items() if code.split("_")[0] in watched}

    seen = json.loads(SEEN_FILE.read_text()) if SEEN_FILE.exists() else None
    if seen is None:
        print(f"First run: recording {len(articles)} existing articles and {len(med_dist)} country statuses.")
        new_articles, changes = [], []
    else:
        known = set(seen.get("articles", []))
        new_articles = [x for x in articles if x["id"] not in known]
        old = seen.get("distribution", {})
        changes = [(code, watched.get(code.split("_")[0], code) + ("" if "_" not in code else f" – {code}"),
                    old.get(code), med_dist.get(code))
                   for code in sorted(set(old) | set(med_dist)) if old.get(code) != med_dist.get(code)]

    for art in reversed(new_articles):
        t, b, labels = article_issue(art)
        print(f"NEW: {t}")
        if not a.dry_run:
            open_issue(t, b, labels)
    if changes:
        t, b, labels = status_issue(changes)
        print(f"CHANGED: {t}")
        if not a.dry_run:
            open_issue(t, b, labels)
    if not new_articles and not changes and seen is not None:
        print("Nothing new on EPPO.")

    if not a.dry_run:
        if new_articles:
            from auto_outbreaks import queue_leads
            queue_leads([{"url": x["url"], "title": x["title"], "verify_tls": True,
                          "source": f"EPPO Reporting Service {x['num']}".strip()} for x in new_articles])
        SEEN_FILE.write_text(json.dumps({
            "checked": dt.date.today().isoformat(),
            "articles": sorted({x["id"] for x in articles} | set((seen or {}).get("articles", [])), key=int),
            "distribution": med_dist,
        }, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
