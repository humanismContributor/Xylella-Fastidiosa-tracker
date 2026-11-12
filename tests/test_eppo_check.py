"""EPPO parsing and change detection on HTML shaped like EPPO's tables."""
import json

import pytest

import auto_outbreaks as ao
import eppo_check as ec

ARTS = [("2026/216", "2026-09", "First report of <i>Xylella fastidiosa</i> subsp. multiplex in Basilicata region (Italy)", 8512),
        ("2026/173", "2026-07", "Potential vectors of Xylella fastidiosa in Argentina", 8469)] + [
        (f"2020/{i:03d}", "2020-01", f"Old article {i}", 7000 + i) for i in range(12)]
DIST = [("Italy", "Present, restricted distribution", "IT"), ("France", "Transient", "FR"),
        ("France", "Present, restricted distribution", "FR_co"), ("Albania", "Absent, confirmed by survey", "AL"),
        ("Türkiye", "Absent, confirmed by survey", "TR"), ("Israel", "Present, few occurrences", "IL"),
        ("Lebanon", "Present, restricted distribution", "LB"), ("Morocco", "Absent, confirmed by survey", "MA"),
        ("Peru", "Present, no details", "PE"), ("Brazil", "Present, no details", "BR_sp"),
        ("Iraq", "Present, no details", "IQ")]


def reporting_html(arts):
    rows = "".join(f'<tr><td>{n}</td><td>{m}</td><td><a href="/reporting/article-{i}">{t}</a></td></tr>'
                   for n, m, t, i in arts)
    return f"<table><tbody>{rows}</tbody></table>"


def dist_html(rows):
    return "<table>" + "".join(f'<tr><td>x</td><td>{n}</td><td></td><td>{s}</td>'
                               f'<td><a href="/taxon/XYLEFA/distribution/{c}">view...</a></td></tr>'
                               for n, s, c in rows) + "</table>"


def test_parsers():
    arts = ec.parse_reporting(reporting_html(ARTS))
    assert len(arts) == len(ARTS)
    assert arts[0] == {"id": "8512", "num": "2026/216", "month": "2026-09",
                       "title": "First report of Xylella fastidiosa subsp. multiplex in Basilicata region (Italy)",
                       "url": "https://gd.eppo.int/reporting/article-8512"}
    d = ec.parse_distribution(dist_html(DIST))
    assert d["FR_co"].startswith("Present") and d["TR"].startswith("Absent") and len(d) == len(DIST)
    assert ec.is_mediterranean(arts[0]["title"]) and not ec.is_mediterranean(arts[1]["title"])


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "SEEN_FILE", tmp_path / "eppo_seen.json")
    monkeypatch.setattr(ao, "LEADS_FILE", tmp_path / "auto_leads.json")
    issues = []
    monkeypatch.setattr(ec, "open_issue", lambda t, b, l: issues.append((t, l, b)))

    def write(name, text):
        p = tmp_path / name
        p.write_text(text)
        return str(p)
    return write, issues, tmp_path


def test_bootstrap_then_changes(files):
    write, issues, tmp = files
    assert ec.main(["--from-files", write("r1.html", reporting_html(ARTS[1:])),
                    write("d1.html", dist_html(DIST))]) == 0
    assert issues == []                                     # first run only records
    seen = json.loads((tmp / "eppo_seen.json").read_text())
    assert "8512" not in seen["articles"] and "FR_co" in seen["distribution"] and "PE" not in seen["distribution"]

    r2, d2 = write("r2.html", reporting_html(ARTS)), write("d2.html", dist_html(DIST + [("Greece", "Present, few occurrences", "GR")]))
    assert ec.main(["--from-files", r2, d2]) == 0
    assert len(issues) == 2
    assert issues[0][0].startswith("EPPO 2026/216") and "mediterranean" in issues[0][1]
    assert "| Greece (GR) | (not listed) | Present, few occurrences |" in issues[1][2]
    leads = json.loads((tmp / "auto_leads.json").read_text())["leads"]
    assert [x["url"] for x in leads] == ["https://gd.eppo.int/reporting/article-8512"]   # queued for the map
    assert leads[0]["source"] == "EPPO Reporting Service 2026/216"
    issues.clear()
    assert ec.main(["--from-files", r2, d2]) == 0 and issues == []


def test_error_page_fails_without_touching_state(files):
    write, _, tmp = files
    bad = write("bad.html", "<html>Service unavailable</html>")
    assert ec.main(["--from-files", bad, bad]) == 1
    assert not (tmp / "eppo_seen.json").exists()
