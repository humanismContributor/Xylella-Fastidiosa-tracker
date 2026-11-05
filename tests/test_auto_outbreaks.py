"""Automatic map additions: the guards, the file format and the queue, with Claude,
the document fetch and the geocoder replaced by stand-ins (no network, no API key)."""
import json
import shutil

import pytest

import auto_outbreaks as ao
import eppo_check as ec
from conftest import ROOT

LEAD = {"url": "https://gd.eppo.int/reporting/article-9000", "title": "New outbreaks in Puglia (Italy)",
        "source": "EPPO Reporting Service 2026/250", "verify_tls": True}
PLACES = {"Andria": (41.227, 16.296), "Bisceglie": (41.241, 16.502), "Valencia": (39.47, -0.376),
          "Massa": (44.035, 10.139)}


def finding(municipality, **kw):
    return {"place": f"{municipality} (XX)", "municipality": municipality, "region": "Puglia", "country_code": "IT",
            "year": 2026, "subspecies": "pauca", "host": "olive", "what_happened": "Infected olive trees found.",
            "evidence": f"X. fastidiosa was confirmed in {municipality}.", "confidence": "high", **kw}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for name in ("official_outbreaks.json", "country_status.json"):
        shutil.copy(ROOT / name, tmp_path / name)
    monkeypatch.setattr(ao, "RECORDS_FILE", tmp_path / "official_outbreaks.json")
    monkeypatch.setattr(ao, "COUNTRIES_FILE", tmp_path / "country_status.json")
    monkeypatch.setattr(ao, "LEADS_FILE", tmp_path / "auto_leads.json")
    issues = []
    monkeypatch.setattr(ec, "open_issue", lambda t, b, l: issues.append((t, b, l)))
    ao.queue_leads([LEAD])
    return tmp_path, issues


def run(findings, argv=(), fetch=lambda url, verify: b"<html>article</html>"):
    return ao.main(list(argv), extract=lambda lead, data: {"document_kind": "new_findings", "findings": findings},
                   geocode=lambda m, r, c: PLACES.get(m), fetch=fetch)


def records(tmp):
    return json.loads((tmp / "official_outbreaks.json").read_text(encoding="utf-8"))["records"]


def test_adds_only_findings_that_pass_the_guards(repo):
    tmp, issues = repo
    before = records(tmp)
    assert run([finding("Andria"),
                finding("Massa", confidence="medium"),                 # not sure enough
                finding("Valencia", country_code="US"),                # not a country on the map
                finding("Bisceglie"),                                  # already on the map
                finding("Andria", year=1990),                          # implausible year
                finding("Nowhere")]) == 0                              # not found by the geocoder
    after = records(tmp)
    assert after[:len(before)] == before                               # hand-written rows untouched
    new = after[len(before):]
    assert [r["place"] for r in new] == ["Andria (XX)"]
    r = new[0]
    assert r["auto"] is True and r["reviewed"] is False and r["country"] == "Italy"
    assert (r["lat"], r["lon"]) == PLACES["Andria"] and r["source_url"] == LEAD["url"]
    assert r["front"] is False                                         # 17 km from Bisceglie: not a jump
    assert len(issues) == 1 and issues[0][2] == ["auto-added"] and "Andria" in issues[0][0]
    assert "already on the map as Bisceglie (BT)" in issues[0][1]
    assert json.loads((tmp / "auto_leads.json").read_text())["leads"] == []


def test_file_keeps_one_row_per_line(repo):
    tmp, _ = repo
    run([finding("Andria")])
    text = (tmp / "official_outbreaks.json").read_text(encoding="utf-8")
    assert text.endswith('"auto": true, "reviewed": false, "added": "' + ao.dt.date.today().isoformat()
                         + '", "evidence": "X. fastidiosa was confirmed in Andria."}\n  ]\n}\n')
    assert f'"updated": "{ao.dt.date.today().isoformat()}"' in text


def test_two_findings_in_one_run_are_not_duplicated(repo):
    tmp, _ = repo
    n = len(records(tmp))
    run([finding("Andria"), finding("Andria", place="Andria – centro")])
    assert len(records(tmp)) == n + 1


def test_long_lists_are_left_for_a_person(repo):
    tmp, issues = repo
    n = len(records(tmp))
    run([finding("Andria")] * (ao.MAX_PER_DOCUMENT + 1))
    assert len(records(tmp)) == n
    assert issues[0][0].startswith("Review by hand") and issues[0][2] == ["official-update"]


def test_failed_documents_are_retried_then_dropped(repo):
    tmp, issues = repo

    def down(url, verify):
        raise RuntimeError("could not fetch")
    for attempt in range(1, ao.MAX_ATTEMPTS):
        run([], fetch=down)
        assert json.loads((tmp / "auto_leads.json").read_text())["leads"][0]["attempts"] == attempt
    run([], fetch=down)
    assert json.loads((tmp / "auto_leads.json").read_text())["leads"] == [] and issues == []


def test_without_api_key_nothing_changes(repo, monkeypatch):
    tmp, issues = repo
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    before = (tmp / "official_outbreaks.json").read_text(encoding="utf-8")
    assert ao.main([]) == 0
    assert (tmp / "official_outbreaks.json").read_text(encoding="utf-8") == before
    assert len(json.loads((tmp / "auto_leads.json").read_text())["leads"]) == 1 and issues == []


def test_dry_run_changes_nothing(repo):
    tmp, issues = repo
    before = (tmp / "official_outbreaks.json").read_text(encoding="utf-8")
    run([finding("Andria")], argv=["--dry-run"])
    assert (tmp / "official_outbreaks.json").read_text(encoding="utf-8") == before and issues == []
    assert len(json.loads((tmp / "auto_leads.json").read_text())["leads"]) == 1


def test_queue_skips_urls_already_queued(repo):
    tmp, _ = repo
    ao.queue_leads([LEAD, {**LEAD, "url": LEAD["url"] + "0"}])
    assert len(json.loads((tmp / "auto_leads.json").read_text())["leads"]) == 2


def test_pdfs_go_to_claude_as_documents():
    assert ao.document_block(b"%PDF-1.7 ...")["type"] == "document"
    block = ao.document_block(b"<html><script>x()</script><p>Focolaio a <b>Andria</b></p></html>")
    assert block == {"type": "text", "text": "Focolaio a Andria"}


def test_every_schema_field_is_required():
    item = ao.SCHEMA["properties"]["findings"]["items"]
    assert set(item["required"]) == set(item["properties"]) and item["additionalProperties"] is False
