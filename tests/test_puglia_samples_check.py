"""Change detection on synthetic CKAN package_show answers shaped like dati.puglia.it's."""
import copy
import json

import pytest

import puglia_samples_check as pc

OLD_URL = ("https://dati.puglia.it/ckan/dataset/6d0f/resource/fe41/download/camp_2020_2022.csv")
NEW_URL = ("https://dati.puglia.it/ckan/dataset/6d0f/resource/2ffb/download/camp_2020_2023.csv")
PACKAGE = {"success": True, "result": {
    "name": "dati-monitoraggio-xylella-fastidiosa", "metadata_modified": "2025-12-23T14:46:40.498388",
    "resources": [
        {"id": "fe41", "name": "CAMP_2020_2022.csv", "url": OLD_URL, "size": 81999815,
         "created": "2024-12-17T17:04:36.097417", "last_modified": "2024-12-17T17:04:36.058306"},
        {"id": "2ffb", "name": "CAMP_2020_2023.csv", "url": "", "size": None,
         "created": "2025-09-12T06:58:40.745686", "last_modified": "2025-09-12T00:00:00"},
    ]}}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs the check on a package dict; returns the issues opened by that run."""
    monkeypatch.setattr(pc, "SEEN_FILE", tmp_path / "seen.json")
    monkeypatch.setattr(pc, "fetch_text", lambda url: pytest.fail("tried the network"))
    issues = []
    monkeypatch.setattr(pc, "open_issue", lambda t, b, l: issues.append((t, b, l)))

    def go(pkg, *flags):
        issues.clear()
        f = tmp_path / "pkg.json"
        f.write_text(json.dumps(pkg), encoding="utf-8")
        code = pc.main(["--from-file", str(f), *flags])
        return code, list(issues)
    go.seen = tmp_path / "seen.json"
    return go


def changed(**edits):
    """PACKAGE with fields of resource 2ffb replaced."""
    pkg = copy.deepcopy(PACKAGE)
    pkg["result"]["resources"][1].update(edits)
    return pkg


def test_first_run_records_only(run):
    assert run(PACKAGE) == (0, [])
    seen = json.loads(run.seen.read_text(encoding="utf-8"))
    assert seen["metadata_modified"] == "2025-12-23T14:46:40.498388"
    assert seen["resources"]["2ffb"] == {"name": "CAMP_2020_2023.csv", "url": "", "size": None,
                                         "created": "2025-09-12T06:58:40.745686",
                                         "last_modified": "2025-09-12T00:00:00"}


def test_nothing_changed(run):
    run(PACKAGE)
    assert run(PACKAGE) == (0, [])
    meta_only = copy.deepcopy(PACKAGE)
    meta_only["result"]["metadata_modified"] = "2026-01-05T10:00:00"
    assert run(meta_only) == (0, [])              # description edits alone are no new data


def test_new_resource(run):
    run(PACKAGE)
    pkg = copy.deepcopy(PACKAGE)
    pkg["result"]["resources"].append({"id": "9abc", "name": "CAMP_2023_2024.csv", "url": NEW_URL,
                                       "size": 25_000_000, "created": "2026-10-20T08:00:00",
                                       "last_modified": "2026-10-20T08:00:00"})
    code, issues = run(pkg)
    assert code == 0 and len(issues) == 1
    title, body, labels = issues[0]
    assert "CAMP_2023_2024.csv" in title and labels == ["sampling-data"]
    assert "New file `CAMP_2023_2024.csv` (dated 2026-10-20, 25.0 MB; download: " + NEW_URL in body
    for step in ("SAMPLES_URL", "python calibrate_scanner.py --download", "sample_osavi.yml",
                 "`all`", "python replay_rules.py"):
        assert step in body
    assert run(pkg) == (0, [])                   # recorded, not reported twice


def test_download_url_appears(run):
    run(PACKAGE)
    code, issues = run(changed(url=NEW_URL, size=96_000_000))
    assert code == 0 and len(issues) == 1
    body = issues[0][1]
    assert f"`CAMP_2020_2023.csv` can now be downloaded: {NEW_URL}" in body
    assert f"[download]({NEW_URL})" in body and "96.0 MB" in body


def test_date_changes(run):
    run(PACKAGE)
    code, issues = run(changed(last_modified="2026-10-01T00:00:00"))
    assert code == 0 and len(issues) == 1
    assert "`CAMP_2020_2023.csv` last_modified changed: 2025-09-12T00:00:00 → 2026-10-01T00:00:00" in issues[0][1]


def test_dry_run_reports_without_issue_or_state(run):
    run(PACKAGE)
    before = run.seen.read_text(encoding="utf-8")
    assert run(changed(url=NEW_URL), "--dry-run") == (0, [])
    assert run.seen.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("answer", [{"success": False, "error": {"message": "Not found"}},
                                    {"success": True, "result": {"resources": []}}])
def test_error_answer_fails_without_touching_state(run, answer):
    assert run(answer) == (1, [])
    assert not run.seen.exists()
