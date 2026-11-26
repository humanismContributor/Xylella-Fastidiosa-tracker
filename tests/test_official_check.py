"""Retries and failure handling of official_check.py, with a fake opener (no network)."""
import email.message
import io
import json
import urllib.error

import pytest

import official_check as oc


def http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://example.org/x", code, "error", headers, io.BytesIO())


class FakeOpener:
    """Raises (or returns) the given outcomes in order, one per request."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def open(self, req, timeout=None):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return io.BytesIO(out)


@pytest.fixture
def opener(monkeypatch):
    sleeps = []
    monkeypatch.setattr(oc.time, "sleep", sleeps.append)

    def install(*outcomes):
        fake = FakeOpener(outcomes)
        monkeypatch.setattr(oc, "OPENERS", {True: fake, False: fake})
        return fake, sleeps
    return install


def test_retries_transient_errors_then_succeeds(opener):
    fake, sleeps = opener(http_error(503), http_error(429), b"ok")
    assert oc.fetch("https://example.org/x") == b"ok"
    assert fake.calls == 3 and sleeps == list(oc.BACKOFF)


def test_honours_retry_after_up_to_a_cap(opener):
    fake, sleeps = opener(http_error(429, retry_after=5), http_error(429, retry_after=9999), b"ok")
    assert oc.fetch("https://example.org/x") == b"ok"
    assert sleeps == [5, oc.MAX_RETRY_AFTER]


def test_transient_errors_through_every_try_mean_unavailable(opener):
    fake, _ = opener(http_error(503), TimeoutError("timed out"), http_error(502))
    with pytest.raises(oc.SourceUnavailable, match="example.org"):
        oc.fetch("https://example.org/x")
    assert fake.calls == 3


@pytest.mark.parametrize("code", [403, 410])
def test_other_client_errors_fail_at_once(opener, code):
    fake, sleeps = opener(http_error(code))
    with pytest.raises(RuntimeError) as exc:
        oc.fetch("https://example.org/x")
    assert not isinstance(exc.value, oc.SourceUnavailable)
    assert fake.calls == 1 and sleeps == []


def test_404_means_the_source_moved(opener):
    opener(http_error(404))
    with pytest.raises(RuntimeError, match="no longer exists"):
        oc.fetch("https://example.org/x")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs main() over sources whose checks return or raise what each test sets in `results`."""
    seen_file = tmp_path / "official_seen.json"
    monkeypatch.setattr(oc, "SEEN_FILE", seen_file)
    results = {}

    def fake_check(src, seen):
        out = results[src["id"]]
        if isinstance(out, Exception):
            raise out
        return out, []
    monkeypatch.setitem(oc.CHECKS, "fake", fake_check)
    monkeypatch.setattr(oc, "load_config", lambda: {"sources": [{"id": i, "kind": "fake", "name": i}
                                                                for i in results]})

    def go():
        code = oc.main([])
        return code, json.loads(seen_file.read_text())
    return results, go


def test_unavailable_source_warns_keeps_state_and_fails_after_some_runs(run, capsys):
    results, go = run
    results.update(up={"h": 1}, down={"h": 2})
    assert go()[0] == 0                                   # first run records both

    results["down"] = oc.SourceUnavailable("food.ec.europa.eu was unavailable after 3 tries (429)")
    for week in range(1, oc.UNAVAILABLE_WEEKS):
        code, seen = go()
        assert code == 0 and seen["unavailable"] == {"down": week}
        assert seen["sources"]["down"] == {"h": 2}        # last good state kept for the next comparison
    assert "::warning::down" in capsys.readouterr().out

    code, seen = go()                                     # still down: now someone should look
    assert code == 1 and seen["unavailable"] == {"down": oc.UNAVAILABLE_WEEKS}

    results["down"] = {"h": 2}                            # back up: the count is cleared
    code, seen = go()
    assert code == 0 and "unavailable" not in seen


def test_broken_source_still_fails_the_first_time(run, capsys):
    results, go = run
    results.update(up={"h": 1}, broken=RuntimeError("no link named 'x' on the page; the page changed"))
    code, seen = go()
    assert code == 1 and "unavailable" not in seen
    assert "Check their entries in official_sources.json" in capsys.readouterr().out
