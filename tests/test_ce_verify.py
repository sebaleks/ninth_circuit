"""Opt-in streaming Cross Verification: label calibration, applies(), model lifecycle, and the
SSE /verify/stream endpoint (backfill, hard cap, not-loaded, not-applicable). The CE itself is
faked so these run fast and offline.
"""
import json

import pytest

from rag_api import ce_verify


# ── label calibration (L-6 thresholds + CE<=0 grey override) ─────────────────

@pytest.mark.parametrize("dense,ce,exp_label,exp_color", [
    (0.50, 5.7, "Strong", "green"),          # real-strong: high+high
    (0.462, 1.00, "Strong", "green"),        # DV underscored: high dense + CE>0.8 med -> protected
    (0.43, -1.5, "Not relevant", "grey"),    # out_vocab: CE<=0 hard override
    (0.28, -1.5, "Not relevant", "grey"),    # weak nonsense: CE<=0 hard override
    (0.43, 0.5, "Moderate", "yellow"),       # high dense + low CE -> 2
    (0.28, 0.5, "Weak", "red"),              # med dense + low CE -> 1
    (0.50, 0.0, "Not relevant", "grey"),     # CE exactly 0 -> override
    (0.43, 0.8, "Moderate", "yellow"),       # CE==0.8 is LOW (boundary)
    (0.43, 0.81, "Strong", "green"),         # just above 0.8 is MED
    (0.28, 3.7, "Moderate", "yellow"),       # CE==3.7 is MED (boundary)
    (0.28, 3.71, "Strong", "green"),         # just above 3.7 is HIGH
])
def test_label_for(dense, ce, exp_label, exp_color):
    r = ce_verify.label_for(dense, ce)
    assert (r["label"], r["color"]) == (exp_label, exp_color)
    assert (r["treatment"] == "grey") == (exp_color == "grey")


def test_applies():
    assert ce_verify.applies("nexus requirement", [{"dense_score": 0.40}]) is True
    assert ce_verify.applies("what was the disposition in 23-2038", [{"dense_score": 0.40}]) is False
    assert ce_verify.applies("credible fear", [{"dense_score": 0.62}]) is False   # confident
    assert ce_verify.applies("anything", []) is False


# ── model lifecycle ──────────────────────────────────────────────────────────

def test_load_unload_state(monkeypatch):
    monkeypatch.setattr(ce_verify, "_model", None)
    assert ce_verify.is_loaded() is False
    monkeypatch.setattr(ce_verify, "_model", object())   # simulate a loaded model
    assert ce_verify.is_loaded() is True
    ce_verify.unload()
    assert ce_verify.is_loaded() is False


def test_load_is_idempotent_noop_when_loaded(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ce_verify, "_model", sentinel)
    ce_verify.load()   # must NOT re-import/replace when already loaded
    assert ce_verify._model is sentinel


# ── /verify endpoints + SSE stream ───────────────────────────────────────────

def _wire(monkeypatch, hits, ce_by_snippet, loaded=True):
    from rag_api import retrieval, guardrails
    monkeypatch.setattr(retrieval, "load", lambda: None)
    monkeypatch.setattr(retrieval, "search_with_rerank", lambda q, fetch_k, return_k: list(hits))
    monkeypatch.setattr(guardrails, "should_refuse", lambda scores: False)
    monkeypatch.setattr(ce_verify, "is_loaded", lambda: loaded)
    monkeypatch.setattr(ce_verify, "score", lambda q, snip: float(ce_by_snippet[snip]))


def _stream(query="conceptual asylum question about persecution", k=5):
    from fastapi.testclient import TestClient
    from rag_api.main import app
    events = []
    with TestClient(app) as client:
        with client.stream("GET", "/verify/stream", params={"query": query, "k": k}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    return events


def _hit(i, dense=0.40):
    return {"case_link": f"c{i}.pdf", "snippet": f"s{i}", "page": 1,
            "dense_score": dense, "score": 0.03}


def test_stream_backfills_to_five_nongrey(monkeypatch):
    # ranks 1-5 scores: [-1, 5, -1, 5, 5] -> 3 non-grey; backfill 6,7 (both 5) -> 5 non-grey, stop.
    hits = [_hit(i) for i in range(8)]
    ce = {"s0": -1, "s1": 5, "s2": -1, "s3": 5, "s4": 5, "s5": 5, "s6": 5, "s7": 5}
    _wire(monkeypatch, hits, ce)
    ev = _stream()
    results = [e for e in ev if e["event"] == "result"]
    done = [e for e in ev if e["event"] == "done"][-1]
    assert len(results) == 7                       # scored 1..7 (stopped after 5th non-grey)
    assert done["scored"] == 7 and done["nongrey"] == 5
    greys = [r for r in results if r["color"] == "grey"]
    assert {g["label"] for g in greys} == {"Not relevant"}
    assert results[0]["rank"] == 1 and results[0]["result_id"] == "c0.pdf"


def test_stream_hard_cap_at_ten(monkeypatch):
    # everything grey -> never reaches 5 non-grey -> stops at the MAX_SCORE cap (10), not the full 12
    hits = [_hit(i) for i in range(12)]
    ce = {f"s{i}": -1 for i in range(12)}
    _wire(monkeypatch, hits, ce)
    ev = _stream()
    results = [e for e in ev if e["event"] == "result"]
    done = [e for e in ev if e["event"] == "done"][-1]
    assert len(results) == 10 and done["scored"] == 10 and done["nongrey"] == 0


def test_stream_not_loaded(monkeypatch):
    _wire(monkeypatch, [_hit(0)], {"s0": 5}, loaded=False)
    ev = _stream()
    assert ev and ev[0] == {"event": "error", "reason": "not_loaded"}


def test_stream_not_applicable_for_lookup(monkeypatch):
    _wire(monkeypatch, [_hit(0)], {"s0": 5})
    ev = _stream(query="what was the disposition in 23-2038")
    assert any(e["event"] == "not_applicable" for e in ev)
    assert not any(e["event"] == "result" for e in ev)


def test_enable_disable_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    from rag_api import retrieval
    monkeypatch.setattr(retrieval, "load", lambda: None)
    monkeypatch.setattr(ce_verify, "load", lambda: monkeypatch.setattr(ce_verify, "_model", object()))
    monkeypatch.setattr(ce_verify, "_model", None)
    with TestClient(app := __import__("rag_api.main", fromlist=["app"]).app) as client:
        assert client.post("/verify/enable").json()["loaded"] is True
        assert client.post("/verify/disable").json()["loaded"] is False
