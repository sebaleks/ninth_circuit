"""Confidence indicator: color model (confidence.py) + CE annotator gating (cross_encoder.py).

Pure-logic tests (no real CE load) — the cross-encoder is faked via monkeypatch so these run
fast and offline.
"""
from rag_api import confidence as conf
from rag_api import cross_encoder


def test_color_anchors():
    # real strong: dense High + CE High -> green
    assert conf.confidence(0.50, 5.7, "nexus requirement")["color"] == "green"
    # DV under-scored: dense High + CE Med (>=0.80) -> green (protected from a CE error)
    assert conf.confidence(0.462, 1.0, "domestic violence partner asylum")["color"] == "green"
    # out_vocab caught (dense High + CE Low) -> yellow; weak-dense nonsense -> red
    assert conf.confidence(0.43, -1.5, "werewolf gang")["color"] == "yellow"
    assert conf.confidence(0.28, -1.5, "ghost reincarnation")["color"] == "red"


def test_band_and_dense_only():
    # above band (dense>0.50): CE ignored, dense alone
    r = conf.confidence(0.62, -5.0, "credible fear")
    assert r["color"] == "green" and r["signals"] == "dense-only"
    # dense Low (shown only if upstream didn't abstain) -> red
    assert conf.confidence(0.10, None, "x")["color"] == "red"


def test_lookup_routing_and_exact_match():
    # exact docket match -> green regardless of dense
    r = conf.confidence(0.28, -9.0, "disposition in 23-2038", exact_match=True)
    assert r["color"] == "green" and r["signals"] == "lookup/exact"
    # lookup w/o exact match -> colored by dense tier (Med -> yellow), CE never used
    r2 = conf.confidence(0.28, -9.0, "matter of A-B-", exact_match=False)
    assert r2["color"] == "yellow" and "lookup" in r2["signals"]


def test_is_lookup():
    assert conf.is_lookup("what was the disposition in 23-2038")
    assert conf.is_lookup("Cordoba v. Garland")
    assert conf.is_lookup("Matter of A-B-")
    assert conf.is_lookup("8 U.S.C. 1158")
    assert not conf.is_lookup("what is the nexus requirement for asylum")


def test_annotate_routes_lookup_around_ce(monkeypatch):
    calls = []

    class FakeCE:
        def rerank(self, q, passages):
            calls.append((q, list(passages)))
            return [9.0] * len(list(passages))

    monkeypatch.setattr(cross_encoder, "_ce", lambda: FakeCE())
    hits = [{"snippet": "s", "case_link": "x/23-2038.pdf", "dense_score": 0.28, "score": 0.03}]
    cross_encoder.annotate("disposition in 23-2038", hits)
    assert calls == []                                 # CE routed around for lookups
    assert hits[0]["ce_score"] is None
    assert hits[0]["confidence"]["color"] == "green"   # exact docket match


def test_annotate_fires_and_batches_on_band_nonlookup(monkeypatch):
    captured = {"n": 0, "batches": 0}

    class FakeCE:
        def rerank(self, q, passages):
            p = list(passages)
            captured["n"] += len(p)
            captured["batches"] += 1
            return [-2.0] * len(p)

    monkeypatch.setattr(cross_encoder, "_ce", lambda: FakeCE())
    hits = [{"snippet": f"s{i}", "case_link": f"x/{i}.pdf", "dense_score": 0.30, "score": 0.03}
            for i in range(7)]
    cross_encoder.annotate("a conceptual question about persecution and nexus", hits)
    assert captured["n"] == 7                           # every hit scored
    assert captured["batches"] == 2                     # batched at BATCH=5 (5 + 2)
    assert all(h["ce_score"] == -2.0 for h in hits)
    # dense Med (0.30) + CE Low (-2.0) -> red
    assert all(h["confidence"]["color"] == "red" for h in hits)


def test_annotate_skips_ce_when_no_in_band_hit(monkeypatch):
    class FakeCE:
        def rerank(self, q, passages):
            raise AssertionError("CE must not fire when all hits are out of band")

    monkeypatch.setattr(cross_encoder, "_ce", lambda: FakeCE())
    hits = [{"snippet": "s", "case_link": "x/1.pdf", "dense_score": 0.62, "score": 0.03}]  # >0.50
    cross_encoder.annotate("a conceptual question about persecution", hits)
    assert hits[0]["ce_score"] is None
    assert hits[0]["confidence"]["color"] == "green"   # dense-only, High
