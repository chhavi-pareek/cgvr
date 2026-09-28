"""The LLM-authored corpus: same field draws as the template corpus, and strict reply validation."""
import io
import json

import numpy as np

import latent.llm_corpus as L
from latent.corpus import _FIELD_SIZES


def test_combinations_are_unique_deterministic_and_the_template_draws():
    a, b = L.combos(500), L.combos(500)
    assert a == b and len(set(a)) == 500
    rng = np.random.default_rng(0)
    first = tuple(int(rng.integers(s)) for s in _FIELD_SIZES)
    assert a[0] == first
    assert all(0 <= v < s for ids in a for v, s in zip(ids, _FIELD_SIZES))


def _reply(items):
    body = json.dumps({"message": {"content": json.dumps({"items": items})}}).encode()
    return io.BytesIO(body)


def test_a_well_formed_reply_is_taken(monkeypatch):
    batch = L.combos(3)
    items = [{"n": k + 1, "text": f"A person walks calmly across the busy plaza number {k}.", "salience": 0.1 * k}
             for k in range(3)]
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda req, timeout=0: _reply(items))
    rows = L.author_batch(batch)
    assert [s for _, s in rows] == [0.0, 0.1, 0.2]


def test_bad_replies_are_retried_then_refused_not_invented(monkeypatch):
    batch = L.combos(2)
    calls = []

    def bad(req, timeout=0):
        calls.append(1)
        # item 2 missing on the first try, salience out of range on the second, too short on the third
        k = len(calls)
        if k == 1:
            return _reply([{"n": 1, "text": "A person walks calmly across the busy plaza.", "salience": 0.3}])
        if k == 2:
            return _reply([{"n": 1, "text": "A person walks calmly across the busy plaza.", "salience": 0.3},
                           {"n": 2, "text": "Someone hurries toward the exit with a bag.", "salience": 1.7}])
        return _reply([{"n": 1, "text": "Walks.", "salience": 0.3}, {"n": 2, "text": "Runs.", "salience": 0.5}])

    monkeypatch.setattr(L.urllib.request, "urlopen", bad)
    assert L.author_batch(batch, retries=3) is None
    assert len(calls) == 3


def test_tags_give_the_sentence_and_one_span_per_field():
    text, spans = L.untag("<company>On her own</company>, a commuter <action>strides for the exit</action> of the "
                          "<place>busy station</place> <manner>impatiently</manner>.")
    assert text == "On her own, a commuter strides for the exit of the busy station impatiently."
    assert spans == {"action": "strides for the exit", "company": "On her own", "gesture": "",
                     "manner": "impatiently", "place": "busy station"}


MEANS = {"strides for the way out": 0, "on her own": 0, "nervously": 1, "calmly": 0}


def _sim(spans, f):
    return np.array([np.eye(len(L.FIELDS[f][1]))[MEANS.get(sp.lower(), 0)] for sp in spans])


def test_a_sentence_passes_only_if_it_carries_every_field(monkeypatch):
    ids = (0, 0, 0, 1, 1)          # exit / alone / no gesture / anxiously / transit hub
    spans = {"action": "strides for the way out", "company": "On her own", "gesture": "",
             "manner": "nervously", "place": "busy station"}
    text = "On her own, a commuter strides for the way out of the busy station nervously, eyes darting about"
    ok = {"text": text, "spans": spans}
    assert L.problems(ids, ok, _sim) == []
    assert "place" in L.problems(ids, {"text": text.replace("station", "plaza"),
                                       "spans": {**spans, "place": "busy plaza"}}, _sim)[0]
    assert "missing" in L.problems(ids, {**ok, "spans": {**spans, "action": ""}}, _sim)[0]
    assert "not in the sentence" in L.problems(ids, {**ok, "spans": {**spans, "manner": "calmly"}}, _sim)[0]
    one = text.replace("strides for the way out", "walking purposefully toward the exit")
    assert L.problems(ids, {"text": one, "spans": {**spans, "action": "walking purposefully toward the exit"}}, _sim) == []
    ids3 = (0, 1, 2, 1, 1)         # exit / with a companion / pointing ahead / anxiously / transit hub
    three = "Walking purposefully toward the exit with a companion, nervously pointing ahead, through the busy station"
    assert "own words" in L.problems(ids3, {"text": three, "spans": {**spans, "action": "Walking purposefully toward "
                                     "the exit", "company": "with a companion", "gesture": "pointing ahead"}}, _sim)[0]
    assert "well-formed" in L.problems(ids, {**ok, "text": text + " <dense crowd>"}, _sim)[0]
    assert "well-formed" in L.problems(ids, {**ok, "text": text + ". In a transit hub."}, _sim)[0]
    monkeypatch.setattr(L, "TOP", 1)
    wrong = {"text": text + " calmly", "spans": {**spans, "manner": "calmly"}}
    assert "read more like" in L.problems(ids, wrong, _sim)[0]


def test_anchor_rewrites_what_fails_keeps_the_best_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "DATA", tmp_path)
    monkeypatch.setattr(L, "PART", tmp_path / "part.jsonl")
    monkeypatch.setattr(L, "ANCHORED", tmp_path / "corpus_llm_anchored.jsonl")
    monkeypatch.setattr(L, "OUT", tmp_path / "none.jsonl")
    monkeypatch.setattr(L, "problems", lambda ids, row: ["bad"] if row["text"].startswith("bad") else [])
    calls = []

    def fake(batch, notes=None):
        calls.append(notes)
        return [{"text": ("bad" if notes[k] is None and k == 1 else "good") + f" {len(calls)}", "salience": 0.5,
                 "spans": {}} for k in range(len(batch))]

    monkeypatch.setattr(L, "anchor_batch", fake)
    L.anchor(n=3, batch=3)
    assert len(calls) == 2 and calls[1] == [("bad 1", "bad")]
    rows = [json.loads(l) for l in L.ANCHORED.open()]
    assert [r["ids"] for r in rows] == [list(i) for i in L.combos(3)]
    assert [(r["text"], r["ok"], r["tries"]) for r in rows] == [("good 1", True, 1), ("good 2", True, 2),
                                                               ("good 1", True, 1)]
    assert (tmp_path / "targets_llm_anchored.npz").exists()
    L.anchor(n=3, batch=3)
    assert len(calls) == 2
