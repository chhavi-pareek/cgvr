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
