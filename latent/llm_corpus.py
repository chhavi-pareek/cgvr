"""The behaviour corpus authored by a language model, as the proposal specifies (section 6.1).

    python -m latent.llm_corpus author            # ~5000 descriptions via qwen2.5:7b (resumable)
    python -m latent.llm_corpus compare           # against the template corpus

The field combinations (intent, social context, gesture, manner, place) are drawn exactly as the
template corpus draws them, so the synthetic decoder targets are the same functions of the same
fields; what changes is who writes the sentence and who rates its salience. The model is given
each combination as a situation, not as phrases to copy, and asked for one sentence in its own
words plus how visually conspicuous the behaviour is, 0-1 -- the salience prior. Batches of ten,
JSON-constrained, validated, retried; written as they finish so a long run can resume.
"""
import json
import sys
import time
import urllib.request

import numpy as np

from latent.corpus import DATA, GESTURE, INTENT, MOOD, PLACE, SOCIAL, _FIELD_SIZES, _salience, make_targets
from llm.policy import URL

MODEL = "qwen2.5:7b"
OUT = DATA / "corpus_llm.jsonl"

PROMPT = """You are writing short descriptions of individual people in a crowd, for a crowd simulation.
For each numbered situation, write ONE natural sentence of 10 to 25 words describing what that person is
doing. Use your own words: do not copy the phrasing given, and vary the sentence structure from item to item.
Then rate how visually conspicuous the behaviour would be to someone watching the crowd, from 0.0 (blends
in completely) to 1.0 (immediately draws the eye).

{items}

Reply with JSON only: {{"items": [{{"n": 1, "text": "...", "salience": 0.0}}, ...]}} with one entry per situation."""


def combos(n=5000, seed=0):
    """Unique field combinations, drawn as latent/corpus.build draws them."""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n:
        ids = tuple(int(rng.integers(s)) for s in _FIELD_SIZES)
        if ids not in seen:
            seen.add(ids)
            out.append(ids)
    return out


def situation(ids):
    i, s, g, m, p = ids
    return (f"intent: {INTENT[i][0]}; company: {SOCIAL[s][0]}; gesture: {GESTURE[g][0]}; "
            f"manner: {MOOD[m][0]}; place: {PLACE[p]}")


def author_batch(batch, temperature=0.8, retries=3):
    items = "\n".join(f"{k + 1}. {situation(ids)}" for k, ids in enumerate(batch))
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(items=items)}], "stream": False,
            "format": "json", "options": {"temperature": temperature, "num_predict": 60 * len(batch) + 80}}
    for _ in range(retries):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            out = json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"])
            got = {int(e["n"]): e for e in out["items"]}
            rows = []
            for k in range(len(batch)):
                e = got[k + 1]
                text, sal = str(e["text"]).strip(), float(e["salience"])
                if not (5 <= len(text.split()) <= 40) or not 0.0 <= sal <= 1.0:
                    raise ValueError(e)
                rows.append((text, sal))
            return rows
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def author(n=5000, batch=10):
    done = [json.loads(l) for l in OUT.open()] if OUT.exists() else []
    todo = combos(n)[len(done):]
    t0, k0 = time.perf_counter(), len(done)
    with OUT.open("a") as f:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            rows = author_batch(chunk)
            if rows is None:                                  # one at a time, as a last resort
                rows = []
                for ids in chunk:
                    r = author_batch([ids])
                    rows.append(r[0] if r else (None, None))
            for ids, (text, sal) in zip(chunk, rows):
                if text is None:
                    print(f"gave up on {ids}", flush=True)
                    continue
                f.write(json.dumps({"text": text, "salience": round(sal, 4), "ids": list(ids)}) + "\n")
            f.flush()
            k = k0 + b + len(chunk)
            if (b // batch) % 10 == 0:
                rate = (k - k0) / max(time.perf_counter() - t0, 1e-9)
                print(f"{k}/{n} descriptions, {rate:.2f}/s, ~{(n - k) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
    finish()


def finish():
    """Targets for the authored corpus: the template's functions of the same field ids."""
    rows = [json.loads(l) for l in OUT.open()]
    ids = np.array([r["ids"] for r in rows])
    tg = make_targets(ids, np.random.default_rng(0))
    np.savez(DATA / "targets_llm.npz", salience=np.array([r["salience"] for r in rows], np.float32), **tg)
    print(f"{len(rows)} authored descriptions; targets written")


def compare(seeds=(0, 1, 2)):
    """Lexical and embedding diversity, near-duplicates, the salience prior against the template's
    hand-set one, and the latent pipeline's own gate (held-out decoder error) on each corpus."""
    import torch

    from latent.corpus import ANIMATION, BEHAVIOUR, load
    from latent.embed import embed
    from latent.train import get_data, head_err, head_var, train

    rows = [json.loads(l) for l in OUT.open()]
    ids = np.array([r["ids"] for r in rows])
    for name in ("template", "llm"):
        texts, _, sal = load(name)
        toks = [t.lower().split() for t in texts]
        uni = {w for t in toks for w in t}
        bi = {(a, b) for t in toks for a, b in zip(t, t[1:])}
        n_tok = sum(len(t) for t in toks)
        e = embed(texts)
        sim = e @ e.T
        np.fill_diagonal(sim, -1.0)
        nn = sim.max(1)
        rng = np.random.default_rng(0)
        pa, pb = rng.integers(len(e), size=(2, 20000))
        print(f"{name:8s}: {len(texts)} texts, {n_tok / len(texts):.1f} words each; distinct words {len(uni)}, "
              f"distinct-1 {len(uni) / n_tok:.3f}, distinct-2 {len(bi) / n_tok:.3f}; embedding dispersion "
              f"(mean pairwise cosine distance) {1 - float((e[pa] * e[pb]).sum(1).mean()):.3f}; nearest-neighbour "
              f"cosine median {np.median(nn):.3f}, share > 0.95 {np.mean(nn > 0.95):.3f}")
    base = np.array([_salience(tuple(i), np.random.default_rng(0)) for i in ids])
    llm_sal = np.array([r["salience"] for r in rows])
    print(f"salience prior: LLM mean {llm_sal.mean():.2f} sd {llm_sal.std():.2f}; hand-set mean {base.mean():.2f} "
          f"sd {base.std():.2f}; Spearman {spearman(llm_sal, base):.2f}")
    for name in ("template", "llm"):
        x, y, tr, te = get_data(name)
        bs, an = [], []
        for seed in seeds:
            m, _, _ = train(seed, True, corpus=name)
            with torch.no_grad():
                pred = m.decode(m.enc(x[te]))
            bs.append(head_err(pred, y, te, BEHAVIOUR, head_var(y, tr, BEHAVIOUR)).item())
            an.append(head_err(pred, y, te, ANIMATION, head_var(y, tr, ANIMATION)).item())
        print(f"{name:8s} latent gate, nested AE, k=16, held-out normalised error over seeds {list(seeds)}: "
              f"behaviour {np.mean(bs):.4f} +- {np.std(bs):.4f}, animation {np.mean(an):.4f} +- {np.std(an):.4f}")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "author"
    {"author": author, "finish": finish, "compare": compare}[cmd]()
