"""The behaviour corpus authored by a language model, as the proposal specifies (section 6.1).

    python -m latent.llm_corpus author            # ~5000 descriptions via qwen2.5:7b (resumable)
    python -m latent.llm_corpus anchor [n] [workers]   # rewrite with quoted evidence per field (t4_corpus.ipynb)
    python -m latent.llm_corpus compare           # against the template corpus

The field combinations (intent, social context, gesture, manner, place) are drawn exactly as the
template corpus draws them, so the synthetic decoder targets are the same functions of the same
fields; what changes is who writes the sentence and who rates its salience. The model is given
each combination as a situation, not as phrases to copy, and asked for one sentence in its own
words plus how visually conspicuous the behaviour is, 0-1 -- the salience prior. Batches of ten,
JSON-constrained, validated, retried; written as they finish so a long run can resume.

The first pass dropped fields: the targets are functions of all five, and the place was missing
from about a third of its sentences. `anchor` asks the model also to quote, for each field, the
words of its sentence that express it, and accepts a sentence only if every quote is in it, the
place words name the right place, and each quote is nearer (MiniLM cosine) to its own option than
to all but TOP-1 others in its list; a rejected sentence is rewritten with the problem stated, up
to TRIES attempts, and the best attempt is kept either way (flagged). A local LLM reader was too
weak to be this check (STATE.md).
"""
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from latent.corpus import DATA, GESTURE, INTENT, MOOD, PLACE, SOCIAL, _FIELD_SIZES, _salience, make_targets
from llm.policy import URL

MODEL = os.environ.get("PARITY_LLM", "qwen2.5:7b")
OUT = DATA / "corpus_llm.jsonl"
ANCHORED = DATA / "corpus_llm_anchored.jsonl"
PART = DATA / "corpus_llm_anchored.part.jsonl"
TOP, TRIES = 3, 4
# "the plaza" in an action contradicts two of the three places, so the action is shown without it
FIELDS = [("action", [(ph.replace("the plaza", "the space"), w) for ph, w in INTENT]), ("company", SOCIAL),
          ("gesture", GESTURE), ("manner", MOOD), ("place", [(p, 0.0) for p in PLACE])]
PLACE_WORDS = [r"plaza|square|piazza|courtyard|fountain|open.air|open space",
               r"transit|station|terminal|hub|platform|concourse|airport|metro|subway|depot",
               r"corridor|hallway|hall|passage|alley|walkway|tunnel|narrow"]

PROMPT = """You are writing short descriptions of individual people in a crowd, for a crowd simulation.
For each numbered situation, write ONE natural sentence of 10 to 25 words describing what that person is
doing. The sentence must convey every element of the situation -- what they intend, who they are with, their
gesture, their manner and the place -- but in your own words: do not copy the phrasing given, and vary the
sentence structure from item to item.
Then rate how visually conspicuous the behaviour would be to someone watching the crowd, from 0.0 (blends
in completely) to 1.0 (immediately draws the eye).

{items}

Reply with JSON only: {{"items": [{{"n": 1, "text": "...", "salience": 0.0}}, ...]}} with one entry per situation."""


ANCHOR = """You are writing short descriptions of individual people in a crowd, for a crowd simulation.
For each numbered situation, write ONE natural sentence of 10 to 30 words describing what that person is
doing. The sentence must convey all five elements of the situation -- action, company, gesture, manner and
place -- in your own words: a sentence that reuses two of the situation's phrases word for word is rejected
(say "amid a packed throng", not "in a dense crowd"); vary the sentence structure from item to item, and do
not add people or objects the situation does not mention. When the gesture is "no gesture", do not mention
gestures at all.
Write the sentence with the words that express each element wrapped in its tag: <action>...</action>,
<company>...</company>, <gesture>...</gesture>, <manner>...</manner>, <place>...</place>, each tag once (no
gesture tag when the gesture is "no gesture"). Two examples, only to show the tags -- do not reuse their words
or their structure:
"<company>On her own</company>, a commuter <action>strides straight for the way out</action> of the <place>busy
station</place>, <gesture>thumbing at her phone</gesture> <manner>with barely contained irritation</manner>."
"<manner>Grinning</manner>, a man <gesture>waves both arms</gesture> as he <action>ambles about without a
plan</action> <place>across the sunny square</place>, <company>his three mates close behind</company>."
Then rate how visually conspicuous the behaviour would be to someone watching the crowd, from 0.0 (blends
in completely) to 1.0 (immediately draws the eye).
Where an earlier attempt is shown, the problem says what to fix.

{items}

Reply with JSON only: {{"items": [{{"n": 1, "tagged": "...", "salience": 0.0}}, ...]}} with one entry per situation."""
TAG = re.compile(r"<(action|company|gesture|manner|place)>(.*?)</\1>", re.S)


def untag(tagged):
    """-> (plain sentence, {field: words}); a field tagged twice keeps its first words."""
    spans = {}
    for m in TAG.finditer(tagged):
        spans.setdefault(m.group(1), m.group(2).strip())
    text = re.sub(r"\s+", " ", re.sub(r"</?\w+>", "", tagged)).strip()
    return text, {name: spans.get(name, "") for name, _ in FIELDS}


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


def _ask(content, fmt, temperature, num_predict):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "stream": False,
            "format": fmt, "options": {"temperature": temperature, "num_predict": num_predict}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"])
    return {int(e["n"]): e for e in out["items"]}


def anchor_batch(batch, notes=None, temperature=0.8, retries=3):
    """notes: per combination None or (earlier sentence, its problems). -> [{text, salience, spans}] or None."""
    lines = []
    for k, ids in enumerate(batch):
        lines.append(f"{k + 1}. " + "; ".join(f"{name}: {opts[i][0]}" for (name, opts), i in zip(FIELDS, ids)))
        if notes and notes[k]:
            lines.append(f'   earlier: "{notes[k][0]}"\n   problem: {notes[k][1]}')
    prompt = ANCHOR.format(items="\n".join(lines))
    for _ in range(retries):
        try:
            got = _ask(prompt, "json", temperature, 110 * len(batch) + 80)
            rows = []
            for k in range(len(batch)):
                e = got[k + 1]
                text, spans = untag(str(e["tagged"]))
                row = {"text": text, "salience": float(e["salience"]), "spans": spans, "model": MODEL}
                if not (5 <= len(row["text"].split()) <= 45) or not 0.0 <= row["salience"] <= 1.0:
                    raise ValueError(e)
                rows.append(row)
            return rows
        except (KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    return None


_st, _opt_emb, _st_lock = None, {}, threading.Lock()


def span_similarity(spans, f):
    """MiniLM cosine of each span to every option of field f: (len(spans), n_options)."""
    global _st
    with _st_lock:
        if _st is None:
            import torch
            from sentence_transformers import SentenceTransformer

            from latent.embed import MODEL as EMB
            _st = SentenceTransformer(EMB, device="cuda" if torch.cuda.is_available() else "cpu")
        if f not in _opt_emb:
            _opt_emb[f] = _st.encode([ph for ph, _ in FIELDS[f][1]], normalize_embeddings=True)
        return _st.encode(spans, normalize_embeddings=True) @ _opt_emb[f].T


def problems(ids, row, sim=span_similarity):
    """What keeps a sentence from carrying its fields; empty if it does."""
    text, out = row["text"].lower(), []
    words = text.split()
    if "<" in text or ">" in text or len(words) < 10 or re.search(r"[.!?;]\s+\S", text):
        return ["write one well-formed sentence of 10 to 30 words, tagging each element with its own tag"]
    if "no gesture" in text:
        out.append('do not write "no gesture"; leave gestures out')
    copied = [name for (name, opts), i in zip(FIELDS[:4], ids)
              if len(opts[i][0].split()) > 1 and opts[i][0].lower() in text]
    if len(copied) >= 2:
        out.append(f"the {', '.join(copied)} copy the situation's wording; put them in your own words")
    for f, ((name, opts), i) in enumerate(zip(FIELDS, ids)):
        span = row["spans"][name]
        if name == "gesture" and i == 0:
            continue
        if not span:
            out.append(f"the {name} ({opts[i][0]}) is missing or untagged")
        elif span.lower() not in text:
            out.append(f'the {name} words "{span}" are not in the sentence')
        elif name == "place":
            hit = [bool(re.search(w, span, re.I)) for w in PLACE_WORDS]
            if not hit[i] or sum(hit) > 1:
                out.append(f'the place words "{span}" do not clearly name {opts[i][0]}')
        else:
            s = sim([span], f)[0]
            if (s > s[i]).sum() >= TOP:
                out.append(f'the {name} words "{span}" read more like "{opts[int(s.argmax())][0]}" than "{opts[i][0]}"')
    return out


def anchor(n=5000, workers=1, batch=3):
    todo_ids = combos(n)
    log = {}
    if PART.exists():
        for l in PART.open():
            a = json.loads(l)
            log.setdefault(a["k"], []).append(a)
    lock = threading.Lock()

    def open_(k):
        v = log.get(k, [])
        return not any(a["ok"] for a in v) and len(v) < TRIES

    def work(ks):
        notes = [(log[k][-1]["text"], "; ".join(log[k][-1]["problems"])) if k in log else None for k in ks]
        rows = anchor_batch([todo_ids[k] for k in ks], notes)
        if rows is None:
            return
        found = [problems(todo_ids[k], r) for k, r in zip(ks, rows)]
        with lock:
            for k, r, p in zip(ks, rows, found):
                a = {"k": k, **r, "problems": p, "ok": not p}
                log.setdefault(k, []).append(a)
                f.write(json.dumps(a) + "\n")
            f.flush()

    t0 = time.perf_counter()
    with PART.open("a") as f:
        for sweep in range(TRIES + 2):                       # +2: batches whose reply never parsed
            todo = [k for k in range(n) if open_(k)]
            if not todo:
                break
            with ThreadPoolExecutor(workers) as ex:
                for j, _ in enumerate(ex.map(work, [todo[b:b + batch] for b in range(0, len(todo), batch)])):
                    if j % 25 == 0:
                        ok = sum(any(a["ok"] for a in v) for v in log.values())
                        tries = sum(len(v) for v in log.values())
                        print(f"sweep {sweep}: {min((j + 1) * batch, len(todo))}/{len(todo)}; {ok}/{n} accepted, "
                              f"{tries} attempts, {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    first = {tuple(r["ids"]): r for r in map(json.loads, OUT.open())} if OUT.exists() else {}
    with ANCHORED.open("w") as f:
        for k, ids in enumerate(todo_ids):
            if k in log:
                a = log[k][min(range(len(log[k])), key=lambda j: (len(log[k][j]["problems"]), j))]
                row = {"text": a["text"], "salience": a["salience"], "spans": a["spans"], "ok": a["ok"],
                       "tries": len(log[k])}
            else:                                            # no reply ever parsed: the first-pass sentence
                row = {"text": first[ids]["text"], "salience": first[ids]["salience"], "spans": {}, "ok": False, "tries": 0}
            f.write(json.dumps({**row, "ids": list(ids)}) + "\n")
    rows = [json.loads(l) for l in ANCHORED.open()]
    print(f"accepted {sum(r['ok'] for r in rows)}/{n}; mean attempts {np.mean([r['tries'] for r in rows]):.2f}")
    finish(ANCHORED, "llm_anchored")


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


def finish(path=OUT, tag="llm"):
    """Targets for the authored corpus: the template's functions of the same field ids."""
    rows = [json.loads(l) for l in path.open()]
    ids = np.array([r["ids"] for r in rows])
    tg = make_targets(ids, np.random.default_rng(0))
    np.savez(DATA / f"targets_{tag}.npz", salience=np.array([r["salience"] for r in rows], np.float32), **tg)
    print(f"{len(rows)} authored descriptions; targets written")


def compare(corpora=("template", "llm", "llm_anchored"), seeds=(0, 1, 2)):
    """Lexical and embedding diversity, near-duplicates, how well each field can be read back from the
    embedding (5-fold logistic probe), the salience prior against the template's hand-set one, and the
    latent pipeline's own gate (held-out decoder error) on each corpus."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from latent.corpus import ANIMATION, BEHAVIOUR, load
    from latent.embed import embed
    from latent.train import get_data, head_err, head_var, train

    corpora = [c for c in corpora if (DATA / ("corpus.jsonl" if c == "template" else f"corpus_{c}.jsonl")).exists()]
    for name in corpora:
        rows = [json.loads(l) for l in (DATA / ("corpus.jsonl" if name == "template" else f"corpus_{name}.jsonl")).open()]
        ids = np.array([r["ids"] for r in rows])
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
        print(f"{name:12s}: {len(texts)} texts, {n_tok / len(texts):.1f} words each; distinct words {len(uni)}, "
              f"distinct-1 {len(uni) / n_tok:.3f}, distinct-2 {len(bi) / n_tok:.3f}; embedding dispersion "
              f"(mean pairwise cosine distance) {1 - float((e[pa] * e[pb]).sum(1).mean()):.3f}; nearest-neighbour "
              f"cosine median {np.median(nn):.3f}, share > 0.95 {np.mean(nn > 0.95):.3f}", flush=True)
        probe = [cross_val_score(LogisticRegression(C=10, max_iter=3000), e, ids[:, f], cv=5).mean() for f in range(5)]
        print(f"{name:12s}: fields read back from the embedding: "
              + "; ".join(f"{fn} {a:.2f}" for (fn, _), a in zip(FIELDS, probe)), flush=True)
        if "ok" in rows[0]:
            print(f"{name:12s}: {np.mean([r['ok'] for r in rows]):.3f} passed every check, "
                  f"mean attempts {np.mean([r['tries'] for r in rows]):.2f}", flush=True)
        if name != "template":
            base = np.array([_salience(tuple(i), np.random.default_rng(0)) for i in ids])
            print(f"{name:12s}: salience prior mean {sal.mean():.2f} sd {sal.std():.2f}; hand-set mean {base.mean():.2f} "
                  f"sd {base.std():.2f}; Spearman {spearman(sal, base):.2f}", flush=True)
    for name in corpora:
        x, y, tr, te = get_data(name)
        bs, an = [], []
        for seed in seeds:
            m, _, _ = train(seed, True, corpus=name)
            with torch.no_grad():
                pred = m.decode(m.enc(x[te]))
            bs.append(head_err(pred, y, te, BEHAVIOUR, head_var(y, tr, BEHAVIOUR)).item())
            an.append(head_err(pred, y, te, ANIMATION, head_var(y, tr, ANIMATION)).item())
        print(f"{name:12s} latent gate, nested AE, k=16, held-out normalised error over seeds {list(seeds)}: "
              f"behaviour {np.mean(bs):.4f} +- {np.std(bs):.4f}, animation {np.mean(an):.4f} +- {np.std(an):.4f}",
              flush=True)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "author"
    {"author": author, "finish": finish, "anchor": anchor, "compare": compare}[cmd](*map(int, sys.argv[2:]))
