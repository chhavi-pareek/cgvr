"""Behaviour corpus: template-grammar descriptions with salience priors and synthetic decoder targets.

Targets are deterministic functions of the categorical fields (seeded random loadings plus one
intent x social interaction and per-sample noise). They stand in for real behaviour/animation
parameters, which do not exist yet.
"""
import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).parent / "data"
N_GESTURE_CLIPS = 8
GROUPS = {"vel": 3, "gaze": 3, "gesture": N_GESTURE_CLIPS, "gait": 5}
BEHAVIOUR = ("vel", "gaze")
ANIMATION = ("gesture", "gait")

# (phrase, salience base)
INTENT = [
    ("walking purposefully toward the exit", 0.55), ("strolling aimlessly", 0.2),
    ("hurrying to catch a departure", 0.8), ("jogging across the open space", 0.75),
    ("waiting patiently in line", 0.25), ("lingering near the fountain", 0.15),
    ("browsing the stalls", 0.3), ("approaching a friend", 0.5),
    ("veering around an obstacle", 0.6), ("turning back the way they came", 0.6),
    ("stopping abruptly", 0.85), ("following a guide", 0.4),
    ("crossing the plaza diagonally", 0.45), ("hugging the edge of the space", 0.2),
    ("scanning the crowd for someone", 0.65), ("dashing through a gap", 0.9),
]
SOCIAL = [
    ("alone", 0.0), ("with a companion", 0.1), ("in a group of four", 0.15),
    ("holding a child's hand", 0.15), ("following a leader", 0.1),
    ("against the flow of oncoming people", 0.2), ("in a dense crowd", 0.05),
    ("beside the queue", 0.05), ("chatting with a neighbour", 0.1), ("ignoring everyone around", 0.0),
]
GESTURE = [
    ("no gesture", 0.0), ("waving", 0.15), ("pointing ahead", 0.12), ("checking a phone", 0.05),
    ("glancing at a watch", 0.05), ("carrying a heavy bag", 0.03), ("gesturing while talking", 0.08),
    ("shrugging", 0.06), ("beckoning someone", 0.14), ("shielding eyes from the sun", 0.04),
]
MOOD = [
    ("calmly", 0.0), ("anxiously", 0.1), ("distractedly", 0.0), ("cheerfully", 0.05),
    ("wearily", -0.05), ("impatiently", 0.1), ("cautiously", 0.02), ("energetically", 0.08),
]
PLACE = ["in a plaza", "in a transit hub", "in a narrow corridor"]
FRAMES = [
    "{mood} {intent} {place}, {social}, {gesture}",
    "{Social}, {mood} {intent}; {gesture} {place}",
    "{Intent} {mood} {place} while {gesture}, {social}",
    "{Gesture}, {social}, {mood} {intent} {place}",
]

_LOAD_SEED = 7  # fixes the loadings; the corpus seed only draws the samples
_FIELD_SIZES = [len(INTENT), len(SOCIAL), len(GESTURE), len(MOOD), len(PLACE)]


def _cap(s):
    return s[0].upper() + s[1:]


def _describe(ids, frame):
    i, s, g, m, p = ids
    return FRAMES[frame].format(
        intent=INTENT[i][0], social=SOCIAL[s][0], gesture=GESTURE[g][0], mood=MOOD[m][0], place=PLACE[p],
        Intent=_cap(INTENT[i][0]), Social=_cap(SOCIAL[s][0]), Gesture=_cap(GESTURE[g][0]),
    )


def _salience(ids, rng):
    i, s, g, m, _ = ids
    return float(np.clip(INTENT[i][1] + SOCIAL[s][1] + GESTURE[g][1] + MOOD[m][1] + rng.normal(0, 0.03), 0, 1))


def make_targets(ids, rng, noise=0.05):
    """ids: (N, 5) int array -> dict of (N, d) float32 arrays per decoder output group."""
    lrng = np.random.default_rng(_LOAD_SEED)
    dim = sum(GROUPS.values())
    load = [lrng.normal(0, 1, (n, dim)) for n in _FIELD_SIZES]
    inter = lrng.normal(0, 0.7, (_FIELD_SIZES[0], _FIELD_SIZES[1], dim))
    z = sum(load[f][ids[:, f]] for f in range(len(load))) + inter[ids[:, 0], ids[:, 1]]
    z = np.tanh(0.6 * z) + rng.normal(0, noise, z.shape)
    out, o = {}, 0
    for name, d in GROUPS.items():
        out[name] = z[:, o:o + d]
        o += d
    out["vel"] = out["vel"] + np.array([1.0, 0.0, 0.0])  # speed scale centred at 1
    e = np.exp(3 * out["gesture"])
    out["gesture"] = e / e.sum(1, keepdims=True)  # blend weights on the simplex
    return {k: v.astype(np.float32) for k, v in out.items()}


def build(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    seen, texts, ids_l, sal = set(), [], [], []
    while len(texts) < n:
        ids = tuple(int(rng.integers(s)) for s in _FIELD_SIZES)
        text = _describe(ids, int(rng.integers(len(FRAMES))))
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
        ids_l.append(ids)
        sal.append(_salience(ids, rng))
    ids = np.array(ids_l)
    return texts, ids, np.array(sal, np.float32), make_targets(ids, rng)


def load():
    texts = [json.loads(l)["text"] for l in (DATA / "corpus.jsonl").open()]
    t = np.load(DATA / "targets.npz")
    return texts, {k: t[k] for k in GROUPS}, t["salience"]


if __name__ == "__main__":
    texts, ids, sal, tg = build()
    DATA.mkdir(exist_ok=True)
    with (DATA / "corpus.jsonl").open("w") as f:
        for t, i, s in zip(texts, ids, sal):
            f.write(json.dumps({"text": t, "salience": round(float(s), 4), "ids": i.tolist()}) + "\n")
    np.savez(DATA / "targets.npz", salience=sal, **tg)
    print(len(texts), "descriptions;", texts[0], "|", texts[1])
