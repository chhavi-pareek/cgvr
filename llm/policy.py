"""The expensive tier (a local LLM, via Ollama) and the cheap ones (surrogates).

LLM policy: the person's situation is written into a prompt with the six actions as lettered
options, and the model's log-probabilities for the first answer token give its EXACT action
distribution -- one forward pass, no sampling noise. Temperature 0, so the same prompt always
gives the same distribution.

Ground truth: `build_table` asks the model about every one of the 576 contexts once and caches
the answers. That table is a MEASURING device -- it is how the benchmark knows what the LLM
would have said at a decision the scheduler served with a surrogate. The scheduler itself never
reads it; everything it knows about the LLM comes from calls it paid for.

Surrogates (cost microseconds, not a model call):
  Marginal   the average of the replies seen so far, per persona
  Distilled  an additive log-linear model -- one weight per (feature value, action), 19 x 6 --
             fitted to the soft labels of every reply received. It cannot memorise the 576-way
             table (it has no interaction terms), so what it gets right it gets by generalising.
"""
import hashlib
import json
import math
import os
import time
import urllib.request

import numpy as np

from .station import (ACTIONS, N_ACT, N_CTX, N_K, N_P, N_Q, N_T, N_Z, PERSONAS, Q_WORDS, T_WORDS, ZONES,
                      ctx_parts)

LETTERS = "ABCDEF"
MODEL = "qwen2.5:7b"
URL = "http://localhost:11434/api/chat"
FLOOR = 1e-4          # probability given to an option outside the model's top-20 tokens
LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "logs")


def prompt(c):
    p, t, k, q, z = (int(v) for v in ctx_parts(c))
    opts = "\n".join(f"{LETTERS[i]}) {a}" for i, a in enumerate(ACTIONS))
    return (f"You are simulating one person in a busy train station. Decide what they do next.\n"
            f"Person: {PERSONAS[p]}.\n"
            f"Their train leaves in {T_WORDS[t]}.\n"
            f"They {'already have' if k else 'do not have'} a ticket.\n"
            f"The ticket queue is {Q_WORDS[q]}.\n"
            f"They are at {ZONES[z]}.\n"
            f"What do they do next?\n{opts}\nAnswer with a single letter.")


def ask(c, model=MODEL, timeout=120):
    """(distribution over the six actions, seconds taken) for context c."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt(c)}], "stream": False,
            "options": {"temperature": 0, "num_predict": 1}, "logprobs": True, "top_logprobs": 20}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    dt = time.perf_counter() - t0
    p = np.full(N_ACT, FLOOR)
    for tok in out["logprobs"][0]["top_logprobs"]:
        s = tok["token"].strip()
        if len(s) == 1 and s in LETTERS:
            p[LETTERS.index(s)] = max(p[LETTERS.index(s)], math.exp(tok["logprob"]))
    return p / p.sum(), dt


def table_path(model=MODEL):
    h = hashlib.sha1((model + prompt(0) + prompt(N_CTX - 1)).encode()).hexdigest()[:10]
    return os.path.join(LOGS, f"llm_table_{model.replace(':', '_')}_{h}.npz")


def build_table(model=MODEL, force=False, log=print):
    """The LLM's exact distribution at every context, plus the latency of every call."""
    path = table_path(model)
    if os.path.exists(path) and not force:
        t = np.load(path)
        return t["P"], t["latency"]
    ask(0, model)                                   # load the model before timing anything
    P = np.empty((N_CTX, N_ACT))
    lat = np.empty(N_CTX)
    for c in range(N_CTX):
        P[c], lat[c] = ask(c, model)
        if log and c % 96 == 95:
            log(f"  {c + 1}/{N_CTX} contexts, median call {np.median(lat[:c + 1]) * 1e3:.0f} ms")
    os.makedirs(LOGS, exist_ok=True)
    np.savez(path, P=P, latency=lat, model=model)
    return P, lat


def kl(q, p):
    """Per-row KL(q || p) in nats."""
    q, p = np.atleast_2d(q), np.atleast_2d(p)
    return (q * (np.log(q) - np.log(p))).sum(-1)


# -- surrogates ---------------------------------------------------------------------------

def features(c):
    """One-hot of each context factor separately: 6 + 4 + 2 + 3 + 4 = 19 columns, no interactions."""
    parts = ctx_parts(np.atleast_1d(c))
    cols = []
    for v, n in zip(parts, (N_P, N_T, N_K, N_Q, N_Z)):
        cols.append(np.eye(n)[v])
    return np.hstack(cols)


class Marginal:
    """The average reply per persona (the global average before a persona has been seen)."""

    def __init__(self):
        self.sum = np.zeros((N_P, N_ACT))
        self.cnt = np.zeros(N_P)

    def observe(self, c, p):
        per = ctx_parts(np.atleast_1d(c))[0]
        np.add.at(self.sum, per, np.atleast_2d(p))
        np.add.at(self.cnt, per, 1)

    def __call__(self, c):
        per = ctx_parts(np.atleast_1d(c))[0]
        glob = self.sum.sum(0) + 1e-9
        q = np.where(self.cnt[per, None] > 0, self.sum[per] / np.maximum(self.cnt[per, None], 1), glob / glob.sum())
        q = np.maximum(q, FLOOR)
        return q / q.sum(-1, keepdims=True)


class Distilled:
    """Additive log-linear model fitted to every reply's soft label (minimises KL(p || q))."""

    def __init__(self, l2=1e-3):
        self.W = np.zeros((19, N_ACT))
        self.b = np.zeros(N_ACT)
        self.X, self.Y = [], []
        self.l2 = l2

    def observe(self, c, p):
        self.X.append(features(c))
        self.Y.append(np.atleast_2d(p))

    def fit(self, iters=200, lr=0.5):
        if not self.X:
            return
        X, Y = np.vstack(self.X), np.vstack(self.Y)
        for _ in range(iters):
            z = X @ self.W + self.b
            z -= z.max(1, keepdims=True)
            q = np.exp(z)
            q /= q.sum(1, keepdims=True)
            g = (q - Y) / len(X)
            self.W -= lr * (X.T @ g + self.l2 * self.W)
            self.b -= lr * g.sum(0)

    def __call__(self, c):
        z = features(c) @ self.W + self.b
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        q = np.maximum(q / q.sum(1, keepdims=True), FLOOR)
        return q / q.sum(1, keepdims=True)
