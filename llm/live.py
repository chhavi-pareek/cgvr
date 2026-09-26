"""A live model server for the station scheduler: real calls to the local LLM, one at a time.

The same interface as llm.schedule.Server (submit / backlog / poll; calls, busy_s, lat, P), so a Run
can be driven in real time -- one simulated second per wall second -- with replies landing when the
model actually answers. The reply is the model's live distribution; at temperature 0 it matches
the table the benchmark measures true drift against, and `mismatch` counts where it did not.
"""
import threading
import time
from collections import deque

import numpy as np

from .policy import ask


class LiveServer:
    def __init__(self, P, latency, t0):
        self.P = [P]
        self.lat = [np.asarray(latency)]          # expected latency: the scheduler's estimate
        self.t0 = t0                              # wall clock of step 0
        self.q = deque()
        self.done = []
        self.cv = threading.Condition()
        self.calls = self.small_calls = 0
        self.busy_s = 0.0
        self.running_since = None
        self.spans = []                           # (start, end) of every call, seconds since t0
        self.mismatch = 0
        self.stopped = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            with self.cv:
                while not self.q and not self.stopped:
                    self.cv.wait()
                if self.stopped:
                    return
                agent, ctx, _, _ = self.q.popleft()
                self.running_since = time.perf_counter() - self.t0
            p, dt = ask(ctx)
            with self.cv:
                end = time.perf_counter() - self.t0
                self.busy_s += dt
                self.spans.append((end - dt, end))
                self.running_since = None
                self.mismatch += int(np.abs(p - self.P[0][ctx]).max() > 1e-3)
                self.done.append((end, agent, ctx, p))

    def submit(self, agent, ctx, step, model=0):
        with self.cv:
            self.q.append((int(agent), int(ctx), float(step), int(model)))
            self.calls += 1
            self.cv.notify()

    def backlog(self, step):
        mean = float(self.lat[0].mean())
        with self.cv:
            now = time.perf_counter() - self.t0
            run = max(self.running_since + mean - now, 0.0) if self.running_since is not None else 0.0
            return run + len(self.q) * mean

    def poll(self, step):
        """Replies that have arrived by now, as (agent, ctx, p, model)."""
        with self.cv:
            now = time.perf_counter() - self.t0
            out = [(a, c, p, 0) for t, a, c, p in self.done if t <= now]
            self.done = [d for d in self.done if d[0] > now]
        return out

    def duty(self, a, b):
        """Share of [a, b) (seconds since t0) the model spent computing."""
        with self.cv:
            return sum(max(0.0, min(e, b) - max(s, a)) for s, e in self.spans) / max(b - a, 1e-9)

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify()
