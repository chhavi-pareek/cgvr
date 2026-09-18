import csv
import os

import numpy as np

HEADER = ["scene", "agent_count", "condition", "frame", "in_view"] + [f"vis{i}" for i in range(4)] + [f"sim{i}" for i in range(4)]


class Logger:
    def __init__(self, path, scene, n, condition):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f, lineterminator="\n")
        self._w.writerow(HEADER)
        self._key = [scene, n, condition]

    def row(self, frame, a):
        vis = np.bincount(a.vis_tier, minlength=4).tolist()
        sim = np.bincount(a.sim_tier, minlength=4).tolist()
        self._w.writerow(self._key + [frame, int(a.in_view.sum())] + vis + sim)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
