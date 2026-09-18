import numpy as np


class Agents:
    def __init__(self, n):
        self.n = n
        z = lambda *s: np.zeros(s, np.float32)
        self.pos = z(n, 2)
        self.vel = z(n, 2)
        self.goal = z(n, 2)
        self.final = z(n, 2)
        self.speed = z(n)
        self.sig = z(n)
        self.slot = np.full(n, -1, np.int32)
        self.queuer = np.zeros(n, bool)
        self.in_view = np.zeros(n, bool)
        self.vis_tier = np.zeros(n, np.int8)
        self.sim_tier = np.zeros(n, np.int8)
