from dataclasses import dataclass

import numpy as np

HIGH, MED, LOW, OFF = 0, 1, 2, 3


@dataclass(frozen=True)
class MassLODConfig:
    vis_bands: tuple = (15.0, 40.0, 90.0)
    sim_bands: tuple = (20.0, 60.0, 140.0)
    vis_caps: tuple = (50, 250, 800)
    sim_caps: tuple = (80, 400, 1200)
    hyst_dist: float = 0.10
    hyst_view: int = 15
    oov_penalty: float = 2.0
    fov: float = np.pi / 2
    view_margin: float = 0.1745
    far: float = 250.0
    cam_h: float = 1.7


def _band_tier(d, bands, prev, h):
    b = np.asarray(bands, np.float32)
    n = b.size
    t = prev.copy()
    for _ in range(n):
        t = t + ((t < n) & (d > b[np.minimum(t, n - 1)] * (1 + h))).astype(np.int8)
    for _ in range(n):
        t = t - ((t > 0) & (d < b[np.maximum(t - 1, 0)] * (1 - h))).astype(np.int8)
    return t


def _cap(tier, order, caps):
    t = tier.copy()
    for lvl, cap in enumerate(caps):
        idx = order[t[order] == lvl]
        t[idx[cap:]] = lvl + 1
    return t


class ThresholdAssigner:
    def __init__(self, cfg=None):
        self.cfg = cfg or MassLODConfig()
        self._vis = self._sim = self._hold = None

    def step(self, a, cam_pos, cam_yaw):
        c = self.cfg
        rel = a.pos - cam_pos
        d2 = np.einsum("ij,ij->i", rel, rel)
        d = np.sqrt(d2 + c.cam_h**2)
        fwd = np.array([np.cos(cam_yaw), np.sin(cam_yaw)], np.float32)
        cosang = (rel @ fwd) / np.maximum(np.sqrt(d2), 1e-6)
        seen = (cosang >= np.cos(c.fov / 2 + c.view_margin)) & (d <= c.far)

        if self._hold is None:
            self._hold = np.zeros(a.n, np.int16)
            zero = np.zeros(a.n, np.int8)
            self._vis = _band_tier(d, c.vis_bands, zero, 0.0)
            self._sim = zero
        self._hold = np.where(seen, c.hyst_view, np.maximum(self._hold - 1, 0)).astype(np.int16)
        a.in_view = self._hold > 0

        a.sig = (d * np.where(a.in_view, 1.0, c.oov_penalty)).astype(np.float32)
        # hysteresis state holds the pre-cap tier so cap pressure never becomes sticky
        self._vis = _band_tier(d, c.vis_bands, self._vis, c.hyst_dist)
        self._sim = _band_tier(a.sig, c.sim_bands, self._sim, c.hyst_dist)

        order = np.argsort(a.sig, kind="stable")
        vis = np.where(a.in_view, self._vis, OFF).astype(np.int8)
        a.vis_tier = _cap(vis, order, c.vis_caps)
        a.sim_tier = _cap(self._sim, order, c.sim_caps)
