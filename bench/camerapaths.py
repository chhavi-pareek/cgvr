"""Five distinct scripted camera paths per scene, keyed on the normalised run time.

Keys are (t in [0,1], x, y, yaw). Every path is defined from the scene size and one focus
point (the chokepoint or centre), so the same five path shapes exist in all three scenes.
"""
import numpy as np

from sim.camera import CameraPath
from sim.scenes import SCENES

FOCUS = {"plaza": (60.0, 60.0), "hub": (60.0, 20.0), "corridor": (30.0, 5.0)}
NAMES = ("orbit", "flythrough", "static_wide", "static_choke", "sweep")


def _yaw_to(x, y, fx, fy):
    return float(np.arctan2(fy - y, fx - x))


def keys(scene, name):
    w, h = SCENES[scene].size
    fx, fy = FOCUS[scene]
    if name == "orbit":
        r = 0.45 * min(w, h)
        out = []
        for k in range(13):
            t = k / 12
            ang = 2 * np.pi * t
            x, y = fx + r * np.cos(ang), fy + r * np.sin(ang)
            out.append((t, x, y, _yaw_to(x, y, fx, fy)))
        return out
    if name == "flythrough":
        return [(0.0, 2.0, h / 2, 0.0), (1.0, w - 2.0, h / 2, 0.0)]
    if name == "static_wide":
        return [(0.0, 1.0, 1.0, _yaw_to(1.0, 1.0, fx, fy)), (1.0, 1.0, 1.0, _yaw_to(1.0, 1.0, fx, fy))]
    if name == "static_choke":
        x, y = fx - 8.0, min(fy + 3.0, h - 0.5)
        return [(0.0, x, y, _yaw_to(x, y, fx, fy)), (1.0, x, y, _yaw_to(x, y, fx, fy))]
    if name == "sweep":  # three fast passes, alternating direction
        out = []
        for p in range(3):
            fwd = p % 2 == 0
            x0, x1 = (2.0, w - 2.0) if fwd else (w - 2.0, 2.0)
            out += [(p / 3, x0, h * 0.25, 0.0 if fwd else np.pi), ((p + 1) / 3 - 1e-3, x1, h * 0.75, 0.0 if fwd else np.pi)]
        out[-1] = (1.0,) + out[-1][1:]
        return out
    raise KeyError(name)


def camera(scene, name, frames):
    return CameraPath(scene, frames, keys(scene, name))
