import numpy as np

_KEYS = {
    "plaza": [(0.0, 20, 20, 0.785), (0.25, 100, 20, 2.356), (0.5, 100, 100, 3.927), (0.75, 20, 100, 5.498), (1.0, 20, 20, 7.069)],
    "hub": [(0.0, 5, 20, 0.0), (0.5, 50, 20, 0.0), (0.75, 80, 8, 0.6), (1.0, 95, 20, 0.0)],
    "corridor": [(0.0, 1, 5, 0.0), (0.6, 35, 5, 0.0), (1.0, 1, 5, 3.1416)],
}


class CameraPath:
    def __init__(self, scene, frames, keys=None):
        k = np.asarray(_KEYS[scene] if keys is None else keys, np.float64)
        self._t = k[:, 0] * max(frames - 1, 1)
        self._k = k

    def at(self, frame):
        x, y, yaw = (float(np.interp(frame, self._t, self._k[:, i])) for i in (1, 2, 3))
        return np.array([x, y], np.float32), yaw
