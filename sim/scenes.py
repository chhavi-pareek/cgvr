import numpy as np

ARRIVE = 1.0


def _arrived(a):
    d = a.goal - a.pos
    return np.flatnonzero(np.einsum("ij,ij->i", d, d) < ARRIVE**2)


class Plaza:
    name = "plaza"
    size = (120.0, 120.0)

    def spawn(self, a, rng):
        hi = np.asarray(self.size)
        a.pos[:] = rng.uniform(0, hi, (a.n, 2))
        a.goal[:] = rng.uniform(0, hi, (a.n, 2))
        a.final[:] = a.goal
        a.speed[:] = rng.uniform(1.0, 1.6, a.n)

    def constrain(self, a, prev):
        np.clip(a.pos, 0, np.asarray(self.size, np.float32), out=a.pos)

    def after_step(self, a, frame, rng):
        idx = _arrived(a)
        if idx.size:
            a.goal[idx] = rng.uniform(0, np.asarray(self.size), (idx.size, 2))


class Hub:
    name = "hub"
    size = (100.0, 40.0)
    window = (60.0, 20.0)
    spacing = 0.8
    service = 45

    def _slot_pos(self, slot):
        s = np.asarray(slot, np.float32)
        return np.stack(
            [self.window[0] - 1.0 - s * self.spacing, np.full(s.shape, self.window[1], np.float32)], -1
        )

    def _exit(self, rng, k):
        w, h = self.size
        return np.stack([np.full(k, w), rng.uniform(0, h, k)], -1)

    def spawn(self, a, rng):
        w, h = self.size
        q = self.qmax = max(1, min(a.n // 5, 60))
        a.pos[:] = rng.uniform(0, (w * 0.6, h), (a.n, 2))
        a.speed[:] = rng.uniform(1.0, 1.6, a.n)
        a.queuer[:q] = True
        a.slot[:q] = np.arange(q)
        a.pos[:q] = self._slot_pos(a.slot[:q]) + rng.uniform(-0.3, 0.3, (q, 2))
        a.goal[:q] = self._slot_pos(a.slot[:q])
        a.goal[q:] = self._exit(rng, a.n - q)
        a.final[:] = a.goal

    def constrain(self, a, prev):
        np.clip(a.pos, 0, np.asarray(self.size, np.float32), out=a.pos)

    def after_step(self, a, frame, rng):
        q = np.flatnonzero(a.slot >= 0)
        if frame % self.service == 0 and q.size:
            front = q[np.argmin(a.slot[q])]
            a.slot[q] -= 1
            a.slot[front] = -1
            a.goal[front] = self._exit(rng, 1)[0]
            q = q[q != front]
            a.goal[q] = self._slot_pos(a.slot[q])
        idx = _arrived(a)
        for i in idx[a.slot[idx] < 0]:
            a.pos[i] = (rng.uniform(0, 5), rng.uniform(0, self.size[1]))
            n_q = int((a.slot >= 0).sum())
            if a.queuer[i] and n_q < self.qmax:
                a.slot[i] = n_q
                a.goal[i] = self._slot_pos(n_q)
            else:
                a.goal[i] = self._exit(rng, 1)[0]


class Corridor:
    name = "corridor"
    size = (60.0, 10.0)
    door_x = (29.7, 30.3)
    gap = (4.2, 5.8)
    exit_x = 58.0
    via = np.asarray((31.0, 5.0), np.float32)

    def _aim(self, a):
        a.goal[:] = np.where((a.pos[:, 0] < 30.0)[:, None], self.via, a.final)

    def spawn(self, a, rng):
        a.pos[:] = rng.uniform(0, (28.0, 10.0), (a.n, 2))
        a.final[:] = np.stack([np.full(a.n, self.exit_x), rng.uniform(3, 7, a.n)], -1)
        a.speed[:] = rng.uniform(1.0, 1.6, a.n)
        self._aim(a)

    def constrain(self, a, prev):
        np.clip(a.pos, 0, np.asarray(self.size, np.float32), out=a.pos)
        x0, x1 = self.door_x
        g0, g1 = self.gap
        y = a.pos[:, 1]
        blocked = (a.pos[:, 0] >= x0) & (prev[:, 0] < x0) & ((y < g0) | (y > g1))
        a.pos[blocked, 0] = x0 - 0.01
        slab = (a.pos[:, 0] >= x0) & (a.pos[:, 0] <= x1)
        a.pos[slab, 1] = np.clip(a.pos[slab, 1], g0, g1)

    def after_step(self, a, frame, rng):
        idx = np.flatnonzero(a.pos[:, 0] > self.exit_x)
        if idx.size:
            a.pos[idx] = rng.uniform(0, (10.0, 10.0), (idx.size, 2))
            a.final[idx, 1] = rng.uniform(3, 7, idx.size)
        self._aim(a)


SCENES = {"plaza": Plaza, "hub": Hub, "corridor": Corridor}
