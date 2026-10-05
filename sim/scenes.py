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


def _palace_size():
    import json
    import pathlib

    from .site import MARGIN

    x0, y0, x1, y1 = json.loads((pathlib.Path(__file__).resolve().parent / "data/mysore_palace_site.json").read_text())["bounds"]
    return (x1 - x0 + 2 * MARGIN, y1 - y0 + 2 * MARGIN)


class Palace:
    """Mysore Palace forecourt on Jamboo Savari day: a seated crowd released at once, then egress
    through the open gates (sim/site.py).

    Seat blocks, their placement, the procession lane, the speed spread and the elderly share are
    ASSUMPTIONS (docs/site/mysore_palace.md, section 6); the mean walking speed is LIT (Rastogi et
    al. 2011, pedestrian precincts). Rows face the palace (west); each row empties to its nearer
    end, into an aisle, and from there along the block side's route to its gate.
    """

    name = "palace"
    size = _palace_size()
    ROWS, COLS = 20, 40  # 800 seats per block
    SEAT, ROW = 0.5, 0.9  # metres between seats in a row, between rows
    AISLE, XGAP = 1.2, 3.0  # aisle at each row end; cross aisle between block columns
    FRONT = (-105.0, 32.0)  # palace front centre, site frame
    LANE = ((-100.0, 32.0), (-34.0, 257.0), 10.0)  # procession lane to the north gate, half-width
    CLEAR_BUILDING, CLEAR_GATE = 10.0, 15.0
    SPEED, SPEED_SD, ELDERLY, ELDERLY_SCALE = 1.06, 0.15, 0.10, 0.75
    PASS = 2.5  # metres from an intermediate waypoint that count as passing it
    STALL_S, STALL_M = 5.0, 0.5  # an agent that moves less than STALL_M in STALL_S seconds ...
    # ... stops aiming at its next waypoint and follows its gate's distance field from wherever
    # the crowd pushed it (a waypoint behind a wall is otherwise a trap)

    def __init__(self, site=None, release=0, waves=0.0, plan="nearest", rates=None):
        """release: frame the procession leaves; waves > 0 releases blocks in order of their
        gate distance, one block every `waves` seconds; plan: 'nearest' gate, 'balanced'
        (each gate's seats proportional to its width) or 'measured' (each block side to the gate
        that would finish it soonest, given `rates`: {gate index: persons per minute sustained,
        as measured by an earlier run})."""
        from .site import Site

        self.site = site or Site()
        self.release, self.waves, self.plan, self.rates = release, waves, plan, rates
        self.left = []  # (agent, gate, frame)
        self._blocks = None

    # --- layout ---------------------------------------------------------------------------
    def blocks(self, n_blocks):
        """Lower-left seat corners of the first n_blocks candidate blocks, nearest the palace front."""
        from scipy import ndimage

        from .site import RES

        st = self.site
        bw, bh = self.ROWS * self.ROW, self.COLS * self.SEAT
        free = ndimage.distance_transform_edt(~st.building) * RES >= self.CLEAR_BUILDING
        a, b, hw = np.asarray(self.LANE[0]) + st.shift, np.asarray(self.LANE[1]) + st.shift, self.LANE[2]
        gates = np.array([g["at"] for g in st.gates])
        out = []
        for x in np.arange(MARGIN_PAD, st.size[0] - bw, bw + self.XGAP):
            for y in np.arange(MARGIN_PAD, st.size[1] - bh, bh + 2 * self.AISLE):
                xs = np.arange(x - 1.0, x + bw + 1.0 + 1e-9, 0.5)
                ys = np.arange(y - self.AISLE - 1.0, y + bh + self.AISLE + 1.0 + 1e-9, 0.5)
                pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
                ij = st.cell(pts)
                if not (st.walkable(pts, st.clear).all() and free[ij[:, 0], ij[:, 1]].all()):
                    continue
                if np.min(np.linalg.norm(gates[:, None] - pts[None], axis=2)) < self.CLEAR_GATE:
                    continue
                ab = b - a
                t = np.clip((pts - a) @ ab / (ab @ ab), 0, 1)
                if np.min(np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)) < hw:
                    continue
                out.append((x, y))
        out = np.asarray(out)
        front = np.asarray(self.FRONT) + st.shift
        out = out[np.argsort(np.linalg.norm(out + (bw / 2, bh / 2) - front, axis=1), kind="stable")]
        if len(out) < n_blocks:
            raise ValueError(f"only {len(out)} blocks fit under the layout assumptions; {n_blocks} needed")
        return out[:n_blocks]

    def _side_routes(self, corners):
        """Route per (block, side) from the side's aisle centre; gate per the plan."""
        st = self.site
        bw, bh = self.ROWS * self.ROW, self.COLS * self.SEAT
        starts = np.concatenate([corners + (bw / 2, -self.AISLE / 2), corners + (bw / 2, bh + self.AISLE / 2)])
        gate, dist = st.nearest_gate(starts)
        if self.plan == "balanced":
            gate = self._balance(starts)
        elif self.plan == "measured":
            gate = self._by_rate(starts, len(corners))
        return starts, gate, [st.route(p, int(k)) for p, k in zip(starts, gate)]

    def _by_rate(self, starts, nb):
        """Greedy: sides nearest their best gate first; each takes the gate with the earliest
        estimated finish, (seats already sent + its own) / measured rate + walk at mean speed."""
        st = self.site
        keys = [k for k in st.open if self.rates.get(k, 0) > 0]
        d = np.stack([st.distance(starts, k) for k in keys], 1)
        rate = np.array([self.rates[k] for k in keys], np.float64)
        load = np.zeros(len(keys))
        seats = self.ROWS * self.COLS / 2
        out = np.zeros(len(starts), np.int64)
        for i in np.argsort(d.min(1), kind="stable"):
            finish = (load + seats) / rate + d[i] / self.SPEED / 60.0
            finish[~np.isfinite(d[i])] = np.inf
            j = int(np.argmin(finish))
            load[j] += seats
            out[i] = keys[j]
        return out

    def _balance(self, starts):
        """Greedy: sides in order of their best-gate distance take the nearest gate with room,
        room being each open gate's share of the total open width."""
        st = self.site
        d = np.stack([st.distance(starts, k) for k in st.open], 1)
        width = np.array([st.gates[k]["width"] for k in st.open])
        room = width / width.sum() * len(starts)
        used = np.zeros(len(st.open))
        out = np.zeros(len(starts), np.int64)
        for i in np.argsort(d.min(1), kind="stable"):
            for j in np.argsort(d[i], kind="stable"):
                if used[j] + 1 <= np.ceil(room[j]) and np.isfinite(d[i, j]):
                    break
            used[j] += 1
            out[i] = st.open[j]
        return out

    # --- scene interface ----------------------------------------------------------------------
    def spawn(self, a, rng):
        per = self.ROWS * self.COLS
        nb = -(-a.n // per)
        corners = self.blocks(nb)
        starts, side_gate, routes = self._side_routes(corners)
        seats = np.stack(np.meshgrid(np.arange(self.ROWS), np.arange(self.COLS), indexing="ij"), -1).reshape(-1, 2)
        pick = np.concatenate([np.full(per, b) for b in range(nb)])[: a.n]
        rc = np.concatenate([seats] * nb)[: a.n]
        if a.n % per:  # the last block is filled at random seats
            last = pick == nb - 1
            rc[last] = seats[rng.choice(per, int(last.sum()), replace=False)]
        b, r, c = pick, rc[:, 0], rc[:, 1]
        seat = corners[b] + np.stack([(r + 0.5) * self.ROW, (c + 0.5) * self.SEAT], 1)
        north = c >= self.COLS // 2
        side = b + north * nb
        end = np.stack([seat[:, 0], np.where(north, corners[b, 1] + self.COLS * self.SEAT + self.AISLE / 2,
                                                corners[b, 1] - self.AISLE / 2)], 1)
        K = 2 + max(len(rt) for rt in routes)
        self.wp = np.zeros((a.n, K, 2))
        self.nwp = np.zeros(a.n, np.int64)
        for i in range(a.n):
            rt = routes[side[i]]
            first = 1 if self.site.visible(end[i], rt[1]) else 0
            w = np.concatenate([seat[i][None], end[i][None], rt[first:]])
            self.wp[i, : len(w)] = w
            self.wp[i, len(w):] = w[-1]
            self.nwp[i] = len(w)
        self.corners, self.block, self.gate = corners, b, side_gate[side]
        self.seat = seat
        self.k = np.zeros(a.n, np.int64)
        self.done = np.zeros(a.n, bool)
        self.follow = np.zeros(a.n, bool)
        self._mark = seat.copy()
        dist = np.array([np.linalg.norm(np.diff(routes[s], axis=0), axis=1).sum() for s in range(len(routes))])
        order = np.argsort(np.minimum(dist[:nb], dist[nb:]), kind="stable")
        rank = np.empty(nb, np.int64)
        rank[order] = np.arange(nb)
        self.t_release = self.release + np.round(rank[b] * self.waves * 30.0).astype(np.int64)
        a.pos[:] = seat
        a.goal[:] = seat
        a.final[:] = self.wp[np.arange(a.n), self.nwp - 1]
        sp = np.clip(rng.normal(self.SPEED, self.SPEED_SD, a.n), 0.6, 1.5)
        sp[rng.random(a.n) < self.ELDERLY] *= self.ELDERLY_SCALE
        a.speed[:] = sp
        a.slot[:] = -1

    def constrain(self, a, prev):
        hold = (self.k == 0) | self.done
        moving = ~hold
        a.pos[moving] = self.site.constrain(a.pos[moving], prev[moving])
        a.pos[hold] = np.where(self.done[hold, None], self._park(np.flatnonzero(hold)), self.seat[hold])
        a.vel[hold] = 0.0

    def _park(self, idx):
        return np.stack([-1000.0 - (idx % 200) * 3.0, -1000.0 - (idx // 200) * 3.0], 1)

    def after_step(self, a, frame, rng):
        go = (self.k == 0) & (frame >= self.t_release) & ~self.done
        if go.any():
            self.k[go] = 1
            a.goal[go] = self.wp[go, 1]
        live = np.flatnonzero((self.k > 0) & ~self.done)
        if not live.size:
            return
        every = int(round(self.STALL_S * 30))
        if frame % every == 0:
            moved = np.linalg.norm(a.pos[live] - self._mark[live], axis=1)
            self.follow[live[(moved < self.STALL_M) & (frame >= self.t_release[live] + every)]] = True
            self._mark[live] = a.pos[live]
        fl = live[self.follow[live]]
        if fl.size:
            a.goal[fl] = a.pos[fl] + 3.0 * self.site.steer(a.pos[fl], self.gate[fl])
        last = self.k[live] == self.nwp[live] - 1
        d = np.linalg.norm(a.goal[live] - a.pos[live], axis=1)
        nxt = live[~last & (d < self.PASS) & ~self.follow[live]]
        self.k[nxt] += 1
        a.goal[nxt] = self.wp[nxt, self.k[nxt]]
        # out once inside any open gate's sink, wherever across its width
        out = live[(last & (d < ARRIVE)) | self.site.walkable(a.pos[live], self.site.sink_any)]
        if out.size:
            self.done[out] = True
            self.left += [(int(i), int(self.gate[i]), frame) for i in out]
            a.pos[out] = self._park(out)
            a.goal[out] = a.pos[out]


MARGIN_PAD = 2.0

SCENES = {"plaza": Plaza, "hub": Hub, "corridor": Corridor, "palace": Palace}
