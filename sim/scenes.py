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
    SPILL_AT = 3.0  # entry: persons/m2 in a 2 m cell above which the zone's rim takes no arrivals
    STALL_S, STALL_M = 5.0, 0.5  # an agent that moves less than STALL_M in STALL_S seconds ...
    # ... stops aiming at its next waypoint and follows its gate's distance field from wherever
    # the crowd pushed it (a waypoint behind a wall is otherwise a trap)

    def __init__(self, site=None, release=0, waves=0.0, plan="nearest", rates=None, mode="egress",
                 window=120.0, peak=45.0, lanes_per_m=1 / 1.5, service=600.0, fps=30):
        """release: frame the procession leaves; waves > 0 releases blocks in order of their
        gate distance, one block every `waves` seconds; plan: 'nearest' gate, 'balanced'
        (each gate's seats proportional to its width) or 'measured' (each block side to the gate
        that would finish it soonest, given `rates`: {gate index: persons per minute sustained,
        as measured by an earlier run}).

        mode 'entry': pass holders arrive at the rim of their gate's approach zone over `window`
        minutes before the procession (triangular, most `peak` minutes before it: ASSUMED), queue
        at the gate's screening line (max(1, round(width * lanes_per_m)) door-frame metal detector
        lanes, `service` persons per hour each: LIT 600-1,200, ASSUMED the low end for frisking),
        then walk to their seats. Frame 0 is the gates opening; the procession starts at
        window minutes. fps: frames per second of the run."""
        from .site import Site

        self.site = site or Site()
        self.release, self.waves, self.plan, self.rates = release, waves, plan, rates
        self.mode, self.window, self.peak, self.fps = mode, window, peak, fps
        self.lanes_per_m, self.service = lanes_per_m, service
        self.left = []  # (agent, gate, frame)
        self.screened = []  # entry: (agent, gate, joined frame, released frame)
        self.seated_at = []  # entry: (agent, frame)
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
        self.t_release = self.release + np.round(rank[b] * self.waves * self.fps).astype(np.int64)
        a.pos[:] = seat
        a.goal[:] = seat
        a.final[:] = self.wp[np.arange(a.n), self.nwp - 1]
        sp = np.clip(rng.normal(self.SPEED, self.SPEED_SD, a.n), 0.6, 1.5)
        sp[rng.random(a.n) < self.ELDERLY] *= self.ELDERLY_SCALE
        a.speed[:] = sp
        a.slot[:] = -1
        if self.mode == "entry":
            self._spawn_entry(a, rng)

    # --- entry: arrival, screening, seating -------------------------------------------------
    def _spawn_entry(self, a, rng):
        """Each agent's way in is its way out reversed: rim of its gate's approach zone, through
        the gate, along its block side's route, to its row end and seat."""
        n = a.n
        self.ewp = np.zeros_like(self.wp)
        for i in range(n):
            w = self.wp[i, : self.nwp[i]][::-1]
            self.ewp[i, : len(w)] = w
            self.ewp[i, len(w):] = w[-1]
        t = rng.triangular(-self.window, -self.peak, 0.0, n)  # minutes before the procession
        self.t_arrive = np.round((t + self.window) * 60.0 * self.fps).astype(np.int64)
        self.t_start = int(round(self.window * 60.0 * self.fps))
        self.phase = np.zeros(n, np.int8)  # 0 not arrived, 1 to the gate, 2 screening, 3 to the seat, 4 seated
        self.snaps = 0
        self.spill_max = 0
        st = self.site
        from .site import RES as _res
        self.zone_cap = {}
        for k in st.open:
            near = np.linalg.norm((np.argwhere(st.zone) + 0.5) * _res - st.gates[k]["at"], axis=1) <= 45.0
            self.zone_cap[k] = int(self.SPILL_AT * near.sum() * _res * _res)
        self._in_zone = {k: 0 for k in st.open}
        self._crowd = np.zeros((int(self.site.size[0] // 2) + 1, int(self.site.size[1] // 2) + 1))
        self.join = np.full(n, -1, np.int64)
        # first waypoint of each way in that lies inside the walls: where a screened agent resumes
        inside = ~self.site.walkable(self.ewp.reshape(-1, 2), self.site.zone).reshape(n, -1)
        inside[np.arange(inside.shape[1])[None, :] >= self.nwp[:, None]] = True
        self.k_in = inside.argmax(1)
        st = self.site
        self.screen_at = {}
        self.lanes = {}
        self.queue = {}
        self.tokens = {}
        for k in st.open:
            g = st.gates[k]
            self.screen_at[k] = g["at"] + g["out"] * 8.0  # door frames at the passage's outer end
            self.lanes[k] = max(1, int(round(g["width"] * self.lanes_per_m)))
            self.queue[k] = []
            self.tokens[k] = 0.0
        a.pos[:] = self._park(np.arange(n))
        a.goal[:] = a.pos
        self.k[:] = -1

    def _after_entry(self, a, frame, rng):
        st = self.site
        # a crowded rim cannot take more: those arrivals wait in the street beyond the modelled
        # approach zone (the spill) and step in when there is room
        if frame % self.fps == 0:
            self._crowd = np.zeros((int(st.size[0] // 2) + 1, int(st.size[1] // 2) + 1))
            w = np.flatnonzero((self.phase >= 1) & (self.phase <= 3))
            if w.size:
                ij = np.clip((a.pos[w] // 2).astype(np.int64), 0, np.array(self._crowd.shape) - 1)
                np.add.at(self._crowd, (ij[:, 0], ij[:, 1]), 0.25)
            self.spill_max = max(self.spill_max, int(((self.phase == 0) & (self.t_arrive <= frame)).sum()))
            self._in_zone = {k: int(((self.phase == 1) | (self.phase == 2))[self.gate == k].sum()) for k in st.open}
        arr = np.flatnonzero((self.phase == 0) & (self.t_arrive <= frame))
        if arr.size:
            ij = np.clip((self.ewp[arr, 0] // 2).astype(np.int64), 0, np.array(self._crowd.shape) - 1)
            arr = arr[self._crowd[ij[:, 0], ij[:, 1]] <= self.SPILL_AT]
        if arr.size:
            # and a zone already holding SPILL_AT persons per m2 of its area takes nobody more
            room = {k: self.zone_cap[k] - self._in_zone[k] for k in st.open}
            take = []
            for i in arr:
                k = int(self.gate[i])
                if room[k] > 0:
                    take.append(i)
                    room[k] -= 1
            arr = np.array(take, np.int64)
        if arr.size:
            jitter = rng.normal(0.0, 1.5, (arr.size, 2))
            p = self.ewp[arr, 0] + jitter
            ok = st.walkable(p)
            p[~ok] = self.ewp[arr[~ok], 0]
            a.pos[arr] = p
            a.vel[arr] = 0.0
            self.phase[arr] = 1
            self.k[arr] = self.k_in[arr]
            # outside, everyone heads for the gate opening itself: the approach zone is open ground
            a.goal[arr] = np.array([st.gates[int(g)]["at"] for g in self.gate[arr]])
        # stepping into the gate passage (the only way in): join that gate's queue, in order of
        # arrival there; the door frames stand at the passage's outer end
        walk = np.flatnonzero(self.phase == 1)
        if walk.size:
            for k in st.open:
                m = walk[self.gate[walk] == k]
                if not m.size:
                    continue
                near = m[st.walkable(a.pos[m], st.gate_passage[k])]
                for i in near:
                    self.phase[i] = 2
                    self.join[i] = frame
                    self.queue[k].append(int(i))
                    a.goal[i] = a.pos[i]
        # screening: each lane passes service/3600 persons a second
        for k in st.open:
            q = self.queue[k]
            self.tokens[k] = min(self.tokens[k] + self.lanes[k] * self.service / 3600.0 / self.fps, float(self.lanes[k]))
            if q and self.tokens[k] >= 1.0:
                # whoever stands furthest in through the gate goes next, not whoever joined first:
                # a released agent never has to push back through the waiting crowd
                inward = -st.gates[k]["out"]
                depth = (a.pos[q] - self.screen_at[k]) @ inward
                order = [q[j] for j in np.argsort(-depth, kind="stable")]
                q[:] = order
            while q and self.tokens[k] >= 1.0:
                i = q.pop(0)
                self.tokens[k] -= 1.0
                self.phase[i] = 3
                self.screened.append((i, k, int(self.join[i]), frame))
                a.goal[i] = self.ewp[i, self.k[i]]
            if q:
                a.goal[q] = a.pos[q]
        # walking: pass waypoints (inside only); seated on reaching the seat
        live = np.flatnonzero(self.phase == 3)
        if not live.size:
            return
        # a screened agent the crowd has pinned against a wall for STALL_S seconds is put back on
        # its route line, as invariant 4 binds a fine position to the core's route
        every = int(round(self.STALL_S * self.fps))
        if frame % every == 0:
            moved = np.linalg.norm(a.pos[live] - self._mark[live], axis=1)
            # only past screening: outside the gate, standing still is queueing
            stuck = live[(moved < self.STALL_M) & (self.phase[live] == 3)]
            if stuck.size:
                p0, p1 = self.ewp[stuck, np.maximum(self.k[stuck] - 1, 0)], self.ewp[stuck, self.k[stuck]]
                ab = p1 - p0
                t = np.clip(np.einsum("ij,ij->i", a.pos[stuck] - p0, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
                q = p0 + t[:, None] * ab
                ok = st.walkable(q)
                a.pos[stuck[ok]] = q[ok]
                self.snaps += int(ok.sum())
            self._mark[live] = a.pos[live]
        last_k = np.array([self.nwp[i] - 1 for i in live])
        last = self.k[live] >= last_k
        d = np.linalg.norm(a.goal[live] - a.pos[live], axis=1)
        nxt = live[~last & (d < self.PASS)]
        self.k[nxt] += 1
        a.goal[nxt] = self.ewp[nxt, self.k[nxt]]
        sit = live[last & (d < ARRIVE) & (self.phase[live] == 3)]
        if sit.size:
            self.phase[sit] = 4
            a.pos[sit] = self.seat[sit]
            self.seated_at += [(int(i), frame) for i in sit]


    def moving(self):
        """Agents the motion step moves this frame."""
        if self.mode == "entry":
            return (self.phase >= 1) & (self.phase <= 3)
        return (self.k > 0) & ~self.done

    def bodies(self):
        """Agents that take up space (movers, and the seated)."""
        if self.mode == "entry":
            # the seated lean back to let late arrivals past along their row: walkers only
            return (self.phase >= 1) & (self.phase <= 3)
        return ~self.done

    def fixed(self):
        """Bodies that never yield: people in their seats."""
        if self.mode == "entry":
            return np.zeros(len(self.phase), bool)
        return self.k == 0

    def constrain(self, a, prev):
        if self.mode == "entry":
            hold = (self.phase == 0) | (self.phase == 4)
            moving = ~hold
            a.pos[moving] = self.site.constrain(a.pos[moving], prev[moving])
            a.pos[hold] = np.where((self.phase[hold] == 4)[:, None], self.seat[hold], self._park(np.flatnonzero(hold)))
            a.vel[hold] = 0.0
            return
        hold = (self.k == 0) | self.done
        moving = ~hold
        a.pos[moving] = self.site.constrain(a.pos[moving], prev[moving])
        a.pos[hold] = np.where(self.done[hold, None], self._park(np.flatnonzero(hold)), self.seat[hold])
        a.vel[hold] = 0.0

    def _park(self, idx):
        return np.stack([-1000.0 - (idx % 200) * 3.0, -1000.0 - (idx // 200) * 3.0], 1)

    def after_step(self, a, frame, rng):
        if self.mode == "entry":
            return self._after_entry(a, frame, rng)
        go = (self.k == 0) & (frame >= self.t_release) & ~self.done
        if go.any():
            self.k[go] = 1
            a.goal[go] = self.wp[go, 1]
        live = np.flatnonzero((self.k > 0) & ~self.done)
        if not live.size:
            return
        every = int(round(self.STALL_S * self.fps))
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
