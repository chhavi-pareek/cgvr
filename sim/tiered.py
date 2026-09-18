"""Tiered crowd simulation for phase 7: behaviour tiers execute.

Conditions
  calib      full fidelity, fine-authoritative, no core: fits the surrogate, mbar and kappa
  reference  full fidelity for every agent plus the invariant core (camera-independent)
  parity     allocator + ledger + surrogate + reconciliation + core
  baseline   MassLOD threshold-and-cap sim tiers tick the process and motion every
             1 / 3 / 10 frames with accumulated dt; OFF agents are frozen; no core
"""
import os
import time

import numpy as np

from alloc.config import AXIS_ERR, N_AXES, build_table
from alloc.costmodel import row_costs_from_theta
from alloc.ledger import ErrorLedger
from alloc.serial import SerialAllocator

from .assign_threshold import ThresholdAssigner
from .behaviour import CHOKEPOINT, NEAR_GOAL, QUEUED, WALKING, BehaviourProcess, lateral_bias, speed_scale, stride
from .invariant import Core
from .reconcile import demote, promote
from .run import DT, SEP_K, SEP_R, _separation
from .scenes import ARRIVE, SCENES
from .surrogate import BASELINE_PERIODS, Surrogate

LOOKAHEAD = 2.0
LOG_EVERY = 30
CORE_US = 4.0  # invariant core, fixed per agent (not allocated)


def _logs(scene, tag):
    d = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench", "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"phase7_{scene}_{tag}")


class World:
    """Scene rules expressed on routes; `auth` is 'core' or 'fine' (who declares arrival)."""

    def __init__(self, scene, a, core, rng):
        self.sc = SCENES[scene]()
        self.scene, self.a, self.core, self.rng = scene, a, core, rng
        self.choke_x = 29.5 if scene == "corridor" else None
        self.trip = np.zeros(a.n, np.int32)
        self.egress = []  # (agent, trip, frame)

    def _route(self, idx, start, end, via=None):
        pts = np.stack([start, end if via is None else via, end], 1)
        self.core.set_route(idx, pts, self.choke_x)

    def spawn(self):
        a, sc, rng = self.a, self.sc, self.rng
        sc.spawn(a, rng)
        if self.scene == "hub":
            self.core.queued[:] = False
        idx = np.arange(a.n)
        if self.scene == "corridor":
            self._route(idx, a.pos, a.final, np.tile([30.0, 5.0], (a.n, 1)))
        else:
            self._route(idx, a.pos, a.goal)

    def _respawn_pos(self, k):
        rng = self.rng
        if self.scene == "hub":
            return np.stack([rng.uniform(0, 5, k), rng.uniform(0, self.sc.size[1], k)], 1)
        if self.scene == "corridor":
            return rng.uniform(0, (10.0, 10.0), (k, 2))
        return None

    def complete(self, idx, frame, teleport):
        """Agents that reached the end of their route. Returns indices that egressed."""
        a, sc, rng, core = self.a, self.sc, self.rng, self.core
        out = []
        for i in idx:
            if self.scene == "hub" and a.slot[i] >= 0:
                core.queued[i] = True  # took a queue slot; not an egress
                continue
            out.append(i)
            self.egress.append((int(i), int(self.trip[i]), frame))
            self.trip[i] += 1
            if self.scene == "plaza":
                a.goal[i] = rng.uniform(0, np.asarray(sc.size), 2)
                start = a.pos[i] if teleport else core.point(core.L[i], i)[0]
                self._route(i, start[None], a.goal[i][None])
                continue
            p = self._respawn_pos(1)[0]
            a.pos[i] = p
            if self.scene == "hub":
                n_q = int((a.slot >= 0).sum())
                if a.queuer[i] and n_q < sc.qmax:
                    a.slot[i] = n_q
                    a.goal[i] = sc._slot_pos(n_q)
                else:
                    a.goal[i] = sc._exit(rng, 1)[0]
                self._route(i, p[None], a.goal[i][None])
            else:
                a.final[i, 1] = rng.uniform(3, 7)
                a.goal[i] = (30.0, 5.0)
                self._route(i, p[None], a.final[i][None], np.array([[30.0, 5.0]]))
        return np.array(out, np.int64)

    def service(self, frame):
        """Hub queue: release the front every `service` frames; everyone shifts one slot."""
        a, sc, core = self.a, self.sc, self.core
        if self.scene != "hub" or frame % sc.service != 0:
            return
        q = np.flatnonzero(a.slot >= 0)
        if not q.size:
            return
        front = q[np.argmin(a.slot[q])]
        a.slot[q] -= 1
        a.slot[front] = -1
        core.queued[front] = False
        a.goal[front] = sc._exit(self.rng, 1)[0]
        start = core.point(core.s[front], front)[0]
        self._route(front, start[None], a.goal[front][None])
        q = q[q != front]
        if q.size:
            core.queued[q] = False
            a.goal[q] = sc._slot_pos(a.slot[q])
            self._route(q, core.point(core.s[q], q), a.goal[q])

    def fine_ctx(self):
        """Context from fine state only (calib and baseline, which have no core)."""
        a = self.a
        ctx = np.full(a.n, WALKING, np.int32)
        tgt = a.final if self.scene == "corridor" else a.goal
        near = np.linalg.norm(tgt - a.pos, axis=1) < 5.0
        ctx[near] = NEAR_GOAL
        if self.scene == "corridor":
            ctx[(a.pos[:, 0] >= 27.0) & (a.pos[:, 0] < 30.0)] = CHOKEPOINT
        if self.scene == "hub":
            ctx[(a.slot >= 0) & (np.linalg.norm(a.goal - a.pos, axis=1) < ARRIVE)] = QUEUED
        return ctx

    def fine_arrivals(self, frame):
        """Baseline / calib arrival rule on fine positions (mirrors sim/scenes.py)."""
        a, sc, rng = self.a, self.sc, self.rng
        if self.scene == "corridor":
            idx = np.flatnonzero(a.pos[:, 0] > sc.exit_x)
        else:
            d = a.goal - a.pos
            idx = np.flatnonzero(np.einsum("ij,ij->i", d, d) < ARRIVE**2)
        out = []
        for i in idx:
            if self.scene == "hub" and a.slot[i] >= 0:
                continue
            out.append(i)
            self.egress.append((int(i), int(self.trip[i]), frame))
            self.trip[i] += 1
            if self.scene == "plaza":
                a.goal[i] = rng.uniform(0, np.asarray(sc.size), 2)
                continue
            a.pos[i] = self._respawn_pos(1)[0]
            if self.scene == "hub":
                n_q = int((a.slot >= 0).sum())
                if a.queuer[i] and n_q < sc.qmax:
                    a.slot[i] = n_q
                    a.goal[i] = sc._slot_pos(n_q)
                else:
                    a.goal[i] = sc._exit(rng, 1)[0]
            else:
                a.final[i, 1] = rng.uniform(3, 7)
        if self.scene == "corridor":
            sc._aim(a)
        return np.array(out, np.int64)

    def fine_service(self, frame):
        a, sc = self.a, self.sc
        if self.scene != "hub" or frame % sc.service != 0:
            return
        q = np.flatnonzero(a.slot >= 0)
        if not q.size:
            return
        front = q[np.argmin(a.slot[q])]
        a.slot[q] -= 1
        a.slot[front] = -1
        a.goal[front] = sc._exit(self.rng, 1)[0]
        q = q[q != front]
        a.goal[q] = sc._slot_pos(a.slot[q])


def _motion(a, idx, target, scale, lat, dt, sc):
    """Goal-seek plus separation for agents idx toward per-agent targets; returns prev pos."""
    prev = a.pos.copy()
    d = target - a.pos[idx]
    dist = np.linalg.norm(d, axis=1)
    fwd = d / np.maximum(dist, 1e-6)[:, None]
    nrm = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
    desired = fwd * (a.speed[idx] * scale)[:, None] + nrm * lat[:, None]
    sep = _separation(a.pos, SEP_R)[idx]
    v = a.vel[idx] + 0.2 * (desired + SEP_K * sep - a.vel[idx])
    sp = np.linalg.norm(v, axis=1)
    v *= np.minimum(1.0, 1.5 * a.speed[idx] * np.maximum(scale, 1.0) / np.maximum(sp, 1e-6))[:, None]
    a.vel[idx] = v
    a.pos[idx] += v * dt
    sc.constrain(a, prev)
    return prev


def measure_costs(scene, n, sur, kappa, reps=20):
    """Telemetry for the behaviour axis: us per agent per frame of the whole fine step (process,
    decode, motion with separation, band binding) with every agent at tier 16 / 8 / 4 / surrogate."""
    r = Run(scene, n, "reference", calib={"surrogate": sur, "kappa": kappa, "theta_beh": np.zeros(4)})
    out = []
    for tier in range(4):
        r.tier[:] = tier
        r._fine_core()  # warm-up
        t = time.perf_counter()
        for _ in range(reps):
            r.core.step(r.a.speed, DT)
            r._fine_core()
        out.append((time.perf_counter() - t) / (reps * n) * 1e6)
    return np.array(out)


def phase7_table(e_sur, nav_err=False):
    """Config table with the behaviour error column replaced by the measured KL rate."""
    tab = build_table()
    err = AXIS_ERR.copy()
    err[0] = (0.0, 0.0, 0.0, e_sur)
    if not nav_err:
        err[1] = 0.0  # navigation tiers do not execute in phase 7 (stated limitation)
    tab.err = err[np.arange(N_AXES), tab.tiers].sum(1)
    return tab


class Run:
    def __init__(self, scene, n, condition, seed=0, cam=None, calib=None, budget_frac=0.25, cap=None,
                 cap_scale=1.0, alloc_cls=SerialAllocator, quality=None):
        self.scene, self.n, self.cond, self.seed = scene, n, condition, seed
        self.rng = np.random.default_rng(seed)
        self.brng = np.random.default_rng(seed + 7)  # behaviour randomness, separate stream
        self.proc = BehaviourProcess(seed=0)
        self.sur = Surrogate(self.proc)
        self.cam = cam
        self.asg = ThresholdAssigner()
        from .state import Agents

        self.a = Agents(n)
        if calib is not None:
            if isinstance(calib["surrogate"], Surrogate):
                self.sur = calib["surrogate"]
            else:
                self.sur.load(calib["surrogate"])
            mbar, kappa, self.theta_beh = self.sur.mbar, float(calib["kappa"]), calib["theta_beh"]
        else:
            mbar, kappa, self.theta_beh = np.ones(4), np.inf, None
        self.core = Core(n, mbar, kappa)
        self.world = World(scene, self.a, self.core, self.rng)
        self.world.spawn()
        self.has_core = condition in ("reference", "parity")
        ctx = self.core.update_ctx() if self.has_core else self.world.fine_ctx()
        self.d = self.proc.init(n, ctx, self.brng)
        self.region = self.sur.region_of(self.d).copy() if self.sur.fitted else np.zeros(n, np.int32)
        self.tier = np.zeros(n, np.int8)  # behaviour tier in effect this frame
        self.phase = self.brng.random(n)
        self.dist_at_demote = np.zeros(n)
        self.lat_off = np.zeros(n)
        self.D_meas = np.zeros(n)
        self.D_meas_reset = np.zeros(n)
        self._prev_x = self.a.pos[:, 0].copy()
        self._prev_tier = np.zeros(n, np.int8)
        self.crossings = 0
        self.band_max = 0.0
        self.rows = []
        self.ctx_log, self.d_log = [], []
        if condition == "parity":
            # ledger accrues the measured per-context rate; the table carries the worst frequent-context
            # rate so the feasibility mask is conservative for every context
            e_sur = float(self.sur.e_rate @ self.sur.ctx_freq)
            e_max = float(self.sur.e_rate[self.sur.ctx_freq > 0.01].max())
            self.table = phase7_table(e_max)
            if quality is not None:  # ablation of invariant 2: authored per-axis quality
                self.table.quality = np.asarray(quality, np.float64)
            # only the behaviour axis executes in phase 7, so only its measured cost enters the
            # budget; the budget is a fraction of the allocatable range above the all-surrogate floor
            th = np.zeros((4, 4))
            th[0, :3] = np.maximum(self.theta_beh[:3] - self.theta_beh[3], 0.0)
            self.cost = row_costs_from_theta(self.table, CORE_US + self.theta_beh[3], th) * 1e-3
            self.budget = n * (self.cost.min() + budget_frac * (self.cost.max() - self.cost.min()))
            self.alloc = alloc_cls(self.table)
            self.e_sur = e_sur
            self.cap = cap if cap is not None else cap_scale * 300.0 * e_sur
            self.ledger = ErrorLedger(n, self.cap, self.rng)
            self.assign = None
            self.infeasible = 0

    # -- per-frame pieces ------------------------------------------------------------
    def _salience(self, frame):
        cp, yaw = self.cam.at(frame)
        self.asg.step(self.a, cp, yaw)
        return 1.0 / (1.0 + self.a.sig / 20.0)

    def _allocate(self, frame):
        s = self._salience(frame)
        r = self.alloc.allocate(s, self.cost, self.budget, headroom=self.ledger.headroom())
        self.infeasible += int(r.infeasible)
        assert self.ledger.check(self.table, r.assign)
        self.assign = r.assign
        new = self.table.tiers[r.assign, 0]
        prom = np.flatnonzero((self.tier == 3) & (new < 3))
        dem = np.flatnonzero((self.tier < 3) & (new == 3))
        if prom.size:
            promote(prom, self.d, self.region, self.sur, self.core, self.a, self.phase, self.dist_at_demote, self.brng)
            self.ledger.reset(prom)
            self.D_meas[prom] = 0.0
        if dem.size:
            demote(dem, self.d, self.region, self.sur, self.core, self.phase, self.dist_at_demote)
            t = self.core.tangent(dem)
            rel = self.a.pos[dem] - self.core.point(self.core.s[dem], dem)
            self.lat_off[dem] = np.clip(-rel[:, 0] * t[:, 1] + rel[:, 1] * t[:, 0], -1.8, 1.8)
        self.tier = new.astype(np.int8)

    def _fine_core(self):
        """Fine step under the core: live process at tiers 0-2, surrogate at tier 3."""
        a, core, ctx = self.a, self.core, self.core.ctx
        live = np.flatnonzero(self.tier < 3)
        sur = np.flatnonzero(self.tier == 3)
        self.proc.step(self.d, ctx, live, self.brng)
        self.sur.step(self.region, ctx, sur, self.brng)
        if live.size:
            k = np.array([16, 8, 4])[self.tier[live]]
            dec = np.empty((live.size, 19))
            for kk in (16, 8, 4):
                m = k == kk
                if m.any():
                    dec[m] = self.proc.decode(self.d[live[m]], kk)
            target = core.point(np.minimum(core.s[live] + LOOKAHEAD, core.L[live]), live)
            queued = core.queued[live]
            scale = np.where(queued, 0.0, speed_scale(dec))
            _motion(a, live, target, scale, np.where(queued, 0.0, lateral_bias(dec)), DT, self.world.sc)
            a.pos[live] = core.bind(a.pos[live], live)
            self.phase[live] = (self.phase[live] + np.linalg.norm(a.vel[live], axis=1) * DT / stride(dec)) % 1.0
        if sur.size:
            t = core.tangent(sur)
            nrm = np.stack([-t[:, 1], t[:, 0]], 1)
            a.pos[sur] = core.point(core.s[sur], sur) + nrm * self.lat_off[sur, None]
            a.vel[sur] = t * (a.speed[sur] * core.mbar[ctx[sur]] * (~core.queued[sur]))[:, None]

    def _fine_free(self, tick_period):
        """Fine-authoritative step (calib / baseline); tick_period per agent, 0 = frozen."""
        a = self.a
        ctx = self.world.fine_ctx()
        for k in sorted(set(tick_period.tolist()) - {0}):
            idx = np.flatnonzero(tick_period == k)
            if not idx.size:
                continue
            for _ in range(k):  # k process steps with accumulated dt: the k-frame kernel is P^k
                self.proc.step(self.d, ctx, idx, self.brng)
            dec = self.proc.decode(self.d[idx], 16)
            tgt = a.goal[idx]
            _motion(a, idx, tgt, speed_scale(dec), lateral_bias(dec), DT * k, self.world.sc)
            self.phase[idx] = (self.phase[idx] + np.linalg.norm(a.vel[idx], axis=1) * DT * k / stride(dec)) % 1.0
        return ctx

    def _kl_baseline(self, tick_period, ticked, ctx):
        r = self.sur.fine[self.d]
        inc = np.zeros(self.n)
        for k in BASELINE_PERIODS[1:]:
            m = tick_period == k
            inc[m & ticked] = self.sur.kl_tick[k][ctx[m & ticked], r[m & ticked]]
            inc[m & ~ticked] = self.sur.kl_frozen[ctx[m & ~ticked], r[m & ~ticked]]
        m = tick_period == 0
        inc[m] = self.sur.kl_frozen[ctx[m], r[m]]
        return inc

    # -- main loop ---------------------------------------------------------------------
    def step(self, frame):
        a, world = self.a, self.world
        if self.cond == "calib":
            ctx = self._fine_free(np.ones(self.n, np.int64))
            self.ctx_log.append(ctx)
            self.d_log.append(self.d.copy())
            crossed = (a.pos[:, 0] >= 30.0) & (self._prev_x < 30.0) if self.scene == "corridor" else None
            self._prev_x = a.pos[:, 0].copy()
            if crossed is not None:
                self.crossings += int(crossed.sum())
            world.fine_service(frame)
            world.fine_arrivals(frame)
            return
        if self.cond == "baseline":
            cp, yaw = self.cam.at(frame)
            self.asg.step(a, cp, yaw)
            per = np.array([1, 3, 10, 0])[a.sim_tier]
            tick = np.where(per > 0, frame % np.maximum(per, 1) == 0, False)
            eff = np.where(tick, per, 0)
            ctx = self._fine_free(eff)
            inc = self._kl_baseline(per, tick, ctx)
            self.D_meas += inc
            self.D_meas_reset += inc
            self.D_meas_reset[(a.sim_tier == 0) & (self._prev_tier > 0)] = 0.0
            self._prev_tier = a.sim_tier.copy()
            world.fine_service(frame)
            world.fine_arrivals(frame)
            self.tier = a.sim_tier.copy()
            return
        # reference / parity: core is authoritative
        if self.cond == "parity":
            self._allocate(frame)
        done = self.core.step(a.speed, DT)
        self._fine_core()
        if self.cond == "parity":
            sur = self.tier == 3
            self.ledger.accrue(np.where(sur, self.sur.e_rate[self.core.ctx], 0.0))
            assert np.all(self.ledger.D <= self.cap + 1e-9)
            self.D_meas[sur] += self.sur.kl_coarse[self.core.ctx[sur], self.region[sur]]
            self.band_max = max(self.band_max, float(np.abs(self.core.project(a.pos) - self.core.s).max()))
        idx = np.flatnonzero(done)
        if idx.size:
            world.complete(idx, frame, teleport=False)
        world.service(frame)

    def run(self, frames, log=True):
        t0 = time.perf_counter()
        for f in range(frames):
            self.step(f)
            if log and (f % LOG_EVERY == LOG_EVERY - 1):
                self.rows.append(self._row(f))
        self.wall = time.perf_counter() - t0
        return self

    def _row(self, f):
        hist = np.bincount(self.tier.astype(np.int64), minlength=4)
        row = {"frame": f, "cond": self.cond, "t0": int(hist[0]), "t1": int(hist[1]), "t2": int(hist[2]),
               "t3": int(hist[3]), "kl_mean": float(self.D_meas.mean()), "kl_p95": float(np.percentile(self.D_meas, 95)),
               "kl_max": float(self.D_meas.max()), "kl_reset_max": float(self.D_meas_reset.max()),
               "egress": len(self.world.egress)}
        if self.cond == "parity":
            row.update(ledger_mean=float(self.ledger.D.mean()), ledger_max=float(self.ledger.D.max()),
                       restorations=self.ledger.restorations, infeasible=self.infeasible, choke_waits=self.core.waits,
                       band_max=self.band_max)
        return row


def calibrate(scene, n=200, frames=6000, seed=0, force=False):
    """Full-fidelity free run: fit the surrogate, mbar, kappa; measure behaviour tier costs."""
    path = _logs(scene, "calib.npz")
    if os.path.exists(path) and not force:
        t = np.load(path)
        return {"surrogate": path, "kappa": float(t["kappa"]), "theta_beh": t["theta_beh"]}
    r = Run(scene, n, "calib", seed=seed).run(frames, log=False)
    ctx_log, d_log = np.stack(r.ctx_log), np.stack(r.d_log)
    r.sur.fit(ctx_log[300:], d_log[300:])
    kappa = r.crossings / max(frames - 300, 1) if scene == "corridor" else np.inf
    theta = measure_costs(scene, n, r.sur, kappa)
    r.sur.save(path)
    t = dict(np.load(path))
    t.update(kappa=kappa, theta_beh=theta, n=n, frames=frames)
    np.savez(path, **t)
    return {"surrogate": path, "kappa": float(kappa), "theta_beh": theta}
