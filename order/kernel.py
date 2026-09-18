"""Tiered neighbour-gather step: the workload the ordering study measures against.

Stand-in kernel, same status as the phase 3 telemetry kernels: the simulator's
tiers do not yet execute tiered work. One thread per agent, in the current sort
order. Three parts:

  gather : uniform grid (cell = R), visit the 3 x 3 cells around the agent,
           read pos[j], vel[j] of every listed agent, accumulate a separation
           force. Coalescing-sensitive (scattered loads), and the neighbour
           loop diverges when lanes in a warp see different list lengths.
  class  : one of four branch bodies chosen by behaviour class, trip counts
           CLS_TRIPS. Coherence-sensitive.
  tier   : one of four branch bodies chosen by the allocator's tier, trip
           counts TIER_TRIPS (tier 3 = none). Coherence-sensitive.

Measurement without Nsight
--------------------------
Warp execution efficiency = lane-instructions executed / (32 * warp-instructions
issued), the ratio Nsight reports as thread_inst_executed / inst_executed. The
instrumented kernel records per lane its neighbour-loop trip count and, per
warp (shared memory, lane 0 reduces), the max and sum of trips and the set of
class and tier bodies present. Every body has a fixed static instruction count
(W_* below, from the loop bodies as written), so issued and executed
instructions follow exactly from those counters: a warp issues the gather loop
max(trips) times and every distinct body its lanes need; a lane executes only
its own. `analyse` computes the same counters on the host from the sort order
and the grid (the Kofler et al. cost model), and the test asserts the two agree.

Bytes moved is computed, not measured: per warp the distinct 32-byte sectors
its lanes touch (own SoA rows, cell ranges, cell lists, gathered pos/vel rows),
i.e. L1-miss sector traffic to L2 assuming a warp's working set stays in L1.
The whole crowd fits in the T4's 4 MB L2, so DRAM traffic is not what the
ordering changes. Kernel time comes from CUDA events on the uninstrumented
kernel on hardware; under the simulator the sweep reports a modelled time
(issue-limited vs sector-limited, T4 constants) and labels it as such.
"""
import math
import types

import numpy as np

import alloc.cuda  # noqa: F401  sets NUMBA_ENABLE_CUDASIM when there is no driver, before numba loads
from numba import config, cuda, int32, njit

SIMULATED = bool(config.ENABLE_CUDASIM)
TPB = 32 if SIMULATED else 256
WARP = 32

R = 1.0
R2 = R * R
CLS_TRIPS = (4, 12, 24, 48)
TIER_TRIPS = (96, 32, 8, 0)

# static instruction counts per region (from the kernel body as written)
W_FIXED = 40   # prologue, cell arithmetic, epilogue stores
W_GATHER = 14  # one neighbour-loop trip: 4 loads, 2 sub, 2 fma, cmp, branch, sqrt, div, 2 fma
W_ENTRY = 4    # branch test and loop setup of a class/tier body
W_TRIP = 6     # one trip of a class/tier body: 4 fma + loop overhead

# T4 constants for the modelled time (simulator only)
T4_SM, T4_SMSP, T4_CLOCK = 40, 4, 1.59e9
T4_IPC_PER_SMSP = 0.5       # sustained issue rate assumed per scheduler
T4_L2_BW = 1.0e12           # bytes / s, L1 <-> L2 aggregate
T4_LAUNCH_US = 4.0


class Grid:
    def __init__(self, pos, size, cell=R):
        self.cell_size = cell
        self.gx = int(math.ceil(size[0] / cell)) + 1
        self.gy = int(math.ceil(size[1] / cell)) + 1
        cx = np.clip(np.floor(pos[:, 0] / cell), 0, self.gx - 1).astype(np.int32)
        cy = np.clip(np.floor(pos[:, 1] / cell), 0, self.gy - 1).astype(np.int32)
        self.cell = (cy * self.gx + cx).astype(np.int32)
        self.items = np.argsort(self.cell, kind="stable").astype(np.int32)
        counts = np.bincount(self.cell, minlength=self.gx * self.gy)
        self.end = np.cumsum(counts).astype(np.int32)
        self.start = (self.end - counts).astype(np.int32)

    def neighbour_cells(self):
        """(N, 9) cell ids around each agent, -1 where outside the grid."""
        cx, cy = self.cell % self.gx, self.cell // self.gx
        out = np.full((len(self.cell), 9), -1, np.int64)
        k = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = cx + dx, cy + dy
                ok = (nx >= 0) & (nx < self.gx) & (ny >= 0) & (ny < self.gy)
                out[ok, k] = (ny * self.gx + nx)[ok]
                k += 1
        return out


def _spin(x, y, n, a):
    for _ in range(n):
        x1 = x * 0.99 + y * a
        y = y * 0.99 - x * a
        x = x1
    return x, y


def _agent(i, pos, vel, cls, tier, cell, start, end, items, gx, gy, out):
    px = pos[i, 0]
    py = pos[i, 1]
    vx = vel[i, 0]
    vy = vel[i, 1]
    c = cell[i]
    cx = c % gx
    cy = c // gx
    fx = 0.0
    fy = 0.0
    trips = 0
    for dy in range(-1, 2):
        ny = cy + dy
        if ny < 0 or ny >= gy:
            continue
        for dx in range(-1, 2):
            nx = cx + dx
            if nx < 0 or nx >= gx:
                continue
            cc = ny * gx + nx
            for k in range(start[cc], end[cc]):
                j = items[k]
                trips += 1
                if j == i:
                    continue
                ddx = px - pos[j, 0]
                ddy = py - pos[j, 1]
                d2 = ddx * ddx + ddy * ddy
                if d2 < R2 and d2 > 1e-12:
                    d = math.sqrt(d2)
                    w = (1.0 - d / R) / d
                    fx += w * ddx + 0.1 * (vx - vel[j, 0])
                    fy += w * ddy + 0.1 * (vy - vel[j, 1])
    x = fx
    y = fy
    b = cls[i]
    if b == 0:
        x, y = _spin(x, y, 4, 0.01)
    elif b == 1:
        x, y = _spin(x, y, 12, 0.02)
    elif b == 2:
        x, y = _spin(x, y, 24, 0.03)
    else:
        x, y = _spin(x, y, 48, 0.04)
    t = tier[i]
    if t == 0:
        x, y = _spin(x, y, 96, 0.05)
    elif t == 1:
        x, y = _spin(x, y, 32, 0.06)
    elif t == 2:
        x, y = _spin(x, y, 8, 0.07)
    out[i, 0] = x
    out[i, 1] = y
    return trips


def _bind(spin):
    """The one Python body compiled twice: CPU oracle and device function."""
    g = dict(_agent.__globals__)
    g["_spin"] = spin
    return types.FunctionType(_agent.__code__, g, _agent.__name__)


_agent_body_cpu = njit(_bind(njit(_spin)))
_agent_dev = cuda.jit(device=True)(_bind(cuda.jit(device=True)(_spin)))


@njit
def step_ref(pos, vel, cls, tier, cell, start, end, items, gx, gy, out, trips):
    for i in range(pos.shape[0]):
        trips[i] = _agent_body_cpu(i, pos, vel, cls, tier, cell, start, end, items, gx, gy, out)


@cuda.jit
def k_step(pos, vel, cls, tier, cell, start, end, items, gx, gy, out):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        _agent_dev(i, pos, vel, cls, tier, cell, start, end, items, gx, gy, out)


@cuda.jit
def k_step_instr(pos, vel, cls, tier, cell, start, end, items, gx, gy, out, trips, wstats):
    """Same kernel plus per-lane trip counts and per-warp counters:
    wstats[w] = (max trips, sum trips, class-body mask, tier-body mask)."""
    sh_t = cuda.shared.array(TPB, int32)
    sh_c = cuda.shared.array(TPB, int32)
    sh_r = cuda.shared.array(TPB, int32)
    i = cuda.grid(1)
    tid = cuda.threadIdx.x
    n = pos.shape[0]
    t = 0
    if i < n:
        t = _agent_dev(i, pos, vel, cls, tier, cell, start, end, items, gx, gy, out)
        trips[i] = t
        sh_c[tid] = cls[i]
        sh_r[tid] = tier[i]
    else:
        sh_c[tid] = -1
        sh_r[tid] = -1
    sh_t[tid] = t
    cuda.syncthreads()
    if tid % WARP == 0 and i < n:
        mx = 0
        sm = 0
        cm = 0
        tm = 0
        for l in range(WARP):
            g = i + l
            if g < n:
                v = sh_t[tid + l]
                sm += v
                if v > mx:
                    mx = v
                cm |= 1 << sh_c[tid + l]
                tm |= 1 << sh_r[tid + l]
        w = i // WARP
        wstats[w, 0] = mx
        wstats[w, 1] = sm
        wstats[w, 2] = cm
        wstats[w, 3] = tm


def _mask_cost(mask, trips):
    return sum(W_ENTRY + trips[b] * W_TRIP for b in range(4) if mask >> b & 1)


def stats_to_wee(wstats, n, cls, tier):
    """Issued / executed instruction totals from the per-warp counters."""
    nw = wstats.shape[0]
    lanes = np.minimum(WARP, n - np.arange(nw) * WARP)
    issued = float(nw * W_FIXED + wstats[:, 0].sum() * W_GATHER)
    issued += sum(_mask_cost(int(m), CLS_TRIPS) for m in wstats[:, 2])
    issued += sum(_mask_cost(int(m), TIER_TRIPS) for m in wstats[:, 3])
    active = float(lanes.sum() * W_FIXED + wstats[:, 1].sum() * W_GATHER)
    ct = np.asarray(CLS_TRIPS)
    tt = np.asarray(TIER_TRIPS)
    active += float((W_ENTRY + ct[cls] * W_TRIP).sum() + (W_ENTRY + tt[tier] * W_TRIP).sum())
    return dict(issued=issued, active=active, wee=active / (WARP * issued))


def _unique_pairs(wid, x):
    return np.unique(wid.astype(np.int64) << np.int64(32) | x.astype(np.int64)).size


def analyse(pos, vel, cls, tier, size):
    """Host model of the counters and sector traffic for agents in the given order."""
    n = len(pos)
    g = Grid(pos, size)
    nc = g.neighbour_cells()
    valid = nc >= 0
    counts = np.where(valid, g.end[np.maximum(nc, 0)] - g.start[np.maximum(nc, 0)], 0)
    trips = counts.sum(1)
    wid = np.arange(n) // WARP
    nw = (n + WARP - 1) // WARP
    wstats = np.zeros((nw, 4), np.int64)
    np.maximum.at(wstats[:, 0], wid, trips)
    wstats[:, 1] = np.bincount(wid, trips, minlength=nw)
    for b in range(4):
        wstats[:, 2] |= (np.bincount(wid[cls == b], minlength=nw) > 0) << b
        wstats[:, 3] |= (np.bincount(wid[tier == b], minlength=nw) > 0) << b
    res = stats_to_wee(wstats, n, cls, tier)
    res["trips"] = trips
    res["wstats"] = wstats

    # sector traffic (32 B): own rows are coalesced
    lanes = np.minimum(WARP, n - np.arange(nw) * WARP)
    own = (np.ceil(lanes * 8 / 32) * 3 + np.ceil(lanes * 4 / 32) + np.ceil(lanes / 32) * 2).sum()
    wc = np.repeat(wid[:, None], 9, 1)[valid]
    cells = nc[valid]
    ranges = 2 * _unique_pairs(wc, cells // 8)
    cnt = counts[valid]
    total = int(cnt.sum())
    k = np.repeat(g.start[cells], cnt) + (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    wk = np.repeat(wc, cnt)
    lists = _unique_pairs(wk, k // 8)
    j = g.items[k]
    gathers = 2 * _unique_pairs(wk, j // 4)
    sectors = float(own + ranges + lists + gathers)
    res.update(sectors=sectors, bytes=32.0 * sectors, bytes_per_agent=32.0 * sectors / n,
               rho=float(trips.mean()), grid=g)
    res["t_model_us"] = model_time_us(res["issued"], sectors)
    return res


def model_time_us(issued, sectors):
    t_issue = issued / (T4_SM * T4_SMSP * T4_IPC_PER_SMSP * T4_CLOCK)
    t_mem = sectors * 32.0 / T4_L2_BW
    return 1e6 * max(t_issue, t_mem) + T4_LAUNCH_US


def _dev_inputs(pos, vel, cls, tier, size):
    g = Grid(pos, size)
    arrs = [np.ascontiguousarray(pos, np.float32), np.ascontiguousarray(vel, np.float32),
            cls.astype(np.int8), tier.astype(np.int8), g.cell, g.start, g.end, g.items]
    return [cuda.to_device(a) for a in arrs] + [g.gx, g.gy], g


def run_ref(pos, vel, cls, tier, size):
    g = Grid(pos, size)
    out = np.zeros((len(pos), 2), np.float32)
    trips = np.zeros(len(pos), np.int32)
    step_ref(np.ascontiguousarray(pos, np.float32), np.ascontiguousarray(vel, np.float32),
             cls.astype(np.int8), tier.astype(np.int8), g.cell, g.start, g.end, g.items, g.gx, g.gy, out, trips)
    return out, trips


def run_instr(pos, vel, cls, tier, size):
    n = len(pos)
    args, g = _dev_inputs(pos, vel, cls, tier, size)
    out = cuda.device_array((n, 2), np.float32)
    trips = cuda.device_array(n, np.int32)
    wstats = cuda.to_device(np.zeros(((n + WARP - 1) // WARP, 4), np.int32))
    k_step_instr[(n + TPB - 1) // TPB, TPB](*args, out, trips, wstats)
    cuda.synchronize()
    return out.copy_to_host(), trips.copy_to_host(), wstats.copy_to_host()


def time_kernel_us(pos, vel, cls, tier, size, reps=20):
    """Median CUDA-event time of the uninstrumented kernel; nan under the simulator."""
    if SIMULATED:
        return float("nan")
    n = len(pos)
    args, g = _dev_inputs(pos, vel, cls, tier, size)
    out = cuda.device_array((n, 2), np.float32)
    grid = (n + TPB - 1) // TPB
    k_step[grid, TPB](*args, out)
    cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = cuda.event(), cuda.event()
        e0.record()
        k_step[grid, TPB](*args, out)
        e1.record()
        e1.synchronize()
        ts.append(cuda.event_elapsed_time(e0, e1) * 1e3)
    return float(np.median(ts))
