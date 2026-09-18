"""Numba-CUDA allocator with a device-resident bisection state machine.

Mapping
-------
One thread per agent, TPB = 256 threads per block, grid = ceil(N / TPB).
Each thread scans all m table rows for its agent (the same j order in every
thread, so table reads are warp-uniform broadcasts). The fixed part of the
table (quality, error rate) lives in constant memory; the per-frame RLS cost
vector is staged into shared memory by every block, because numba binds
constant memory at compile time (a CUDA-C port would cudaMemcpyToSymbol it).

Reduction
---------
Per-agent selected cost -> shared-memory tree within the block -> one partial
per block in global memory. The last block to finish (threadfence + atomic
ticket) sums the partials in fixed order, so the total is deterministic, and
its thread 0 advances the bracketing/bisection state machine and writes the
next lambda into device memory. The host only launches kernels in batches and
reads a done flag between batches; lambda itself never leaves the device, and
the warm-start multiplier persists on the device between frames.

Arithmetic
----------
The serial oracle computes score = (s*q) - (lam*c) with three IEEE fp64
roundings. NVVM contracts a*b-c into an FMA by default, which changes the
rounding, so on real hardware the products go through libdevice dmul_rn /
dadd_rn, which are never contracted. Under the simulator plain Python ops are
already unfused.
"""
import math

import numpy as np
from numba import config, cuda, float64, int32

from alloc.common import finish, prepare

SIMULATED = bool(config.ENABLE_CUDASIM)
TPB = 32 if SIMULATED else 256  # small blocks keep the simulator usable
MAXM = 256  # >= table rows (180); shared-memory staging size for the cost vector

# T4 / sm_75 hardware limits used for the occupancy statement
T4 = dict(sm=40, regs_per_sm=65536, smem_per_sm=64 * 1024, smem_per_block=48 * 1024,
          const_bytes=64 * 1024, warps_per_sm=32, blocks_per_sm=16, threads_per_sm=1024,
          warp=32)

# state layout
S_LAM, S_LO, S_HI, S_TLO, S_THI, S_LMAX, S_BUDGET, S_SLACK, S_RTOL, S_BTOL, S_LAMF, S_TF = range(12)
I_PHASE, I_DONE, I_CUR, I_LOB, I_HIB, I_EVALS, I_ITER, I_INF, I_TICKET, I_MAXIT, I_FDONE, \
    I_FSTEPS, I_RES, I_FILL, I_NODC = range(15)

if SIMULATED:
    def _mul(a, b):
        return a * b

    def _sub(a, b):
        return a - b
else:
    from numba.cuda import libdevice

    @cuda.jit(device=True)
    def _mul(a, b):
        return libdevice.dmul_rn(a, b)

    @cuda.jit(device=True)
    def _sub(a, b):
        return libdevice.dadd_rn(a, -b)


@cuda.jit(device=True)
def _block_sum(part, tid):
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            part[tid] += part[tid + stride]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True)
def _block_argmax(r, idx, jj, tid):
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            o = tid + stride
            if r[o] > r[tid] or (r[o] == r[tid] and idx[o] < idx[tid]):
                r[tid] = r[o]
                idx[tid] = idx[o]
                jj[tid] = jj[o]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True)
def _swap(ist, a, b):
    t = ist[a]
    ist[a] = ist[b]
    ist[b] = t


@cuda.jit(device=True)
def _finish(st, ist, warm, res, lam, t, infeasible, update_warm):
    ist[I_RES] = res
    st[S_LAMF] = lam
    st[S_TF] = t
    ist[I_INF] = infeasible
    st[S_SLACK] = st[S_BUDGET] - t
    ist[I_DONE] = 1
    ist[I_FDONE] = 0 if (ist[I_FILL] != 0 and infeasible == 0) else 1
    if update_warm:
        warm[0] = lam


@cuda.jit(device=True)
def _advance(st, ist, warm, t):
    """Mirror of alloc.common.drive, one evaluation at a time. Thread 0 only."""
    ist[I_EVALS] += 1
    ph = ist[I_PHASE]
    budget = st[S_BUDGET]
    lam_max = st[S_LMAX]
    if ph == 0:  # evaluated lam = 0
        if t <= budget:
            _finish(st, ist, warm, ist[I_CUR], 0.0, t, 0, True)
            return
        st[S_LO] = 0.0
        st[S_TLO] = t
        _swap(ist, I_LOB, I_CUR)
        if ist[I_NODC] != 0:
            _finish(st, ist, warm, ist[I_LOB], 0.0, t, 1, False)
            ist[I_FDONE] = 1
            return
        st[S_LAM] = lam_max
        ist[I_PHASE] = 1
        return
    if ph == 1:  # evaluated lam_max
        if t > budget:
            _finish(st, ist, warm, ist[I_CUR], lam_max, t, 1, True)
            return
        st[S_HI] = lam_max
        st[S_THI] = t
        _swap(ist, I_HIB, I_CUR)
        w = warm[0]
        if w <= 0.0:
            ph = -1  # cold: straight to the bisection check
        else:
            st[S_LO] = min(w / 2, lam_max)
            st[S_HI] = min(w * 2, lam_max)
            st[S_LAM] = st[S_LO]
            ist[I_PHASE] = 2
            return
    if ph == 2:  # evaluated warm lo
        st[S_TLO] = t
        _swap(ist, I_LOB, I_CUR)
        st[S_LAM] = st[S_HI]
        ist[I_PHASE] = 3
        return
    if ph == 3:  # evaluated hi (warm or expand-up)
        st[S_THI] = t
        _swap(ist, I_HIB, I_CUR)
        if st[S_THI] > budget:
            st[S_LO] = st[S_HI]
            st[S_TLO] = st[S_THI]
            _swap(ist, I_LOB, I_HIB)
            st[S_HI] = min(st[S_HI] * 2, lam_max)
            st[S_LAM] = st[S_HI]
            return
        if st[S_TLO] <= budget:
            st[S_HI] = st[S_LO]
            st[S_THI] = st[S_TLO]
            _swap(ist, I_LOB, I_HIB)
            st[S_LO] = st[S_LO] / 2 if st[S_LO] > 1e-12 else 0.0
            st[S_LAM] = st[S_LO]
            ist[I_PHASE] = 5
            return
        ph = -1
    elif ph == 5:  # evaluated lo (expand-down)
        st[S_TLO] = t
        _swap(ist, I_LOB, I_CUR)
        if st[S_LO] != 0.0 and st[S_TLO] <= budget:
            st[S_HI] = st[S_LO]
            st[S_THI] = st[S_TLO]
            _swap(ist, I_LOB, I_HIB)
            st[S_LO] = st[S_LO] / 2 if st[S_LO] > 1e-12 else 0.0
            st[S_LAM] = st[S_LO]
            return
        ph = -1
    elif ph == 6:  # evaluated mid
        if t <= budget:
            st[S_HI] = st[S_LAM]
            st[S_THI] = t
            _swap(ist, I_HIB, I_CUR)
        else:
            st[S_LO] = st[S_LAM]
            st[S_TLO] = t
            _swap(ist, I_LOB, I_CUR)
        ist[I_ITER] += 1
    # bisection step or stop
    ist[I_PHASE] = 6
    lo = st[S_LO]
    hi = st[S_HI]
    if ist[I_ITER] >= ist[I_MAXIT] or hi - lo <= st[S_RTOL] * hi \
            or budget - st[S_THI] <= st[S_BTOL] * budget:
        _finish(st, ist, warm, ist[I_HIB], hi, st[S_THI], 0, True)
        return
    st[S_LAM] = 0.5 * (lo + hi)


def _build(quality, err):
    Q = np.ascontiguousarray(quality, np.float64)
    E = np.ascontiguousarray(err, np.float64)

    @cuda.jit
    def k_eval(s, h, cost, abuf, st, ist, warm, partials):
        cq = cuda.const.array_like(Q)
        ce = cuda.const.array_like(E)
        sh_c = cuda.shared.array(MAXM, float64)
        part = cuda.shared.array(TPB, float64)
        flag = cuda.shared.array(1, int32)
        if ist[I_DONE] != 0:
            return
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        nb = cuda.gridDim.x
        n = s.shape[0]
        m = cost.shape[0]
        j = tid
        while j < m:
            sh_c[j] = cost[j]
            j += TPB
        cuda.syncthreads()
        lam = st[S_LAM]
        cur = ist[I_CUR]
        i = bid * TPB + tid
        mine = 0.0
        if i < n:
            si = s[i]
            hi_ = h[i]
            best = -math.inf
            bj = -1
            bc = math.inf
            bq = -math.inf
            for j in range(m):
                if ce[j] <= hi_:
                    cj = sh_c[j]
                    qj = cq[j]
                    sc = _sub(_mul(si, qj), _mul(lam, cj))
                    if sc > best or (sc == best and (cj < bc or (cj == bc and qj > bq))):
                        best = sc
                        bj = j
                        bc = cj
                        bq = qj
            abuf[cur, i] = bj
            mine = bc
        part[tid] = mine
        cuda.syncthreads()
        _block_sum(part, tid)
        if tid == 0:
            partials[bid] = part[0]
            cuda.threadfence()
            ticket = cuda.atomic.add(ist, I_TICKET, 1)
            flag[0] = 1 if ticket == nb - 1 else 0
        cuda.syncthreads()
        if flag[0] == 1:
            cuda.threadfence()
            acc = 0.0
            k = tid
            while k < nb:
                acc += partials[k]
                k += TPB
            part[tid] = acc
            cuda.syncthreads()
            _block_sum(part, tid)
            if tid == 0:
                ist[I_TICKET] = 0
                _advance(st, ist, warm, part[0])

    @cuda.jit
    def k_fill(s, h, cost, abuf, st, ist, pr, pi, pj):
        cq = cuda.const.array_like(Q)
        ce = cuda.const.array_like(E)
        sh_c = cuda.shared.array(MAXM, float64)
        shr = cuda.shared.array(TPB, float64)
        shi = cuda.shared.array(TPB, int32)
        shj = cuda.shared.array(TPB, int32)
        flag = cuda.shared.array(1, int32)
        if ist[I_FDONE] != 0:
            return
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        nb = cuda.gridDim.x
        n = s.shape[0]
        m = cost.shape[0]
        j = tid
        while j < m:
            sh_c[j] = cost[j]
            j += TPB
        cuda.syncthreads()
        res = ist[I_RES]
        slack = st[S_SLACK]
        i = bid * TPB + tid
        best = -math.inf
        bj = -1
        if i < n:
            si = s[i]
            hi_ = h[i]
            ai = abuf[res, i]
            cu = _mul(si, cq[ai])
            cc = sh_c[ai]
            bc = math.inf
            bq = -math.inf
            for j in range(m):
                if ce[j] <= hi_:
                    cj = sh_c[j]
                    qj = cq[j]
                    du = _sub(_mul(si, qj), cu)
                    dc = cj - cc
                    if du > 0.0 and dc > 0.0 and dc <= slack:
                        r = du / dc
                        if r > best or (r == best and (cj < bc or (cj == bc and qj > bq))):
                            best = r
                            bj = j
                            bc = cj
                            bq = qj
        shr[tid] = best
        shi[tid] = i if bj >= 0 else 0x7FFFFFFF
        shj[tid] = bj
        cuda.syncthreads()
        _block_argmax(shr, shi, shj, tid)
        if tid == 0:
            pr[bid] = shr[0]
            pi[bid] = shi[0]
            pj[bid] = shj[0]
            cuda.threadfence()
            ticket = cuda.atomic.add(ist, I_TICKET, 1)
            flag[0] = 1 if ticket == nb - 1 else 0
        cuda.syncthreads()
        if flag[0] == 1:
            cuda.threadfence()
            br = -math.inf
            bi = 0x7FFFFFFF
            bjj = -1
            k = tid
            while k < nb:
                if pr[k] > br or (pr[k] == br and pi[k] < bi):
                    br = pr[k]
                    bi = pi[k]
                    bjj = pj[k]
                k += TPB
            shr[tid] = br
            shi[tid] = bi
            shj[tid] = bjj
            cuda.syncthreads()
            _block_argmax(shr, shi, shj, tid)
            if tid == 0:
                ist[I_TICKET] = 0
                if shj[0] < 0:
                    ist[I_FDONE] = 1
                else:
                    bi = shi[0]
                    bjj = shj[0]
                    d = sh_c[bjj] - sh_c[abuf[res, bi]]
                    abuf[res, bi] = bjj
                    st[S_SLACK] = slack - d
                    ist[I_FSTEPS] += 1

    return k_eval, k_fill


class CudaAllocator:
    def __init__(self, table, rtol=1e-4, btol=1e-3, max_iter=64, fill=True, batch=None):
        self.table = table
        self.rtol = rtol
        self.btol = btol
        self.max_iter = max_iter
        self.fill = fill
        self.batch = batch if batch is not None else (2 if SIMULATED else 8)
        self.lam = None
        self.k_eval, self.k_fill = _build(table.quality, table.err)
        self.d_warm = cuda.to_device(np.zeros(1, np.float64))
        self.d_st = cuda.to_device(np.zeros(16, np.float64))
        self.d_ist = cuda.to_device(np.zeros(16, np.int32))
        self._n = -1
        self.launches = 0

    def _buffers(self, n):
        if n != self._n:
            nb = (n + TPB - 1) // TPB
            self.d_abuf = cuda.device_array((3, n), np.int32)
            self.d_part = cuda.device_array(nb, np.float64)
            self.d_pi = cuda.device_array(nb, np.int32)
            self.d_pj = cuda.device_array(nb, np.int32)
            self._n = n
            self._nb = nb
        return self._nb

    def reset_warm(self):
        self.d_warm.copy_to_device(np.zeros(1, np.float64))
        self.lam = None

    def allocate(self, salience, cost, budget, headroom=None):
        s, q, c, e, h, lam_max = prepare(self.table, salience, cost, headroom)
        n = len(s)
        nb = self._buffers(n)
        d_s = cuda.to_device(s)
        d_h = cuda.to_device(h)
        d_c = cuda.to_device(c)
        st = np.zeros(16, np.float64)
        st[S_LMAX] = lam_max
        st[S_BUDGET] = float(budget)
        st[S_RTOL] = self.rtol
        st[S_BTOL] = self.btol
        ist = np.zeros(16, np.int32)
        ist[I_CUR], ist[I_LOB], ist[I_HIB] = 0, 1, 2
        ist[I_MAXIT] = self.max_iter
        ist[I_FILL] = 1 if self.fill else 0
        ist[I_FDONE] = 1
        ist[I_NODC] = 1 if lam_max < 0 else 0
        self.d_st.copy_to_device(st)
        self.d_ist.copy_to_device(ist)
        args = (d_s, d_h, d_c, self.d_abuf, self.d_st, self.d_ist)
        self.launches = 0
        while True:
            for _ in range(self.batch):
                self.k_eval[nb, TPB](*args, self.d_warm, self.d_part)
                self.launches += 1
            ist = self.d_ist.copy_to_host()
            if ist[I_DONE]:
                break
        while not ist[I_FDONE]:
            for _ in range(self.batch):
                self.k_fill[nb, TPB](*args, self.d_part, self.d_pi, self.d_pj)
                self.launches += 1
            ist = self.d_ist.copy_to_host()
        st = self.d_st.copy_to_host()
        a = self.d_abuf[int(ist[I_RES])].copy_to_host()
        self.lam = float(self.d_warm.copy_to_host()[0])
        return finish(s, q, c, a, float(st[S_LAMF]), int(ist[I_EVALS]),
                      bool(ist[I_INF]), int(ist[I_FSTEPS]))


def occupancy_report(alloc=None):
    """Occupancy of k_eval against T4 limits. Measured on hardware, analytic otherwise."""
    smem_eval = 8 * MAXM + 8 * TPB + 4
    smem_fill = 8 * MAXM + 8 * TPB + 4 * TPB * 2 + 4
    const_bytes = 2 * 8 * MAXM
    lines = [f"TPB={TPB} warps/block={TPB // T4['warp']} static smem: eval={smem_eval} B fill={smem_fill} B "
             f"const={const_bytes} B (of {T4['const_bytes']} B)"]
    regs = None
    if alloc is not None and not SIMULATED:
        try:
            sig = list(alloc.k_eval.overloads)[0]
            regs = alloc.k_eval.get_regs_per_thread(sig)
            ov = alloc.k_eval.overloads[sig]
            blocks = cuda.current_context().get_active_blocks_per_multiprocessor(
                ov._codelibrary.get_cufunc(), TPB, smem_eval) if hasattr(
                cuda.current_context(), "get_active_blocks_per_multiprocessor") else None
            lines.append(f"measured: regs/thread={regs} active blocks/SM={blocks}")
        except Exception as ex:  # noqa: BLE001
            lines.append(f"measured occupancy unavailable: {ex!r}")
    for r in ([regs] if regs else [32, 40, 48, 64]):
        by_regs = T4["regs_per_sm"] // (r * TPB)
        by_smem = T4["smem_per_sm"] // smem_eval
        by_thr = T4["threads_per_sm"] // TPB
        b = min(by_regs, by_smem, by_thr, T4["blocks_per_sm"])
        lines.append(f"analytic regs={r}: blocks/SM by regs={by_regs} smem={by_smem} threads={by_thr} "
                     f"-> {b} blocks/SM = {b * TPB // T4['warp']}/{T4['warps_per_sm']} warps "
                     f"({100 * b * TPB // T4['threads_per_sm']}%)")
    return "\n".join(lines)
