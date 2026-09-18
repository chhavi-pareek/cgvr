"""Parallel allocators must reproduce the serial oracle bit for bit."""
import shutil

import numpy as np
import pytest

from alloc.config import build_table
from alloc.costmodel import row_costs_from_theta
from alloc.serial import SerialAllocator
from alloc.threaded import ThreadedAllocator
from tests.test_alloc import _theta, _ticks

TABLE = build_table()


def _instance(seed, n, ticks=True):
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, 0.8, n)
    core, th = _theta(rng)
    cost = row_costs_from_theta(TABLE, core, th)
    if ticks:
        cost = _ticks(cost).astype(np.float64)
    return s, cost


def _same(r, ref):
    assert np.array_equal(r.assign, ref.assign)
    assert r.infeasible == ref.infeasible
    assert r.evals == ref.evals
    assert r.fill_steps == ref.fill_steps
    assert r.lam == ref.lam
    assert r.cost == ref.cost
    assert r.utility == ref.utility


def _sweep(make, seed, n, ticks, fracs=(0.9, 0.55, 0.4, 0.42, 0.25, 0.05), headroom=False):
    """Warm-started budget sweep: exercises cold start, expand-up, expand-down, bisection."""
    s, cost = _instance(seed, n, ticks)
    ref = SerialAllocator(TABLE)
    par = make()
    h = None
    if headroom:
        h = np.random.default_rng(seed + 100).uniform(0.2, 1.6, n)
        h = np.maximum(h, TABLE.err.min())
    t0 = ref.allocate(s, cost, np.inf, headroom=h).cost
    par.allocate(s, cost, np.inf, headroom=h)
    for f in fracs:
        a = ref.allocate(s, cost, f * t0, headroom=h)
        b = par.allocate(s, cost, f * t0, headroom=h)
        _same(b, a)
    return ref, par


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("ticks", [True, False])
def test_threaded_matches_serial(seed, ticks):
    _sweep(lambda: ThreadedAllocator(TABLE, threads=4), seed, 1000, ticks)


def test_threaded_headroom_and_thread_count():
    _sweep(lambda: ThreadedAllocator(TABLE, threads=3), 7, 500, True, headroom=True)
    _sweep(lambda: ThreadedAllocator(TABLE, threads=1), 8, 300, True)


def test_threaded_degenerate_cases():
    s, cost = _instance(5, 200)
    ref, par = SerialAllocator(TABLE), ThreadedAllocator(TABLE, threads=2)
    floor = cost.min() * len(s)
    _same(par.allocate(s, cost, floor * 0.5), ref.allocate(s, cost, floor * 0.5))
    flat = np.full_like(cost, 3.0)
    _same(par.allocate(s, flat, 100.0), ref.allocate(s, flat, 100.0))
    _same(par.allocate(s, cost, floor * 2), ref.allocate(s, cost, floor * 2))
    assert par.lam == ref.lam


# -- CUDA (simulator on machines without a driver) -----------------------------

@pytest.mark.parametrize("seed,ticks", [(0, True), (1, False)])
def test_cuda_matches_serial(seed, ticks):
    from alloc.cuda import CudaAllocator
    # N=40 spans two 32-thread simulator blocks, so the last-block reduction runs
    _sweep(lambda: CudaAllocator(TABLE), seed, 40, ticks, fracs=(0.42, 0.45))


def test_cuda_headroom_and_degenerate():
    from alloc.cuda import CudaAllocator
    _sweep(lambda: CudaAllocator(TABLE), 3, 40, True, fracs=(0.4,), headroom=True)
    s, cost = _instance(5, 40)
    ref, par = SerialAllocator(TABLE), CudaAllocator(TABLE)
    floor = cost.min() * len(s)
    _same(par.allocate(s, cost, floor * 0.5), ref.allocate(s, cost, floor * 0.5))
    flat = np.full_like(cost, 3.0)
    _same(par.allocate(s, flat, 100.0), ref.allocate(s, flat, 100.0))
    assert par.lam == ref.lam


# -- OpenACC (serial-C build on machines without nvc) --------------------------

@pytest.fixture(scope="module")
def acc_lib():
    from alloc import openacc
    if not openacc.available():
        if shutil.which("cc") is None:
            pytest.skip("no C compiler")
        openacc.build("serial")
    return openacc


@pytest.mark.parametrize("seed,ticks", [(0, True), (1, False), (2, True)])
def test_openacc_matches_serial(acc_lib, seed, ticks):
    _sweep(lambda: acc_lib.OpenACCAllocator(TABLE), seed, 1000, ticks)


def test_openacc_headroom_and_degenerate(acc_lib):
    _sweep(lambda: acc_lib.OpenACCAllocator(TABLE), 3, 500, True, headroom=True)
    s, cost = _instance(5, 200)
    ref, par = SerialAllocator(TABLE), acc_lib.OpenACCAllocator(TABLE)
    floor = cost.min() * len(s)
    _same(par.allocate(s, cost, floor * 0.5), ref.allocate(s, cost, floor * 0.5))
    flat = np.full_like(cost, 3.0)
    _same(par.allocate(s, flat, 100.0), ref.allocate(s, flat, 100.0))
    assert par.lam == ref.lam
