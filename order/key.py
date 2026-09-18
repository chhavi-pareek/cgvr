"""Composite sort key: coarse Morton cell, then the allocator's state fields, then fine Morton.

key(w) = [top k Morton bits] [tier (2 bits)] [class (2 bits)] [remaining 20-k Morton bits] [agent id]
with k = round(w * BITS). w = 1 is pure Morton order (state only breaks exact
cell ties); w = 0 is pure state order: state alone decides which agents share a
warp, Morton only orders inside each state group. In between, agents are
grouped by state inside each of 4^k coarse Morton cells and spatially ordered
within the group. `blind=True` drops the Morton bits entirely (state, then
agent id), a reference with no spatial information. The tier field is the
allocator's output from the previous frame (invariant 5).
"""
import numpy as np

BITS = 20  # 10 bits per axis, 1024 x 1024 cells over the scene
TIER_BITS = 2
CLS_BITS = 2
STATE_BITS = TIER_BITS + CLS_BITS
NCLS = 1 << CLS_BITS
ID_BITS = 24


def _spread(v):
    v = v.astype(np.uint32) & 0x3FF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def morton(pos, size):
    """20-bit Morton code of positions quantised onto a 1024^2 grid over `size`."""
    q = np.clip(pos / np.asarray(size, np.float32) * 1024.0, 0, 1023).astype(np.uint32)
    return _spread(q[:, 0]) | (_spread(q[:, 1]) << np.uint32(1))


def morton_bits(w):
    return int(round(float(w) * BITS))


def composite_key(pos, tier, cls, w, size, blind=False):
    k = morton_bits(w)
    n = len(pos)
    assert n < (1 << ID_BITS)
    m = morton(pos, size).astype(np.uint64)
    if blind:
        m = np.zeros(n, np.uint64)
    coarse = m >> np.uint64(BITS - k)
    fine = m & np.uint64((1 << (BITS - k)) - 1)
    state = (tier.astype(np.uint64) << np.uint64(CLS_BITS)) | cls.astype(np.uint64)
    ident = np.arange(n, dtype=np.uint64)
    lo = BITS - k + ID_BITS
    return (coarse << np.uint64(STATE_BITS + lo)) | (state << np.uint64(lo)) | (fine << np.uint64(ID_BITS)) | ident


def order(pos, tier, cls, w, size, blind=False):
    """Permutation that sorts agents by the composite key."""
    return np.argsort(composite_key(pos, tier, cls, w, size, blind), kind="stable")
