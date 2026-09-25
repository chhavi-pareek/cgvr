"""A second ledger, for what the viewer sees change: a hard bound on LOD popping.

Every change of an agent's view pair (animation or geometry tier) while it is on screen is a
visible pop. The classic remedy is hysteresis -- a margin a switch must clear -- which makes
pops rarer but bounds nothing: an agent sitting on a salience boundary can still flicker at
any rate the margin does not happen to cover. This is the error ledger's construction applied
to that quantity instead of to behavioural divergence: a token bucket per (viewer, agent),
capacity C, refilled at r per frame, a visible change spends one token, and an agent in view
with less than one token has its view pair HELD for the frame (alloc.factored's view_lock).

Guarantee: over any window of T frames, the visible pops of one agent in one view number at
most C + r * T, plus any hold the allocator had to release because the frame budget could not
be met otherwise (FactoredAllocator.released). The ordering of the three constraints is the
design rule of the whole system: the divergence cap is safety and is never broken, the frame
budget yields only to it, and the pop budget yields to both.

Changes while the agent is off screen are free: nobody sees them.
"""
import numpy as np


class PopLedger:
    def __init__(self, shape, capacity=2.0, refill=1.0 / 300.0):   # a token per 5 s, as the engine
        self.capacity, self.refill = float(capacity), float(refill)
        self.tokens = np.full(shape, self.capacity)
        self.prev = np.full(shape, -1, np.int64)
        self.pops = 0

    def holds(self, in_view):
        """View pair to hold per agent (its current one), or -1 where it may change."""
        hold = np.asarray(in_view, bool) & (self.tokens < 1.0) & (self.prev >= 0)
        return np.where(hold, self.prev, -1)

    def update(self, view_pair, in_view):
        """Record this frame's view pairs; returns the mask of visible pops."""
        vp = np.asarray(view_pair, np.int64)
        popped = (self.prev >= 0) & (vp != self.prev) & np.asarray(in_view, bool)
        self.tokens = np.minimum(self.capacity, self.tokens - popped + self.refill)
        self.pops += int(popped.sum())
        self.prev = vp.copy()
        return popped
