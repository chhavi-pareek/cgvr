"""The station (llm/station.py) with each traveller's own memory in the prompt.

As in llm/museum_mem.py: five shared factors give 576 contexts, which one run's model budget nearly
covers, so a reply cache serves almost every decision. One memory factor of eight first-person
facts, drawn on arrival and changed by what the traveller does and what happens to them, makes
4608 contexts.
"""
import numpy as np

from . import station
from .station import A_FOOD, A_INFO, ACTIONS, N_ACT, N_Z, PERSONAS, Q_WORDS, T_WORDS, ZONES

MEMORY = (
    "Nothing in particular is on their mind.",
    "They are carrying a heavy suitcase.",
    "They have not eaten since this morning and are very hungry.",
    "Their phone battery is almost dead.",
    "They already missed one train today.",
    "They arranged to meet a friend at the information desk.",
    "They are not sure which platform their train leaves from.",
    "They heard an announcement that trains are running a few minutes late.",
)
MEMORY_W = np.array([0.30, 0.12, 0.12, 0.08, 0.06, 0.08, 0.14, 0.10])
M_NONE, M_SUITCASE, M_HUNGRY, M_PHONE, M_MISSED, M_FRIEND, M_PLATFORM, M_LATE = range(len(MEMORY))
N_M = len(MEMORY)
SIZES = station.SIZES + (N_M,)
N_CTX = station.N_CTX * N_M


def ctx_index(p, t, k, q, z, m):
    return station.ctx_index(p, t, k, q, z) * N_M + np.asarray(m)


def ctx_parts(c):
    c = np.asarray(c)
    return station.ctx_parts(c // N_M) + (c % N_M,)


def prompt(c, rot=0):
    p, t, k, q, z, m = (int(v) for v in ctx_parts(c))
    opts = "\n".join(f"{'ABCDEF'[i]}) {ACTIONS[(i + rot) % N_ACT]}" for i in range(N_ACT))
    return (f"You are simulating one person in a busy train station. Decide what they do next.\n"
            f"Person: {PERSONAS[p]}.\n"
            f"Their train leaves in {T_WORDS[t]}.\n"
            f"They {'already have' if k else 'do not have'} a ticket.\n"
            f"The ticket queue is {Q_WORDS[q]}.\n"
            f"{MEMORY[m]}\n"
            f"They are at {ZONES[z]}.\n"
            f"What do they do next?\n{opts}\nAnswer with a single letter.")


def _settled(act, mem):
    """Memory once an errand is done: food ends hunger, the information desk finds the friend and
    the platform."""
    mem = mem.copy()
    mem[(act == A_FOOD) & (mem == M_HUNGRY)] = M_NONE
    mem[(act == A_INFO) & np.isin(mem, (M_FRIEND, M_PLATFORM))] = M_NONE
    return mem


class StationMem(station.Station):
    """The station's state plus each traveller's memory: drawn on arrival, settled by the errand
    that answers it, and set to 'missed a train' when their train leaves without them."""

    def __init__(self, n, rng):
        self.mem = np.zeros(n, np.int64)
        super().__init__(n, rng)

    def respawn(self, idx, step, stagger=False):
        super().respawn(idx, step, stagger)
        idx = np.atleast_1d(idx)
        if idx.size:
            self.mem[idx] = self.rng.choice(N_M, idx.size, p=MEMORY_W)

    def context(self, step, idx=None, zone=None, ticket=None, mem=None, at_step=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        base = super().context(step, idx, zone, ticket, at_step)
        return base * N_M + np.asarray(self.mem[idx] if mem is None else mem)

    def next_state(self, idx):
        """(zone, ticket, memory) at the next decision. A missed train cannot be foreseen."""
        zone, ticket = super().next_state(idx)
        idx = np.atleast_1d(idx)
        return zone, ticket, _settled(self.activity[idx], self.mem[idx])

    def apply(self, idx, actions, step):
        idx, actions = np.atleast_1d(idx), np.atleast_1d(actions)
        super().apply(idx, actions, step)
        self.mem[idx] = _settled(actions, self.mem[idx])

    def tick(self, step):
        before, gen = self.train_at.copy(), self.gen.copy()
        super().tick(step)
        self.mem[(self.train_at > before) & (self.gen == gen)] = M_MISSED


World = StationMem
