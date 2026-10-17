"""The museum evacuation (llm/museum.py) with each visitor's own memory in the prompt.

With five shared factors the museum has 648 contexts, fewer than the model can answer in one run,
so a reply cache covers nearly every decision and rationing barely matters. LLM agents carry
memory (what they came for, what they saw, what they left behind), so their prompts rarely repeat;
one memory factor of eight first-person facts, assigned on arrival and changed by what the visitor
does, makes 5184 contexts, a few times what one run can buy.
"""
import numpy as np

from . import museum
from .museum import (A_BROWSE, A_COAT, ACTIONS, ALARM_WORDS, COMPANY_WORDS, EXIT_WORDS, N_ACT, N_Z, PERSONAS,
                     Z_SIDE, ZONES)

MEMORY = (
    "Nothing in particular is on their mind.",
    "They came to see one particular exhibit and have not found it yet.",
    "They have been on their feet for two hours and are very tired.",
    "They left a bag with their laptop and passport in the cloakroom.",
    "They have a train to catch in twenty minutes.",
    "Earlier they overheard a guard say the fire alarm would be tested today.",
    "A few minutes ago they walked past the emergency exit in the side gallery.",
    "Their phone battery is dead, so they cannot call anyone.",
)
MEMORY_W = np.array([0.30, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07, 0.10])
M_NONE, M_EXHIBIT, M_TIRED, M_BAG, M_TRAIN, M_DRILL, M_SAW_EXIT, M_PHONE = range(len(MEMORY))
N_M = len(MEMORY)
SIZES = museum.SIZES + (N_M,)
N_CTX = museum.N_CTX * N_M
SEE_EXIT_P, FIND_EXHIBIT_P = 0.5, 0.3
ALARM_S, SMOKE_S = museum.ALARM_S, museum.SMOKE_S
EVENT_PART = museum.EVENT_PART                           # memory is appended last, so the alarm keeps its index


def ctx_index(p, a, x, g, z, m):
    return museum.ctx_index(p, a, x, g, z) * N_M + np.asarray(m)


def ctx_parts(c):
    c = np.asarray(c)
    return museum.ctx_parts(c // N_M) + (c % N_M,)


def prompt(c, rot=0):
    p, a, x, g, z, m = (int(v) for v in ctx_parts(c))
    opts = "\n".join(f"{'ABCDEF'[i]}) {ACTIONS[(i + rot) % N_ACT]}" for i in range(N_ACT))
    return (f"You are simulating one person in a large museum. Decide what they do next.\n"
            f"Person: {PERSONAS[p]}.\n"
            f"{ALARM_WORDS[a]}\n"
            f"{EXIT_WORDS[x]}\n"
            f"{COMPANY_WORDS[g]}\n"
            f"{MEMORY[m]}\n"
            f"They are {ZONES[z]}.\n"
            f"What do they do next?\n{opts}\nAnswer with a single letter.")


class MuseumMem(museum.Museum):
    """The museum's state plus each visitor's memory: drawn on arrival, then a browse into the
    side gallery may show them its emergency exit, a browse may find the exhibit they came for,
    and collecting the coat retrieves a bag left in the cloakroom."""

    def __init__(self, n, rng):
        self.mem = np.zeros(n, np.int64)
        super().__init__(n, rng)

    def respawn(self, idx, step, stagger=False):
        super().respawn(idx, step, stagger)
        idx = np.atleast_1d(idx)
        if idx.size:
            self.mem[idx] = self.rng.choice(N_M, idx.size, p=MEMORY_W)

    def context(self, step, idx=None, zone=None, comp=None, mem=None, at_step=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        base = super().context(step, idx, zone, comp, at_step)
        return base * N_M + np.asarray(self.mem[idx] if mem is None else mem)

    def next_state(self, idx):
        """(zone, companions, memory) at the next decision: memory as now, except a coat errand
        retrieves the bag."""
        zone, comp = super().next_state(idx)
        idx = np.atleast_1d(idx)
        mem = self.mem[idx].copy()
        mem[(self.activity[idx] == A_COAT) & (mem == M_BAG)] = M_NONE
        return zone, comp, mem

    def apply(self, idx, actions, step):
        idx, actions = np.atleast_1d(idx), np.atleast_1d(actions)
        super().apply(idx, actions, step)
        r = self.rng
        for i, a in zip(idx, actions):
            if a == A_COAT and self.mem[i] == M_BAG:
                self.mem[i] = M_NONE
            elif a == A_BROWSE:
                if self.mem[i] == M_EXHIBIT and r.random() < FIND_EXHIBIT_P:
                    self.mem[i] = M_NONE
                elif self.zone[i] == Z_SIDE and self.mem[i] == M_NONE and r.random() < SEE_EXIT_P:
                    self.mem[i] = M_SAW_EXIT


World = MuseumMem
