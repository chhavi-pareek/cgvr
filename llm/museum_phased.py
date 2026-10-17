"""The museum with memory (llm/museum_mem.py) with the alarm phased zone by zone.

The contexts, prompts and answer tables are those of museum_mem; only the timing of the shift changes.
The alarm, and later the smell of smoke, reaches the four zones one after another, PHASE_S apart, as in
a phased evacuation, so at any moment only some visitors face the new situation. A mechanism test: does
PARITY's advantage in the museum come from a shift that makes every context new at once? The refill
stop, the people counted at the alarm and the evacuation curve still start at ALARM_S.
"""
import numpy as np

from . import museum_mem as M
from .museum_mem import *  # noqa: F401,F403  (same contexts, prompt and factors, hence the same answer tables)

PHASE_S = 60                                   # zone z hears the alarm at ALARM_S + PHASE_S * z


class MuseumPhased(M.MuseumMem):
    def context(self, step, idx=None, zone=None, comp=None, mem=None, at_step=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        s = np.full(idx.size, step) if at_step is None else np.broadcast_to(np.asarray(at_step), idx.shape)
        z = np.asarray(self.zone[idx] if zone is None else zone)
        off = PHASE_S * np.clip(z, 0, None)
        a = (s >= M.ALARM_S + off).astype(np.int64) + (s >= M.SMOKE_S + off)
        g = np.asarray(self.comp[idx] if comp is None else comp)
        m = np.asarray(self.mem[idx] if mem is None else mem)
        return M.ctx_index(self.persona[idx], a, np.full(idx.size, self.xb), g, z, m)


World = MuseumPhased
