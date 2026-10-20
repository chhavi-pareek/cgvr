"""An airport terminal: events that reach one flight's passengers at a time.

The station (llm/station.py) is steady and the museum (llm/museum.py) shifts all at once at the
alarm. Here every passenger has a flight, and each flight has its own announcements -- boarding,
and for some flights a gate change or a delay -- at its own time. The shift is local and it
recurs: at any moment a few flights' passengers face a new situation while the rest do not. Each
passenger also carries one of eight memories, so few prompts repeat (3456 contexts).

Context: persona, flight status (the known event), time to departure, how crowded the gate is,
where they are, and what they remember. Passengers interact through the gate: everyone who walks
to it or joins the line makes it more crowded for the rest of that flight. A passenger who boards
or misses the flight is replaced by a new one, so the number of slots stays the same. A scheduler
forecasting a context knows the departure time but not an announcement still to come.

One step is one second; a run lasts 900 s, as in the museum, so that the model can answer only a
fraction of the contexts the crowd meets (with 1800 s it could answer most of them, as in the station).
"""
import numpy as np

PERSONAS = (
    "a frequent business traveller who knows this airport well",
    "a first-time flyer who is nervous about flying",
    "a parent travelling with two young children",
    "an elderly traveller who walks slowly",
    "a student backpacker on a tight budget",
    "a passenger who has just arrived on a late connecting flight",
)
PERSONA_W = np.array([0.25, 0.15, 0.15, 0.10, 0.20, 0.15])

STATUS_WORDS = ("Their flight is on time and boarding has not started yet.",
                "Their flight is now boarding.",
                "Their flight has just been moved to a different gate.",
                "Their flight has been delayed by about an hour.")
S_ONTIME, S_BOARDING, S_GATE, S_DELAYED = range(4)
T_WORDS = ("more than 45 minutes", "between 20 and 45 minutes", "less than 20 minutes")
GATE_WORDS = ("The area at their gate is quiet.", "Their gate is crowded and a long line has formed.")
ZONES = ("among the shops and restaurants", "in a corridor far from their gate", "at a seat near their gate")
Z_SHOPS, Z_FAR, Z_GATE = range(3)
MEMORY = (
    "Nothing in particular is on their mind.",
    "They have not eaten since early morning and are very hungry.",
    "Their phone battery is almost dead.",
    "They promised to buy a gift for someone at home.",
    "They heard a rumour that flights to their destination may be cancelled.",
    "They are not sure their carry-on bag is small enough for the cabin.",
    "They agreed to meet a friend at a cafe before boarding.",
    "They are worried they left something behind at the security check.",
)
MEMORY_W = np.array([0.30, 0.12, 0.10, 0.10, 0.08, 0.10, 0.10, 0.10])
M_NONE, M_HUNGRY, M_PHONE, M_GIFT, M_RUMOUR, M_BAG, M_FRIEND, M_SECURITY = range(8)
# browsing first, as in the museum: the models rarely pick a late option
ACTIONS = ("browse the shops or get something to eat", "walk to their gate", "join the line to board the plane",
           "ask at the airline's service desk", "sit down and wait where they are",
           "check the departures board and their phone for updates")
A_BROWSE, A_GATE, A_LINE, A_DESK, A_WAIT, A_CHECK = range(6)

N_P, N_S, N_T, N_G, N_Z, N_M = len(PERSONAS), 4, 3, 2, len(ZONES), len(MEMORY)
SIZES = (N_P, N_S, N_T, N_G, N_Z, N_M)
N_CTX = int(np.prod(SIZES))
N_ACT = len(ACTIONS)
EVENT_PART = 1                                   # ctx_parts index of the flight status, a known event

FIRST_DEP, FLIGHT_EVERY, N_FLIGHTS = 300, 150, 60   # a departure every 150 s from 300 s on
P_GATE, P_DELAY = 0.25, 0.15                     # share of flights with a gate change, with a delay
DELAY_S, BOARD_S, CLOSE_S, GATE_SHOWN_S = 3600, 1800, 120, 600
WALK = (120, 180, 0)                             # zone -> seconds to the gate
DWELL = {A_BROWSE: 180, A_DESK: 120, A_WAIT: 150, A_CHECK: 30}
BOARD_PER_S = 0.5                                # each boarding gate lets one passenger on every 2 s
CROWD_ON, CROWD_OFF = 0.55, 0.45                 # share of a flight's passengers at its gate
NEVER = 10 ** 12


def ctx_index(p, s, t, g, z, m):
    return (((((np.asarray(p) * N_S + s) * N_T + t) * N_G + g) * N_Z + z) * N_M + m)


def ctx_parts(c):
    c = np.asarray(c)
    m = c % N_M; c = c // N_M
    z = c % N_Z; c = c // N_Z
    g = c % N_G; c = c // N_G
    t = c % N_T; c = c // N_T
    s = c % N_S
    return c // N_S, s, t, g, z, m


def prompt(c, rot=0):
    p, s, t, g, z, m = (int(v) for v in ctx_parts(c))
    opts = "\n".join(f"{'ABCDEF'[i]}) {ACTIONS[(i + rot) % N_ACT]}" for i in range(N_ACT))
    return (f"You are simulating one passenger in a busy airport terminal. Decide what they do next.\n"
            f"Passenger: {PERSONAS[p]}.\n"
            f"{STATUS_WORDS[s]}\n"
            f"Their flight leaves in {T_WORDS[t]}.\n"
            f"{GATE_WORDS[g]}\n"
            f"{MEMORY[m]}\n"
            f"They are {ZONES[z]}.\n"
            f"What do they do next?\n{opts}\nAnswer with a single letter.")


def t_bucket(seconds):
    m = np.asarray(seconds) / 60.0
    return np.where(m > 45, 0, np.where(m >= 20, 1, 2))


class Airport:
    """Vectorised state for n passenger slots. A slot with busy_until <= step needs a decision."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        r = rng
        k = np.arange(N_FLIGHTS)
        self.dep0 = FIRST_DEP + k * FLIGHT_EVERY
        self.gate_at = np.where(r.random(N_FLIGHTS) < P_GATE, self.dep0 - r.integers(25 * 60, 40 * 60, N_FLIGHTS), NEVER)
        self.delay_at = np.where(r.random(N_FLIGHTS) < P_DELAY, self.dep0 - r.integers(35 * 60, 50 * 60, N_FLIGHTS), NEVER)
        self.closed = np.zeros(N_FLIGHTS, bool)
        self.gb = np.zeros(N_FLIGHTS, np.int64)          # each gate's crowding as passengers see it
        self.lines = [[] for _ in range(N_FLIGHTS)]
        self.credit = np.zeros(N_FLIGHTS)
        self.persona = np.zeros(n, np.int64)
        self.flight = np.zeros(n, np.int64)
        self.zone = np.zeros(n, np.int64)
        self.mem = np.zeros(n, np.int64)
        self.busy_until = np.zeros(n, np.int64)
        self.activity = np.full(n, -1, np.int64)
        self.gen = np.zeros(n, np.int64)
        self.boarded = self.missed = 0
        self.respawn(np.arange(n), 0, stagger=True)

    def dep(self, f, now):
        """Departure of flights f as known at time now (a delay counts once announced)."""
        f = np.asarray(f)
        return self.dep0[f] + DELAY_S * (self.delay_at[f] <= now)

    def status(self, f, now, at=None):
        """Status of flights f at time at, from what has been announced by time now."""
        f = np.asarray(f)
        at = now if at is None else np.asarray(at)
        left = self.dep(f, now) - at
        delayed = self.delay_at[f] <= now
        moved = (self.gate_at[f] <= now) & (at < self.gate_at[f] + GATE_SHOWN_S)
        return np.where(delayed & (left > BOARD_S), S_DELAYED,
                        np.where(moved, S_GATE, np.where(left <= BOARD_S, S_BOARDING, S_ONTIME)))

    def respawn(self, idx, step, stagger=False):
        idx = np.atleast_1d(idx)
        if idx.size == 0:
            return
        r = self.rng
        self.gen[idx] += 1
        self.persona[idx] = r.choice(N_P, idx.size, p=PERSONA_W)
        self.mem[idx] = r.choice(N_M, idx.size, p=MEMORY_W)
        lo, hi = (5 * 60, 75 * 60) if stagger else (15 * 60, 75 * 60)
        ok = np.flatnonzero((self.dep0 - step >= lo) & (self.dep0 - step <= hi) & ~self.closed)
        self.flight[idx] = r.choice(ok, idx.size)
        self.zone[idx] = r.integers(0, N_Z, idx.size) if stagger else Z_FAR     # new arrivals come through security
        self.activity[idx] = -1
        self.busy_until[idx] = step + (r.integers(0, 120, idx.size) if stagger else 0)

    def present(self):
        return self.n

    def context(self, step, idx=None, zone=None, mem=None, at_step=None):
        """Context index, optionally predicted: the zone and memory the passenger WILL have and the
        step they will decide at. Announcements are those made by now; the gate's crowding is the
        current one."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        at = np.full(idx.size, step) if at_step is None else np.broadcast_to(np.asarray(at_step), idx.shape)
        f = self.flight[idx]
        s = self.status(f, step, at)
        t = t_bucket(self.dep(f, step) - at)
        z = np.asarray(self.zone[idx] if zone is None else zone)
        m = np.asarray(self.mem[idx] if mem is None else mem)
        return ctx_index(self.persona[idx], s, t, self.gb[f], z, m)

    def next_state(self, idx):
        """(zone, memory) each busy passenger will have when they next decide."""
        idx = np.atleast_1d(idx)
        zone, mem = self.zone[idx].copy(), self.mem[idx].copy()
        act = self.activity[idx]
        zone[act == A_BROWSE] = Z_SHOPS
        zone[act == A_GATE] = Z_GATE
        mem[(act == A_BROWSE) & np.isin(mem, (M_HUNGRY, M_FRIEND))] = M_NONE
        mem[(act == A_DESK) & np.isin(mem, (M_BAG, M_RUMOUR, M_SECURITY))] = M_NONE
        return zone, mem

    def next_decision_step(self, step, idx):
        return self.busy_until[np.atleast_1d(idx)].copy()

    def apply(self, idx, actions, step):
        """Carry out one decision per passenger."""
        r = self.rng
        for i, a in zip(np.atleast_1d(idx), np.atleast_1d(actions)):
            self.activity[i] = a
            f = self.flight[i]
            if a in (A_GATE, A_LINE):
                walk = WALK[self.zone[i]]
                self.zone[i] = Z_GATE
                if a == A_LINE and walk == 0 and self.status(f, step) == S_BOARDING:
                    self.lines[f].append(int(i))
                    self.busy_until[i] = NEVER                 # until boarded
                else:
                    # walking to the gate; one who wanted the line decides again on arrival
                    self.busy_until[i] = step + max(walk, 30 if a == A_LINE else 0) + (60 if a == A_GATE else 0)
            elif a == A_BROWSE:
                self.zone[i] = Z_SHOPS
                if self.mem[i] == M_HUNGRY and r.random() < 0.8:
                    self.mem[i] = M_NONE
                elif self.mem[i] == M_FRIEND and r.random() < 0.6:
                    self.mem[i] = M_NONE
                elif self.mem[i] == M_GIFT and r.random() < 0.5:
                    self.mem[i] = M_NONE
                self.busy_until[i] = step + DWELL[a]
            elif a == A_DESK:
                if self.mem[i] in (M_BAG, M_RUMOUR, M_SECURITY) and r.random() < 0.7:
                    self.mem[i] = M_NONE
                self.busy_until[i] = step + DWELL[a]
            else:
                self.busy_until[i] = step + DWELL[a]

    def outcomes(self):
        return dict(boarded=self.boarded, missed=self.missed)

    # post hoc (round 10): a decision carried out by the surrogate can be revised while the passenger is still doing it
    REVISE = ("zone", "busy_until", "activity", "mem")

    def snap(self, i):
        return {k: getattr(self, k)[i].copy() for k in self.REVISE}

    def revisable(self, i, a, step):
        return self.activity[i] == a and self.busy_until[i] > step

    def unapply(self, i, pre, post, a, step):
        """Undo apply() of action a for passenger i (still doing it); fields changed since by the world stay."""
        line = self.lines[self.flight[i]]
        if i in line:
            line.remove(i)
        for k, v in pre.items():
            if getattr(self, k)[i] == post[k]:
                getattr(self, k)[i] = v

    def tick(self, step):
        """Announcements, boarding, gate closing and each gate's crowding."""
        moved = np.flatnonzero(self.gate_at == step)
        for f in moved:                                      # the gate moved: those at it must walk again
            here = np.flatnonzero((self.flight == f) & (self.zone == Z_GATE))
            self.zone[here] = Z_FAR
            for i in self.lines[f]:
                self.busy_until[i] = step
                self.activity[i] = -1
            self.lines[f] = []
            wait = here[self.busy_until[here] > step]
            self.busy_until[wait] = np.minimum(self.busy_until[wait], step + 30)
        late = np.flatnonzero(self.delay_at == step)
        for f in late:                                       # a delay sets off rumours
            mine = np.flatnonzero((self.flight == f) & (self.mem == M_NONE))
            self.mem[mine[self.rng.random(mine.size) < 0.3]] = M_RUMOUR
        for f in np.flatnonzero(~self.closed & (self.dep(np.arange(N_FLIGHTS), step) - CLOSE_S <= step)):
            self.closed[f] = True                            # the gate closes: the line boards, the rest miss it
            board = np.array(self.lines[f], np.int64)
            self.boarded += board.size
            self.lines[f] = []
            rest = np.flatnonzero(self.flight == f)
            rest = rest[~np.isin(rest, board)]
            self.missed += rest.size
            self.respawn(board, step)
            self.respawn(rest, step)
        for f in np.flatnonzero(~self.closed & (self.status(np.arange(N_FLIGHTS), step) == S_BOARDING)):
            if not self.lines[f]:
                self.credit[f] = 0.0
                continue
            self.credit[f] += BOARD_PER_S
            while self.credit[f] >= 1.0 and self.lines[f]:
                self.credit[f] -= 1.0
                i = self.lines[f].pop(0)
                self.boarded += 1
                self.respawn(i, step)
        on = np.bincount(self.flight, minlength=N_FLIGHTS)
        at = np.bincount(self.flight[self.zone == Z_GATE], minlength=N_FLIGHTS)
        share = at / np.maximum(on, 1)
        self.gb = np.where(self.gb == 0, share >= CROWD_ON, share >= CROWD_OFF).astype(np.int64)


World = Airport
