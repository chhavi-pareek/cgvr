"""A museum evacuation: the rationing problem when the situation changes all at once.

Visitors browse a museum and decide, now and then, what to do next. At ALARM_S a fire alarm
sounds; at SMOKE_S smoke reaches the halls. Every context after the alarm is one the model has
not been asked about -- the surrogate was trained on calm browsing -- so the whole crowd's
contexts shift at once, the model can serve a few percent of the decisions, and those decisions
decide who gets out and how fast. It is the distribution-shift test of the scheduler; the station
(llm/station.py) is the steady one.

Context: persona, alarm state, how the main exit looks, the person's companions, and where they
are (648 contexts). People interact through the exits: everyone heading for the main exit
lengthens its queue, and a jammed main exit changes what the next person decides. Exit capacity
and the queue lengths people perceive scale with the crowd (a bigger museum has wider doors), so
crowd sizes are comparable. Before the alarm a visitor who leaves is replaced by a new one; after
it nobody comes in and the crowd drains. The API is the station's (llm/schedule.py drives both).

One step is one second.
"""
import numpy as np

PERSONAS = (
    "a regular visitor who knows the building well",
    "a tourist on a first visit who does not know the layout",
    "a parent with two young children",
    "an elderly visitor who walks slowly with a cane",
    "a teenager visiting with a group of friends",
    "a museum guard on duty",
)
PERSONA_W = np.array([0.25, 0.25, 0.15, 0.10, 0.20, 0.05])
COMPANY_P = np.array([0.3, 0.6, 1.0, 0.5, 1.0, 0.0])     # chance of coming with someone

ALARM_WORDS = ("Everything is calm; there has been no alarm.",
               "A fire alarm is ringing, but there is no sign of fire.",
               "A fire alarm is ringing and they can smell smoke.")
EXIT_WORDS = ("The main exit is clear.", "There is a queue at the main exit.",
              "The main exit is jammed with people pushing.")
COMPANY_WORDS = ("They came alone.", "The people they came with are beside them.",
                 "They have lost sight of the people they came with.")
ZONES = ("in a gallery far from the exits", "in the central hall", "in a side gallery next to an emergency exit",
         "at the main exit")
Z_FAR, Z_HALL, Z_SIDE, Z_MAIN = range(4)
# Browsing comes first: the 7B rarely picks a late option (the station showed the same position
# bias), and with browsing last (v1) calm visitors kept leaving, so the crowd was mostly out
# before the alarm and the shift barely happened.
ACTIONS = ("keep looking at the exhibits", "walk to the main exit", "walk to the nearest emergency exit",
           "look for the people they came with", "stay where they are and wait for instructions",
           "go to the cloakroom to collect their coat and bag")
A_BROWSE, A_MAIN, A_EMERG, A_SEARCH, A_WAIT, A_COAT = range(6)

N_P, N_A, N_X, N_G, N_Z = len(PERSONAS), 3, 3, 3, len(ZONES)
N_CTX = N_P * N_A * N_X * N_G * N_Z
N_ACT = len(ACTIONS)
SIZES = (N_P, N_A, N_X, N_G, N_Z)

ALARM_S, SMOKE_S = 300, 480
WALK = np.array([[60, 40], [30, 35], [45, 8], [5, 40]])  # zone -> seconds to (main, emergency) exit
DWELL = {A_SEARCH: 30, A_WAIT: 30, A_COAT: 70, A_BROWSE: 90}
MAIN_PER_1000, EMERG_PER_1000 = 2.0, 1.0                  # people per second through each exit
X_EDGES = (0.015, 0.06)                                  # main-exit queue, as a share of the crowd
X_HYST = 0.004
LOSE_P = (0.02, 0.15)                                    # a walk separates a group: calm, alarm
FIND_P = 0.5
NEVER = 10 ** 12


def ctx_index(p, a, x, g, z):
    return (((np.asarray(p) * N_A + a) * N_X + x) * N_G + g) * N_Z + z


def ctx_parts(c):
    c = np.asarray(c)
    z = c % N_Z; c = c // N_Z
    g = c % N_G; c = c // N_G
    x = c % N_X; c = c // N_X
    a = c % N_A
    return c // N_A, a, x, g, z


def prompt(c, rot=0):
    p, a, x, g, z = (int(v) for v in ctx_parts(c))
    opts = "\n".join(f"{'ABCDEF'[i]}) {ACTIONS[(i + rot) % N_ACT]}" for i in range(N_ACT))
    return (f"You are simulating one person in a large museum. Decide what they do next.\n"
            f"Person: {PERSONAS[p]}.\n"
            f"{ALARM_WORDS[a]}\n"
            f"{EXIT_WORDS[x]}\n"
            f"{COMPANY_WORDS[g]}\n"
            f"They are {ZONES[z]}.\n"
            f"What do they do next?\n{opts}\nAnswer with a single letter.")


class Museum:
    """Vectorised state for n visitor slots. A slot with busy_until <= step needs a decision; an
    evacuated slot is never busy-free again."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.persona = np.zeros(n, np.int64)
        self.zone = np.zeros(n, np.int64)
        self.comp = np.zeros(n, np.int64)
        self.busy_until = np.zeros(n, np.int64)
        self.activity = np.full(n, -1, np.int64)
        self.inside = np.ones(n, bool)
        self.gen = np.zeros(n, np.int64)
        self.queues = ([], [])                 # main, emergency: agent ids, FIFO
        self.credit = [0.0, 0.0]
        self.rate = (MAIN_PER_1000 * n / 1000.0, EMERG_PER_1000 * n / 1000.0)
        self.xb = 0
        self.alarm = 0
        self.at_alarm = None                   # people inside when the alarm sounded
        self.out_t = []                        # seconds after the alarm of each evacuation
        self.coats = self.left = 0
        self.respawn(np.arange(n), 0, stagger=True)

    def respawn(self, idx, step, stagger=False):
        idx = np.atleast_1d(idx)
        if idx.size == 0:
            return
        r = self.rng
        self.gen[idx] += 1
        self.persona[idx] = r.choice(N_P, idx.size, p=PERSONA_W)
        self.comp[idx] = (r.random(idx.size) < COMPANY_P[self.persona[idx]]).astype(np.int64)
        self.zone[idx] = r.integers(0, 3, idx.size) if stagger else Z_HALL
        self.activity[idx] = -1
        self.busy_until[idx] = step + (r.integers(0, 120, idx.size) if stagger else 0)

    def present(self):
        return int(self.inside.sum())

    def context(self, step, idx=None, zone=None, comp=None, at_step=None):
        """Context index, optionally predicted (the zone and companions the agent WILL have). The
        alarm is read at the step the decision is made; the exit queue is the current one."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        s = np.full(idx.size, step) if at_step is None else np.broadcast_to(np.asarray(at_step), idx.shape)
        a = (s >= ALARM_S).astype(np.int64) + (s >= SMOKE_S)
        z = np.asarray(self.zone[idx] if zone is None else zone)
        g = np.asarray(self.comp[idx] if comp is None else comp)
        return ctx_index(self.persona[idx], a, np.full(idx.size, self.xb), g, z)

    def next_state(self, idx):
        """(zone, companions) each busy agent will have when it next decides."""
        idx = np.atleast_1d(idx)
        zone, comp = self.zone[idx].copy(), self.comp[idx].copy()
        act = self.activity[idx]
        zone[act == A_COAT] = Z_HALL
        comp[(act == A_SEARCH) & (comp == 2)] = 1          # the likelier outcome of a search
        return zone, comp

    def next_decision_step(self, step, idx):
        return self.busy_until[np.atleast_1d(idx)].copy()

    def _walk(self, i, step):
        if self.comp[i] == 1 and self.rng.random() < LOSE_P[int(step >= ALARM_S)]:
            self.comp[i] = 2

    def apply(self, idx, actions, step):
        """Carry out one decision per agent."""
        r = self.rng
        for i, a in zip(np.atleast_1d(idx), np.atleast_1d(actions)):
            self.activity[i] = a
            if a in (A_MAIN, A_EMERG):
                self._walk(i, step)
                self.busy_until[i] = step + WALK[self.zone[i], int(a == A_EMERG)]   # queues on arrival
            elif a == A_SEARCH:
                if self.comp[i] == 2 and r.random() < FIND_P:
                    self.comp[i] = 1
                    self.zone[i] = r.integers(0, 3)
                self.busy_until[i] = step + DWELL[a]
            elif a == A_COAT:
                self._walk(i, step)
                if step >= ALARM_S:
                    self.coats += 1
                self.zone[i] = Z_HALL
                self.busy_until[i] = step + DWELL[a]
            elif a == A_BROWSE:
                self._walk(i, step)
                self.zone[i] = r.integers(0, 3)
                self.busy_until[i] = step + DWELL[a]
            else:
                self.busy_until[i] = step + DWELL[a]

    def _exit(self, i, step):
        if self.at_alarm is None:
            self.left += 1
            self.respawn(i, step)                                    # a new visitor comes in
            return
        self.inside[i] = False
        self.gen[i] += 1                                             # ends the person's ledger
        self.zone[i] = -1
        self.busy_until[i] = NEVER
        self.activity[i] = -1
        self.out_t.append(step - ALARM_S)

    def perceive_exit(self):
        share = len(self.queues[0]) / self.n
        while self.xb < len(X_EDGES) and share >= X_EDGES[self.xb] + X_HYST:
            self.xb += 1
        while self.xb > 0 and share < X_EDGES[self.xb - 1] - X_HYST:
            self.xb -= 1

    def tick(self, step):
        """Arrivals at the exits, the exits' service, the alarm."""
        if step == ALARM_S:
            self.at_alarm = self.present()
        arrive = np.flatnonzero(np.isin(self.activity, (A_MAIN, A_EMERG)) & (self.busy_until <= step) & self.inside)
        for i in arrive:
            e = int(self.activity[i] == A_EMERG)
            self.queues[e].append(int(i))
            self.zone[i] = Z_MAIN if e == 0 else Z_SIDE
            self.busy_until[i] = NEVER
            self.activity[i] = 10 + e                                # queued
        for e in (0, 1):
            self.credit[e] += self.rate[e]
            while self.credit[e] >= 1.0 and self.queues[e]:
                self.credit[e] -= 1.0
                self._exit(self.queues[e].pop(0), step)
            if not self.queues[e]:
                self.credit[e] = min(self.credit[e], 1.0)
        self.perceive_exit()

    def outcomes(self):
        """Evacuation from the alarm: seconds until half and 90% of the people inside at the
        alarm were out (nan if never), how many were still inside at the end, how many went back
        for their coat after the alarm, and the evacuation curve (bench/llm_agents.py compares it
        with the reference's: W1 between evacuation-time distributions, capped at 600 s)."""
        t = np.sort(self.out_t)
        m = self.at_alarm or 0
        q = lambda f: float(t[int(np.ceil(f * m)) - 1]) if m and len(t) >= np.ceil(f * m) else float("nan")  # noqa: E731
        # share of the people inside at the alarm who are out by 30, 60, ... 600 s after it
        grid = np.arange(30, 601, 30)
        curve = np.searchsorted(t, grid, side="right") / max(m, 1)
        return dict(t50=q(0.5), t90=q(0.9), inside=self.present(), coats=self.coats, left=self.left, curve=curve)


World = Museum
