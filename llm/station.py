"""A station full of people who each decide, now and then, what to do next.

The expensive behaviour tier here is an LLM: every decision can be made by prompting a local
language model with the person's situation, or by a cheap surrogate. Decisions are discrete
(six actions), so a policy is a distribution over actions and the divergence between two
policies at one decision is an exact KL. By the chain rule the KL between two trajectory laws
is the sum of those per-decision KLs along the trajectory -- the same accounting the ledger
already does for the latent jump process, one decision at a time instead of one frame.

Context (what the LLM is told): persona, minutes to the person's train, whether they hold a
ticket, how long the ticket queue is, and where they are. Discretised so the LLM's exact
answer can be tabulated for every context as ground truth for MEASURING divergence; the
scheduler never reads that table (see llm/schedule.py). People interact through the queue:
everyone who decides to queue lengthens it for everyone else, and a long queue changes what
the next person decides.

One step is one second.
"""
import numpy as np

PERSONAS = (
    "a hurried daily commuter who knows the station well",
    "a tourist who has never been to this station",
    "a student travelling on a tight budget",
    "an anxious elderly traveller who walks slowly",
    "a parent travelling with a small child",
    "a business traveller on a phone call",
)
# how often each persona arrives, and how likely each already holds a ticket or pass
PERSONA_W = np.array([0.30, 0.15, 0.20, 0.10, 0.10, 0.15])
TICKET_P = np.array([0.90, 0.15, 0.50, 0.40, 0.40, 0.80])

ACTIONS = ("join the ticket queue", "go to the platform gates", "ask at the information desk",
           "sit on a bench and wait", "buy a coffee or a snack", "leave the station")
A_QUEUE, A_GATES, A_INFO, A_WAIT, A_FOOD, A_LEAVE = range(6)

ZONES = ("the station entrance", "the main concourse", "the ticket queue area", "the platform gates")
Z_ENTRANCE, Z_CONCOURSE, Z_QUEUE, Z_GATES = range(4)

T_EDGES = (3, 10, 20)            # minutes to train: <3, 3-10, 10-20, >20
T_WORDS = ("less than 3 minutes", "about 6 minutes", "about 15 minutes", "more than 20 minutes")
Q_EDGES = (4, 12)                # people in the queue: short, medium, long
Q_HYST = 2                       # nobody perceives a queue of 11 vs 12; the bucket moves 2 past an edge
Q_WORDS = ("short (about a minute)", "moderate (about five minutes)", "very long (over ten minutes)")

N_P, N_T, N_K, N_Q, N_Z = len(PERSONAS), 4, 2, 3, len(ZONES)
N_CTX = N_P * N_T * N_K * N_Q * N_Z
N_ACT = len(ACTIONS)

WALK = {A_QUEUE: 15, A_GATES: 30, A_INFO: 20, A_WAIT: 0, A_FOOD: 20, A_LEAVE: 20}
DWELL = {A_INFO: 60, A_WAIT: 120, A_FOOD: 150}
SERVICE_S = 25                   # the ticket window serves one person every 25 s


def ctx_index(p, t, k, q, z):
    return (((np.asarray(p) * N_T + t) * N_K + k) * N_Q + q) * N_Z + z


def ctx_parts(c):
    c = np.asarray(c)
    z = c % N_Z; c = c // N_Z
    q = c % N_Q; c = c // N_Q
    k = c % N_K; c = c // N_K
    t = c % N_T
    return c // N_T, t, k, q, z


def t_bucket(minutes):
    return np.searchsorted(T_EDGES, np.asarray(minutes), side="right")


def q_bucket(people):
    return np.searchsorted(Q_EDGES, np.asarray(people), side="right")


class Station:
    """Vectorised state for n people. An agent with busy_until <= step needs a decision."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.persona = np.zeros(n, np.int64)
        self.zone = np.zeros(n, np.int64)
        self.ticket = np.zeros(n, bool)
        self.train_at = np.zeros(n, np.int64)
        self.busy_until = np.zeros(n, np.int64)
        self.activity = np.full(n, -1, np.int64)
        self.queue = []                        # agent ids, FIFO
        self.qb = 0                            # the queue length as people perceive it (bucket)
        self.gen = np.zeros(n, np.int64)       # bumped when a new person takes the slot
        self.next_service = SERVICE_S
        self.boarded = self.missed = self.left = self.turned_away = 0
        self.respawn(np.arange(n), 0, stagger=True)

    def respawn(self, idx, step, stagger=False):
        idx = np.atleast_1d(idx)
        if idx.size == 0:
            return
        r = self.rng
        self.gen[idx] += 1
        self.persona[idx] = r.choice(N_P, idx.size, p=PERSONA_W)
        self.ticket[idx] = r.random(idx.size) < TICKET_P[self.persona[idx]]
        self.zone[idx] = Z_ENTRANCE
        self.train_at[idx] = step + r.integers(4 * 60, 40 * 60, idx.size)
        self.activity[idx] = -1
        self.busy_until[idx] = step + (r.integers(0, 120, idx.size) if stagger else 0)

    def minutes_to_train(self, step, idx=None):
        idx = np.arange(self.n) if idx is None else idx
        return np.maximum(self.train_at[idx] - step, 0) / 60.0

    def context(self, step, idx=None, zone=None, ticket=None, at_step=None):
        """Context index, optionally predicted: the zone and ticket the agent WILL have and the
        step it will decide at. The queue length is the current one -- the part a prediction
        can get wrong."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        s = step if at_step is None else at_step
        t = t_bucket(self.minutes_to_train(np.asarray(s), idx))
        k = np.asarray(self.ticket[idx] if ticket is None else ticket).astype(np.int64)
        z = np.asarray(self.zone[idx] if zone is None else zone)
        q = np.full(idx.size, self.qb)
        return ctx_index(self.persona[idx], t, k, q, z)

    def next_state(self, idx):
        """(zone, ticket) each busy agent will be in when it next decides."""
        idx = np.atleast_1d(idx)
        zone, ticket = self.zone[idx].copy(), self.ticket[idx].copy()
        act = self.activity[idx]
        zone[act == A_QUEUE] = Z_CONCOURSE
        ticket[act == A_QUEUE] = True
        zone[np.isin(act, (A_INFO, A_WAIT, A_FOOD))] = Z_CONCOURSE
        return zone, ticket

    def next_decision_step(self, step, idx):
        """When each agent will next decide; the queue's is an estimate from its position."""
        idx = np.atleast_1d(idx)
        out = self.busy_until[idx].copy()
        pos = {a: i for i, a in enumerate(self.queue)}
        for j, a in enumerate(idx):
            if a in pos:
                out[j] = self.next_service + pos[a] * SERVICE_S
        return out

    def apply(self, idx, actions, step):
        """Carry out one decision per agent."""
        for i, a in zip(np.atleast_1d(idx), np.atleast_1d(actions)):
            self.activity[i] = a
            if a == A_QUEUE:
                self.zone[i] = Z_QUEUE
                self.queue.append(int(i))
                self.busy_until[i] = 10 ** 12          # until served
            elif a == A_GATES:
                if self.ticket[i]:
                    self.zone[i] = Z_GATES
                    self.busy_until[i] = max(step + WALK[a], self.train_at[i])   # board at the train
                else:
                    self.turned_away += 1
                    self.zone[i] = Z_CONCOURSE
                    self.busy_until[i] = step + 2 * WALK[a]
                    self.activity[i] = -1
            elif a == A_LEAVE:
                self.left += 1
                self.respawn(i, step + WALK[a])
            else:
                self.zone[i] = Z_CONCOURSE
                self.busy_until[i] = step + WALK[a] + DWELL[a]

    def perceive_queue(self):
        L = len(self.queue)
        while self.qb < len(Q_EDGES) and L >= Q_EDGES[self.qb] + Q_HYST:
            self.qb += 1
        while self.qb > 0 and L < Q_EDGES[self.qb - 1] - Q_HYST:
            self.qb -= 1

    def tick(self, step):
        """Serve the queue, board trains, and roll missed trains forward."""
        self.perceive_queue()
        if step >= self.next_service:
            self.next_service = step + SERVICE_S
            if self.queue:
                i = self.queue.pop(0)
                self.ticket[i] = True
                self.zone[i] = Z_CONCOURSE
                self.busy_until[i] = step
                self.activity[i] = -1
        at_gate = np.flatnonzero((self.activity == A_GATES) & (self.zone == Z_GATES) & (self.busy_until <= step))
        if at_gate.size:
            board = at_gate[self.train_at[at_gate] <= step + 1]
            self.boarded += board.size
            self.respawn(board, step)
        late = np.flatnonzero(self.train_at < step - 60)
        late = late[~np.isin(late, self.queue)]
        if late.size:
            self.missed += late.size
            self.train_at[late] = step + 20 * 60      # the next one, twenty minutes later
