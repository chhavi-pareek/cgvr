"""Three crowd densities for the ordering study, drawn from the phase 1 simulator.

sparse : plaza, 120 x 120 m, agents wandering (~0.14 / m^2 at N = 2000)
mixed  : hub, 100 x 40 m, one queue plus through traffic (~0.5 / m^2)
dense  : queue variant of the hub, 40 x 20 m, half the crowd in 20 parallel
         queue lanes at 0.4 m spacing (~2.5 / m^2)

Each frame yields positions, velocities, the previous frame's simulation tier
(the field the composite key uses; here from the MassLOD-style assigner, in the
allocated condition from CudaAllocator.allocate) and a behaviour class:
0 queued, 1 walking, 2 arriving, 3 idle.
"""
import numpy as np

from sim.assign_threshold import ThresholdAssigner
from sim.camera import CameraPath
from sim.run import _step
from sim.scenes import Hub, Plaza
from sim.state import Agents

from .key import NCLS


class DenseQueue(Hub):
    name = "dense"
    size = (40.0, 20.0)
    window = (30.0, 4.0)
    spacing = 0.4
    lanes = 20

    def _slot_pos(self, slot):
        s = np.asarray(slot, np.int64)
        lane, depth = s % self.lanes, s // self.lanes
        return np.stack(
            [self.window[0] - 1.0 - depth * self.spacing, self.window[1] + lane * self.spacing], -1
        ).astype(np.float32)

    def spawn(self, a, rng):
        Hub.spawn(self, a, rng)
        q = self.qmax = max(1, a.n // 2)
        a.queuer[:] = False
        a.slot[:] = -1
        a.queuer[:q] = True
        a.slot[:q] = np.arange(q)
        a.pos[:q] = self._slot_pos(a.slot[:q]) + rng.uniform(-0.1, 0.1, (q, 2))
        a.goal[:q] = self._slot_pos(a.slot[:q])
        a.goal[q:] = self._exit(rng, a.n - q)
        a.final[:] = a.goal


DENSITIES = {"sparse": (Plaza, "plaza"), "mixed": (Hub, "hub"), "dense": (DenseQueue, "hub")}


def behaviour_class(a):
    d = a.goal - a.pos
    near = np.einsum("ij,ij->i", d, d) < 25.0
    sp = np.sqrt(np.einsum("ij,ij->i", a.vel, a.vel))
    walking = sp > 0.5 * a.speed
    cls = np.full(a.n, 3, np.int8)
    cls[walking] = 1
    cls[near & ~walking] = 2
    cls[a.slot >= 0] = 0
    assert cls.max() < NCLS
    return cls


class Frame:
    __slots__ = ("pos", "vel", "tier", "cls", "size", "density", "n", "index")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def tile(fr, k, seed=0):
    """k x k copies of a frame at the same density, agents shuffled so the id
    tie-break of the key carries no spatial information. N = k^2 * n."""
    if k <= 1:
        return fr
    w, h = fr.size
    off = np.array([(i * w, j * h) for i in range(k) for j in range(k)], np.float32)
    pos = (fr.pos[None] + off[:, None]).reshape(-1, 2)
    rep = lambda a: np.tile(a, (k * k,) + (1,) * (a.ndim - 1))
    p = np.random.default_rng(seed).permutation(len(pos))
    return Frame(pos=pos[p], vel=rep(fr.vel)[p], tier=rep(fr.tier)[p], cls=rep(fr.cls)[p],
                 size=(w * k, h * k), density=fr.density, n=len(pos), index=fr.index)


def frames(density, n, seed=0, warm=90, count=1, stride=30):
    """Yield `count` snapshot frames after `warm` warm-up frames, `stride` frames apart."""
    scene_cls, cam_name = DENSITIES[density]
    rng = np.random.default_rng(seed)
    sc, a = scene_cls(), Agents(n)
    sc.spawn(a, rng)
    total = warm + stride * count
    cam, asg = CameraPath(cam_name, total), ThresholdAssigner()
    scale = np.asarray(sc.size, np.float32) / np.asarray(Hub.size if cam_name == "hub" else sc.size, np.float32)
    tier = np.zeros(n, np.int8)
    for f in range(total):
        cp, yaw = cam.at(f)
        asg.step(a, cp * scale, yaw)
        if f >= warm and (f - warm) % stride == 0:
            yield Frame(pos=a.pos.copy(), vel=a.vel.copy(), tier=tier.copy(), cls=behaviour_class(a),
                        size=sc.size, density=density, n=n, index=f)
        tier = a.sim_tier.copy()  # previous-frame tier feeds the next key (invariant 5)
        _step(a, sc, f, rng)
