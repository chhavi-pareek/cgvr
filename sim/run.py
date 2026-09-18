import argparse

import numpy as np

from bench.log import Logger

from .assign_threshold import ThresholdAssigner
from .camera import CameraPath
from .scenes import SCENES
from .state import Agents

DT = 1.0 / 30.0
SEP_R, SEP_K, CHUNK = 1.0, 1.5, 512


def _separation(pos, r):
    n = len(pos)
    out = np.zeros_like(pos)
    idx = np.arange(n)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        diff = pos[s:e, None, :] - pos[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        m = (d2 < r * r) & (idx[s:e, None] != idx[None, :])
        d = np.sqrt(d2)
        w = np.where(m, (1 - d / r) / np.maximum(d, 1e-6), 0)
        out[s:e] = np.einsum("ij,ijk->ik", w, diff)
    return out


def _step(a, sc, frame, rng):
    prev = a.pos.copy()
    d = a.goal - a.pos
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    desired = d / np.maximum(dist, 1e-6)[:, None] * a.speed[:, None]
    a.vel += 0.2 * (desired + SEP_K * _separation(a.pos, SEP_R) - a.vel)
    sp = np.sqrt(np.einsum("ij,ij->i", a.vel, a.vel))
    a.vel *= np.minimum(1.0, 1.5 * a.speed / np.maximum(sp, 1e-6))[:, None]
    a.pos += a.vel * DT
    sc.constrain(a, prev)
    sc.after_step(a, frame, rng)


def run(scene, agents, condition="baseline", seed=0, frames=300, out=None):
    rng = np.random.default_rng(seed)
    sc, a = SCENES[scene](), Agents(agents)
    sc.spawn(a, rng)
    cam, asg = CameraPath(scene, frames), ThresholdAssigner()
    out = out or f"bench/logs/{scene}_{agents}_{condition}_{seed}.csv"
    with Logger(out, scene, agents, condition) as log:
        for f in range(frames):
            cp, yaw = cam.at(f)
            asg.step(a, cp, yaw)
            log.row(f, a)
            _step(a, sc, f, rng)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=list(SCENES), default="plaza")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--condition", choices=["baseline"], default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--out")
    args = p.parse_args()
    print(run(args.scene, args.agents, args.condition, args.seed, args.frames, args.out))


if __name__ == "__main__":
    main()
