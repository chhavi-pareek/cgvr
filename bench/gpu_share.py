"""Shared GPU, step 1: how LLM inference and crowd rendering interfere on one machine.

    python -m bench.gpu_share                      # rates 0, 0.5, 1 calls/s and back-to-back
    python -m bench.gpu_share --joint              # step 2, from the logs of the runs above
    python -m bench.gpu_share --live               # step 3: the station live beside the renderer

The crowd renders in the standalone bench (its own process, as a game would be) while this
process asks the local LLM (qwen2.5:7b through Ollama) the station's questions at a fixed rate --
Poisson arrivals, or back to back ("sat"). Frame times come from the bench's CSV; call latencies
from here. MassLOD at fixed caps does fixed rendering work, so its frame-time rise is the
interference itself; knapsack and PARITY hold a target, so what they give up in image quality is
what sharing costs a controlled renderer. The LLM is also timed with nothing rendering.

Needs Ollama running and the Mac bench build (tools/build_bench.sh mac). Long: ~4 min per rate
plus cooldowns, since the machine throttles under sustained load.
"""
import argparse
import csv
import os
import subprocess
import threading
import time

import numpy as np

from llm.policy import ask
from llm.station import N_CTX

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.environ.get("PARITY_BENCH_APP", "/Users/user/UnityProjects/Parity3D/Builds/mac/ParityBench.app/Contents/MacOS/ParityBench")
OUT = os.path.join(HERE, "bench", "logs", "gpu_share")


class Load(threading.Thread):
    """Asks the model at `rate` calls per second (Poisson; None = back to back) until stopped."""

    def __init__(self, rate, seed=0):
        super().__init__(daemon=True)
        self.rate, self.rng = rate, np.random.default_rng(seed)
        self.stop = threading.Event()
        self.calls = []                      # (start, end) wall-clock seconds

    def run(self):
        nxt = time.perf_counter()
        while not self.stop.is_set():
            if self.rate:
                nxt += float(self.rng.exponential(1.0 / self.rate))
                wait = nxt - time.perf_counter()
                if wait > 0 and self.stop.wait(wait):
                    break
            t0 = time.perf_counter()
            ask(int(self.rng.integers(N_CTX)))
            self.calls.append((t0, time.perf_counter()))


def bench(tag, n, targets, policies, seed):
    os.makedirs(OUT, exist_ok=True)
    args = [BUILD, "-batchmode", "-parityBench", "-overlap", "-sizes", str(n), "-targets", targets,
            "-policies", policies, "-seed", str(seed), "-out", OUT, "-tag", tag,
            "-logFile", os.path.join(OUT, f"{tag}.log")]
    t0 = time.perf_counter()
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(os.path.join(OUT, f"frame_bench_{tag}.csv"))))
    return rows, t0, time.perf_counter()


def duty_curve():
    """(duty, frame mean, frame p95) per measured LLM rate: duty = calls served per second x their
    median latency, the share of time the model held the GPU; frames from the fixed-work MassLOD
    cells, whose rendering does not change with load."""
    import re
    pts = [(0.0, None)]
    for log in ("gpu_share.log", "gpu_share_low.log"):
        path = os.path.join(HERE, "bench", "logs", log)
        if not os.path.exists(path):
            continue
        for m in re.finditer(r"LLM\s+(\S+) calls/s asked -> ([0-9.]+) served during the bench, latency median (\d+) ms", open(path).read()):
            pts.append((float(m.group(2)) * float(m.group(3)) / 1000.0, m.group(1)))
    out = []
    for duty, r in pts:
        rows = list(csv.DictReader(open(os.path.join(OUT, f"frame_bench_share_{r if r else '0'}.csv"))))
        mass = [x for x in rows if x["policy"] == "masslod"]
        out.append((duty, float(np.mean([float(x["ft_mean"]) for x in mass])), float(np.mean([float(x["ft_p95"]) for x in mass]))))
    return sorted(set(out))


def joint(a):
    """Step 2: the LLM's GPU duty is the budget the frame rate pays for. For each duty, what every
    scheduler achieves for the crowd's drift, and what the frames cost; and the duty PARITY's
    ledger actually needs when it asks only for the calls the cap requires."""
    from llm.policy import build_table
    from llm.schedule import Run
    curve = duty_curve()
    print("duty (share of time the LLM holds the GPU) -> crowd frames (fixed-work cells):")
    for d, fm, fp in curve:
        print(f"  {d:5.2f}: mean {fm:6.1f} ms  p95 {fp:6.1f} ms")
    frame = lambda d: (float(np.interp(d, [c[0] for c in curve], [c[1] for c in curve])),  # noqa: E731
                       float(np.interp(d, [c[0] for c in curve], [c[2] for c in curve])))
    P, lat = build_table()
    lat = lat * (a.shared_latency / float(np.median(lat)))      # calls take this long beside the renderer
    for n in a.agents:
        print(f"\nN = {n}, calls {a.shared_latency:.2f} s beside the renderer; {a.seconds // 60} simulated minutes, seeds {a.seeds}")
        print("  policy        duty budget  duty used  frame mean/p95   worst/cap  overflow  steady drift  waits/agent-h")
        for util in a.duties:
            for pol in ("parity", "view_lod", "round_robin"):
                if util == 0 and pol != "parity":
                    continue
                rs = [Run(n, pol, P, lat, seed=s, cap=a.cap, util=max(util, 1e-6)).run(a.seconds) for s in a.seeds]
                used = float(np.mean([r.server.busy_s / a.seconds for r in rs]))
                ss = [r.summary() for r in rs]
                m = lambda k: float(np.mean([x[k] for x in ss]))  # noqa: E731
                fm, fp = frame(used)
                name = "parity (ledger only)" if util == 0 else pol
                print(f"  {name:20s} {util:5.2f} {used:10.3f}   {fm:5.1f} / {fp:5.1f}   {max(x['dmax'] for x in ss) / a.cap:6.2f}x"
                      f" {100 * m('overflow_frac'):7.2f}% {m('kl_steady_per_agent_h'):12.2f} {m('stall_per_agent_h'):13.1f}")


def live(a):
    """Step 3: one live system. The station runs in real time, its scheduler making real calls to
    the model (llm/live.py), while the crowd renders a fixed-work cell beside it for `window`
    seconds. Conditions: no LLM; PARITY asking only for the calls the cap requires; round-robin
    given exactly the duty PARITY used; round-robin saturated."""
    from llm.live import LiveServer
    from llm.policy import build_table
    from llm.schedule import Run
    P, lat = build_table()
    lat = lat * (a.shared_latency / float(np.median(lat)))
    ask(0)
    conds = [("no LLM", None, 0.0), ("parity (ledger only)", "parity", 1e-6),
             ("round_robin, PARITY's duty", "round_robin", None), ("round_robin, saturated", "round_robin", 0.9)]
    parity_duty = None
    print(f"live: {a.live_agents} agents deciding in real time, crowd N={a.agents[0]} at MassLOD caps x{a.live_caps} "
          f"for {a.window} s per condition", flush=True)
    for name, pol, util in conds:
        time.sleep(a.cool)
        if util is None:
            util = parity_duty
        tag = "live_" + name.split(" ")[0].replace(",", "") + ("" if util in (0.0, 1e-6) else f"_{util:.2f}")
        logf = os.path.join(OUT, f"{tag}.log")
        if os.path.exists(logf):
            os.remove(logf)
        proc = subprocess.Popen([BUILD, "-batchmode", "-parityBench", "-overlap", "-sizes", str(a.agents[0]), "-seed", "1",
                                 "-policies", "masslod", "-knobs", str(a.live_caps), "-seconds", str(a.window),
                                 "-out", OUT, "-tag", tag, "-logFile", logf],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.perf_counter()
        run = None
        if pol:
            run = Run(a.live_agents, pol, P, lat, seed=0, cap=a.cap, util=util)
            run.server = LiveServer(P, lat, t0)
        cell_start, t = None, 0
        while proc.poll() is None:
            if cell_start is None and os.path.exists(logf) and "cell order" in open(logf, errors="ignore").read():
                cell_start = time.perf_counter() - t0 + 2.0          # after the warm-up frames
            if run is not None:
                wait = t0 + t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                run.step(t)
                t += 1
            else:
                time.sleep(1.0)
        row = list(csv.DictReader(open(os.path.join(OUT, f"frame_bench_{tag}.csv"))))[0]
        line = f"  {name:28s} frames mean {float(row['ft_mean']):6.1f} p95 {float(row['ft_p95']):6.1f} ms (gpu wait {float(row['gpu_wait_ms']):5.1f})"
        if run is not None:
            run.server.stop()
            s = run.summary()
            duty = run.server.duty(cell_start or 0.0, (cell_start or 0.0) + a.window)
            overall = run.server.busy_s / max(t, 1)
            if pol == "parity":
                parity_duty = overall
            line += (f" | {t} s live, {run.server.calls} calls, duty {overall:.3f} (in the frame window {duty:.3f}), "
                     f"worst/cap {s['dmax'] / a.cap:.2f}x, overflow {100 * s['overflow_frac']:.2f}%, "
                     f"steady drift {s['kl_steady_per_agent_h']:.2f}, waits/agent-h {s['stall_per_agent_h']:.1f}, "
                     f"live replies off the table {run.server.mismatch}")
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--live-agents", type=int, default=300)
    ap.add_argument("--live-caps", default="2")
    ap.add_argument("--window", type=int, default=480)
    ap.add_argument("--duties", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.5, 0.9])
    ap.add_argument("--shared-latency", type=float, default=1.85)
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--rates", nargs="+", default=["0", "0.5", "1", "sat"])
    ap.add_argument("--agents", type=int, nargs="+", default=[8000])
    ap.add_argument("--targets", default="20")
    ap.add_argument("--policies", default="masslod,knapsack_fullsim,parity")
    ap.add_argument("--cool", type=int, default=120, help="seconds between conditions")
    a = ap.parse_args()
    if a.joint:
        joint(a)
        return
    if a.live:
        live(a)
        return
    a.agents = a.agents[0]

    ask(0)                                   # load the model into memory before anything is timed
    alone = Load(None)
    alone.start(); time.sleep(60); alone.stop.set(); alone.join()
    lat0 = np.array([e - s for s, e in alone.calls])
    print(f"LLM alone: {len(lat0)} calls in 60 s, latency median {1e3 * np.median(lat0):.0f} ms, "
          f"p95 {1e3 * np.quantile(lat0, 0.95):.0f} ms -> saturates at {len(lat0) / 60:.2f} calls/s")

    summary = []
    for r in a.rates:
        time.sleep(a.cool)
        rate = None if r == "sat" else float(r)
        load = Load(rate, seed=1) if rate != 0 else None
        if load:
            load.start()
        rows, t0, t1 = bench(f"share_{r}", a.agents, a.targets, a.policies, seed=1)
        if load:
            load.stop.set(); load.join()
        lat = np.array([e - s for s, e in (load.calls if load else []) if s >= t0 and e <= t1])
        got = len(lat) / max(t1 - t0, 1e-9)
        print(f"\nLLM {r:>4} calls/s asked -> {got:.2f} served during the bench"
              + (f", latency median {1e3 * np.median(lat):.0f} ms p95 {1e3 * np.quantile(lat, 0.95):.0f} ms" if len(lat) else ""))
        for row in rows:
            f = lambda k: float(row[k])  # noqa: E731
            print(f"  {row['policy']:17s} {f('knob'):5g}: frame mean {f('ft_mean'):6.2f} p95 {f('ft_p95'):6.2f} ms "
                  f"(cpu {f('cpu_ms'):5.2f}, gpu wait {f('gpu_wait_ms'):5.2f})  image {f('quality_image'):.3f}  "
                  f"state {f('quality_state'):.3f}  thermo {f('thermo_ms'):.1f}")
            summary.append(dict(rate=r, served=got, lat_ms=1e3 * np.median(lat) if len(lat) else float("nan"), **row))
    with open(os.path.join(OUT, f"gpu_share_{'_'.join(a.rates)}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nwrote {os.path.join(OUT, f'gpu_share_{chr(95).join(a.rates)}.csv')}")


if __name__ == "__main__":
    main()
