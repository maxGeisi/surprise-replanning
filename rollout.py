"""Run the policy in many environments at once (batched policy calls).

Replanning strategies:
  every k steps                 execute k actions of each 16-step plan, then plan again
  every k steps + trigger       ... and also plan again as soon as the trigger fires
     trigger = "surprise"       1-step world model error above a threshold
     trigger = "jump"           block moved more than a threshold in one step (no model)
     trigger = "random"         fires with a fixed probability per step (control)
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from pusht import Sim, keypoints
from policy import T_OBS


@dataclass
class Strategy:
    k: int = 8
    trigger: str = None       # None, "surprise", "jump", "random"
    thr: float = np.inf       # threshold, or probability for "random"
    fix_hist: bool = True     # after a trigger, overwrite the old block pose in the observation history

    @property
    def name(self):
        return f"k={self.k}" + (f" + {self.trigger}" if self.trigger else "") + (" (no hist fix)" if self.trigger and not self.fix_hist else "")


KICKS = {
    # (distance px, rotation deg). for comparison: while being pushed normally the block
    # moves up to ~17 px per step (mean keypoint displacement)
    "small": ((4, 10), (2, 6)),
    "medium": ((10, 20), (5, 12)),
    "large": ((20, 40), (10, 30)),
}


def kick_schedule(seed, size="large", max_steps=300, gap=(40, 80), first=(20, 60)):
    """when and how hard the block gets pushed by 'somebody'. depends only on the seed,
    so every strategy sees exactly the same disturbances at the same time steps"""
    dist, dangle = KICKS[size]
    rng = np.random.default_rng(seed + 777)
    t, out = rng.integers(*first), {}
    while t < max_steps:
        out[int(t)] = (rng.uniform(*dist), np.radians(rng.uniform(*dangle)))
        t += rng.integers(*gap)
    return out


@dataclass
class Episode:
    seed: int
    sim: Sim = None
    hist: list = None
    queue: list = field(default_factory=list)
    done: bool = False
    success_step: int = -1
    max_cov: float = 0.0
    calls: int = 0
    triggers: int = 0
    kicks: dict = None
    cov: list = field(default_factory=list)
    log: dict = field(default_factory=lambda: {"state": [], "action": [], "plan": [], "queue": [], "kick": [],
                                               "surprise": [], "replan": [], "cov": []})


def run(policy, strategy, seeds, wm=None, kicks=None, max_steps=300, keep_log=False, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    eps = []
    for sd in seeds:
        e = Episode(seed=int(sd), sim=Sim())
        o = e.sim.reset(sd)
        e.hist = [o] * T_OBS
        e.kicks = kick_schedule(sd, kicks, max_steps) if kicks else {}
        e.kick_rng = np.random.default_rng(sd + 999)
        eps.append(e)

    for t in range(max_steps):
        live = [e for e in eps if not e.done]
        if not live:
            break
        need = [e for e in live if not e.queue]
        if need:
            plans = policy(np.stack([np.stack(e.hist[-T_OBS:]) for e in need]))
            for e, p in zip(need, plans):
                e.queue = list(p[:strategy.k])
                e.calls += 1
                if keep_log:
                    e.log["plan"].append((t, p))
        S, A, S1, pre = [], [], [], []
        for e in live:
            S.append(e.sim.state())
            A.append(e.queue.pop(0))
            e.sim.step(A[-1])
            pre.append(None)
            if t in e.kicks:
                pre[-1] = e.sim.block_pose()
                e.sim.kick(e.kick_rng, *e.kicks[t])
            S1.append(e.sim.state())
        S, A, S1 = np.array(S), np.array(A), np.array(S1)
        sur = wm.surprise(S, A, S1)["weighted"] if wm is not None else np.full(len(live), np.nan)
        jump = np.linalg.norm(keypoints_batch(S1[:, 4:7]) - keypoints_batch(S[:, 4:7]), axis=-1).mean(-1)

        for i, e in enumerate(live):
            o = e.sim.obs()
            e.hist.append(o)
            cov = e.sim.coverage()
            e.max_cov = max(e.max_cov, cov)
            e.cov.append(cov)
            if strategy.trigger == "surprise":
                fire = sur[i] > strategy.thr
            elif strategy.trigger == "jump":
                fire = jump[i] > strategy.thr
            elif strategy.trigger == "random":
                fire = rng.random() < strategy.thr
            else:
                fire = False
            if fire and e.queue:
                e.queue = []
                e.triggers += 1
                if strategy.fix_hist:
                    # the 2-step history straddles the jump: looks like the block flew 30 px in
                    # one step, never happens in the demos. pretend it was always where it is now
                    prev = e.hist[-2].copy()
                    prev[2:5] = o[2:5]
                    e.hist[-2] = prev
            if keep_log:
                for k, v in [("state", S[i]), ("action", A[i]), ("surprise", sur[i]), ("replan", fire), ("cov", cov),
                             ("queue", np.array(e.queue)), ("kick", pre[i])]:
                    e.log[k].append(v)
            if cov > 0.95:
                e.done, e.success_step = True, t + 1
    return eps


def keypoints_batch(poses):
    return np.stack([keypoints(p) for p in poses])


def summary(eps, max_steps=300):
    succ = np.array([e.success_step > 0 for e in eps])
    steps = np.array([e.success_step if e.success_step > 0 else max_steps for e in eps])
    return dict(
        success=float(succ.mean()),
        max_cov=float(np.mean([e.max_cov for e in eps])),
        steps=float(steps.mean()),
        calls=float(np.mean([e.calls for e in eps])),
        calls_per_100=float(100 * np.sum([e.calls for e in eps]) / steps.sum()),
        triggers=float(np.mean([e.triggers for e in eps])),
    )
