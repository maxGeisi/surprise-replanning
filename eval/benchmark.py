"""Success rate of every replanning strategy under every kind of disturbance.
Saves after each strategy (eval/results/episodes/<condition>/<strategy>.npz) and skips what is
already there, so it can be stopped and restarted."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run from the repo root

import argparse
import json
import time

import numpy as np
import torch
torch.set_num_threads(2)

from policy import Policy
from pusht import keypoints
from rollout import Strategy, run, summary
from worldmodel import WorldModel

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=300)
p.add_argument("--conditions", nargs="+", default=["clean", "small", "medium", "large"])
p.add_argument("--fa", type=float, default=0.01)    # false trigger rate per step on clean rollouts
args = p.parse_args()

policy = Policy("runs/policy/ema.pt")
wm = WorldModel()
wm.load_state_dict(torch.load("runs/wm/model.pt"))
CAL = np.arange(20000, 20100)            # calibration seeds, different from the test seeds
TEST = np.arange(10000, 10000 + args.n)
OUT = "eval/results/benchmark.json"

if os.path.exists(OUT):
    res = json.load(open(OUT))
    print("resuming", {c: len(v) for c, v in res["conditions"].items()}, flush=True)
else:
    # ---- thresholds from clean rollouts ----
    t0 = time.time()
    eps = run(policy, Strategy(k=8), CAL, wm=wm, keep_log=True)
    sur, jump = [], []
    for e in eps:
        S = np.array(e.log["state"] + [e.sim.state()])
        sur += e.log["surprise"]
        jump += [np.linalg.norm(keypoints(b) - keypoints(a), axis=1).mean() for a, b in zip(S[:-1, 4:7], S[1:, 4:7])]
    thr = {"surprise": float(np.quantile(sur, 1 - args.fa)), "jump": float(np.quantile(jump, 1 - args.fa))}
    print("thresholds", thr, f"({len(sur)} clean steps, {time.time() - t0:.0f}s)", flush=True)
    res = {"thr": thr, "n": args.n, "conditions": {}}
thr = res["thr"]


def go(cond, st, rng_seed=0):
    out = res["conditions"].setdefault(cond, {})
    if st.name in out:
        return out[st.name]
    t0 = time.time()
    eps = run(policy, st, TEST, wm=wm, kicks=None if cond == "clean" else cond, rng_seed=rng_seed)
    out[st.name] = summary(eps)
    cov = np.full((len(eps), 300), np.nan)
    for i, e in enumerate(eps):
        cov[i, :len(e.cov)] = e.cov
    os.makedirs(f"eval/results/episodes/{cond}", exist_ok=True)
    np.savez_compressed(f"eval/results/episodes/{cond}/{st.name}.npz", cov=cov,
                        success_step=np.array([e.success_step for e in eps]), calls=np.array([e.calls for e in eps]))
    json.dump(res, open(OUT, "w"), indent=1)
    print(cond, st.name, {k: round(v, 3) for k, v in out[st.name].items()}, f"{time.time() - t0:.0f}s", flush=True)
    return out[st.name]


for cond in args.conditions:
    # most important comparisons first
    for k in (8, 16):
        go(cond, Strategy(k))
    for k in (16, 8):
        s = go(cond, Strategy(k, "surprise", thr["surprise"]))
        # control: replan at random steps, same rate as the surprise trigger
        go(cond, Strategy(k, "random", s["triggers"] / s["steps"]), rng_seed=1)
        go(cond, Strategy(k, "jump", thr["jump"]))
    for k in (4, 2, 1):
        go(cond, Strategy(k))
