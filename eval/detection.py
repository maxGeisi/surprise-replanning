"""How many of the pushes does each detector notice, at the threshold that gives 1% false
alarms per step on clean rollouts? Policy runs with k=8, nothing is triggered, only measured."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run from the repo root

import json

import numpy as np
import torch
torch.set_num_threads(1)

from policy import Policy
from pusht import keypoints
from rollout import Strategy, run
from worldmodel import WorldModel

thr = json.load(open("eval/results/benchmark.json"))["thr"]
policy = Policy("runs/policy/ema.pt")
wm = WorldModel()
wm.load_state_dict(torch.load("runs/wm/model.pt"))
seeds = np.arange(10000, 10100)

out = {}
for cond in ["small", "medium", "large"]:
    eps = run(policy, Strategy(8), seeds, wm=wm, kicks=cond, keep_log=True)
    hit = {"surprise": [], "jump": []}
    fa = {"surprise": [], "jump": []}
    for e in eps:
        S = np.array(e.log["state"] + [e.sim.state()])
        sur = np.array(e.log["surprise"])
        jump = np.array([np.linalg.norm(keypoints(b) - keypoints(a), axis=1).mean() for a, b in zip(S[:-1, 4:7], S[1:, 4:7])])
        pushed = np.array([k is not None for k in e.log["kick"]])
        for name, x in [("surprise", sur), ("jump", jump)]:
            hit[name] += list(x[pushed] > thr[name])
            fa[name] += list(x[~pushed] > thr[name])
    out[cond] = {n: {"detected": float(np.mean(hit[n])), "false_alarms": float(np.mean(fa[n])), "pushes": len(hit[n])}
                 for n in hit}
    print(cond, out[cond], flush=True)
json.dump(out, open("eval/results/detection.json", "w"), indent=1)
