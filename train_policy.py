import argparse
import copy
import os
import time

import numpy as np
import torch
torch.set_num_threads(1)

from policy import Diffusion, obs_feat, act_norm, T_OBS, T_PRED

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--out", default="runs/policy")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)

d = np.load("data/demos.npz")
S, A, ends = d["states"], d["actions"], d["ep_end"]
O = obs_feat(S[:, [0, 1, 4, 5, 6]])
An = act_norm(A)

# all windows. obs (t-1, t), actions t .. t+15. pad with the first obs / last action
# like the original implementation
idx_obs, idx_act = [], []
start = 0
for end in ends:
    for t in range(start, end):
        idx_obs.append([max(t - k, start) for k in reversed(range(T_OBS))])
        idx_act.append([min(t + k, end - 1) for k in range(T_PRED)])
    start = end
idx_obs, idx_act = np.array(idx_obs), np.array(idx_act)
print(len(ends), "episodes", len(idx_obs), "windows")

model = Diffusion()
ema = copy.deepcopy(model)
print(sum(q.numel() for q in model.parameters()) / 1e6, "M params")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.03)

log = open(f"{args.out}/log.csv", "w")
log.write("step,loss,time\n")
t0, run = time.time(), 0
for step in range(1, args.steps + 1):
    i = rng.integers(0, len(idx_obs), args.bs)
    obs = torch.from_numpy(O[idx_obs[i]])
    act = torch.from_numpy(An[idx_act[i]])
    loss = model.loss(obs, act)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    decay = min(0.999, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm, alpha=1 - decay)
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 500 == 0:
        print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        log.write(f"{step},{run:.5f},{time.time() - t0:.0f}\n"); log.flush()
        torch.save(ema.state_dict(), f"{args.out}/ema.pt")
