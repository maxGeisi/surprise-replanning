"""Forward model of the robot + block: (state, action) -> next state, plus a second
head that predicts how large its own error is going to be (log variance per dim).

surprise = squared prediction error / predicted variance  ("precision weighted")
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn


def feat(s):
    """state [..., 7] = (agent xy, agent vel, block xy, block angle) -> [..., 8]
    agent velocity is proprioception, a real arm knows it"""
    s = torch.as_tensor(s, dtype=torch.float32)
    return torch.cat([s[..., 0:2] / 256 - 1, s[..., 2:4] / 500, s[..., 4:6] / 256 - 1,
                      torch.cos(s[..., 6:7]), torch.sin(s[..., 6:7])], -1)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class WorldModel(nn.Module):
    def __init__(self, d=8, adim=2, h=256):
        super().__init__()
        self.pred = mlp(d + adim, h, d)
        self.logvar = mlp(d + adim, 128, d)
        self.register_buffer("scale", torch.ones(d))     # running mean of the squared error

    def forward(self, s, a):
        x = torch.cat([feat(s), torch.as_tensor(a, dtype=torch.float32) / 256 - 1], -1)
        return feat(s) + self.pred(x), self.logvar(x)

    @torch.no_grad()
    def surprise(self, s, a, s_next):
        mu, lv = self(s, a)
        err = (mu - feat(s_next)) ** 2 / self.scale
        return {"raw": err.mean(-1).numpy(), "weighted": (err / lv.exp()).mean(-1).numpy()}


def transitions(path="data/demos.npz"):
    d = np.load(path)
    S, A, ends = d["states"], d["actions"], d["ep_end"]
    # next state is the first state of the next step inside the same episode
    last = np.zeros(len(S), bool)
    last[ends - 1] = True
    i = np.where(~last)[0]
    return S[i], A[i], S[i + 1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--out", default="runs/wm")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    S, A, S1 = transitions()
    n_val = len(S) // 10            # last 10% of the transitions (= last ~30 episodes)
    tr, va = slice(0, -n_val), slice(-n_val, None)
    m = WorldModel()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(0)
    N = len(S) - n_val
    for it in range(args.steps + 1):
        i = rng.integers(0, N, 512)
        mu, lv = m(S[i], A[i])
        e2 = (mu - feat(S1[i])) ** 2
        with torch.no_grad():
            m.scale.mul_(0.99).add_(0.01 * e2.mean(0))
        # NLL on the detached error, so the variance head cannot change the predictor
        nll = 0.5 * (e2.detach() / m.scale / lv.exp() + lv).mean()
        loss = e2.mean() + nll
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0:
            with torch.no_grad():
                mu, _ = m(S[va], A[va])
                v = ((mu - feat(S1[va])) ** 2).mean(0)
            # in pixels for the block position, to have a feeling for it
            print(f"{it} train {e2.mean().item():.2e}  val {v.mean().item():.2e}  "
                  f"val block rmse {v[4:6].mean().sqrt().item() * 256:.2f}px", flush=True)
    torch.save(m.state_dict(), f"{args.out}/model.pt")
