import argparse
import time

import numpy as np

from expert import demo

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=300)
p.add_argument("--out", default="data/demos.npz")
args = p.parse_args()

S, A, ep_end, n_ok = [], [], [], 0
t0 = time.time()
for seed in range(args.n):        # eval uses seeds >= 10000
    s, a, cov = demo(seed)
    if cov[-1] < 0.95:
        continue                  # only keep successful demos
    n_ok += 1
    S.append(s); A.append(a)
    ep_end.append(sum(len(x) for x in S))
    if seed % 50 == 0:
        print(seed, f"{time.time() - t0:.0f}s")

np.savez_compressed(args.out, states=np.concatenate(S), actions=np.concatenate(A), ep_end=np.array(ep_end))
lens = np.diff([0] + ep_end)
print(f"{n_ok}/{args.n} successful, {ep_end[-1]} steps, episode length {lens.mean():.0f} +- {lens.std():.0f}")
