"""Side-by-side clip: same start, same pushes from outside, two replanning strategies.

The seed in the README (30033) is hand-picked: I ran seeds 30000-30059 (not test seeds) and took one
where only the right one solves it and there are few pushes, so the clip stays short."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run from the repo root

import argparse
import json

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from policy import Policy
from pusht import Sim, keypoints
from rollout import Strategy, run
from worldmodel import WorldModel

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=10003)
p.add_argument("--kicks", default="large")
p.add_argument("--out", default="media/demo.mp4")
p.add_argument("--fps", type=int, default=12)
p.add_argument("--seconds", type=float, default=17)   # cut the clip here
p.add_argument("--card", action="store_true")          # numbers at the end
args = p.parse_args()

thr = json.load(open("eval/results/benchmark.json"))["thr"]["surprise"]
policy = Policy("runs/policy/ema.pt")
wm = WorldModel()
wm.load_state_dict(torch.load("runs/wm/model.pt"))

runs = []
for st, title in [(Strategy(16), "replan every 16 steps"),
                  (Strategy(16, "surprise", thr), "+ replan on surprise")]:
    # same sampling noise for both, so they do exactly the same thing until the first trigger
    policy.gen = torch.Generator().manual_seed(args.seed)
    e = run(policy, st, [args.seed], wm=wm, kicks=args.kicks, keep_log=True)[0]
    e.log["state"].append(e.sim.state())
    runs.append((e, title))
    print(title, "success step", e.success_step)

S = 448
F = lambda n: ImageFont.load_default(size=n)
RED, ORANGE, GRAY = (220, 50, 50), (245, 140, 30), (120, 120, 120)
sim = Sim()
T = min(max(len(e.log["action"]) for e, _ in runs) + 1, int(args.seconds * 10))


def panel(e, title, i):
    n = len(e.log["action"])
    j = min(i, n)                       # after the end, freeze on the last frame
    sim.set_state(e.log["state"][j])
    img = Image.fromarray(sim.frame(512)).resize((S, S), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    sc = S / 512
    # block pose before the last push from outside, for a few frames
    for back in range(0, 8):
        k = j - 1 - back
        if 0 <= k < n and e.log["kick"][k] is not None:
            ghost = keypoints(e.log["kick"][k]) * sc
            for poly in (ghost[:4], ghost[4:]):
                d.polygon([tuple(q) for q in poly], outline=RED, width=2)
            break
    # what is left of the current plan
    if 0 < j <= n:
        q = e.log["queue"][j - 1]
        if len(q):
            pts = [tuple(x * sc) for x in q]
            if len(pts) > 1:
                d.line(pts, fill=ORANGE, width=2)
            for x in pts:
                d.ellipse([x[0] - 3, x[1] - 3, x[0] + 3, x[1] + 3], fill=ORANGE)
    # trigger
    for back in range(0, 4):
        k = j - 1 - back
        if 0 <= k < n and e.log["replan"][k]:
            d.rectangle([0, 0, S - 1, S - 1], outline=RED, width=6)
            break
    d.text((10, 10), title, font=F(20), fill=(30, 30, 30))
    if e.success_step > 0 and i >= e.success_step:
        d.text((10, S - 34), f"solved, {e.success_step / 10:.1f} s", font=F(20), fill=(30, 150, 60))
    else:
        cov = e.log["cov"][j - 1] if j > 0 else 0
        d.text((10, S - 34), f"{j / 10:.1f} s   {cov * 100:.0f}%", font=F(18), fill=GRAY)
    return np.array(img)


def trace(e, i, used):
    W, H = S, 130
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    s = np.minimum(np.array(e.log["surprise"], float) / thr, 3)     # in units of the threshold
    x = lambda t: 10 + t / 300 * (W - 20)
    y = lambda v: H - 26 - v / 3 * (H - 40)
    d.line([(10, y(1)), (W - 10, y(1))], fill=(200, 200, 200), width=1)
    for t in range(min(i, len(s))):
        if e.log["kick"][t] is not None:
            d.line([(x(t), 8), (x(t), H - 26)], fill=(240, 170, 170), width=2)
    pts = [(x(t), y(v)) for t, v in enumerate(s[:i])]
    if len(pts) > 1:
        d.line(pts, fill=(60, 110, 200) if used else (170, 170, 170), width=2)
    if used:
        for t in range(min(i, len(s))):
            if e.log["replan"][t]:
                d.ellipse([x(t) - 4, y(s[t]) - 4, x(t) + 4, y(s[t]) + 4], fill=RED)
    label = "surprise" + ("" if used else " (not used)")
    d.text((10, H - 20), label, font=F(14), fill=GRAY)
    return np.array(img)


frames = []
for i in range(T):
    cols = []
    for k, (e, title) in enumerate(runs):
        cols.append(np.concatenate([panel(e, title, i), trace(e, i, used=k == 1)], 0))
    gap = np.full((cols[0].shape[0], 12, 3), 255, np.uint8)
    body = np.concatenate([cols[0], gap, cols[1]], 1)
    frames.append(body)

# end card with the numbers over all test episodes
res = json.load(open("eval/results/benchmark.json"))["conditions"][args.kicks]
card = Image.new("RGB", (frames[0].shape[1], frames[0].shape[0]), "white")
d = ImageDraw.Draw(card)
lines = [(f"{args.kicks} pushes, 300 episodes", 26, (30, 30, 30)), ("", 10, None)]
for name, label in [("k=16", "every 16 steps"), ("k=16 + random", "+ random replans"),
                    ("k=16 + surprise", "+ replan on surprise"), ("k=8", "every 8 steps"),
                    ("k=8 + surprise", "+ replan on surprise")]:
    lines.append((f"{100 * res[name]['success']:3.0f}%   {label}", 22, RED if "surprise" in name else (60, 60, 60)))
lines += [("", 10, None), ("success within 30 s, +-7 points", 16, GRAY)]
yy = 120
for text, size, col in lines:
    if text:
        d.text((60, yy), text, font=F(size), fill=col)
    yy += size + 14
if args.card:
    frames += [np.array(card)] * (4 * args.fps)

w = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)
for f in frames:
    w.append_data(f)
w.close()
print("wrote", args.out, len(frames), "frames")
