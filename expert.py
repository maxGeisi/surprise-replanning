"""Scripted expert for PushT, only used to generate demonstrations.

It cheats: it has a copy of the simulator. Every few steps it tries all push
primitives (contact point on the T x push length) in the copy, takes the one that
brings the T closest to the goal, drives around the block to that contact point
and pushes. Greedy, no lookahead.
"""
import numpy as np

from pusht import Sim, keypoints, kp_error, point_poly_dist, rot, AGENT_R


def contact_candidates():
    """(point, outward normal) in block coordinates"""
    c = []
    for x in range(-55, 56, 10):                                   # top of the bar
        c.append(((x, 0), (0, -1)))
    for sx in (-1, 1):
        for y in (8, 15, 22):                                      # ends of the bar
            c.append(((60 * sx, y), (sx, 0)))
        for x in (36, 46, 56):                                     # underside of the bar
            c.append(((x * sx, 30), (0, 1)))
        for y in range(50, 116, 10):                               # sides of the stem
            c.append(((15 * sx, y), (sx, 0)))
    for x in (-10, 0, 10):                                         # end of the stem
        c.append(((x, 120), (0, 1)))
    return [(np.array(p, float), np.array(n, float)) for p, n in c]


CONTACTS = contact_candidates()
LENGTHS = (10, 25, 50, 90)
FINE = (4, 5, 6, 7, 9, 12, 16)      # used close to the goal. pre is 3 px off the surface, so L=5 pushes ~2 px
SPEED = 20          # px per control step, about what the human demos in pusht do


def path_actions(start, waypoints, speed=SPEED):
    """targets along the polyline at constant speed"""
    pts = [np.asarray(start, float)] + [np.asarray(w, float) for w in waypoints]
    out, carry = [], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(b - a)
        if seg < 1e-6:
            continue
        d = speed - carry
        while d <= seg:
            out.append(a + (b - a) * d / seg)
            d += speed
        carry = seg - (d - speed)
    if not out or np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


def clear(a, b, pose, margin):
    n = max(2, int(np.linalg.norm(b - a) / 4))
    return all(point_poly_dist(a + (b - a) * t, pose) >= margin for t in np.linspace(0, 1, n))


def route(agent, target, pose, margin=AGENT_R + 8, R=108):
    """waypoints from the agent to target without touching the block: straight if
    possible, otherwise out to a circle around the block, along it, and back in"""
    if clear(agent, target, pose, margin):
        return [target]
    C = pose[:2] + rot(pose[2]) @ np.array([0, 45.0])            # center of the T
    a0 = np.arctan2(*(agent - C)[::-1])
    a1 = np.arctan2(*(target - C)[::-1])
    da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    pts = []
    if np.linalg.norm(agent - C) < R:
        pts.append(C + R * np.array([np.cos(a0), np.sin(a0)]))
    n = int(np.ceil(abs(da) / np.radians(25)))
    for i in range(1, n + 1):
        a = a0 + da * i / n
        pts.append(C + R * np.array([np.cos(a), np.sin(a)]))
    pts.append(target)
    return [np.clip(p, 10, 502) for p in pts]


class Expert:
    def __init__(self, nav_cost=0.03):
        self.sim = Sim()           # private copy for planning
        self.nav_cost = nav_cost
        self.queue = []

    def push(self, cw, nw, L):
        """approach point and actions of one push primitive.
        The agent lags its PD target by ~0.1 s, so every push starts with the agent at
        rest at `far` (the expert waits there for 2 steps) and the planner simulates
        exactly the actions that are executed from there."""
        pre = cw + nw * (AGENT_R + 3)
        far = cw + nw * (AGENT_R + 30)
        end = pre - nw * L
        return far, path_actions(far, [pre, end]) + [end, end]

    def plan(self, state):
        pose = state[4:7]
        R = rot(pose[2])
        now = kp_error(pose)
        lengths = FINE if now < 12 else LENGTHS
        best = (np.inf, None)
        for c, n in CONTACTS:
            cw, nw = pose[:2] + R @ c, R @ n
            nav = np.linalg.norm(cw - state[:2])
            for L in lengths:
                far, acts = self.push(cw, nw, L)
                if far.min() < 0 or far.max() > 512:
                    continue
                s = state.copy()
                s[0:2], s[2:4] = far, 0
                self.sim.set_state(s)
                for a in acts:
                    self.sim.step(a)
                err = kp_error(self.sim.block_pose())
                if err > now - 0.05:
                    continue
                # walking far for a slightly better push is not worth it, unless we are close
                cost = err + self.nav_cost * nav * min(1, now / 20)
                if cost < best[0]:
                    best = (cost, (cw, nw, L))
        return best[1]

    def __call__(self, state):
        if not self.queue:
            p = self.plan(state)
            agent, pose = state[:2], state[4:7]
            if p is None:
                # nothing helps from here, go somewhere else and try again
                self.queue = path_actions(agent, route(agent, np.random.uniform(80, 432, 2), pose))
            else:
                cw, nw, L = p
                far, acts = self.push(cw, nw, L)
                self.queue = path_actions(agent, route(agent, far, pose)) + [far, far] + acts
                self.queue.append(acts[-1] + nw * (L + 25))      # back off
        return self.queue.pop(0)


def demo(seed, max_steps=300):
    sim, expert = Sim(), Expert()
    sim.reset(seed)
    S, A, cov = [], [], []
    for t in range(max_steps):
        s = sim.state()
        a = expert(s)
        S.append(s); A.append(a)
        sim.step(a)
        cov.append(sim.coverage())
        if cov[-1] > 0.95:
            break
    return np.array(S), np.array(A), np.array(cov)


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = []
    for seed in range(20):
        S, A, cov = demo(seed)
        res.append((cov[-1] > 0.95, len(A), cov.max()))
        print(seed, res[-1], f"{time.time() - t0:.0f}s")
    r = np.array(res, float)
    print("success", r[:, 0].mean(), "steps", r[:, 1].mean(), "max cov", r[:, 2].mean())
