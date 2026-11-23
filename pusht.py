import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from gym_pusht.envs.pusht import PushTEnv

GOAL = np.array([256.0, 256.0, np.pi / 4])
AGENT_R = 15
# T outline in block coordinates (bar on top, stem below), same as gym-pusht
BAR = np.array([[-60, 0], [60, 0], [60, 30], [-60, 30]], dtype=float)
STEM = np.array([[-15, 30], [15, 30], [15, 120], [-15, 120]], dtype=float)
KP_LOCAL = np.concatenate([BAR, STEM])


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def keypoints(pose):
    """8 corners of the T in world coordinates. pose = (x, y, angle)"""
    pose = np.asarray(pose, dtype=float)
    return KP_LOCAL @ rot(pose[2]).T + pose[:2]


GOAL_KP = keypoints(GOAL)


def kp_error(pose):
    return np.linalg.norm(keypoints(pose) - GOAL_KP, axis=1).mean()


class Sim:
    """gym-pusht with direct access to the state.

    state = [agent x, y, agent vx, vy, block x, y, block angle]
    obs   = [agent x, y, block x, y, block angle]  (what the policy gets)
    """

    def __init__(self):
        self.env = PushTEnv(obs_type="state")
        self.env.reset(seed=0)
        self.n_sub = 10    # 100 Hz physics, 10 Hz control

    def reset(self, seed):
        self.env.reset(seed=int(seed))
        return self.obs()

    def state(self):
        a, b = self.env.agent, self.env.block
        return np.array([*a.position, *a.velocity, *b.position, b.angle])

    def set_state(self, s):
        a, b = self.env.agent, self.env.block
        a.position = (s[0], s[1])
        a.velocity = (s[2], s[3])
        b.angle = s[6]          # angle first! setting it rotates around the COG
        b.position = (s[4], s[5])
        b.velocity = (0, 0)
        b.angular_velocity = 0
        self.env.space.reindex_shapes_for_body(a)
        self.env.space.reindex_shapes_for_body(b)

    def obs(self):
        s = self.state()
        return np.array([s[0], s[1], s[4], s[5], s[6]])

    def block_pose(self):
        b = self.env.block
        return np.array([*b.position, b.angle])

    def step(self, action):
        # same as PushTEnv.step without the reward computation
        e = self.env
        act = np.asarray(action, dtype=float)
        for _ in range(self.n_sub):
            acc = e.k_p * (act - np.array(e.agent.position)) - e.k_v * np.array(e.agent.velocity)
            e.agent.velocity += tuple(acc * e.dt)
            e.space.step(e.dt)
        return self.obs()

    def coverage(self):
        return self.env._get_coverage()

    def kick(self, rng, dist, dangle, tries=30):
        """Move the block by `dist` px in a random direction and rotate it by +-dangle,
        like somebody grabbing it. Rejects poses that overlap the agent or leave the table."""
        s = self.state()
        for _ in range(tries):
            phi = rng.uniform(0, 2 * np.pi)
            new = s.copy()
            new[4:6] += dist * np.array([np.cos(phi), np.sin(phi)])
            new[6] += dangle * rng.choice([-1, 1])
            kp = keypoints(new[4:7])
            if kp.min() < 15 or kp.max() > 497:
                continue
            if point_poly_dist(new[:2], new[4:7]) < AGENT_R + 3:
                continue
            self.set_state(new)
            return True
        return False

    def frame(self, size=512):
        import cv2
        import pygame
        screen = self.env._draw()
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2)).copy()
        if size != 512:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img


def point_poly_dist(p, pose):
    """distance from point p to the T at `pose` (0 if inside)"""
    q = rot(-pose[2]) @ (np.asarray(p) - pose[:2])       # into block coordinates
    d = []
    for lo, hi in [(BAR[0], BAR[2]), (STEM[0], STEM[2])]:
        dx = max(lo[0] - q[0], 0, q[0] - hi[0])
        dy = max(lo[1] - q[1], 0, q[1] - hi[1])
        d.append(np.hypot(dx, dy))
    return min(d)
