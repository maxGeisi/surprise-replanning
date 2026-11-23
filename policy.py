"""Small state-based Diffusion Policy (Chi et al. 2023), written from the paper and
the official colab notebook. 1D conditional UNet over the action sequence,
FiLM conditioning on the last 2 observations, DDPM training, DDIM sampling."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_OBS = 2       # observation horizon
T_PRED = 16     # prediction horizon
OBS_DIM = 6     # agent xy, block xy, cos/sin of block angle
ACT_DIM = 2


def obs_feat(obs):
    """raw obs [..., 5] = (agent x, y, block x, y, angle) -> [..., 6] in [-1, 1]"""
    obs = np.asarray(obs, dtype=np.float32)
    return np.concatenate([obs[..., 0:4] / 256 - 1, np.cos(obs[..., 4:5]), np.sin(obs[..., 4:5])], -1)


def act_norm(a):
    return np.asarray(a, dtype=np.float32) / 256 - 1


def act_unnorm(a):
    return (a + 1) * 256


# ---- network ----------------------------------------------------------------

class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        x = t.float()[:, None] * f[None]
        return torch.cat([x.sin(), x.cos()], -1)


def conv_block(i, o, k=5, groups=8):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(groups, o), nn.Mish())


class ResBlock(nn.Module):
    def __init__(self, i, o, cond_dim):
        super().__init__()
        self.b1, self.b2 = conv_block(i, o), conv_block(o, o)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * o))
        self.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, cond):
        h = self.b1(x)
        scale, bias = self.film(cond)[..., None].chunk(2, dim=1)
        h = self.b2(scale * h + bias)
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, dims=(64, 128, 256), t_dim=64):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmb(t_dim), nn.Linear(t_dim, 4 * t_dim), nn.Mish(), nn.Linear(4 * t_dim, t_dim))
        cond = t_dim + T_OBS * OBS_DIM
        chans = [ACT_DIM] + list(dims)
        self.down = nn.ModuleList()
        for k, (i, o) in enumerate(zip(chans[:-1], chans[1:])):
            last = k == len(dims) - 1
            self.down.append(nn.ModuleList([ResBlock(i, o, cond), ResBlock(o, o, cond),
                                            nn.Conv1d(o, o, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([ResBlock(dims[-1], dims[-1], cond), ResBlock(dims[-1], dims[-1], cond)])
        self.up = nn.ModuleList()
        for k, (o, i) in enumerate(zip(reversed(dims[:-1]), reversed(dims[1:]))):
            self.up.append(nn.ModuleList([ResBlock(2 * i, o, cond), ResBlock(o, o, cond),
                                          nn.ConvTranspose1d(o, o, 4, 2, 1)]))
        self.out = nn.Sequential(conv_block(dims[0], dims[0]), nn.Conv1d(dims[0], ACT_DIM, 1))

    def forward(self, x, t, obs):
        # x [B, T_PRED, 2], t [B], obs [B, T_OBS, 6]
        cond = torch.cat([self.t_emb(t), obs.flatten(1)], -1)
        x = x.transpose(1, 2)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, cond)
        for r1, r2, up in self.up:
            x = torch.cat([x, skips.pop()], 1)
            x = up(r2(r1(x, cond), cond))
        return self.out(x).transpose(1, 2)


# ---- diffusion ----------------------------------------------------------------

def cosine_betas(n, s=0.008):
    # "squaredcos_cap_v2" in diffusers
    f = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return torch.tensor([min(1 - f((i + 1) / n) / f(i / n), 0.999) for i in range(n)])


class Diffusion(nn.Module):
    def __init__(self, n_train=100):
        super().__init__()
        self.net = UNet1D()
        self.n_train = n_train
        self.register_buffer("abar", torch.cumprod(1 - cosine_betas(n_train), 0))

    def loss(self, obs, act):
        B = act.shape[0]
        t = torch.randint(0, self.n_train, (B,))
        noise = torch.randn_like(act)
        ab = self.abar[t][:, None, None]
        x = ab.sqrt() * act + (1 - ab).sqrt() * noise
        return F.mse_loss(self.net(x, t, obs), noise)

    @torch.no_grad()
    def sample(self, obs, n_steps=10, generator=None):
        """DDIM (eta = 0)"""
        B = obs.shape[0]
        x = torch.randn(B, T_PRED, ACT_DIM, generator=generator)
        ts = torch.linspace(self.n_train - 1, 0, n_steps).round().long()
        for i, t in enumerate(ts):
            ab = self.abar[t]
            eps = self.net(x, t.expand(B), obs)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            if i == len(ts) - 1:
                return x0
            ab_prev = self.abar[ts[i + 1]]
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps


class Policy:
    """wraps the model: raw observation history in, actions in pixels out"""

    def __init__(self, path="runs/policy/ema.pt", n_steps=10, seed=0):
        self.model = Diffusion()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.n_steps = n_steps
        self.gen = torch.Generator().manual_seed(seed)
        self.calls = 0

    def __call__(self, obs_hist):
        """obs_hist [B, T_OBS, 5] -> actions [B, T_PRED, 2]"""
        self.calls += len(obs_hist)
        o = torch.from_numpy(obs_feat(obs_hist))
        return act_unnorm(self.model.sample(o, self.n_steps, self.gen).numpy())
