# Here will be defined sequence /beta and its derivatives 

import torch.nn.functional as F
import torch
import config
import math


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    xx = torch.linspace(0, timesteps, steps)

    ft = torch.cos(((xx / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


def linear_beta_schedule(timesteps):
    beta_1 = 1e-4
    beta_T = 0.02
    return torch.linspace(beta_1, beta_T, timesteps)


cfg = config.load("./config/config.yaml")

if cfg.schedule == 'linear':
    betas = linear_beta_schedule(timesteps=cfg.timesteps)
elif cfg.schedule == 'cosine':
    betas = cosine_beta_schedule(timesteps=cfg.timesteps)
else:
    raise NotImplementedError()

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# (1, 0) - padding element number on the left and right side - for dimension consistence
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_inv = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# aposterior probability q(x_{t-1}|x_t,x_0) to approximate revere process
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
