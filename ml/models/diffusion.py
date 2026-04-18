from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .common import init_weights


@dataclass
class DiffusionLosses:
    noise_loss: float


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)))
        args = t[:, None].float() * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.net(emb)


class TinyDiffusionUNet(nn.Module):
    def __init__(self, channels: int = 64, timesteps_dim: int = 64):
        super().__init__()
        self.time_mlp = TimeEmbedding(timesteps_dim)
        self.input = nn.Conv2d(6, channels, 3, padding=1)
        self.down1 = nn.Sequential(nn.Conv2d(channels, channels * 2, 4, 2, 1), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(channels * 2, channels * 4, 4, 2, 1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(nn.Conv2d(channels * 4, channels * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(channels, 3, 3, padding=1)
        self.time_proj = nn.Linear(timesteps_dim, channels * 4)

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if noisy_target.shape[-1] != 64:
            noisy_target = F.interpolate(noisy_target, size=(64, 64), mode="bilinear", align_corners=False)
        if condition.shape[-1] != 64:
            condition = F.interpolate(condition, size=(64, 64), mode="bilinear", align_corners=False)
        x = torch.cat([noisy_target, condition], dim=1)
        x = self.input(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        time_emb = self.time_proj(self.time_mlp(timesteps)).view(timesteps.shape[0], -1, 1, 1)
        x_mid = self.mid(x2 + time_emb)
        x = self.up1(x_mid)
        x = self.up2(x)
        return self.out(x)


class DiffusionFeasibilityModel(nn.Module):
    def __init__(self, timesteps: int = 100):
        super().__init__()
        self.model = TinyDiffusionUNet()
        self.model.apply(init_weights)
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.loss_fn = nn.MSELoss()

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, DiffusionLosses]:
        batch_size = target.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=target.device)
        target_small = F.interpolate(target, size=(64, 64), mode="bilinear", align_corners=False)
        noise = torch.randn_like(target_small)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy_target = torch.sqrt(alpha_bar) * target_small + torch.sqrt(1 - alpha_bar) * noise
        pred_noise = self.model(noisy_target, condition, t)
        loss = self.loss_fn(pred_noise, noise)
        return loss, DiffusionLosses(noise_loss=float(loss.detach().cpu()))

    @torch.no_grad()
    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        x = torch.randn(condition.shape[0], 3, 64, 64, device=condition.device)
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((condition.shape[0],), timestep, device=condition.device, dtype=torch.long)
            pred_noise = self.model(x, condition, t)
            beta_t = self.betas[timestep]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alphas_cumprod[timestep]
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
            if timestep > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        return x.clamp(-1, 1)
