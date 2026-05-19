from dataclasses import dataclass

import torch
from torch import nn

from .common import PatchDiscriminator, UNetGenerator, init_weights


@dataclass
class Pix2PixLosses:
    generator_total: float
    discriminator_total: float
    adversarial: float
    reconstruction: float


class Pix2PixModel(nn.Module):
    def __init__(self, lambda_l1: float = 100.0):
        super().__init__()
        self.generator = UNetGenerator()
        self.discriminator = PatchDiscriminator()
        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def discriminator_loss(self, pre: torch.Tensor, post: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_logits = self.discriminator(pre, post)
        fake_logits = self.discriminator(pre, fake.detach())
        real_loss = self.bce(real_logits, torch.ones_like(real_logits))
        fake_loss = self.bce(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (real_loss + fake_loss)

    def generator_loss(self, pre: torch.Tensor, post: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        adv_logits = self.discriminator(pre, fake)
        adv_loss = self.bce(adv_logits, torch.ones_like(adv_logits))
        l1_loss = self.l1(fake, post)
        total = adv_loss + self.lambda_l1 * l1_loss
        return total, float(adv_loss.detach().cpu()), float(l1_loss.detach().cpu())

