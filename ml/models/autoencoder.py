from dataclasses import dataclass

import torch
from torch import nn

from .common import UNetGenerator, init_weights


@dataclass
class AutoencoderLosses:
    reconstruction: float


class AutoencoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = UNetGenerator()
        self.generator.apply(init_weights)
        self.l1 = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def step(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, AutoencoderLosses]:
        prediction = self(pre)
        loss = self.l1(prediction, post)
        return loss, AutoencoderLosses(reconstruction=float(loss.detach().cpu()))

