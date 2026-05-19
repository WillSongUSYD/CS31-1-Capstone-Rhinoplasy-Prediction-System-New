from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .common import UnconditionalPatchDiscriminator, UNetGenerator, init_weights


@dataclass
class CycleGANLosses:
    generator_total: float
    discriminator_total: float
    cycle: float
    identity: float


class CycleGANModel(nn.Module):
    def __init__(self, lambda_cycle: float = 10.0, lambda_identity: float = 5.0):
        super().__init__()
        # Generators: g_xy maps domain X (pre) -> Y (post); g_yx maps Y -> X
        self.g_xy = UNetGenerator()
        self.g_yx = UNetGenerator()
        # Discriminators operate on a single domain each (unconditional PatchGAN
        # with InstanceNorm - see UnconditionalPatchDiscriminator for rationale).
        self.d_x = UnconditionalPatchDiscriminator()
        self.d_y = UnconditionalPatchDiscriminator()
        for module in [self.g_xy, self.g_yx, self.d_x, self.d_y]:
            module.apply(init_weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_xy(x)

    def generator_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, CycleGANLosses, torch.Tensor, torch.Tensor]:
        """Compute the generator loss and return the produced fakes (detached)
        so the training loop can feed them into the discriminator step without
        re-running the generators. Returns (total_loss, losses, fake_y_detached,
        fake_x_detached).
        """
        # Forward translations (in-graph: gradients will flow back to G)
        fake_y = self.g_xy(x)
        fake_x = self.g_yx(y)
        # Cycle reconstructions
        cycle_x = self.g_yx(fake_y)
        cycle_y = self.g_xy(fake_x)
        # Identity mappings
        identity_x = self.g_yx(x)
        identity_y = self.g_xy(y)

        # Adversarial losses: generators try to fool discriminators
        # (discriminators are expected to be frozen during G step)
        pred_fake_y = self.d_y(fake_y)
        pred_fake_x = self.d_x(fake_x)
        adv_y = self.bce(pred_fake_y, torch.ones_like(pred_fake_y))
        adv_x = self.bce(pred_fake_x, torch.ones_like(pred_fake_x))

        cycle_loss = self.l1(cycle_x, x) + self.l1(cycle_y, y)
        identity_loss = self.l1(identity_x, x) + self.l1(identity_y, y)
        total = adv_x + adv_y + self.lambda_cycle * cycle_loss + self.lambda_identity * identity_loss

        losses = CycleGANLosses(
            generator_total=float(total.detach().item()),
            discriminator_total=0.0,
            cycle=float(cycle_loss.detach().item()),
            identity=float(identity_loss.detach().item()),
        )
        return total, losses, fake_y.detach(), fake_x.detach()

    def discriminator_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fake_y: torch.Tensor = None,
        fake_x: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the discriminator loss. If fake_y/fake_x are provided (from
        generator_loss), reuse them to avoid redundant generator forward passes.
        Otherwise generate fresh fakes under no_grad.
        """
        if fake_y is None or fake_x is None:
            with torch.no_grad():
                fake_y = self.g_xy(x)
                fake_x = self.g_yx(y)

        # D_y distinguishes real post-op (y) from generated post-op (fake_y)
        pred_real_y = self.d_y(y)
        pred_fake_y = self.d_y(fake_y)
        loss_y = 0.5 * (
            self.bce(pred_real_y, torch.ones_like(pred_real_y))
            + self.bce(pred_fake_y, torch.zeros_like(pred_fake_y))
        )

        # D_x distinguishes real pre-op (x) from generated pre-op (fake_x)
        pred_real_x = self.d_x(x)
        pred_fake_x = self.d_x(fake_x)
        loss_x = 0.5 * (
            self.bce(pred_real_x, torch.ones_like(pred_real_x))
            + self.bce(pred_fake_x, torch.zeros_like(pred_fake_x))
        )

        return loss_x + loss_y
