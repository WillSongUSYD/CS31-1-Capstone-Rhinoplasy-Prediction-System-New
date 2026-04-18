from typing import Iterable

import torch
from torch import nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.InstanceNorm2d):
        # InstanceNorm2d only has affine weights when affine=True
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def _norm_layer(num_channels: int, norm: str) -> nn.Module:
    """Build a normalisation layer. InstanceNorm avoids the cross-phase
    running-stat pollution that BatchNorm exhibits in GAN training when
    generator and discriminator alternate and see different input statistics.
    """
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=False)
    raise ValueError(f"Unknown norm type: {norm!r} (expected 'batch' or 'instance')")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, down: bool = True, use_norm: bool = True, dropout: float = 0.0, norm: str = "batch"):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not use_norm)]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=not use_norm)]
        if use_norm:
            layers.append(_norm_layer(out_channels, norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.down1 = ConvBlock(in_channels, 64, use_norm=False)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        self.down5 = ConvBlock(512, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.up1 = ConvBlock(512, 512, down=False, dropout=0.5)
        self.up2 = ConvBlock(1024, 512, down=False, dropout=0.5)
        self.up3 = ConvBlock(1024, 256, down=False)
        self.up4 = ConvBlock(512, 128, down=False)
        self.up5 = ConvBlock(256, 64, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.bottleneck(d5)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d5], dim=1))
        up3 = self.up3(torch.cat([up2, d4], dim=1))
        up4 = self.up4(torch.cat([up3, d3], dim=1))
        up5 = self.up5(torch.cat([up4, d2], dim=1))
        return self.final(torch.cat([up5, d1], dim=1))


def _disc_body(in_channels: int, norm: str) -> nn.Sequential:
    """Shared PatchGAN discriminator body. `norm` selects batch or instance norm."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        _norm_layer(128, norm),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        _norm_layer(256, norm),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 512, 4, 1, 1, bias=False),
        _norm_layer(512, norm),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 1, 4, 1, 1),
    )


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator for pix2pix (takes concat(x, y)).

    Default norm is 'batch' for backward compatibility with existing checkpoints.
    New trainings should prefer norm='instance' to avoid running-stat pollution
    when generator/discriminator alternate between different input distributions.
    """

    def __init__(self, in_channels: int = 6, norm: str = "batch"):
        super().__init__()
        self.model = _disc_body(in_channels, norm)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, y], dim=1))


class UnconditionalPatchDiscriminator(nn.Module):
    """Unconditional PatchGAN discriminator for CycleGAN (takes single image).

    Defaults to InstanceNorm, which matches the canonical CycleGAN implementation
    and is more robust to the small-batch / alternating-phase training schedule.
    """

    def __init__(self, in_channels: int = 3, norm: str = "instance"):
        super().__init__()
        self.model = _disc_body(in_channels, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def set_requires_grad(modules: Iterable[nn.Module], value: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = value
