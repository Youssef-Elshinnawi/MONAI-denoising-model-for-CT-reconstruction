"""
FBPConvNet-style residual denoiser: a MONAI UNet wrapped so it predicts the
noise/artifact residual and adds it back to the input, rather than predicting
the clean image directly.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class ResidualDenoiser(nn.Module):
    """output = input + UNet(input) -- the UNet predicts the residual, not the clean image."""

    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.unet(x)


if __name__ == "__main__":
    model = ResidualDenoiser()

    dummy_input = torch.rand(4, 1, 512, 512)
    output = model(dummy_input)

    print(f"input shape:  {dummy_input.shape}")
    print(f"output shape: {output.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable parameters: {num_params:,}")
