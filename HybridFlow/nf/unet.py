import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dZeros(nn.Conv2d):
    """A conv layer with weights and bias initialized to zero."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv, using transposed conv + skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use in_channels here (not in_channels//2) for the transposed conv input
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        # After concat with the skip connection, channels = in_channels//2 + skip_channels.
        # In the typical U-Net pattern skip_channels == in_channels//2, so total = in_channels.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_down, x_skip):
        # x_down has shape [B, in_channels, H, W] 
        x_down = self.up(x_down)
        # x_down now [B, in_channels//2, 2H, 2W]

        # Pad if the spatial shapes differ
        diffY = x_skip.size(2) - x_down.size(2)
        diffX = x_skip.size(3) - x_down.size(3)
        if diffY != 0 or diffX != 0:
            x_down = F.pad(
                x_down, 
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )
        # Concat skip
        x = torch.cat([x_skip, x_down], dim=1)
        # x now has shape [B, in_channels//2 + skip_channels, 2H, 2W]
        return self.conv(x)

class Unet(nn.Module):
    """Example U-Net."""
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels * 2)

        # Bottleneck
        self.bottleneck = DoubleConv(hidden_channels * 2, hidden_channels * 4)

        # Decoder
        self.up1 = Up(hidden_channels * 4, hidden_channels * 2)
        self.up2 = Up(hidden_channels * 2, hidden_channels)

        # Final output (zero-init if desired)
        # self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.final_conv = Conv2dZeros(hidden_channels, out_channels)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)                # [B, hidden_channels, H, W]
        x2 = self.down1(x1)             # [B, hidden_channels*2, H/2, W/2]

        # Bottleneck
        x_bottleneck = self.bottleneck(x2)  # [B, hidden_channels*4, H/2, W/2]

        # Decoder
        x = self.up1(x_bottleneck, x2)      # [B, hidden_channels*2, H/2, W/2]
        x = self.up2(x, x1)                 # [B, hidden_channels, H, W]

        # Final
        out = self.final_conv(x)
        return out
