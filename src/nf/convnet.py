import torch.nn as nn
from nf.layers import Conv2d, Conv2dZeros


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.cnn = self.st_net(in_channels, out_channels, hidden_channels)

    def st_net(self, in_channels, out_channels, hidden_channels):
        block = nn.Sequential(Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False),
                              Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)), nn.ReLU(inplace=False),
                              Conv2dZeros(hidden_channels, out_channels))
        
        return block

    def __call__(self, x):
        return self.cnn(x)


class SmallResBlock(nn.Module):
    """
    A simple residual block:
      x -> conv(3x3) -> ReLU -> conv(3x3) -> +x
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity


class ConvNet_with_Resnet(nn.Module):
    """
    Example usage in place of your st_net.
    """
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.cnn = self.st_net(in_channels, out_channels, hidden_channels)

    def st_net(self, in_channels, out_channels, hidden_channels):
        block = nn.Sequential(
            # 1) First conv to go from in_channels -> hidden_channels
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 2) One or two small residual blocks for more expressive power
            SmallResBlock(hidden_channels),
            
            # 3) Optional: a 1x1 conv or another small block
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            
            # 4) Final zero-initialized conv to produce out_channels
            Conv2dZeros(hidden_channels, out_channels)
        )
        return block

    def forward(self, x):
        return self.cnn(x)


