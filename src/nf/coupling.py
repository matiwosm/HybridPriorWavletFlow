
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import torch
import torch.nn as nn
from nflows.transforms.splines import rational_quadratic_spline, unconstrained_rational_quadratic_spline
from utilities import split_feature
from nf.convnet import ConvNet, ConvNet_with_Resnet
from nf.layers import SqueezeLayer
from nf.unet import Unet
import numpy as np


class Affine(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, conditional):
        super().__init__()
        
        if out_channels % 2 != 0:
            out_channels += 1
            
        if conditional:
            self.block = ConvNet(in_channels // 2 + in_channels // 3, out_channels, hidden_channels)
        else:
            self.block = ConvNet(in_channels // 2, out_channels, hidden_channels)

    def get_param(self, x, conditioning):
        z1, z2 = split_feature(x, "split")
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        if reverse:
            z2 = z2 / s
            z2 = z2 - t
            logdet = -torch.sum(torch.log(s), dim=[1, 2, 3]) + logdet
        else:
            s, t, z1, z2 = self.get_param(x, conditioning)
            z2 = z2 + t
            z2 = z2 * s
            logdet = torch.sum(torch.log(s), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        return z, logdet

class MyFullyActiveRQS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 H, W,
                 hidden_channels,
                 conditional=True,
                 net_type='ConvNet',
                 num_bins=8,
                 rqs_left=-10.0,
                 rqs_right=10.0,
                 min_bin_width=1e-3,
                 min_bin_height=1e-3,
                 min_derivative=1e-3):
        super().__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
        # Each pixel+channel needs (3*num_bins - 1) parameters: widths, heights, derivs
        out_channels_spline = out_channels * (3*self.num_bins - 1)

        in_channels_net = in_channels
        if conditional:
            # Concat the "frozen" half + conditioning
            in_channels_net = in_channels + (in_channels // 3)  # your custom logic

        # Build your chosen network (ConvNet, UNet, ResNet, etc.)
        if net_type == 'ConvNet':
            self.block = ConvNet(in_channels_net, out_channels_spline, hidden_channels)
        elif net_type == 'UNet':
            self.block = Unet(in_channels_net, out_channels_spline, hidden_channels)
        elif net_type == 'ResNet':
            self.block = ConvNet_with_Resnet(in_channels_net, out_channels_spline, hidden_channels)
        else:
            raise ValueError(f"Unknown net_type={net_type}")

    def forward(self, x, logdet, conditioning=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, logdet, conditioning)
        else:
            return self.reverse_flow(x, logdet, conditioning)

    def normal_flow(self, x, logdet, conditioning):
        """
        Vectorized forward pass, no mask:
          1) Build network input
          2) Flatten x => (B*C*H*W,)
          3) Flatten params => (B*C*H*W, 3*num_bins -1)
          4) RQ-spline forward
        """
        B, C, H, W = x.shape

        # 1) build net input
        if conditioning is not None:
            net_in = torch.cat([x, conditioning], dim=1)
        else:
            net_in = x

        # 2) get parameters from the net
        # shape => (B, C*(3*num_bins -1), H, W)
        params = self.block(net_in)
        
        # Flatten x
        x_flat = x.reshape(-1)  # shape (B*C*H*W,)

        # Flatten params to match x_flat
        # => (B, C*(3*num_bins -1), H, W) -> (B, C, (3*num_bins -1), H, W)
        # -> permute -> reshape => (B*C*H*W, 3*num_bins -1)
        params = params.view(B, C, (3*self.num_bins - 1), H, W)
        params = params.permute(0, 1, 3, 4, 2)  # => (B, C, H, W, (3*num_bins -1))
        params_flat = params.reshape(B*C*H*W, (3*self.num_bins - 1))

        # 3) slice out widths, heights, derivatives
        step = self.num_bins
        widths      = params_flat[:, 0*step : 1*step]
        heights     = params_flat[:, 1*step : 2*step]
        derivatives = params_flat[:, 2*step : (3*self.num_bins - 1)]

        # 4) single pass through the RQ-spline forward
        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            x_flat,
            widths,
            heights,
            derivatives,
            inverse=False,
            tails="linear",
            tail_bound=3.0,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative
        )

        # reshape y_flat => (B, C, H, W)
        y = y_flat.view(B, C, H, W)

        # sum up logdet over all elements
        # logabsdet_flat => (B*C*H*W,)
        logabsdet_bchw = logabsdet_flat.view(B, -1)  # => (B, C*H*W)
        logdet_out = logdet + logabsdet_bchw.sum(dim=1)

        return y, logdet_out

    def reverse_flow(self, y, logdet, conditioning):
        """
        Inverse pass:
          1) build net input from y
          2) flatten y
          3) flatten net output => RQ-spline inverse
        """
        B, C, H, W = y.shape

        if conditioning is not None:
            net_in = torch.cat([y, conditioning], dim=1)
        else:
            net_in = y

        # get params
        params = self.block(net_in)
        params = params.view(B, C, (3*self.num_bins - 1), H, W)
        params = params.permute(0, 1, 3, 4, 2)
        params_flat = params.reshape(B*C*H*W, (3*self.num_bins - 1))

        # flatten y
        y_flat = y.reshape(-1)

        step = self.num_bins
        widths      = params_flat[:, 0*step : 1*step]
        heights     = params_flat[:, 1*step : 2*step]
        derivatives = params_flat[:, 2*step : (3*self.num_bins - 1)]

        x_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            y_flat,
            widths,
            heights,
            derivatives,
            inverse=True,
            tails="linear",
            tail_bound=3.0,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative
        )

        x_out = x_flat.view(B, C, H, W)

        logabsdet_bchw = logabsdet_flat.view(B, -1)
        logdet_out = logdet + logabsdet_bchw.sum(dim=1)

        return x_out, logdet_out

class MyCheckerboardRQS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 H, W,
                 hidden_channels,
                 parity,
                 conditional=True,
                 net_type='ConvNet',
                 num_bins=8,
                 rqs_left=-10.0,
                 rqs_right=10.0,
                 min_bin_width=1e-3,
                 min_bin_height=1e-3,
                 min_derivative=1e-3,
                 mask_type='checkerboard'):

        super().__init__()
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Decide which mask to use
        if mask_type == 'checkerboard':
            self.mask = self.make_checker_mask((1, 1, H, W), parity)
        elif mask_type == 'channel':
            self.mask = self.make_channel_mask(in_channels, parity)
        else:
            raise ValueError(f"Unknown mask_type={mask_type}")

        # Each pixel+channel needs (3*num_bins - 1) parameters: widths, heights, derivs
        out_channels_spline = out_channels * (3*self.num_bins - 1)

        in_channels_net = in_channels
        if conditional:
            # Concat the "frozen" half + conditioning
            in_channels_net = in_channels + (in_channels // 3)  # your custom logic

        # Build your chosen network (ConvNet, UNet, ResNet, etc.)
        if net_type == 'ConvNet':
            self.block = ConvNet(in_channels_net, out_channels_spline, hidden_channels)
        elif net_type == 'UNet':
            self.block = Unet(in_channels_net, out_channels_spline, hidden_channels)
        elif net_type == 'ResNet':
            self.block = ConvNet_with_Resnet(in_channels_net, out_channels_spline, hidden_channels)
        else:
            raise ValueError(f"Unknown net_type={net_type}")

    def make_checker_mask(self, shape, parity):
        """
        shape: (1, 1, H, W), parity in {0,1}
        """
        checker = torch.ones(shape, dtype=torch.uint8) - parity
        checker[:, :, ::2, ::2] = parity
        checker[:, :, 1::2, 1::2] = parity
        return checker.cuda()

    def make_channel_mask(self, in_channels, parity):
        """
        Zero out half the channels (freeze). E.g. even or odd channels
        """
        mask = torch.zeros((1, in_channels, 1, 1), dtype=torch.uint8)
        mask[:, parity::2, :, :] = 1
        return mask.cuda()

    def forward(self, x, logdet, conditioning=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, logdet, conditioning)
        else:
            return self.reverse_flow(x, logdet, conditioning)

    def normal_flow(self, x, logdet, conditioning):
        """
        Vectorized forward pass:
          1) freeze the mask part (x_frozen)
          2) gather RQ-spline params from the net
          3) flatten (B, C, H, W) -> (B*C*H*W), do one big spline call
          4) reshape back
        """
        B, C, H, W = x.shape

        # 1) Separate "frozen" & "active"
        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x

        # 2) Build network input to get RQS params
        if conditioning is not None:
            z1_c = torch.cat([x_frozen, conditioning], dim=1)  # (B, in_channels_net, H, W)
            params = self.block(z1_c)
        else:
            params = self.block(x_frozen)  # shape: (B, out_channels_spline, H, W)

        # Now we have shape (B, C*(3*num_bins - 1), H, W).
        # We'll flatten "active" data for the transform:
        # x_active: (B, C, H, W) -> x_flat: (B*C*H*W,)
        x_flat = x_active.reshape(-1)

        # Meanwhile flatten params to match (B*C*H*W, 3*num_bins -1):
        # Original shape: (B, C*(3*num_bins-1), H, W)
        # We want a row for each of the B*C*H*W "positions".
        # Step a) reshape to (B, C, (3*num_bins -1), H, W)
        # Step b) permute => (B, C, H, W, (3*num_bins -1))
        # Step c) final reshape => (B*C*H*W, 3*num_bins -1)

        params = params.view(B, C, (3*self.num_bins - 1), H, W)
        params = params.permute(0, 1, 3, 4, 2)  # => (B, C, H, W, (3*num_bins-1))
        params_flat = params.reshape(B*C*H*W, (3*self.num_bins - 1))

        # Now each *pixel* in x_flat has dimension=1, but we have a distinct set of parameters
        # for each pixel * channel. So effectively, we treat B*C*H*W "1D splines."

        # 3) Slice out widths, heights, derivatives
        # Because each "channel" chunk has (3*num_bins -1) params for 1D,
        # we combine them for all channels => B*H*W*C distinct sub-splines.
        # Actually, for a truly "channelwise" approach, you usually do a loop, but
        # here we treat them all at once in a single vector: (B*C*H*W).

        step = self.num_bins
        widths     = params_flat[:, 0*step : 1*step]
        heights    = params_flat[:, 1*step : 2*step]
        derivatives= params_flat[:, 2*step : (3*self.num_bins - 1)]

        # 4) Single pass through nflows' spline
        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            x_flat,
            widths,                # (B*C*H*W, num_bins)
            heights,               # (B*C*H*W, num_bins)
            derivatives,           # (B*C*H*W, num_bins -1)
            inverse=False,
            tails="linear",
            tail_bound=3.0,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative
        )
        # 5) Reshape back to (B, C, H, W)
        y = y_flat.view(B, C, H, W)

        # Combine with frozen part
        x_out = x_frozen + y * (1 - self.mask)

        # The logabsdet_flat is (B*C*H*W,). We need to sum over the "pixels" dimension
        # to get the total logdet for each batch item.
        # We'll reshape to (B, C*H*W) so we can sum over C*H*W:
        logabsdet_bchw = logabsdet_flat.view(B, C, H, W)
        logabsdet_bchw *= (1 - self.mask)
        # sum(dim=1) => shape (B,) => add to logdet
        logdet_out = logdet + logabsdet_bchw.sum(dim=[1,2,3])
        return x_out, logdet_out

    def reverse_flow(self, y, logdet, conditioning):
        """
        Inverse pass with the same vectorization strategy:
          1) freeze y_frozen, transform y_active
          2) flatten -> single call to rational_quadratic_spline(inverse=True)
          3) reshape
        """
        B, C, H, W = y.shape

        y_frozen = self.mask * y
        y_active = (1 - self.mask) * y

        if conditioning is not None:
            z1_c = torch.cat([y_frozen, conditioning], dim=1)
            params = self.block(z1_c)
        else:
            params = self.block(y_frozen)

        # shape => (B, C, (3*num_bins-1), H, W)
        params = params.view(B, C, (3*self.num_bins - 1), H, W)
        params = params.permute(0, 1, 3, 4, 2)
        params_flat = params.reshape(B*C*H*W, (3*self.num_bins - 1))

        y_flat = y_active.reshape(-1)

        step = self.num_bins
        widths     = params_flat[:, 0*step : 1*step]
        heights    = params_flat[:, 1*step : 2*step]
        derivatives= params_flat[:, 2*step : (3*self.num_bins - 1)]

        # Inverse = True
        x_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            y_flat,
            widths,
            heights,
            derivatives,
            inverse=True,
            tails="linear",
            tail_bound=3.0,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative
        )

        x_out = y_frozen.clone()
        x_out_active = x_flat.view(B, C, H, W)
        x_out += (x_out_active * (1 - self.mask))  # fill in the “active” portion

        logabsdet_bchw = logabsdet_flat.view(B, C, H, W)
        logabsdet_bchw *= (1 - self.mask)
        logdet_out = logdet + logabsdet_bchw.sum(dim=[1,2,3])

        return x_out, logdet_out


class MyCheckerboardRQS_per_channel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 H, W,
                 hidden_channels,
                 parity,
                 conditional=True,
                 net_type='ConvNet',
                 num_bins=8,  # e.g. default to 8
                 rqs_left=-3.0,
                 rqs_right=3.0,
                 min_bin_width=1e-3,
                 min_bin_height=1e-3,
                 min_derivative=1e-3,
                 mask_type='checkerboard'):
        super().__init__()

        self.num_bins = num_bins
        self.rqs_left = rqs_left
        self.rqs_right = rqs_right
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Decide which mask to use (checkerboard vs. channel-wise)
        if mask_type == 'checkerboard':
            self.mask = self.make_checker_mask((1, 1, H, W), parity)
        elif mask_type == 'channel':
            self.mask = self.make_channel_mask(in_channels, parity)
        else:
            raise ValueError(f"Unknown mask_type={mask_type}")

        # The network that outputs the RQ-spline parameters
        # Instead of output_channels=out_channels*2, we need
        # out_channels = (out_channels_active * (3 * num_bins - 1))
        # because each dimension needs: widths, heights, derivatives
        # for each bin: total 3*num_bins -1
        # But we only transform the active portion (the masked out).
        out_channels_spline = (out_channels) * (3 * self.num_bins - 1)

        in_channels_net = in_channels
        if conditional:
            # Concat the "frozen" half + conditioning into the net.
            in_channels_net = in_channels + (in_channels // 3)  # or however you defined it

        if net_type == 'ConvNet':
            self.block = ConvNet(in_channels_net, out_channels_spline, hidden_channels)
        elif net_type == 'UNet':
            self.block = Unet(in_channels_net, out_channels_spline, hidden_channels)
        elif net_type == 'ResNet':
            self.block = ConvNet_with_Resnet(in_channels_net, out_channels_spline, hidden_channels)
        else:
            raise ValueError(f"Unknown net_type={net_type}")

    def make_checker_mask(self, shape, parity):
        """
        shape: (1, 1, H, W)
        parity: 0 or 1
        """
        checker = torch.ones(shape, dtype=torch.uint8) - parity
        checker[:, :, ::2, ::2] = parity
        checker[:, :, 1::2, 1::2] = parity
        return checker.cuda()

    def make_channel_mask(self, in_channels, parity):
        """
        For simplicity, a channel-wise mask that zeroes half the channels.
        E.g., channels [0,2,4,...] or [1,3,5,...].
        """
        mask = torch.zeros((1, in_channels, 1, 1), dtype=torch.uint8)
        mask[:, parity::2, :, :] = 1  # e.g. freeze even or odd channels
        return mask.cuda()

    def forward(self, x, logdet, conditioning=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, logdet, conditioning)
        else:
            return self.reverse_flow(x, logdet, conditioning)

    def normal_flow(self, x, logdet, conditioning):
        """
        Forward pass: we freeze the mask part, transform the active part
        using RQ-spline. We update logdet with the sum of log|df/dx| from 
        the transform.
        """
        # 1) separate the "frozen" part & "active" part
        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x

        # 2) compute the network output (RQS parameters) from the frozen part + cond
        if conditioning is not None:
            z1_c = torch.cat([x_frozen, conditioning], dim=1)
            params = self.block(z1_c)
        else:
            params = self.block(x_frozen)

        # shape of params: (B, (C_active*(3*num_bins-1)), H, W)
        # we must reshape so that each (pixel, channel) gets its own RQ-spline
        B, _, H, W = params.shape
        C_active = x_active.shape[1]

        # We split `params` into per-channel chunks
        # each chunk has size (3*num_bins - 1)
        params = params.view(B, C_active*(3*self.num_bins - 1), H, W)

        # We'll process each channel/pixel in a loop or vectorized manner
        # We'll do it mostly vectorized for speed:
        x_out = x.clone()
        logdet_out = logdet

        # Index for slicing out each channel’s parameters
        # (3*num_bins - 1) per channel
        step = (3*self.num_bins - 1)

        for c_idx in range(C_active):
            # Get channel c_idx from x_active
            # Make sure to get the "active" channel c_idx from the correct portion
            # If your mask is checkerboard, c_idx might not be straightforwardly indexing 
            # the “active channels.”  But we’ll assume you are freezing the same 
            # spatial positions for all channels.
            channel_params = params[:, c_idx*step:(c_idx+1)*step, :, :]

            # x_current: the values of x_active in channel c_idx
            x_current = x_active[:, c_idx, :, :]

            # We'll call rational_quadratic_spline(...) on this x_current
            # but we have to flatten or keep shape consistent
            x_flat = x_current.reshape(-1)

            params_flat = channel_params.reshape(-1, step)

            # Now we parse out the RQ-spline parameters: 
            #   widths, heights, derivatives
            # nflows has a helper called `utils.splines.unconstrained_rational_quadratic_spline`,
            # which we can call directly. It returns (y, logabsdet)
            y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
                x_flat,
                params_flat[:, :self.num_bins],           # unnormalized widths
                params_flat[:, self.num_bins:2*self.num_bins],  # unnormalized heights
                params_flat[:, 2*self.num_bins:],        # unnormalized derivatives
                inverse=False,
                tails="linear",
                tail_bound=3.0,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative
            )

            # reshape back
            y_current = y_flat.view(B, H, W)
            x_out[:, c_idx, :, :] = x_frozen[:, c_idx, :, :] + y_current  # "active" is replaced

            # accumulate logdet
            logabsdet = logabsdet_flat.view(B, H, W)
            # sum over all pixels for batch
            logdet_out = logdet_out + logabsdet.sum(dim=[1,2])

        return x_out, logdet_out

    def reverse_flow(self, y, logdet, conditioning):
        """
        Inverse pass: given y, we retrieve x by inverting the RQ-spline with 
        the *same* parameters predicted by the "frozen" + conditioning.
        """
        y_frozen = self.mask * y
        y_active = (1 - self.mask) * y

        if conditioning is not None:
            z1_c = torch.cat([y_frozen, conditioning], dim=1)
            params = self.block(z1_c)
        else:
            params = self.block(y_frozen)

        B, _, H, W = params.shape
        C_active = y_active.shape[1]
        step = (3*self.num_bins - 1)
        params = params.view(B, C_active*step, H, W)

        x_out = y.clone()
        logdet_out = logdet

        for c_idx in range(C_active):
            channel_params = params[:, c_idx*step:(c_idx+1)*step, :, :]
            y_current = y_active[:, c_idx, :, :]
            y_flat = y_current.reshape(-1)
            params_flat = channel_params.reshape(-1, step)

            # We just call unconstrained_rational_quadratic_spline with inverse=True

            x_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
                x_flat,
                params_flat[:, :self.num_bins],           # unnormalized widths
                params_flat[:, self.num_bins:2*self.num_bins],  # unnormalized heights
                params_flat[:, 2*self.num_bins:],        # unnormalized derivatives
                inverse=True,
                tails="linear",
                tail_bound=3.0,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative
            )
            x_current = x_flat.view(B, H, W)
            x_out[:, c_idx, :, :] = y_frozen[:, c_idx, :, :] + x_current

            logabsdet = logabsdet_flat.view(B, H, W)
            logdet_out = logdet_out + logabsdet.sum(dim=[1,2])

        return x_out, logdet_out


class RadialMask(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, hidden_channels, idx, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

        center = (int(w/2), int(h/2))
        radius = min(center[0], center[1], w - center[0], w - center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius // 2
        idx = (idx + 1) % 2

        if idx == 0:
            self.mask_in = nn.Parameter(torch.tensor(mask, dtype=torch.float), requires_grad=False)
            self.mask_out = nn.Parameter(torch.tensor(~mask, dtype=torch.float), requires_grad=False)
        elif idx == 1:
            self.mask_in = nn.Parameter(torch.tensor(~mask, dtype=torch.float), requires_grad=False)
            self.mask_out = nn.Parameter(torch.tensor(mask, dtype=torch.float), requires_grad=False)

    def get_param(self, x, conditioning):
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * self.mask_out
        t = t * self.mask_out
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet

class HorizontalChain(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, hidden_channels, idx, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))
        split_h = h // 8
        idx = (idx + 1) % 8
        self.mask_in = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        self.mask_out = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        if idx != 7:
            self.mask_in[:, :, idx*split_h:(idx+1)*split_h, :] = 1
            self.mask_out[:, :, (idx+1)*split_h:(idx+2)*split_h, :] = 1
        else:
            self.mask_in[:, :, idx*split_h:(idx+1)*split_h, :] = 1
            self.mask_out[:, :, 0:split_h, :] = 1

    def get_param(self, x, conditioning):
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * self.mask_out
        t = t * self.mask_out
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet

class CycleMask(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, hidden_channels, idx, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))
        split_h = h // 2
        split_w = w // 2
        idx = (idx + 1) % 4

        self.mask_in = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        self.mask_out = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        if idx == 0:
            self.mask_in[:, :, :split_h, :split_w] = 1
            self.mask_out[:, :, :split_h, split_w:] = 1
        elif idx == 1:
            self.mask_in[:, :, :split_h, split_w:] = 1
            self.mask_out[:, :, split_h:, split_w:] = 1
        elif idx == 2:
            self.mask_in[:, :, split_h:, split_w:] = 1
            self.mask_out[:, :, split_h:, :split_w] = 1
        elif idx == 3:
            self.mask_in[:, :, split_h:, :split_w] = 1
            self.mask_out[:, :, :split_h, :split_w] = 1

    def get_param(self, x, conditioning):
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * self.mask_out
        t = t * self.mask_out
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet
    
class MyCheckerboard(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, hidden_channels, parity, conditional=True, net_type='ConvNet'):
        super().__init__()
        if conditional:
            if net_type == 'ConvNet':
                self.block = ConvNet(in_channels + int(in_channels/3), out_channels * 2, hidden_channels)
            elif net_type == 'UNet':
                self.block = Unet(in_channels + int(in_channels/3), out_channels * 2, hidden_channels)
            elif net_type == 'ResNet':
                self.block = ConvNet_with_Resnet(in_channels + int(in_channels/3), out_channels * 2, hidden_channels)
        else:
            if net_type == 'ConvNet':
                self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)
            elif net_type == 'UNet':
                self.block = Unet(in_channels, out_channels * 2, hidden_channels)
            elif net_type == 'ResNet':
                self.block = ConvNet_with_Resnet(in_channels, out_channels * 2, hidden_channels)
        self.mask = self.make_checker_mask((1, 1, H, W), parity)
    def make_checker_mask(self, shape, parity):
        checker = torch.ones(shape, dtype=torch.uint8) - parity
        checker[:, :, ::2, ::2] = parity
        checker[:, :, 1::2, 1::2] = parity
        return checker.to('cuda')
        
    def forward(self, x, logdet, conditioning=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, logdet, conditioning)
        else:
            return self.reverse_flow(x, logdet, conditioning)
        
    def normal_flow(self, x, logdet, conditioning):
        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x
        if conditioning is not None:
            z1_c = torch.cat([x_frozen, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(x_frozen)
        s, t = split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ + logdet
    
    def reverse_flow(self, fx, logdet, conditioning):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        if conditioning is not None:
            z1_c = torch.cat([fx_frozen, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(fx_frozen)
        s, t = split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask)*(-s), dim=tuple(axes))
        return x, logJ + logdet

class Checkerboard(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, hidden_channels, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        checkerboard = [[((i % 2) + j) % 2 for j in range(W)] for i in range(H)]
        self.mask = torch.tensor(checkerboard, requires_grad=False).view(1, 1, H, W)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_param(self, x, conditioning):
        self.mask = self.mask.to(x.get_device())
        z1 = x * self.mask
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x/exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z = (x + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        return z, logdet



class Checkerboard3D(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, hidden_channels, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        checkerboard = np.indices((in_channels, H, W)).sum(axis=0) % 2
        self.mask = torch.tensor(checkerboard, requires_grad=False).view(1, in_channels, H, W)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_param(self, x, conditioning):
        self.mask = self.mask.to(x.get_device())
        z1 = x * self.mask
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet + logdet
        else:
            z = (x + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        return z, logdet

class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x, reverse=False):
        if reverse:
            return x / self.weight  # Inverse scaling
        else:
            return x * self.weight  # Forward scaling