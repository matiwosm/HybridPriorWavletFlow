import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 
import torch
import torch.nn as nn
from HybridFlow.utilities import *
from HybridFlow.nf.coupling import *
from HybridFlow.nf.layers import *
import numpy as np

class FlowStep(nn.Module):
    def __init__(self, params, C, H, W, idx, conditional, device):
        super().__init__()
        hidden_channels = params.hiddenChannels
        self.flow_permutation = params.perm
        self.flow_coupling = params.coupling
        self.actnorm = ActNorm2d(C, params.actNormScale)

        # # 2. permute
        if self.flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(C, LU_decomposed=params.LU)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif self.flow_permutation == "shuffle":
            self.shuffle = Permute2d(C, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (self.shuffle(z, rev), logdet,)
        else:
            self.reverse = Permute2d(C, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (self.reverse(z, rev), logdet,)

        # 3. coupling
        if params.coupling == 'affine':
            self.coupling = Affine(C, C, hidden_channels, conditional)
        elif params.coupling == 'checker':
            self.coupling = MyCheckerboard(C, C, H, W, hidden_channels, idx%2, conditional, params.net_type, device=device)
        elif params.coupling == 'rqs':
            self.coupling = MyCheckerboardRQS(C, C, H, W, hidden_channels, idx%2, conditional, params.net_type)
        elif params.coupling == 'rqs_per_c':
            self.coupling = MyCheckerboardRQS_per_channel(C, C, H, W, hidden_channels, idx%2, conditional, params.net_type)
        elif params.coupling == 'fully_active_rqs':
            self.coupling = MyFullyActiveRQS(C, C, H, W, hidden_channels, conditional, params.net_type)
        elif params.coupling == 'checker3d':
            self.coupling = Checkerboard3D(C, C, H, W, hidden_channels, conditional)
        elif params.coupling == 'cycle':
            self.coupling = CycleMask(C, C, H, W, hidden_channels, idx, conditional)
        elif params.coupling == 'radial':
            self.coupling = RadialMask(C, C, H, W, hidden_channels, idx, conditional)
        elif params.coupling == 'horizontal':
            self.coupling = HorizontalChain(C, C, H, W, hidden_channels, idx, conditional)
    
    def forward(self, input, conditioning, logdet, reverse=False):
        if not reverse:
            return self.normal_flow(input, conditioning, logdet)
        else:
            return self.reverse_flow(input, conditioning, logdet)

    def normal_flow(self, input, conditioning, logdet):
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet)
        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)
        # 3. coupling
        z, logdet = self.coupling(z, logdet, conditioning)
        return z, logdet

    def reverse_flow(self, input, conditioning, logdet):

        # 1.coupling
        z, logdet = self.coupling(input, logdet, conditioning, reverse=True)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, params, shape, conditional, device):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = params.K
        self.L = params.L
        C, H, W = shape

        for idx in range(self.K):
            self.layers.append(FlowStep(params, C, H, W, idx, conditional, device))
            self.output_shapes.append([-1, C, H, W])

    def forward(self, input, conditioning, logdet=None, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, conditioning, temperature)
        else:
            return self.encode(input, conditioning, logdet)

    def encode(self, z, conditioning, logdet):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, SqueezeLayer):
                z, logdet = layer(z, logdet, reverse=False)
            elif isinstance(layer, FlowStep):
                z, logdet = layer(z, conditioning, logdet, reverse=False)
            else:
                z, logdet = layer(z, logdet, reverse=False)
            
        return z, logdet

    def decode(self, z, conditioning, logdet, temperature=None):
        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            if isinstance(layer, SqueezeLayer):
                z, logdet = layer(z, logdet=0, reverse=True)
            elif isinstance(layer, FlowStep):
                z, logdet = layer(z, conditioning, logdet=0, reverse=True)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z, logdet



class Glow(nn.Module):
    def __init__(self, params, shape, conditional, corr_prior, priortype, normalize=False, norm_constants=None, normalization_type='min_max', device='cpu'):
        super().__init__()
        self.flow = FlowNet(params, shape, conditional, device)
        self.y_classes = params.y_classes
        self.y_condition = params.y_condition
        self.learn_top = params.y_learn_top
        self.prior_dist = corr_prior
        self.prior_type = priortype
        self.normalize = normalize
        self.norm_constants = norm_constants
        self.normalization_type = normalization_type
        # print('Initilizing Glow Using ', self.prior_type, "prior")
        # learned prior
        if self.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if self.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(self.y_classes, 2 * C)
            self.project_class = LinearZeros(C, self.y_classes)

        self.register_buffer("prior_h", torch.zeros(
            [1, self.flow.output_shapes[-1][1] * 2, self.flow.output_shapes[-1][2],
                self.flow.output_shapes[-1][3], ]), )

    def forward(self, x=None, n_batch=64, conditioning=None, y_onehot=None, z=None, temperature=None, reverse=False, **kwargs):
        if reverse:
            return self.reverse_flow(x, n_batch, conditioning, y_onehot, temperature)
        else:
            return self.normal_flow(x, conditioning, y_onehot)
        
    def prior(self, data, n_batch=64, y_onehot=None):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(n_batch, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)
        return split_feature(h, "split")

    def apply_normalization_for_training(self, x, conditioning, norm_constants, normalization_type):
        """
        Normalizes x and/or the conditioning. 
        If conditioning is not None, it is treated as the low-frequency component (low_freq_x),
        and x contains the high-frequency components that also need normalization.
        If conditioning is None, x is assumed to be low-frequency data only.
        """

        # Decide which data is considered "low-frequency"
        if conditioning is not None:
            low_freq_x = conditioning
        else:
            low_freq_x = x

        # 1) Normalize low-frequency components
        normalized_low_freq = torch.empty_like(low_freq_x)
        N_channels = low_freq_x.shape[1]
        for ch in range(N_channels):
            if normalization_type == 'min_max':
                const = max(
                    -1 * norm_constants['low']['min'][ch],
                    norm_constants['low']['max'][ch]
                )
            elif normalization_type == 'std':
                const = norm_constants['low']['mean_std'][ch]
            else:
                raise ValueError("normalization_type must be 'min_max' or 'std'.")

            normalized_low_freq[:, ch, :, :] = low_freq_x[:, ch, :, :] / const

        # 2) If we have conditioning, then x is actually the high-frequency part
        if conditioning is not None:
            N_channels = x.shape[1] // 3
            n_high_types = 3  # e.g. ['high_horizontal', 'high_vertical', 'high_diagonal']
            high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
            normalized_high_freq = torch.empty_like(x)
            for ch in range(N_channels):
                for ht_idx, ht in enumerate(high_types):
                    idx = ch * n_high_types + ht_idx
                    if normalization_type == 'min_max':
                        const = max(
                            -1 * norm_constants[ht]['min'][ch],
                            norm_constants[ht]['max'][ch]
                        )
                    elif normalization_type == 'std':
                        const = norm_constants[ht]['mean_std'][ch]
                    else:
                        raise ValueError("normalization_type must be 'min_max' or 'std'.")

                    normalized_high_freq[:, idx, :, :] = x[:, idx, :, :] / const

            # Return (high-frequency, low-frequency) if conditioning was given
            return normalized_high_freq, normalized_low_freq
        else:
            # Return (low-frequency, None) if conditioning was None
            return normalized_low_freq, conditioning


    def apply_normalization_before_sampling(self, conditioning, norm_constants, normalization_type):
        """
        Normalizes the conditioning before sampling if it is not None.
        This is typically the low-frequency component.
        """

        if conditioning is not None:
            # Low-frequency components
            N_channels = conditioning.shape[1]
            normalized_data = torch.empty_like(conditioning)
            for ch in range(N_channels):
                if normalization_type == 'min_max':
                    const = max(
                        -1 * norm_constants['low']['min'][ch],
                        norm_constants['low']['max'][ch]
                    )
                elif normalization_type == 'std':
                    const = norm_constants['low']['mean_std'][ch]
                else:
                    raise ValueError("normalization_type must be 'min_max' or 'std'.")

                normalized_data[:, ch, :, :] = conditioning[:, ch, :, :] / const

            return normalized_data
        else:
            return conditioning


    def unnormalization_after_sampling(self, x, conditioning, norm_constants, normalization_type):
        """
        Unnormalizes the sampled x. 
        If conditioning is None, then x is assumed to be the low-frequency component.
        Otherwise, x is the high-frequency component.
        """

        if conditioning is None:
            # Unnormalize low-frequency component
            N_channels = x.shape[1]
            unnormalized_data = torch.empty_like(x)
            for ch in range(N_channels):
                if normalization_type == 'min_max':
                    const = max(
                        -1 * norm_constants['low']['min'][ch],
                        norm_constants['low']['max'][ch]
                    )
                elif normalization_type == 'std':
                    const = norm_constants['low']['mean_std'][ch]
                else:
                    raise ValueError("normalization_type must be 'min_max' or 'std'.")

                unnormalized_data[:, ch, :, :] = x[:, ch, :, :] * const

        else:
            # Unnormalize high-frequency components
            N_channels = x.shape[1] // 3
            n_high_types = 3  # ['high_horizontal', 'high_vertical', 'high_diagonal']
            high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
            unnormalized_data = torch.empty_like(x)
            for ch in range(N_channels):
                for ht_idx, ht in enumerate(high_types):
                    idx = ch * n_high_types + ht_idx
                    if normalization_type == 'min_max':
                        const = max(
                            -1 * norm_constants[ht]['min'][ch],
                            norm_constants[ht]['max'][ch]
                        )
                    elif normalization_type == 'std':
                        const = norm_constants[ht]['mean_std'][ch]
                    else:
                        raise ValueError("normalization_type must be 'min_max' or 'std'.")

                    unnormalized_data[:, idx, :, :] = x[:, idx, :, :] * const

        return unnormalized_data


    def normal_flow(self, x, conditioning, y_onehot):
        if self.normalize:
            x, conditioning = self.apply_normalization_for_training(x, conditioning, self.norm_constants, self.normalization_type)

        b, c, h, w = x.shape

        # x, logdet = uniform_binning_correction(x)
        logdet = torch.zeros_like(x)[:, 0, 0, 0]
        z, objective = self.flow(x, conditioning, logdet=logdet, reverse=False)
        objective += self.prior_dist.log_prob(z) 

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)
        return {"latent": z, "likelihood": bpd, "y_logits": y_logits}

    def reverse_flow(self, z, n_batch, conditioning, y_onehot, temperature):
        # unnormalized_conditioning = conditioning
        if self.normalize:
            conditioning = self.apply_normalization_before_sampling(conditioning, self.norm_constants, self.normalization_type)

        with torch.no_grad():
            if z is None:
                z = self.prior_dist.sample_n(n_batch).type(torch.float64)
            logdet = torch.zeros(z.shape[0]).to(z.device)
            x, logdet = self.flow(z, conditioning, logdet=logdet, temperature=temperature, reverse=True)
            if self.normalize:
                x = self.unnormalization_after_sampling(x, conditioning, self.norm_constants, self.normalization_type)

            # z1, obj = self.flow(x, unnormalized_conditioning, logdet=logdet, temperature=temperature)
            # if self.normalize:
            #     z1 = self.unnormalization_after_sampling(z1, conditioning, self.norm_constants, self.normalization_type)
            #     print('unnormalizing', z1.shape)
            # print('x = ', x.shape)
            # print(unnormalized_conditioning.shape if unnormalized_conditioning is not None else unnormalized_conditioning)
            # print('z, z1, reverseflow', torch.allclose(z1, z, atol=1e-5))
            # print('end of reverse flow')
        return x, logdet
    
    def sample_latents(self, n_batch=64,temperature=1.0):
        z = self.prior_dist.sample_n(n_batch).type(torch.float64)
        return z
    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
