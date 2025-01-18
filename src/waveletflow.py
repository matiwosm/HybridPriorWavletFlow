import torch.nn as nn
from src.dwt.dwt import Dwt
from src.dwt.wavelets import Haar
from src.nf.glow import Glow
import math
import torch
import numpy as np
from utilities import normalize_dwt_components, unnormalize_dwt_components
from corr_prior import *
class WaveletFlow(nn.Module):
    def __init__(self, cf, cond_net, partial_level=-1, prior=None, stds=None, priortype='WN'):
        super().__init__()
        self.n_levels = cf.nLevels
        self.base_level = cf.baseLevel
        self.partial_level = partial_level
        self.wavelet = Haar()
        self.dwt = Dwt(wavelet=self.wavelet)
        self.conditioning_network = cond_net
        self.stds = stds #perlevel for training
        self.normalize = cf.normalize[partial_level]
        self.normalization_type = cf.norm_type[partial_level]
        self.prior_type = priortype
        if partial_level == -1 or partial_level == self.base_level:
            base_size = 2 ** self.base_level
            cf.K = cf.stepsPerResolution[partial_level - 1]
            cf.L = cf.stepsPerResolution_L[partial_level - 1]
            shape = (cf.imShape[0], base_size, base_size)
            self.base_flow = Glow(cf, shape, False, prior, self.prior_type, self.normalize, self.stds, self.normalization_type)
        else:
            self.base_flow = None
        
        start_flow_padding = [None] * self.base_level
        self.sub_flows = start_flow_padding + [self.base_flow]
        
        for level in range(self.base_level + 1, self.n_levels + 1):
            if partial_level != -1 and partial_level != level:
                self.sub_flows.append(None)
            else:
                h = 2**(level-1)
                w = 2**(level-1)
                cf.K = cf.stepsPerResolution[level-1]
                cf.L = cf.stepsPerResolution_L[level-1]
                shape = (cf.imShape[0] * 3, h, w)
                self.sub_flows.append(Glow(cf, shape, cf.conditional, prior, self.prior_type, self.normalize, self.stds, self.normalization_type))
        self.sub_flows = nn.ModuleList(self.sub_flows)

    def forward(self, x, std=None, partial_level=-1):

        latents = []
        low_freq = x 
        for level in range(self.n_levels, self.base_level-1, -1):
            if level == partial_level or partial_level == -1:
                if level == self.base_level:
                    flow = self.base_flow
                    conditioning = None
                    # self.stds = std[str(5 - int(np.log2(dwt_components['low'].shape[-1])))]
                    low_freq = dwt_components['low']
                    res = flow.forward(low_freq, conditioning=conditioning)
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                    conditioning = self.conditioning_network.encoder_list[level](low_freq)
                    flow = self.sub_flows[level]
                    high_freq = dwt_components['high']
                    res = flow.forward(high_freq, conditioning=conditioning)
                latents.append(res["latent"])
                b, c, h, w = low_freq.shape
                res["likelihood"] -= (c * h * w * torch.log(torch.tensor(0.5)) * (self.n_levels - level)) /  (math.log(2.0) * c * h * w)
                x = torch.abs(dwt_components['high'])
                if partial_level != -1:
                    break 
            else:
                if self.partial_level <= 8 and level > 8:
                    pass
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                latents.append(None)

        return {"latent":latents, "likelihood":res["likelihood"]}

    
    def sample_latents(self,n_batch=64,temperature=1.0):
        latents = [self.base_flow.sample_latents(n_batch=n_batch,temperature=temperature)]
        for level in range(1,self.n_levels+1):
            flow = self.sub_flows[level]
            if flow == None:
                latents.append(None)
            else:
                latents.append(flow.sample_latents(n_batch=n_batch,temperature=temperature))
        return latents
    
    def sample(self, mean_stds_all_levels, target=None, latents=None, partial_level=-1, comp='low', cond_on_target=False, n_batch=64):
        return self.sample_unnorm(target, latents, partial_level, comp, cond_on_target, n_batch)
        
    def sample_norm(self, mean_stds_all_levels, target=None, latents=None, partial_level=-1, comp='low', cond_on_target=False, n_batch=64):
        
        # If target is provided, we condition on it by performing a DWT decomposition
        data = []
        if target is not None:
            data.append(target)
            # Decompose target down to the base level for conditioning if needed
            for i in range(self.base_level+1, self.n_levels+1):
                target = self.dwt.forward(target)['low']
                data.append(target)
            data.reverse()

        if latents is None:
            latents = self.sample_latents(n_batch=n_batch, temperature=1.0)
        # print(latents[0].shape, latents[0][:2])
        samples = []
        samples_high_freq = []  # To store unnormalized high-frequency components

        # Base level reconstruction
        base_norm = self.base_flow.forward(latents[0], conditioning=None, temperature=1.0, reverse=True)[0]
        std = mean_stds_all_levels[str(5 - int(np.log2(base_norm.shape[-1])))]

        if cond_on_target:
            # If conditioning on target, overwrite base_norm with normalized target low band
            base_norm = normalize_dwt_components(data[0], std, 'low')

        # Unnormalize base
        base = unnormalize_dwt_components(base_norm, std, 'low')
        # print(base[:2], base_norm[:2], base.shape, base_norm.shape)
        samples.append(base)  # Append the base level reconstruction
        samples_high_freq.append(base)

        # Reconstruct upwards from base_level+1 to n_levels
        final_level = self.base_level if partial_level == -1 else partial_level
        for level in range(self.base_level+1, self.n_levels+1):
            flow = self.sub_flows[level]
            # print(std, level, self.base_level+1, self.n_levels+1)
            if flow is not None:
               
                base_norm = normalize_dwt_components(base, std, 'low')
                # print('base norm', base_norm[0])
                conditioning = self.conditioning_network.encoder_list[level](base_norm)

                latent = latents[level]  # Align indexing as per old code
                x_norm = flow.forward(latent, conditioning=conditioning, temperature=1.0, reverse=True)[0]

                # Unnormalize the 'high' frequency component
                x_unnorm_high = unnormalize_dwt_components(x_norm, std, 'high')
                samples_high_freq.append(x_unnorm_high.clone())
                # if level == 3:
                #     print('level 3', x_unnorm_high.shape, std['low']['mean_std'])
                
                # Inverse DWT to merge high and low frequencies
                x = self.dwt.inverse({'low': base, 'high': x_unnorm_high})
                if level < self.n_levels:
                    std = mean_stds_all_levels[str(5-level)]
                # If conditioning on target at lower levels:
                if cond_on_target and level < 5:
                    # Use the target's data instead of the reconstructed base if desired
                    x = data[level - self.base_level]
                # print('flow not none', x.shape)
                base = x  # Update base for next iteration
                samples.append(x)  # Append reconstruction at this level

            else:
                # If no flow at this level, just append the current base if desired
                print('wrong place to be')
                break
                samples.append(base)
        if comp == 'low':
            return samples
        else:
            return samples_high_freq

    def sample_unnorm(self, target=None, latents=None, partial_level=-1, comp='low', cond_on_target=False, n_batch=64):
        
        # If target is provided, we condition on it by performing a DWT decomposition
        data = []
        if target is not None:
            data.append(target)
            # Decompose target down to the base level for conditioning if needed
            for i in range(self.base_level+1, self.n_levels+1):
                target = self.dwt.forward(target)['low']
                data.append(target)
            data.reverse()

        if latents is None:
            latents = self.sample_latents(n_batch=n_batch, temperature=1.0)

        samples = []
        samples_high_freq = []  # To store unnormalized high-frequency components

        # Base level reconstruction
        base = self.base_flow.forward(latents[0], conditioning=None, temperature=1.0, reverse=True)[0]

        if cond_on_target:
            # If conditioning on target, overwrite base_norm with normalized target low band
            base = data[0]
        samples.append(base)  # Append the base level reconstruction
        samples_high_freq.append(base)

        # Reconstruct upwards from base_level+1 to n_levels
        final_level = self.base_level if partial_level == -1 else partial_level
        for level in range(self.base_level+1, self.n_levels+1):
            flow = self.sub_flows[level]
            # print(std, level, self.base_level+1, self.n_levels+1)
            if flow is not None:
                conditioning = self.conditioning_network.encoder_list[level](base)

                latent = latents[level]  # Align indexing as per old code
                x_unnorm = flow.forward(latent, conditioning=conditioning, temperature=1.0, reverse=True)[0]

                samples_high_freq.append(x_unnorm.clone())
                
                # Inverse DWT to merge high and low frequencies
                x = self.dwt.inverse({'low': base, 'high': x_unnorm})

                # If conditioning on target at lower levels:
                if cond_on_target and level < 5:
                    # Use the target's data instead of the reconstructed base if desired
                    x = data[level - self.base_level]
                # print('flow not none', x.shape)
                base = x  # Update base for next iteration
                samples.append(x)  # Append reconstruction at this level

            else:
                # If no flow at this level, just append the current base if desired
                print('wrong place to be')
                break
                samples.append(base)
        if comp == 'low':
            return samples
        else:
            return samples_high_freq
