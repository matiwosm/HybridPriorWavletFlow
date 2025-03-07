import torch.nn as nn
from src.dwt.dwt import Dwt
from src.dwt.wavelets import Haar
from src.nf.glow import Glow
import math
import torch
import numpy as np

class WaveletFlow(nn.Module):
    def __init__(self, cf, cond_net, partial_level=-1):
        super().__init__()
        self.n_levels = cf.nLevels
        self.base_level = cf.baseLevel
        self.partial_level = partial_level
        self.wavelet = Haar()
        self.dwt = Dwt(wavelet=self.wavelet)
        self.conditioning_network = cond_net
        
        if partial_level == -1 or partial_level == self.base_level:
            base_size = 2 ** self.base_level
            cf.K = cf.stepsPerResolution[partial_level]
            cf.L = cf.stepsPerResolution_L[partial_level]
            shape = (cf.imShape[0], base_size, base_size)
            self.base_flow = Glow(cf, shape, False)
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
                self.sub_flows.append(Glow(cf, shape, cf.conditional))

        self.sub_flows = nn.ModuleList(self.sub_flows)
        # print(self.sub_flows)
    def forward(self, x, partial_level=-1):

        latents = []
        low_freq = x 
        # print('before everything', x.shape, 'partial_level', partial_level, '\n')
        for level in range(self.n_levels, self.base_level-1, -1):
            # print('level', level, '\n')
            if level == partial_level or partial_level == -1:
                if level == self.base_level:
                    flow = self.base_flow
                    conditioning = None
                    res = flow.forward(dwt_components['low'], conditioning=conditioning)
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                    conditioning = self.conditioning_network.encoder_list[level](low_freq)
                    flow = self.sub_flows[level]
                    res = flow.forward(dwt_components['high'], conditioning=conditioning)
                # print('res', res['latent'].shape, '\n')
                latents.append(res["latent"])
                b, c, h, w = low_freq.shape
                res["likelihood"] -= (c * h * w * torch.log(torch.tensor(0.5)) * (self.n_levels - level)) /  (math.log(2.0) * c * h * w)
                x = torch.abs(dwt_components['high'])
                if partial_level != -1:
                    break 
            
            else:
                # print('in else level = ', level)
                if self.partial_level <= 8 and level > 8:
                    # print('in else if level = ', level)
                    pass
                else:
                    # print('in else else level = ', level)
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                latents.append(None)

        return {"latent":latents, "likelihood":res["likelihood"]}
    
    def sample_latents(self,n_batch=1,temperature=1.0):
        latents = [None]*self.base_level

        for flow in self.sub_flows:
            if flow == None:
                latents.append(None)
            else:
                latents.append(flow.sample_latents(n_batch=n_batch,temperature=temperature))
        return latents
    
    def sample(self, partial_level=-1):
        samples = []
        latents = self.sample_latents(n_batch=32,temperature=1.0)
        base = self.base_flow.forward(latents[0], conditioning=None,temperature=1.0, reverse=True)[0]
        print('base = ', base.shape)
        # print(latents[0])
        level = self.base_level if partial_level == -1 else partial_level
        # print(latents)
        for level in range(self.base_level+1,self.n_levels+1):
            latent = latents[level]
            flow = self.sub_flows[level]
            if flow is not None:
                # print(level, 'latent', latent.shape)
                conditioning = self.conditioning_network.encoder_list[level](base)
                # print(conditioning.device, latent.device)
                x = flow.forward(latent, conditioning=conditioning,temperature=1.0, reverse=True)
                print('shape', x[0].shape, x[1].shape, len(x))
                if level > self.base_level:
                    print('before dwt', base.shape, x[0].shape)
                    x = self.dwt.inverse({'low': base, 'high': x[0]})
                print('after x shape', x.shape)
                base = x
            samples.append(x)
        return samples

