from importlib import machinery
import mailcap
from operator import inv
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
class Dwt(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        self.kernel = None
        self.inv_kernel = None
        self.f = self.wavelet.factor
        self.m = self.wavelet.multiplier
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)  # 1 level DWT using Haar
        self.idwt = DWTInverse(wave='haar', mode='zero').to(device)
        # print('device', device)
    def forward(self, x):
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        assert H % H_w == 0 and W % W_w == 0, '({},{}) not dividing by {} nicely'.format(H, W, self.f)
        high_freq_list = []
        low_freq, high_freq = self.dwt(x)
        high_freq = high_freq[0]
        high_freq = torch.cat([high_freq[:, :, 1:2:, :], high_freq[:, :, 0:1:, :],
                              high_freq[:, :, 2:3:, :]], dim=2)
        # print('in forward', low_freq.shape, high_freq.shape)
        # high_freq = high_freq.transpose(1, 2)
        
#         for i in range(high_freq.shape[2]):
#             high_freq_list.append(high_freq[:, :, i, :, :])
            
#         high_freq = torch.cat(high_freq_list, dim=1)
        # Flatten the directional dimensions into channel dimension
        high_freq = high_freq.reshape(low_freq.shape[0], -1, low_freq.shape[2], low_freq.shape[3])

        # print(low_freq.shape, high_freq.shape)
        components = {"low": low_freq, "high": high_freq}
        # print('in forward', low_freq.shape, high_freq.shape)

        return components

    def inverse(self, components):
        low = components['low']
        high = components['high']
        C = low.shape[1]
        # high_freq = torch.cat([high[:, :, 1:2:, :], high_freq[0][:, :, 0:1:, :],
        #                       high_freq[0][:, :, 2:3:, :]], dim=2)
        
        # Reshape back into [B, C, 3, W, H]
        high = high.reshape(low.shape[0], C, 3, low.shape[2], low.shape[3])
        
        # Now, permute back to the original form expected by IDWT
        # high = high.permute(0, 2, 1, 3, 4)  # revert the earlier permute to [B, C, 3, W, H]
        high = torch.cat([high[:, :, 1:2:, :], high[:, :, 0:1:, :],
                              high[:, :, 2:3:, :]], dim=2)
        components = (low, [high])
        # print('in inverse', low.shape, high.shape)
        x_reconstructed = self.idwt(components)
        

        return x_reconstructed
    
    
    


class Dwt_custom(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        self.kernel = None
        self.inv_kernel = None
        self.f = self.wavelet.factor
        self.m = self.wavelet.multiplier

    def forward(self, x):
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        high_freq = []
        low_freq = []
        ldj = 0

        assert H % H_w == 0 and W % W_w == 0, '({},{}) not dividing by {} nicely'.format(H, W, self.f)
        forward_kernel = self.make_forward_kernel(C).to(x.get_device())
        y = nn.functional.conv2d(x, forward_kernel, None, (2,2), 'valid')
        for i in range(C):
            low_freq.append(y[:, i*self.m:i*self.m+1, : ,:])
            high_freq.append(y[:, i*self.m+1:i*self.m+self.m, : ,:])
        
        high_freq = torch.cat(high_freq, dim=1)
        low_freq = torch.cat(low_freq, dim=1)
        # print('low', low_freq.shape, 'high', high_freq.shape)
        components = {"low": low_freq, "high": high_freq}
          
        return components

    def make_forward_kernel(self, C):
        if self.kernel is not None:
            return self.kernel
        
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        k = self.wavelet.kernels

        kernel = torch.zeros((C*self.m, C, H_w, W_w))
        
        for i in range(C):
            for j in range(self.m):
                kernel[i*self.m+j, i, :, :] = torch.tensor(k[j])
        
        self.kernel = kernel
        return kernel

    
    def make_inverse_kernel(self, C):
        inv_kernel = torch.zeros((C, 2 * C, self.wavelet.h, self.wavelet.w), dtype=torch.float32)
        for i in range(C):
            inv_kernel[i, i, :, :] = torch.tensor(self.wavelet.kernels[0], dtype=torch.float32).flip([0, 1]) * 4
            inv_kernel[i, C + i, :, :] = torch.tensor(self.wavelet.kernels[1], dtype=torch.float32).flip([0, 1]) * 4

        self.inv_kernel = inv_kernel
        return inv_kernel

    def inverse(self, components):
        C = components['low'].shape[1]
        inverse_kernel = self.make_inverse_kernel(C).to(components['low'].device)
        print("Inverse Kernel Shape:", inverse_kernel.shape)
        combined = torch.cat((components['low'], components['high']), dim=1)
        print("Combined Shape:", combined.shape)
        x_reconstructed = nn.functional.conv_transpose2d(
            combined, inverse_kernel, None, stride=(2, 2), padding=0, output_padding=0)
        print("Reconstructed Shape:", x_reconstructed.shape)
        return x_reconstructed
    
    
    