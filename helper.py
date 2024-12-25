import math
import numpy as np
import os

class utils():
    def __init__(self):
        self.dummy_x = 0
    def get_cdf(self, data):
        data = np.array(data).flatten()
        y = np.arange(1, len(data)+1)/len(data)
        x = np.sort(data)
        return x, y

    def torch_mod(self, x):
        return torch.remainder(x, 2*np.pi)
    def torch_wrap(self, x):
        return torch_mod(x+np.pi) - np.pi
    def grab(self, var):
        return var.detach().cpu().numpy()
    def get_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))
    
    def get_2d_power(self, nx, dx, r1, r2=None, num_bins=100, scale='linear'):
        if (np.any(np.isnan(r1)) or np.any(np.isinf(r1))):
            print("whyyyyyy", r1.shape)
            
        if (r2 is None):
            r2 = r1
        lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,
                                np.fft.fftfreq( nx, dx )*2.*np.pi )
        ell = np.sqrt(lx**2 + ly**2)
        ell = ell[~np.isnan(ell)]
        lbins = np.sort(np.unique(ell.flatten())) 
        if scale == 'linear':
             lbins = np.linspace(np.min(ell), np.max(ell), num_bins)
        elif scale == 'log':
            lbins = np.logspace(np.log10(np.min(ell[np.where(ell > 0)])), np.log10(np.max(ell)), num=num_bins)
            # print(lbins, np.min(ell[np.where(ell > 0)]))
        else:
            print('scale should be linear or log')
            exit(0)

        # print('lbins', lbins, np.min(lbins), np.max(lbins))
        wvec = np.ones(ell.flatten().shape)

        tfac = 1 #np.sqrt((dx * dx) / (nx * nx))
        FMap1 = np.fft.rfft2(r1) * tfac

       
        FMap2 = np.fft.rfft2(r2) * tfac
        # FMap1 = np.fft.ifft2(np.fft.fftshift(r1))
        cvec = (np.conj(FMap1) * (FMap2)).flatten()

        wvec[ np.isnan(cvec) ] = 0.0
        cvec[ np.isnan(cvec) ] = 0.0

        norm, bins = np.histogram(ell, bins=lbins, weights=wvec); norm[ np.where(norm != 0.0) ] = 1./norm[ np.where(norm != 0.0) ]
        clrr, bins = np.histogram(ell, bins=lbins, weights=cvec*wvec); clrr *= norm
        # print('clrr', clrr)
        return bins, clrr
    
    #power spectrum with no binning
    def get_2d_power_prior(self, nx, dx, r1, r2=None, num_bins=100):
        if (np.any(np.isnan(r1)) or np.any(np.isinf(r1))):
            print("whyyyyyy", r1.shape)
            
        if (r2 is None):
            r2 = r1
        lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,
                                np.fft.fftfreq( nx, dx )*2.*np.pi )
        ell = np.sqrt(lx**2 + ly**2)
        ell = ell[~np.isnan(ell)]
        lbins = np.sort(np.unique(ell.flatten())) 


        wvec = np.ones(ell.flatten().shape)

        tfac = 1 #np.sqrt((dx * dx) / (nx * nx))
        FMap1 = np.fft.rfft2(r1) * tfac

       
        FMap2 = np.fft.rfft2(r2) * tfac
        # FMap1 = np.fft.ifft2(np.fft.fftshift(r1))
        cvec = (np.conj(FMap1) * (FMap2)).flatten()

        wvec[ np.isnan(cvec) ] = 0.0
        cvec[ np.isnan(cvec) ] = 0.0

        return lbins, cvec
    