import base64
import io
import time
import pickle
import math
import numpy as np
import pylab as pl
import torch
import json
import pandas as pd
torch.manual_seed(0)

import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
  raise RuntimeError('Torch versions lower than 1.5.0 not supported')

class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(loc, var)
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x)
        return torch.sum(logp, dim=[1, 2, 3])
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)
    
class CorrelatedNormal_dwt:
    def __init__(self, loc, var,nx, dx,cl_theo,torch_device, freq='high', prior_type='CC'):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        self.freq = freq
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        self.level = int(np.log2(nx))
        print('level = ', self.level)
        #create the array to multiply the fft with to get the desired power spectrum
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten().astype(np.float32)
        
        ell = self.get_ell(self.nx, self.dx)
        
        self.lbins1 = cl_theo[:, 0]
        self.l_comp_kap = cl_theo[:, 1]
        self.l_comp_cib = cl_theo[:, 2]
        self.h_comp_kap1 = cl_theo[:, 3]
        self.h_comp_kap2 = cl_theo[:, 4]
        self.h_comp_kap3 = cl_theo[:, 5]
        self.h_comp_cib1 = cl_theo[:, 6]
        self.h_comp_cib2 = cl_theo[:, 7]
        self.h_comp_cib3 = cl_theo[:, 8]
        if prior_type == 'CC':
            self.l_comp_kap_cib = cl_theo[:, 9]
            self.h_comp1_kap_cib = cl_theo[:, 10]
            self.h_comp2_kap_cib = cl_theo[:, 11]
            self.h_comp3_kap_cib = cl_theo[:, 12]
            
        
        # Reshape the components to match the required shape (H, W)
        l_comp1 = self.l_comp_kap.reshape(self.rfourier_shape[1:3])   # Shape: (H, W)
        l_comp2 = self.l_comp_cib.reshape(self.rfourier_shape[1:3])   # Shape: (H, W)

        h_comp1 = self.h_comp_kap1.reshape(self.rfourier_shape[1:3])  # Shape: (H, W)
        h_comp2 = self.h_comp_kap2.reshape(self.rfourier_shape[1:3])
        h_comp3 = self.h_comp_kap3.reshape(self.rfourier_shape[1:3])

        h_comp4 = self.h_comp_cib1.reshape(self.rfourier_shape[1:3])
        h_comp5 = self.h_comp_cib2.reshape(self.rfourier_shape[1:3])
        h_comp6 = self.h_comp_cib3.reshape(self.rfourier_shape[1:3])

        if prior_type == 'CC':
            l_comp_kap_cib = self.l_comp_kap_cib.reshape(self.rfourier_shape[1:3])
            h_comp1_kap_cib = self.h_comp1_kap_cib.reshape(self.rfourier_shape[1:3])
            h_comp2_kap_cib = self.h_comp2_kap_cib.reshape(self.rfourier_shape[1:3])
            h_comp3_kap_cib = self.h_comp3_kap_cib.reshape(self.rfourier_shape[1:3])
        elif prior_type == 'C':
            zero_array = np.zeros_like(l_comp1)
            l_comp_kap_cib = zero_array
            h_comp1_kap_cib = zero_array
            h_comp2_kap_cib = zero_array
            h_comp3_kap_cib = zero_array
        else:
            raise ValueError("Invalid prior type")
        
        if freq == 'high':
            H, W = h_comp1.shape  # Spatial dimensions
            # Initialize a 4D covariance matrix: (6 components, 6 components, H, W)
            clfactor = np.zeros((6, 6, H, W), dtype=np.complex64)

            # Set diagonal elements (auto-spectra)
            clfactor[0, 0, :, :] = h_comp1
            clfactor[1, 1, :, :] = h_comp2
            clfactor[2, 2, :, :] = h_comp3
            clfactor[3, 3, :, :] = h_comp4
            clfactor[4, 4, :, :] = h_comp5
            clfactor[5, 5, :, :] = h_comp6

            # Set cross-correlation terms between corresponding components
            clfactor[0, 3, :, :] = h_comp1_kap_cib
            clfactor[3, 0, :, :] = h_comp1_kap_cib  # Symmetric element

            clfactor[1, 4, :, :] = h_comp2_kap_cib
            clfactor[4, 1, :, :] = h_comp2_kap_cib  # Symmetric element

            clfactor[2, 5, :, :] = h_comp3_kap_cib
            clfactor[5, 2, :, :] = h_comp3_kap_cib  # Symmetric element

            # All other elements remain zero
        else:
            H, W = l_comp1.shape  # Spatial dimensions
            # Initialize a 4D covariance matrix for low-frequency components
            clfactor = np.zeros((2, 2, H, W), dtype=np.complex64)
            clfactor[0, 0, :, :] = l_comp1
            clfactor[1, 1, :, :] = l_comp2
            if prior_type == 'CC':
                clfactor[0, 1, :, :] = l_comp_kap_cib
                clfactor[1, 0, :, :] = l_comp_kap_cib
            else:
                clfactor[0, 1, :, :] = zero_array
                clfactor[1, 0, :, :] = zero_array

        # Now you can convert clfactor to a torch tensor
        self.cov = torch.from_numpy(clfactor).to(torch_device)
        self.inv_cov = torch.zeros_like(self.cov).to(torch_device)

        
        self.det = torch.zeros((self.cov.shape[-2], self.cov.shape[-1])).to(torch_device)
        for j in range(self.cov.shape[-2]):
            for k in range(self.cov.shape[-1]):
                self.inv_cov[:, :, j, k] = torch.linalg.inv(self.cov[:, :, j, k])
                self.det[j, k] = torch.linalg.det(self.cov[:, :, j, k])      
        self.clfactor = np.copy(clfactor)
        self.clfactor = self._cholesky(self.clfactor)
        self.clfactor = (torch.from_numpy(self.clfactor).float().to(torch_device))
        #masks for rfft symmetries
        a_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)
        a_mask[int(self.nx/2+1):, 0] = False
        a_mask[int(self.nx/2+1):, int(nx/2)] = False
        b_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)    
        b_mask[0,0] = False
        b_mask[0,int(self.nx/2)] = False
        b_mask[int(self.nx/2),0] = False
        b_mask[int(self.nx/2),int(self.nx/2)] = False
        b_mask[int(self.nx/2+1):, 0] = False
        b_mask[int(self.nx/2+1):, int(self.nx/2)] = False
        self.a_mask = a_mask
        self.b_mask = b_mask
        
        #how many mask elements
        a_nr = self.a_mask.sum()
        b_nr = self.b_mask.sum()

        #make distributions with the right number of elements for each re and im mode.
        a_shape = (a_nr)
        loc = torch.zeros(a_shape)
        var = torch.ones(a_shape)
        self.a_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.a_dist = self.a_dist.expand(torch.flatten(loc).shape)
        b_shape = (b_nr)
        loc = torch.zeros(b_shape)
        var = torch.ones(b_shape)
        self.b_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.b_dist = self.b_dist.expand(torch.flatten(loc).shape)
        #estimate scalar fudge factor to make unit variance.
        print('estimating fudge factor')
        s = self.sample_n(1000)
        for i in range(s.shape[1]):
            print('std', s.shape, torch.std(s[:, i, :, :]))
        
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi ) 

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)    
        
    def log_prob(self, x):
        fft1 = torch.fft.rfftn(x,dim=[-2,-1]) * np.sqrt(2.)
        
        x = torch.view_as_real(fft1)

        #correct: use symmetries
        a = x[:, :, :, :, 0]
        b = x[:, :, :, :, 1]
        
 
        clinvfactor = self.inv_cov.type(torch.float64)
        logp_real = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 0], clinvfactor), (x[:, :, :, :, 0])) 
        
        logp_imaj = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 1], clinvfactor), (x[:, :, :, :, 1]))
        # print(clinvfactor.type())
        # print(a.type())
        # print(b.type())
        # print(x[torch.isnan(x)].shape)
        # print(x.shape, clinvfactor.shape, a.shape, b.shape)
        logp_real = logp_real[:, self.a_mask]
        logp_imaj = logp_imaj[:, self.b_mask]
        
        logp = torch.sum(logp_imaj, dim=[-1]) + torch.sum(logp_real, dim=[-1])
        # print(logp_real.type(), logp.type(), logp_real.shape, logp)
        return logp
    
    def sample_n(self, batch_size):
        #https://pytorch.org/docs/stable/complex_numbers.html
        
        #draw random rfft modes
        x = self.dist.sample((batch_size,)).to(self.torch_device)
        
        #test logp
        #logptemp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        #print("logp temp", torch.sum(logptemp, dim=1))
        
        #reshape to rfft format
        x = x.reshape(batch_size, *self.rfourier_shape)
        #make complex data type
        fft = torch.view_as_complex(x)/ np.sqrt(2.)
         
        #enforce rfft constraints
        #from quicklens
        fft[:,:,0,0] = np.sqrt(2.) * fft[:,:,0,0].real #fft.real
        fft[:,:,int(self.nx/2+1):, 0] = torch.conj( torch.flip(fft[:,:,1:int(self.nx/2),0], (2,)) ) 
        
        #extra symmetries (assuming th rfft output format is as in numpy)
        fft[:,:,0,int(self.nx/2)] = fft[:,:,0,int(self.nx/2)].real * np.sqrt(2.)
        fft[:,:,int(self.nx/2),0] = fft[:,:,int(self.nx/2),0].real * np.sqrt(2.)
        fft[:,:,int(self.nx/2),int(self.nx/2)] = fft[:,:,int(self.nx/2),int(self.nx/2)].real * np.sqrt(2.)
        fft[:,:,int(self.nx/2+1):, int(self.nx/2)] = torch.conj( torch.flip(fft[:,:,1:int(self.nx/2),int(self.nx/2)], (2,)) ) 
        
        #flip from https://github.com/pytorch/pytorch/issues/229
        #https://pytorch.org/docs/stable/generated/torch.flip.html#torch.flip
        
        #TODO: check normalization of irfftn. see quicklens maps.py line 907 and the new options of irfftn.
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        #for now scalar fudge factor to make unit variance.
        # print(self.clfactor.type(torch.complex64).device, fft.type(torch.complex64).device)
        fft = torch.einsum('nplm,knlm->kplm', self.clfactor.type(torch.complex64), (fft.type(torch.complex64))) 

        rmap = torch.fft.irfftn(fft,dim=[-2, -1])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        return rmap
    
    def _cholesky(self, array):
        for x in range(0, array.shape[-2]):
            for y in range(0, array.shape[-1]):
                u, t, v = np.linalg.svd(array[:, :, x, y])
                array[:, :, x, y] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return array
    