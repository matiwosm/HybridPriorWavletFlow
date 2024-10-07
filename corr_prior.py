import base64
import io
import time
import pickle
import math
import numpy as np
import pylab as pl
import torch

import pandas as pd
torch.manual_seed(0)

import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
  raise RuntimeError('Torch versions lower than 1.5.0 not supported')


class CorrelatedNormal_dwt:
    def __init__(self, loc, var,nx, dx,cl_theo,torch_device, freq='high'):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        
        #create the array to multiply the fft with to get the desired power spectrum
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten().astype(np.float32)
        
        ell = self.get_ell(self.nx, self.dx)
        
        self.lbins1 = cl_theo[:, 0]
        self.l_comp1 = cl_theo[:, 1]
        self.h_comp1 = cl_theo[:, 2]
        self.h_comp2 = cl_theo[:, 3]
        self.h_comp3 = cl_theo[:, 4]
        
        l_comp1 = self.l_comp1.reshape( self.rfourier_shape[1:3] )
        h_comp1 = self.h_comp1.reshape( self.rfourier_shape[1:3] )
        h_comp2 = self.h_comp2.reshape( self.rfourier_shape[1:3] )
        h_comp3 = self.h_comp3.reshape( self.rfourier_shape[1:3] )
        
        if freq == 'high':
            clfactor = np.array([[h_comp1, h_comp1*0, h_comp1*0],
                                 [h_comp1*0, h_comp2, h_comp1*0],
                                 [h_comp1*0, h_comp1*0, h_comp3]])
        elif freq == 'low':
            clfactor = np.array([[l_comp1]])
        else:
            raise ValueError("the only options are 'low' and 'high'")
        

        self.cov = torch.from_numpy(np.copy(clfactor)).type(torch.complex64).to(torch_device)
        self.inv_cov = torch.zeros(self.cov.shape).type(torch.complex64).to(torch_device)
        
        self.det = torch.zeros((self.cov.shape[-2], self.cov.shape[-1])).to(torch_device)
        print(self.cov.shape)
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
        self.comp1 = 1.
        self.comp2 = 1.
        self.comp3 = 1.
        samples = self.sample_n(10000)
        self.comp1 = 1./np.std(samples[:, 0, :, :].cpu().numpy())
        self.comp2 = 1./np.std(samples[:, 1, :, :].cpu().numpy())
        self.comp3 = 1./np.std(samples[:, 2, :, :].cpu().numpy())
        samples = self.sample_n(10000)
        print("rescale = ", self.comp1, self.comp2, self.comp3, np.std(samples[:, 0, :, :].cpu().numpy()),
              np.std(samples[:, 1, :, :].cpu().numpy()), np.std(samples[:, 2, :, :].cpu().numpy()))
        # print(self.ells_flat[self.ells_flat == 0].shape)
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi ) 

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)    
        
    def log_prob(self, x):
        fft1 = torch.fft.rfftn(x,dim=[-2,-1]) * np.sqrt(2.)
        fft1[:, 0, :, :] = fft1[:, 0, :, :]/self.comp1
        fft1[:, 1, :, :] = fft1[:, 1, :, :]/self.comp2
        fft1[:, 2, :, :] = fft1[:, 2, :, :]/self.comp3
        
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
        fft = torch.view_as_complex(x) / np.sqrt(2.)
         
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
        fft[:, 0, :, :] *= self.comp1
        fft[:, 1, :, :] *= self.comp2
        fft[:, 2, :, :] *= self.comp3

        rmap = torch.fft.irfftn(fft,dim=[-2, -1])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        
        return rmap
    
    def _cholesky(self, array):
        for x in range(0, array.shape[-2]):
            for y in range(0, array.shape[-1]):
                u, t, v = np.linalg.svd(array[:, :, x, y])
                array[:, :, x, y] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return array
    
 
class CorrelatedNormal_single:
    def __init__(self, loc, var,nx, dx,cl_theo,torch_device, freq='low'):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        
        #create the array to multiply the fft with to get the desired power spectrum
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten().astype(np.float32)
        
        ell = self.get_ell(self.nx, self.dx)
        
        self.lbins1 = cl_theo[:, 0]
        self.l_comp1 = cl_theo[:, 1]
        self.h_comp1 = cl_theo[:, 2]
        self.h_comp2 = cl_theo[:, 3]
        self.h_comp3 = cl_theo[:, 4]
        
        l_comp1 = self.l_comp1.reshape( self.rfourier_shape[1:3] )
        h_comp1 = self.h_comp1.reshape( self.rfourier_shape[1:3] )
        h_comp2 = self.h_comp2.reshape( self.rfourier_shape[1:3] )
        h_comp3 = self.h_comp3.reshape( self.rfourier_shape[1:3] )
        
        if freq == 'high':
            clfactor = np.array([[h_comp1, h_comp1*0, h_comp1*0],
                                 [h_comp1*0, h_comp2, h_comp1*0],
                                 [h_comp1*0, h_comp1*0, h_comp3]])
        elif freq == 'low':
            print('low')
            clfactor = np.array([[l_comp1]])
        else:
            raise ValueError("the only options are 'low' and 'high'")
        

        self.cov = torch.from_numpy(np.copy(clfactor)).type(torch.complex64).to(torch_device)
        print('cov = ', self.cov)
        self.inv_cov = torch.zeros(self.cov.shape).type(torch.complex64).to(torch_device)
        
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
        self.rescale_kappa = 1.
        # self.rescale_cib = 1.
        samples = self.sample_n(10000)
        self.rescale_kappa = 1./np.std((samples[:, 0, :, :].cpu().numpy()))
        samples = self.sample_n(10000)
        # # self.rescale_cib = 1./np.std(util_obj.grab(samples[:, 1, :, :]))
        print("rescale = ", self.rescale_kappa, 'corr prior single', np.std((samples[:, 0, :, :].cpu().numpy())))
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi ) 

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)    
        
    def log_prob(self, x):
        fft1 = torch.fft.rfftn(x,dim=[-2,-1]) * np.sqrt(2.)
        fft1[:, 0, :, :] = fft1[:, 0, :, :]/self.rescale_kappa
        
        x = torch.view_as_real(fft1) 
        #correct: use symmetries
        a = x[:, :, :, :, 0]
        b = x[:, :, :, :, 1]
        
 
        clinvfactor = self.inv_cov.type(torch.float64)
        logp_real = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 0], clinvfactor), (x[:, :, :, :, 0])) 
        
        logp_imaj = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 1], clinvfactor), (x[:, :, :, :, 1]))
        
        logp_real = logp_real[:, self.a_mask]
        logp_imaj = logp_imaj[:, self.b_mask]
        
        logp = torch.sum(logp_imaj, dim=[-1]) + torch.sum(logp_real, dim=[-1])
        return logp
    
    def sample_n(self, batch_size):
        # print('sampling from CorrelatedNormal_single', self.rescale_kappa)
        #https://pytorch.org/docs/stable/complex_numbers.html
        
        #draw random rfft modes
        x = self.dist.sample((batch_size,)).to(self.torch_device)
        
        #test logp
        #logptemp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        #print("logp temp", torch.sum(logptemp, dim=1))
        
        #reshape to rfft format
        x = x.reshape(batch_size, *self.rfourier_shape)
        #make complex data type
        fft = torch.view_as_complex(x) / np.sqrt(2.)
         
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
        
        fft = torch.einsum('nplm,knlm->kplm', self.clfactor.type(torch.complex64), (fft.type(torch.complex64))) 

        fft[:, 0, :, :] *= self.rescale_kappa
        rmap = torch.fft.irfftn(fft,dim=[-2, -1])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        
        return rmap
    
    def _cholesky(self, array):
        for x in range(0, array.shape[-2]):
            for y in range(0, array.shape[-1]):
                u, t, v = np.linalg.svd(array[:, :, x, y])
                array[:, :, x, y] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return array
    
    


class CorrelatedNormal:
    def __init__(self, loc, var,nx, dx,cl_theo,torch_device):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        
        #create the array to multiply the fft with to get the desired power spectrum
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten().astype(np.float32)
        
        ell = self.get_ell(self.nx, self.dx)
        
        self.lbins1 = cl_theo[:, 0]
        self.Dlkap_kap = cl_theo[:, 1]
        self.Dlcib_cib = cl_theo[:, 3]
        self.Dlkap_cross = cl_theo[:, 2]

        clfactor_kap = self.Dlkap_kap.reshape( self.rfourier_shape[1:3] )
        clfactor_cib = self.Dlcib_cib.reshape( self.rfourier_shape[1:3] )
        clfactor_cross = self.Dlkap_cross.reshape( self.rfourier_shape[1:3] )

        clfactor = np.array([[clfactor_kap]])
        

        self.cov = torch.from_numpy(np.copy(clfactor)).type(torch.complex64).to(torch_device)
        self.inv_cov = torch.zeros(self.cov.shape).type(torch.complex64).to(torch_device)
        
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
        self.rescale_kappa = 1.
        self.rescale_cib = 1.
        # samples = self.sample_n(10000)
        # self.rescale_kappa = 1./np.std(utilities.grab(samples[:, 0, :, :]))
        # self.rescale_cib = 1./np.std(utilities.grab(samples[:, 1, :, :]))
        print("rescale = ", self.rescale_kappa)
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi ) 

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)    
        
    def log_prob(self, x):
        fft1 = torch.fft.rfftn(x,dim=[-2,-1]) * np.sqrt(2.)
        fft1[:, 0, :, :] = fft1[:, 0, :, :]/self.rescale_kappa
        # fft1[:, 1, :, :] = fft1[:, 1, :, :]/self.rescale_cib
        
        x = torch.view_as_real(fft1)  
        #correct: use symmetries
        a = x[:, :, :, :, 0]
        b = x[:, :, :, :, 1]
        
 
        clinvfactor = self.inv_cov.type(torch.float64)
        logp_real = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 0], clinvfactor), (x[:, :, :, :, 0])) 
        
        logp_imaj = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 1], clinvfactor), (x[:, :, :, :, 1]))
        
        
        logp_real = logp_real[:, self.a_mask]
        logp_imaj = logp_imaj[:, self.b_mask]
        
        logp = torch.sum(logp_imaj, dim=[-1]) + torch.sum(logp_real, dim=[-1])
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
        fft = torch.view_as_complex(x) / np.sqrt(2.)
         
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
        
        fft = torch.einsum('nplm,knlm->kplm', self.clfactor.type(torch.complex64), (fft.type(torch.complex64))) 
        fft[:, 0, :, :] *= self.rescale_kappa
        # fft[:, 1, :, :] *= self.rescale_cib
        
        rmap = torch.fft.irfftn(fft,dim=[-2, -1])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        # rmap = torch.ones(rmap.shape).to('cuda')
        return rmap
    
    def _cholesky(self, array):
        for x in range(0, array.shape[-2]):
            for y in range(0, array.shape[-1]):
                u, t, v = np.linalg.svd(array[:, :, x, y])
                array[:, :, x, y] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return array
    
    
    