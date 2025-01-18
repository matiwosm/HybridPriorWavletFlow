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

class CorrelatedNormalDWTGeneral:
    def __init__(
        self, 
        loc, 
        var,
        nx, 
        dx,
        cl_theo,             # <-- This is the 2D NumPy array (power_spec)
        colname_to_index,    # <-- Dictionary mapping "low_auto_ch0" -> column idx
        torch_device,
        freq='high',
        n_channels=5,
        prior_type='CC',
        norm_std=None
    ):
        """
        A more general version that handles n_channels and wavelet sub-bands.
        
        Parameters
        ----------
        loc : torch.Tensor
            Mean in Fourier domain (rfftn shape).
        var : torch.Tensor
            Variance in Fourier domain (rfftn shape).
        nx : int
            Size of the map (spatial dimension).
        dx : float
            Pixel size (or step size).
        cl_theo : np.ndarray
            Theoretical power spectra (power_spec), shape (N_ell, N_columns).
        colname_to_index : dict
            Dictionary mapping column names to integer indices in cl_theo.
        torch_device : torch.device
            Device to use.
        freq : str
            'low' or 'high'.
        n_channels : int
            Number of channels to handle for each sub-band.
        prior_type : str
            'C' or 'CC'. If 'C', off-diagonal cross-terms set to 0. If 'CC', read from cl_theo.
        """

        self.torch_device = torch_device
        self.nx = nx
        self.dx = dx
        self.freq = freq
        self.n_channels = n_channels
        self.prior_type = prior_type
        
        # Store the power spectrum array and the name->index dictionary
        self.cl_theo = cl_theo
        self.colname_to_index = colname_to_index

        #normalization constants
        self.norm_std = norm_std
        
        # Normal distribution for random Fourier modes
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), 
            torch.flatten(var)
        )
        self.rfourier_shape = loc.shape
        self.level = int(np.log2(nx))
        print('level = ', self.level)
        
        # create the array of ells for reference (optional usage)
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten().astype(np.float32)
        
        # Decide how many sub-bands
        if freq == 'low':
            self.wavelet_count = 1
        elif freq == 'high':
            self.wavelet_count = 3
        else:
            raise ValueError(f"Unsupported freq '{freq}' - must be 'low' or 'high'.")
        
        # Build the covariance
        H, W = self.rfourier_shape[1:3]
        self.cov = self._build_cov_from_cl_theo(self.cl_theo, H, W)
        
        # Convert to torch
        self.cov = torch.from_numpy(self.cov).to(torch_device)
        
        # Invert covariance
        self.inv_cov = torch.zeros_like(self.cov)
        self.det = torch.zeros((H, W), device=torch_device)
        for j in range(H):
            for k in range(W):
                self.inv_cov[:, :, j, k] = torch.linalg.inv(self.cov[:, :, j, k])
                self.det[j, k] = torch.linalg.det(self.cov[:, :, j, k].real)
        
        # Cholesky / sqrt factor
        self.clfactor = np.copy(self.cov.cpu().numpy())
        self.clfactor = self._cholesky(self.clfactor)
        self.clfactor = torch.from_numpy(self.clfactor.real).float().to(torch_device)
        
        # Prepare masks for RFFT symmetries
        a_mask, b_mask = self._create_rfft_masks(nx)
        self.a_mask = a_mask
        self.b_mask = b_mask
        
        # Distributions for real/imag parts
        a_nr = self.a_mask.sum()
        b_nr = self.b_mask.sum()
        self.a_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        ).expand((a_nr,))
        self.b_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        ).expand((b_nr,))
        
        # print('estimating fudge factor')
        # s = self.sample_n(3)
        # for i in range(s.shape[1]):
        #     print(f'std channel {i}:', torch.std(s[:, i, :, :]))
        #     print(f'max channel {i}:', torch.max(s[:, i, :, :]))
        #     print(f'min channel {i}:', torch.min(s[:, i, :, :]))

    def get_spectrum_by_name(self, name):
        """
        Retrieve the 1D power spectrum from self.cl_theo by column name.

        Parameters
        ----------
        name : str
            e.g. "low_auto_ch0", "high_horizontal_cross_ch1_ch3", etc.

        Returns
        -------
        spec_1d : np.ndarray, shape (N_ell,)
            The 1D array corresponding to this power spectrum column.
        """
        # Look up the column index in our dictionary
        col_index = self.colname_to_index.get(name, None)
        if col_index is None:
            raise KeyError(f"No column found for '{name}' in power spectrum data.")
        
        # Return the entire column (skipping any pre-processing, if needed)
        return self.cl_theo[:, col_index]  # shape (N_ell,)

    def _build_cov_from_cl_theo(self, cl_theo, H, W):
        """
        Build the 4D covariance array of shape 
          (wavelet_count*n_channels, wavelet_count*n_channels, H, W)
        by parsing auto/cross columns from cl_theo.

        If prior_type == 'C', cross terms are set to zero.
        If prior_type == 'CC', cross terms are read from cl_theo.
        """
        cov_size = self.wavelet_count * self.n_channels
        clfactor = np.zeros((cov_size, cov_size, H, W), dtype=np.complex64)
        
        # Identify sub-band labels
        if self.freq == 'low':
            subband_labels = ['low']  # single
        else:
            subband_labels = ['high_horizontal', 'high_vertical', 'high_diagonal']
        
        # Fill diagonal (auto terms)
        for s_idx, subb in enumerate(subband_labels):
            for ch in range(self.n_channels):
                col_name = f"{subb}_auto_ch{ch}"
                # Use our new get_spectrum_by_name
                spectrum_1d = self.get_spectrum_by_name(col_name)  # shape: (N_ell,)
                spectrum_2d = spectrum_1d.reshape(H, W).astype(np.complex64)
                
                rowcol = ch * self.wavelet_count + s_idx
                clfactor[rowcol, rowcol, :, :] = spectrum_2d
        
        # Fill off-diagonal (cross terms)
        for s_idx, subb in enumerate(subband_labels):
            for ch1 in range(self.n_channels):
                for ch2 in range(ch1+1, self.n_channels):
                    row1 = ch1 * self.wavelet_count + s_idx
                    row2 = ch2 * self.wavelet_count + s_idx
                    
                    if self.prior_type == 'CC':
                        col_name = f"{subb}_cross_ch{ch1}_ch{ch2}"
                        spectrum_1d = self.get_spectrum_by_name(col_name)
                        spectrum_2d = spectrum_1d.reshape(H, W).astype(np.complex64)
                        # Symmetric fill
                        clfactor[row1, row2, :, :] = spectrum_2d
                        clfactor[row2, row1, :, :] = spectrum_2d
                    elif self.prior_type == 'C':
                        clfactor[row1, row2, :, :] = 0.
                        clfactor[row2, row1, :, :] = 0.
                    else:
                        raise ValueError("Invalid prior type. Must be 'C' or 'CC'.")
        
        return clfactor

    def _cholesky(self, array):
        for x in range(array.shape[-2]):
            for y in range(array.shape[-1]):
                u, s, v = np.linalg.svd(array[:, :, x, y])
                array[:, :, x, y] = np.dot(
                    u,
                    np.dot(np.diag(np.sqrt(s)), v)
                )
        return array

    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid(
            np.fft.fftfreq(nx, dx)[0:int(nx/2+1)]*2.*np.pi,
            np.fft.fftfreq(nx, dx)*2.*np.pi
        )

    def get_ell(self, nx, dx):
        """ returns the wavenumber l = sqrt(lx^2 + ly^2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)

    def _create_rfft_masks(self, nx):
        a_mask = np.ones((nx, int(nx/2+1)), dtype=bool)
        a_mask[int(nx/2+1):, 0] = False
        a_mask[int(nx/2+1):, int(nx/2)] = False
        
        b_mask = np.ones((nx, int(nx/2+1)), dtype=bool)
        b_mask[0,0] = False
        b_mask[0,int(nx/2)] = False
        b_mask[int(nx/2),0] = False
        b_mask[int(nx/2),int(nx/2)] = False
        b_mask[int(nx/2+1):, 0] = False
        b_mask[int(nx/2+1):, int(nx/2)] = False
        
        return a_mask, b_mask

    def log_prob(self, x):
        fft1 = torch.fft.rfftn(x,dim=[-2,-1]) * np.sqrt(2.)

        #unnormalization
        if self.norm_std is not None:
            if self.freq == 'low':
                for ch in range(self.n_channels):
                    std = self.norm_std['low']['mean_std'][ch]
                    # min_max = max(-1*self.norm_std['low']['min'][ch], self.norm_std['low']['max'][ch])
                    fft1[:, ch, :, :]  = (fft1[:, ch, :, :]/ std) #* min_max
            if self.freq == 'high':
                n_high_types = 3
                high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
                for ch in range(self.n_channels):
                    for ht_idx, ht in enumerate(high_types):
                        idx = ch * n_high_types + ht_idx
                        std = self.norm_std[ht]['mean_std'][ch]
                        # min_max = max(-1*self.norm_std[ht]['min'][ch], self.norm_std[ht]['max'][ch])
                        fft1[:, idx, :, :] = (fft1[:, idx, :, :]/ std) #* min_max

        
        x = torch.view_as_real(fft1)

        #correct: use symmetries
        a = x[:, :, :, :, 0]
        b = x[:, :, :, :, 1]
        
 
        clinvfactor = self.inv_cov.type(x.dtype)
        logp_real = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 0], clinvfactor), (x[:, :, :, :, 0])) 
        
        logp_imaj = (-1/2)*torch.einsum('kplm,kplm->klm', torch.einsum('nplm,pklm->nklm', x[:, :, :, :, 1], clinvfactor), (x[:, :, :, :, 1]))
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
        #unnormalization
        if self.norm_std is not None:
            if self.freq == 'low':
                for ch in range(self.n_channels):
                    std = self.norm_std['low']['mean_std'][ch]
                    # min_max = max(-1*self.norm_std['low']['min'][ch], self.norm_std['low']['max'][ch])
                    rmap[:, ch, :, :]  = (rmap[:, ch, :, :] * std) #/min_max

            if self.freq == 'high':
                n_high_types = 3
                high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
                for ch in range(self.n_channels):
                    for ht_idx, ht in enumerate(high_types):
                        idx = ch * n_high_types + ht_idx
                        std = self.norm_std[ht]['mean_std'][ch]
                        # min_max = max(-1*self.norm_std[ht]['min'][ch], self.norm_std[ht]['max'][ch])
                        rmap[:, idx, :, :] = (rmap[:, idx, :, :] * std) #/min_max

        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        return rmap
    
    def _cholesky(self, array):
        for x in range(0, array.shape[-2]):
            for y in range(0, array.shape[-1]):
                u, t, v = np.linalg.svd(array[:, :, x, y])
                array[:, :, x, y] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return array
    
    
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
    