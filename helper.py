import math
import numpy as np
import os
from numpy.fft import fft2, fftshift, fftfreq

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
            print("bad r1", r1.shape)

            
        if (r2 is None):
            r2 = r1
        else:
            if (np.any(np.isnan(r2)) or np.any(np.isinf(r2))):
                print("bad r2", r2.shape)
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


    def get_2d_bispectrum_monte_carlo(self, nx, dx, map_data, nbins, scale='log'):
        """
        Compute the 2D auto-bispectrum using the flat-sky approximation.
        
        Parameters:
        - nx: Size of the map (nx x nx)
        - dx: Pixel scale in radians
        - map_data: Input 2D map
        - nbins: Number of logarithmic bins
        - scale: Binning scale ('log' or 'linear')
        
        Returns:
        - ell_centers: Bin centers in multipole space
        - bispectrum: Binned bispectrum values
        """
        # Compute FFT and shift to center
        fft_map = fftshift(fft2(map_data))
        
        # Generate Fourier grid
        kx = 2 * np.pi * fftshift(fftfreq(nx, d=dx))
        ky = 2 * np.pi * fftshift(fftfreq(nx, d=dx))
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        ell_grid = np.sqrt(kx_grid**2 + ky_grid**2).real
        
        # Avoid zero frequency (DC component)
        ell_nonzero = ell_grid.copy()
        ell_nonzero[ell_nonzero == 0] = np.inf
        
        # Define bins
        if scale == 'log':
            ell_min = np.min(ell_nonzero)
            ell_max = np.max(ell_grid)
            bins = np.logspace(np.log10(ell_min), np.log10(ell_max), nbins + 1)
        else:
            bins = np.linspace(0, np.max(ell_grid), nbins + 1)
        
        # Digitize ell grid into bins
        bin_indices = np.digitize(ell_grid, bins) - 1  # Zero-based indices
        
        # Compute bispectrum by sampling valid triangles
        bispectrum = np.zeros(nbins)
        counts = np.zeros(nbins)
        
        # Precompute valid indices (excluding out-of-bounds and DC component)
        valid_mask = (bin_indices >= 0) & (bin_indices < nbins)
        valid_indices = np.argwhere(valid_mask)
        
        # Monte Carlo sampling to reduce computation (adjust sample_size as needed)
        sample_size = 10000  # Reduce for speed, increase for accuracy
        sample_size = min(10000, len(valid_indices)) 
        np.random.seed(42)
        samples = valid_indices[np.random.choice(len(valid_indices), sample_size, replace=False)]
        
        for idx in samples:
            i, j = idx
            k1x = kx_grid[i, j]
            k1y = ky_grid[i, j]
            bin1 = bin_indices[i, j]
            
            # Randomly sample k2
            k2_idx = valid_indices[np.random.randint(0, len(valid_indices))]
            k2x = kx_grid[k2_idx[0], k2_idx[1]]
            k2y = ky_grid[k2_idx[0], k2_idx[1]]
            bin2 = bin_indices[k2_idx[0], k2_idx[1]]
            
            # Compute k3 = -k1 -k2
            k3x = -k1x - k2x
            k3y = -k1y - k2y
            
            # Find nearest grid point for k3
            dist = (kx_grid - k3x)**2 + (ky_grid - k3y)**2
            k3_i, k3_j = np.unravel_index(np.argmin(dist), dist.shape)
            bin3 = bin_indices[k3_i, k3_j]
            
            if bin3 < 0 or bin3 >= nbins:
                continue
            
            # Compute bispectrum contribution
            product = fft_map[i, j] * fft_map[k2_idx[0], k2_idx[1]] * fft_map[k3_i, k3_j]
            bispectrum[bin3] += np.real(product)
            counts[bin3] += 1
        
        # Average and handle empty bins
        bispectrum = np.where(counts > 0, bispectrum / counts, 0)
        ell_centers = (bins[1:] + bins[:-1]) / 2
        
        return ell_centers, bispectrum


    def get_2d_bispectrum(self, nx, dx, map_data, nbins, scale='log'):
        """
        Vectorized bispectrum computation using grid symmetry and array broadcasting.
        - map_data: 2D input map (square array)
        - dx: Pixel scale in radians
        - nbins: Number of radial bins
        - scale: 'log' or 'linear' binning
        Returns:
        - ell_centers: Bin centers
        - bispectrum: Binned bispectrum values
        """
        nx = map_data.shape[0]
        fft_map = fftshift(fft2(map_data))
        
        # Precompute Fourier grid components
        kx = 2 * np.pi * fftshift(fftfreq(nx, d=dx))
        ky = 2 * np.pi * fftshift(fftfreq(nx, d=dx))
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        ell_grid = np.sqrt(kx_grid**2 + ky_grid**2)
        dk = kx[1] - kx[0]  # Wavevector spacing

        # Create mask for valid wavevectors (k > 0)
        valid_mask = (ell_grid > 0)
        valid_indices = np.argwhere(valid_mask)
        n_valid = len(valid_indices)

        # Precompute all valid k-vectors and their FFT values
        kx_flat = kx_grid[valid_mask]
        ky_flat = ky_grid[valid_mask]
        fft_flat = fft_map[valid_mask]

        # Calculate all possible k3 vectors from pairs (k1, k2)
        k3x = -kx_flat[:, None] - kx_flat[None, :]  # Shape: (n_valid, n_valid)
        k3y = -ky_flat[:, None] - ky_flat[None, :]

        # Find nearest grid indices for k3 vectors using vectorized rounding
        i3 = np.round((k3x + kx.max()) / dk).astype(int)
        j3 = np.round((k3y + ky.max()) / dk).astype(int)

        # Create validity mask for k3 indices
        valid_k3 = (i3 >= 0) & (i3 < nx) & (j3 >= 0) & (j3 < nx)
        i3_valid = i3[valid_k3]
        j3_valid = j3[valid_k3]

        # Get FFT values at k3 locations using advanced indexing
        fft_k3 = fft_map[i3_valid, j3_valid]

        # Calculate triple products only for valid triangles
        k1_indices, k2_indices = np.where(valid_k3)
        products = (fft_flat[k1_indices] * 
                    fft_flat[k2_indices] * 
                    fft_k3).real

        # Get ell values for binning
        ell3 = ell_grid[i3_valid, j3_valid]

        # Create bins
        if scale == 'log':
            bins = np.logspace(np.log10(ell3.min()), np.log10(ell3.max()), nbins + 1)
        else:
            bins = np.linspace(0, ell3.max(), nbins + 1)

        # Bin the bispectrum values
        bin_indices = np.digitize(ell3, bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < nbins)
        
        bispectrum = np.bincount(bin_indices[valid_bins], 
                                weights=products[valid_bins], 
                                minlength=nbins)
        counts = np.bincount(bin_indices[valid_bins], 
                            minlength=nbins)

        # Normalize and return
        with np.errstate(divide='ignore', invalid='ignore'):
            bispectrum = np.where(counts > 0, bispectrum / counts, 0)
        
        ell_centers = (bins[1:] + bins[:-1]) / 2
        return ell_centers, bispectrum
        
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
    
