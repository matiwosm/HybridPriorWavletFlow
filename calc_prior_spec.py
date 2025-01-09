import os
from collections import defaultdict
import re
import sys
import torch
import random
import numpy as np
from data_loader import My_lmdb, yuuki_256
from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
import torch.optim as optim
from src.nf.glow import Glow
from src.waveletflow import WaveletFlow
from src.conditioning_network import Conditioning_network
import argparse
import time
import lmdb
import corr_prior
import pandas as pd
from helper import utils as util
from src.dwt.wavelets import Haar
from src.dwt.dwt import Dwt
def get_2d_power(nx, dx, r1, r2=None, num_bins=100):
        if (np.any(np.isnan(r1)) or np.any(np.isinf(r1))):
            print("whyyyyyy")
            
        if (r2 is None):
            r2 = r1
        lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,
                                np.fft.fftfreq( nx, dx )*2.*np.pi )
        ell = np.sqrt(lx**2 + ly**2)
        ell = ell[~np.isnan(ell)]
        lbins = ell.flatten()
        FMap1 = np.fft.rfft2(r1)

       
        FMap2 = np.fft.rfft2(r2)
        # FMap1 = np.fft.ifft2(np.fft.fftshift(r1))
        cvec = (np.conj(FMap1) * (FMap2)).flatten()
        
        return lbins, cvec

def compute_and_accumulate_spectra(comp_a, comp_b, nx, dx):
    """
    Computes the sum of power spectra for the given components over a batch.

    Args:
        comp_a: numpy array of shape [batch_size, H, W]
        comp_b: numpy array of shape [batch_size, H, W]
        nx: int, size of the map
        dx: float, pixel size

    Returns:
        spectra_sum: numpy array, sum of spectra over the batch
        ell: numpy array, ell values (from the last computation)
    """
    batch_size = comp_a.shape[0]
    spectra_sum = None
    ell = None

    for sample_idx in range(batch_size):
        map_a = comp_a[sample_idx, :, :]
        map_b = comp_b[sample_idx, :, :]
        ell, bin_spectrum = get_2d_power(nx, dx, map_a, map_b)
        bin_spectrum = np.real(bin_spectrum)
        if spectra_sum is None:
            spectra_sum = bin_spectrum
        else:
            spectra_sum += bin_spectrum

    return spectra_sum, ell


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = 'configs/example_config_hcc_prior.py'
cf = SourceFileLoader('cf', config_file).load_module()


wavelet = Haar().to(device)
dwt = Dwt(wavelet=wavelet).to(device)

#replace with your dataset
bdir = cf.dataset_path
file = "data.mdb"

# Create the dataset
dataset = My_lmdb(
    db_path=bdir,
    file_path=file,
    transformer=None,
    num_classes=1,
    class_cond=False,
    channels_to_use=cf.channels_to_get,
    noise_dict=cf.noise_dict,        # noise only these channels
    apply_scaling=True,           # do the scaling
    data_shape=cf.data_shape       
)

# Number of DWT levels to compute
max_m = 5 # Adjust as needed

# including the 'low' components and high-frequency components.
for m in range(0, 5):
    loader = DataLoader(dataset, batch_size=8096)
    util_obj = util()
    
    # Initialize dictionaries to hold accumulated spectra and total std deviations
    comp_auto_spectra_sum = {}
    comp_cross_spectra_sum = {}
    comp_total_std = {}
    total_samples = 0  # Total number of samples processed
    num_batches = 0    # Number of batches processed
    
    # Define high-frequency component types
    high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
    n_high_types = len(high_types)

    # Initialize ell to None
    ell = None

    for i, x in enumerate(loader):
        x = x.to(device).to(torch.float32)
        # Apply DWT m+1 times
        for k in range(m + 1):
            dwt_result = dwt.forward(x)
            x1 = dwt_result['low']  # x1 shape: (batch_size, N_channels, H, W)
            x2 = dwt_result['high']  # x2 shape: (batch_size, N_channels * 3, H, W)
            x = x1  # Update x for the next level

        nx = x1.shape[-1]
        dx = (0.5 / 60. * np.pi / 180.) * 2 ** (m + 1)
        ell, bin_spectrum = get_2d_power(nx, dx, x1.cpu().numpy(), x1.cpu().numpy())
        print(m, i, x1.shape, x2.shape)

        N_channels = x1.shape[1]  # Number of input channels

        # Initialize dictionaries on the first batch
        if i == 0:
            # Initialize for 'low' components
            comp_types = ['low'] + high_types
            for comp_type in comp_types:
                comp_auto_spectra_sum[comp_type] = [0 for _ in range(N_channels)]
                comp_cross_spectra_sum[comp_type] = {}
                comp_total_std[comp_type] = [0 for _ in range(N_channels)]
                # Initialize cross spectra dictionaries
                for i_ch in range(N_channels):
                    for j_ch in range(i_ch + 1, N_channels):
                        comp_cross_spectra_sum[comp_type][(i_ch, j_ch)] = 0

        # Process 'low' components
        x1_np = x1.cpu().numpy()  # Shape: [batch_size, N_channels, H, W]
        batch_size = x1_np.shape[0]

        for ch in range(N_channels):
            comp_ch = x1_np[:, ch, :, :]  # Shape: [batch_size, H, W]
            # Compute standard deviation over the batch
            std_batch = np.std(comp_ch)
            comp_total_std['low'][ch] += std_batch
            # Compute auto spectra and accumulate
            spectra_sum, ell_current = compute_and_accumulate_spectra(comp_ch, comp_ch, nx, dx)
            comp_auto_spectra_sum['low'][ch] += spectra_sum
            if ell is None:
                ell = ell_current  # Store ell from the first computation
            # Compute cross spectra with other channels
            for ch2 in range(ch + 1, N_channels):
                comp_ch2 = x1_np[:, ch2, :, :]
                cross_spectra_sum, _ = compute_and_accumulate_spectra(comp_ch, comp_ch2, nx, dx)
                comp_cross_spectra_sum['low'][(ch, ch2)] += cross_spectra_sum

        # Process high-frequency components
        x2_np = x2.cpu().numpy()  # Shape: [batch_size, N_channels * 3, H, W]

        for ht_idx, ht in enumerate(high_types):
            for ch in range(N_channels):
                idx = ch * n_high_types + ht_idx  # Index in x2
                comp_ch = x2_np[:, idx, :, :]  # Shape: [batch_size, H, W]
                # Compute standard deviation over the batch
                std_batch = np.std(comp_ch)
                comp_total_std[ht][ch] += std_batch
                # Compute auto spectra and accumulate
                spectra_sum, _ = compute_and_accumulate_spectra(comp_ch, comp_ch, nx, dx)
                comp_auto_spectra_sum[ht][ch] += spectra_sum
                # Compute cross spectra with other channels
                for ch2 in range(ch + 1, N_channels):
                    idx2 = ch2 * n_high_types + ht_idx
                    comp_ch2 = x2_np[:, idx2, :, :]
                    cross_spectra_sum, _ = compute_and_accumulate_spectra(comp_ch, comp_ch2, nx, dx)
                    comp_cross_spectra_sum[ht][(ch, ch2)] += cross_spectra_sum

        total_samples += batch_size
        num_batches += 1  # Increment the number of batches

    # After processing all batches, compute mean standard deviations
    comp_mean_std = {}
    for comp_type in comp_total_std.keys():
        comp_mean_std[comp_type] = []
        for ch in range(N_channels):
            mean_std = comp_total_std[comp_type][ch] / num_batches
            comp_mean_std[comp_type].append(mean_std)

    # Compute average spectra
    comp_mean_spectra = {}
    comp_cross_mean_spectra = {}

    for comp_type in comp_auto_spectra_sum.keys():
        # Auto spectra
        comp_mean_spectra[comp_type] = []
        for ch in range(N_channels):
            print(comp_type, comp_mean_std[comp_type][ch])
            # Normalize by total samples and squared mean standard deviation
            mean_spectrum = comp_auto_spectra_sum[comp_type][ch] / total_samples
            mean_std = comp_mean_std[comp_type][ch]
            mean_spectrum /= mean_std ** 2
            comp_mean_spectra[comp_type].append(mean_spectrum)
        # Cross spectra
        comp_cross_mean_spectra[comp_type] = []
        for (i_ch, j_ch), cross_spectrum_sum in comp_cross_spectra_sum[comp_type].items():
            mean_cross_spectrum = cross_spectrum_sum / total_samples
            mean_std_i = comp_mean_std[comp_type][i_ch]
            mean_std_j = comp_mean_std[comp_type][j_ch]
            mean_cross_spectrum /= (mean_std_i * mean_std_j)
            comp_cross_mean_spectra[comp_type].append((i_ch, j_ch, mean_cross_spectrum))

    # Save results to a file
    filename = f'ps/64x64_kappa_noise_{0.025}_yuuki_{dataset[0].shape[0]}comps_dwtlevel{nx}x{nx}.dat'
    with open(filename, 'w') as file:
        print('SAVING')
        # Write column headers
        headers = ['ell']
        # Auto spectra headers
        for comp_type in comp_mean_spectra.keys():
            for ch in range(N_channels):
                headers.append(f'{comp_type}_auto_ch{ch}')
        # Cross spectra headers
        for comp_type in comp_cross_mean_spectra.keys():
            for (i_ch, j_ch, _) in comp_cross_mean_spectra[comp_type]:
                headers.append(f'{comp_type}_cross_ch{i_ch}_ch{j_ch}')
        file.write('; '.join(headers) + '\n')

        # Assuming ell is the same for all spectra
        for idx in range(len(ell)):
            line = f"{ell[idx]}"
            # Auto spectra
            for comp_type in comp_mean_spectra.keys():
                for ch in range(N_channels):
                    line += f"; {comp_mean_spectra[comp_type][ch][idx]}"
            # Cross spectra
            for comp_type in comp_cross_mean_spectra.keys():
                for (_, _, cross_spectrum_mean) in comp_cross_mean_spectra[comp_type]:
                    line += f"; {cross_spectrum_mean[idx]}"
            line += "\n"
            file.write(line)
