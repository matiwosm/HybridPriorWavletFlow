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
from tqdm import tqdm

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

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="ps/256x256_kappa_noise_0.01_yuuki_2comps", help='missing output dir to save power spectra')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ps_path = args.output_dir

#dir to save plots
if not os.path.exists(ps_path):
        os.makedirs(ps_path)

print(ps_path)


#replace with your dataset
bdir = '/sdf/group/kipac/users/mati/yuuki_sim_train_64x64/'
file = "data.mdb"

# Create the dataset
dataset = My_lmdb(
    db_path=bdir,
    file_path=file,
    transformer=None,
    num_classes=1,
    class_cond=False,
    channels_to_use=['kappa', 'ksz', 'tsz', 'cib', 'rad'],
    noise_dict={},        # noise only these channels
    apply_scaling=True,           # do the scaling
    data_shape=[5, 64, 64]  
)

# Number of DWT levels to compute
n_comps = 5

loader = DataLoader(dataset, batch_size=1024)
util_obj = util()

# Initialize dictionaries to hold accumulated spectra and total std deviations
comp_auto_spectra_sum = {}
comp_cross_spectra_sum = {}
comp_total_std = {}
total_samples = 0  # Total number of samples processed
num_batches = 0    # Number of batches processed

# Initialize ell to None
ell = None

for i, x in enumerate(tqdm(loader, desc=f"Processing ", unit="batch")):
    x = x.to(device).to(torch.float32)

    nx = x.shape[-1]
    dx = (0.5 / 60. * np.pi / 180.)
    ell, bin_spectrum = get_2d_power(nx, dx, x.cpu().numpy(), x.cpu().numpy())

    N_channels = x.shape[1]  # Number of input channels

    # Initialize dictionaries on the first batch
    if i == 0:
        comp_auto_spectra_sum = [0 for _ in range(N_channels)]
        comp_cross_spectra_sum = {}
        comp_total_std = [0 for _ in range(N_channels)]
        # Initialize cross spectra dictionaries
        for i_ch in range(N_channels):
            for j_ch in range(i_ch + 1, N_channels):
                comp_cross_spectra_sum[(i_ch, j_ch)] = 0

    x1_np = x.cpu().numpy()  # Shape: [batch_size, N_channels, H, W]
    batch_size = x1_np.shape[0]

    for ch in range(N_channels):
        comp_ch = x1_np[:, ch, :, :]  # Shape: [batch_size, H, W]
        # Compute standard deviation over the batch
        std_batch = np.std(comp_ch)
        comp_total_std[ch] += std_batch
        # Compute auto spectra and accumulate
        spectra_sum, ell_current = compute_and_accumulate_spectra(comp_ch, comp_ch, nx, dx)
        comp_auto_spectra_sum[ch] += spectra_sum
        if ell is None:
            ell = ell_current  # Store ell from the first computation
        # Compute cross spectra with other channels
        for ch2 in range(ch + 1, N_channels):
            comp_ch2 = x1_np[:, ch2, :, :]
            cross_spectra_sum, _ = compute_and_accumulate_spectra(comp_ch, comp_ch2, nx, dx)
            comp_cross_spectra_sum[(ch, ch2)] += cross_spectra_sum


    total_samples += batch_size
    num_batches += 1  # Increment the number of batches

# After processing all batches, compute mean standard deviations
comp_mean_std = {}
comp_mean_std = []
for ch in range(N_channels):
    mean_std = comp_total_std[ch] / num_batches
    comp_mean_std.append(mean_std)

# Compute average spectra
comp_mean_spectra = {}
comp_cross_mean_spectra = {}


# Auto spectra
comp_mean_spectra = []
for ch in range(N_channels):
    # Normalize by total samples and squared mean standard deviation
    mean_spectrum = comp_auto_spectra_sum[ch] / total_samples
    mean_std = comp_mean_std[ch]
    print(mean_std)
    mean_spectrum /= mean_std ** 2
    comp_mean_spectra.append(mean_spectrum)
# Cross spectra
comp_cross_mean_spectra = []
for (i_ch, j_ch), cross_spectrum_sum in comp_cross_spectra_sum.items():
    mean_cross_spectrum = cross_spectrum_sum / total_samples
    mean_std_i = comp_mean_std[i_ch]
    mean_std_j = comp_mean_std[j_ch]
    mean_cross_spectrum /= (mean_std_i * mean_std_j)
    comp_cross_mean_spectra.append((i_ch, j_ch, mean_cross_spectrum))

# Save results to a file
filename = f'{ps_path}/map_size_{nx}x{nx}.dat'
with open(filename, 'w') as file:
    print('SAVING')
    # Write column headers
    headers = ['ell']
    # Auto spectra headers
    for ch in range(N_channels):
            headers.append(f'auto_ch{ch}')
    # Cross spectra headers
    for (i_ch, j_ch, _) in comp_cross_mean_spectra:
            headers.append(f'cross_ch{i_ch}_ch{j_ch}')
    file.write('; '.join(headers) + '\n')

    # Assuming ell is the same for all spectra
    for idx in range(len(ell)):
        line = f"{ell[idx]}"
        # Auto spectra
        for ch in range(N_channels):
                line += f"; {comp_mean_spectra[ch][idx]}"
        # Cross spectra
        for (_, _, cross_spectrum_mean) in comp_cross_mean_spectra:
                line += f"; {cross_spectrum_mean[idx]}"
        line += "\n"
        file.write(line)
