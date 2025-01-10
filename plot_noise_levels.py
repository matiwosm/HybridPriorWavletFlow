import os
from collections import defaultdict
import re
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
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helper import utils as util
from utilities import *
import corr_prior
import pandas as pd
import json
from quantimpy import minkowski
plt.style.use('ggplot')

def plot_power(iter_loader, samples_dataloader, prior, step, savedir, batch_size, batch_num):
    """
    Plots the power spectrum and fractional differences for target, sample, and prior data.

    Parameters:
    - target: object with method sample_n(batch_size)
    - sample: object with method sample_n(batch_size)
    - prior: object with method sample_n(batch_size)
    - step: int, current step or epoch
    - savedir: str, directory to save plots
    - batch_size: int, number of samples per batch
    - n_samples: int, total number of samples to process
    """
    import matplotlib.pyplot as plt

    dx = (0.5 / 60.) * (np.pi / 180.)  # Pixel size

    target = next(iter_loader).cpu().numpy()
    # Get the shape information
    sample_shape = target.shape  # Assuming shape (B, C, H, W)
    C = sample_shape[1]
    if C == 2:
        comps = ['kap', 'cib']
    elif C == 3:
        comps = ['kap', 'tsz', 'cib']
    else:
        comps = ['kap', 'ksz', 'tsz', 'cib', 'radio']

    nx = sample_shape[2]

    size = 15  # Number of bins for power spectrum
    ell = None  # To store the ell values from the first computation

    # Initialize accumulators for means and variances
    spectra_sum_target = np.zeros((C, C, size))
    spectra_sum_sample = np.zeros((C, C, size))
    if prior is not None:
        spectra_sum_prior = np.zeros((C, C, size))
    spectra_sq_sum_target = np.zeros((C, C, size))
    spectra_sq_sum_sample = np.zeros((C, C, size))

    total_samples = 0
    model_samples = 0

    batch_num_after_smaple = 0
    # Loop over batches
    for batch_idx in range(batch_num + batch_num_after_smaple):
        if (batch_idx % 10) == 0:
            print(batch_idx, batch_num+batch_num_after_smaple)
        current_batch_size = batch_size

        # Sample data
        target_data = next(iter_loader).cpu().numpy()  # Shape: (B, C, H, W)
        # target = next(iter_loader)
        # std = 0.3
        # noise = torch.randn_like(target)  # Noise ~ N(0, 1)
        # target_data = (target + std * noise).cpu().numpy()
        #sample_prior
        if prior is not None:
            prior_data = prior.sample_n(batch_size).cpu().numpy()


        if batch_idx <= batch_num:
            sample_data = next(samples_dataloader).cpu().numpy()
        

        # Loop over the batch
        for i in range(current_batch_size):
            total_samples += 1
            if batch_idx <= batch_num:
                model_samples += 1

            # Loop over components
            for j in range(C):
                for k in range(C):
                    if (j >= k):
                        # Extract images
                        target_image_j = target_data[i, j, :, :]
                        target_image_k = target_data[i, k, :, :]
                        if batch_idx <= batch_num:
                            sample_image_j = sample_data[i, j, :, :]
                            sample_image_k = sample_data[i, k, :, :]
                        if prior is not None:
                            prior_image_j = prior_data[i, j, :, :]
                            prior_image_k = prior_data[i, k, :, :]

                        # Compute power spectra
                        util_obj = util()
                        ell_temp, bin_spectrum_target = util_obj.get_2d_power(nx, dx, target_image_j, target_image_k, size+1, scale='log')
                        if batch_idx <= batch_num:
                            _, bin_spectrum_sample = util_obj.get_2d_power(nx, dx, sample_image_j, sample_image_k, size+1, scale='log')
                        if prior is not None:
                            _, bin_spectrum_prior = util_obj.get_2d_power(nx, dx, prior_image_j, prior_image_k, size+1, scale='log')

                        # Store ell values once
                        if ell is None:
                            ell = ell_temp[1:]  # Exclude the first bin (zero frequency)

                        # Accumulate sums and sums of squares
                        bin_spectrum_target_real = np.real(bin_spectrum_target)  # Exclude the first bin
                        spectra_sum_target[j, k, :] += bin_spectrum_target_real
                        spectra_sq_sum_target[j, k, :] += bin_spectrum_target_real**2

                        if prior is not None:
                            bin_spectrum_prior_real = np.real(bin_spectrum_prior)
                            spectra_sum_prior[j, k, :] += bin_spectrum_prior_real

                        


                        if batch_idx <= batch_num:
                            bin_spectrum_sample_real = np.real(bin_spectrum_sample)
                            spectra_sum_sample[j, k, :] += bin_spectrum_sample_real
                            spectra_sq_sum_sample[j, k, :] += bin_spectrum_sample_real**2

    # Compute means
    mean_spectrum_target = spectra_sum_target / total_samples
    mean_spectrum_sample = spectra_sum_sample / model_samples
    if prior is not None:
        mean_spectrum_prior = spectra_sum_prior / total_samples

    # Compute standard errors
    std_spectrum_target = np.sqrt((spectra_sq_sum_target / total_samples) - mean_spectrum_target**2) / np.sqrt(total_samples)
    std_spectrum_sample = np.sqrt((spectra_sq_sum_sample / model_samples) - mean_spectrum_sample**2) / np.sqrt(model_samples)

    print('number of model samples = ', model_samples)
    print('number of target and prior samples = ', total_samples)
    fig_diff, axs_diff = plt.subplots(C, C, figsize=(3*C, 3*C))  # For fractional difference plots
    fig_pow, axs_pow   = plt.subplots(C, C, figsize=(3*C, 3*C))  # For power spectrum plots

    for j in range(C):
        for k in range(C):
            # We only plot for j >= k (as per your condition)
            if j >= k:
                # ---------- Compute fractional difference ----------
                y = (mean_spectrum_sample[j, k, :] - mean_spectrum_target[j, k, :]) / mean_spectrum_target[j, k, :]

                partial_X1 = 1 / mean_spectrum_target[j, k, :]
                partial_X2 = - (mean_spectrum_sample[j, k, :] - mean_spectrum_target[j, k, :]) / (mean_spectrum_target[j, k, :]**2)
                y_err = np.sqrt((partial_X1 * std_spectrum_sample[j, k, :])**2 +
                                (partial_X2 * std_spectrum_target[j, k, :])**2)

                # ---------- Plot fractional difference in subplot ----------
                ax_diff = axs_diff[j, k]  # Access the subplot at row j, col k
                ax_diff.errorbar(ell, y, yerr=y_err, fmt='.-', ecolor='red', label='(Noised - Unnoised)/Unnoised')
                ax_diff.semilogx(ell, np.zeros_like(ell), 'k--')
                ax_diff.set_ylim(-0.2, 0.1)
                ax_diff.set_ylabel('Fractional Difference')
                ax_diff.set_xlabel(r'$\ell$')
                ax_diff.set_title(f"{comps[j]} x {comps[k]}")
                ax_diff.legend()

                # ---------- Plot power spectra in subplot ----------
                ax_pow = axs_pow[j, k]
                ax_pow.errorbar(ell, mean_spectrum_target[j, k, :],
                                yerr=std_spectrum_target[j, k, :],
                                fmt='.-', label='Unnoised')
                ax_pow.errorbar(ell, mean_spectrum_sample[j, k, :],
                                yerr=std_spectrum_sample[j, k, :],
                                fmt='.-', label='Noised')
                if prior is not None:
                    ax_pow.plot(ell, mean_spectrum_prior[j, k, :], label='Prior')
                ax_pow.set_xscale('log')
                if j == k:
                    ax_pow.set_yscale('log')
                ax_pow.set_ylabel(r'$C_{\ell}$')
                ax_pow.set_xlabel(r'$\ell$')
                ax_pow.set_title(f"{comps[j]} x {comps[k]}")
                ax_pow.legend()

    # Tight layout for each figure
    fig_diff.tight_layout()
    fig_pow.tight_layout()

    # Save the figures
    diff_path = os.path.join(savedir, f"{step}_all_fractional_difference.png")
    pow_path  = os.path.join(savedir, f"{step}_all_power_spectra.png")

    print(f"Saving fractional difference figure at: {diff_path}")
    fig_diff.savefig(diff_path)
    plt.close(fig_diff)

    print(f"Saving power spectrum figure at: {pow_path}")
    fig_pow.savefig(pow_path)
    plt.close(fig_pow)

config_file = 'configs/HCC_prior_best_model_256x256_all_levels.py'
cf = SourceFileLoader('cf', config_file).load_module()

bdir = cf.val_dataset_path
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

loader_unnoised = DataLoader(dataset, batch_size=cf.sample_batch_size, shuffle=True, pin_memory=True)
iter_unnoised = iter(loader_unnoised)

config_file = 'configs/HCC_prior_best_model_256x256_all_levels_kap_noise_0.01.py'
cf = SourceFileLoader('cf', config_file).load_module()

bdir = cf.val_dataset_path
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
loader_noised = DataLoader(dataset, batch_size=cf.sample_batch_size, shuffle=True, pin_memory=True)
iter_noised = iter(loader_noised)

#dir to save plots
if not os.path.exists(cf.plotSaveDir):
    os.makedirs(cf.plotSaveDir)
plot_power(iter_unnoised, iter_noised, None, 10, cf.plotSaveDir, cf.sample_batch_size, 100)