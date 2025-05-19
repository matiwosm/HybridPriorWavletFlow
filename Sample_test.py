import os
from collections import defaultdict
import re
import torch
import random
import numpy as np
from data_loader import *
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
import tqdm

plt.style.use('ggplot')

seed = 42 #786
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.autograd.set_detect_anomaly(True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

torch.set_default_dtype(torch.float64)

def setup_colormap_and_norm(data, j, min_max):
    min_val = np.amin(data[:, j, :, :])
    max_val = np.amax(data[:, j, :, :])
    
    min_val = -1 * min_max[j]
    max_val = min_max[j]
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    
    if j == 4:
        min_val = np.amin(data[0, j, :, :])
        max_val = min_max[j]
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        cmap = plt.get_cmap("gray")
    elif j == 0:
        cmap = plt.get_cmap("RdBu_r")
    else:
        cmap = plt.get_cmap("RdYlBu_r")
    
    return cmap, norm

def plot_samples(data, level):
    COMPS = ['kap', 'cib']
    MIN_MAX = [3, 3, 50, 50, 50]

    fig, axs = plt.subplots(data.shape[0], data.shape[1], figsize=(2 * data.shape[1], 2 * data.shape[0]), sharex='row', sharey='row', dpi=200)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            axis = np.arange(0, data.shape[2], 1)
            cmap, norm = setup_colormap_and_norm(data, j, MIN_MAX)
            axs[i, j].contourf(axis, axis, data[i, j, :, :], cmap=cmap, norm=norm, levels=100, zorder=0)
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.show()


def compute_and_plot_minkowski_functionals_no_prior(
    model, 
    loader, 
    cf, 
    mean_stds_all_levels, 
    device, 
    cond_on_target=False,
    n_thresholds=50,
    max_iterations=3000
):
    """
    1) Iterates over model and iter_loader to collect final reconstructed maps (Target, Sample).
    2) Computes and accumulates Minkowski functionals (V0, V1, V2) at multiple thresholds for each channel.
    3) Produces 2*C figures total if there are C channels:
       - For each channel, one figure (3 subplots) for (V0, V1, V2) comparing Target vs. Sample
       - Another figure (3 subplots) for fractional differences of each Minkowski functional

    Parameters
    ----------
    model : YourModelClass
        The model from which we can get final sample maps via
            samples = model.sample(...)[-1]
    iter_loader : iterator / DataLoader
        Yields batches of data for the model to process.
    cf : config object
    mean_stds_all_levels : ...
        Stats for normalization if needed by model.sample().
    device : torch.device
        Device for PyTorch computations.
    cond_on_target : bool
        Whether model.sample() uses cond_on_target = True/False. Only set True for debugging.
    n_thresholds : int
        Number of thresholds to sample between the global min and max of each channel.
    max_iterations : int
        Number of iterations to run from iter_loader.

    Saves
    -----
    - For each channel, two figures:
      1) Minkowski_Functionals_Channel_{c_i}_Absolute.png  (V0, V1, V2 vs threshold)
      2) Minkowski_Functionals_Channel_{c_i}_FracDiff.png  ((Sample - Target)/Target)
    """

    Minkowski2D = minkowski.functionals
    ##########################################################################
    # 1) First Pass: find global min & max per channel across the dataset
    ##########################################################################
    global_min = None
    global_max = None
    count_batches = 0

    iter_loader = iter(loader)
    for i in range(max_iterations):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break

        if i % 10 == 0:
            print(f"first pass Processed {i} iterations...")

        count_batches += 1
        target = batch.to(device)
        #noising data
        # target = apply_noise_torch_vectorized(target, cf.channels_to_get, cf.noise_dict)
        # target = scale_data_pt(target, cf.channels_to_get)

        # Get final maps (Target, Sample)
        tar_map, smp_map = get_final_maps_from_model(
            model, target, mean_stds_all_levels, cf, cond_on_target
        )

        tar_map = tar_map.detach().cpu().numpy()  # (B, C, H, W)
        smp_map = smp_map.detach().cpu().numpy()

        # Combine them to get a universal min/max
        combined = np.concatenate([tar_map, smp_map], axis=0)  # shape (2B, C, H, W)
        q_min = 2.0   # or 5, or 2, etc.
        q_max = 98.0  # or 95, etc.

        # after you get 'combined' for each batch:
        if global_min is None:
            # shape (C,)
            global_min = np.percentile(combined, q_min, axis=(0, 2, 3))
            global_max = np.percentile(combined, q_max, axis=(0, 2, 3))
        else:
            global_min = np.minimum(global_min, np.percentile(combined, q_min, axis=(0, 2, 3)))
            global_max = np.maximum(global_max, np.percentile(combined, q_max, axis=(0, 2, 3)))

    if global_min is None or global_max is None:
        print("No data found in iter_loader.")
        return

    n_channels = len(global_min)
    thresholds = []
    for c in range(n_channels):
        thr = np.linspace(global_min[c], global_max[c], n_thresholds)
        thresholds.append(thr)


    ##########################################################################
    # 2) Second Pass: accumulate Minkowski functionals
    ##########################################################################
    # We'll store sums and sums-of-squares for V0, V1, V2 for Target, Sample
    # shape: (C, n_thresholds)
    MF = {
        'target': {
            'V0_sum':   np.zeros((n_channels, n_thresholds)),
            'V0_sum2':  np.zeros((n_channels, n_thresholds)),
            'V1_sum':   np.zeros((n_channels, n_thresholds)),
            'V1_sum2':  np.zeros((n_channels, n_thresholds)),
            'V2_sum':   np.zeros((n_channels, n_thresholds)),
            'V2_sum2':  np.zeros((n_channels, n_thresholds)),
        },
        'sample': {
            'V0_sum':   np.zeros((n_channels, n_thresholds)),
            'V0_sum2':  np.zeros((n_channels, n_thresholds)),
            'V1_sum':   np.zeros((n_channels, n_thresholds)),
            'V1_sum2':  np.zeros((n_channels, n_thresholds)),
            'V2_sum':   np.zeros((n_channels, n_thresholds)),
            'V2_sum2':  np.zeros((n_channels, n_thresholds)),
        },
    }
    sample_count = 0
    iter_loader = iter(loader)

    for i in range(max_iterations):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
        
        if i % 10 == 0:
            print(f"Second pass Processed {i} iterations...")

        target = batch.to(device)
        # target = apply_noise_torch_vectorized(target, cf.channels_to_get, cf.noise_dict)
        # target = scale_data_pt(target, cf.channels_to_get)
        
        tar_map, smp_map = get_final_maps_from_model(
            model, target, mean_stds_all_levels, cf, cond_on_target
        )

        tar_map = tar_map.detach().cpu().numpy()
        smp_map = smp_map.detach().cpu().numpy()
        B2, C2, H2, W2 = tar_map.shape
        sample_count += B2

        for b_i in range(B2):
            for c_i in range(C2):
                map_tar = tar_map[b_i, c_i]  # (H, W)
                map_smp = smp_map[b_i, c_i]
                thr_array = thresholds[c_i]

                # Compute Minkowski functionals for each threshold
                V0_t, V1_t, V2_t = compute_MFs_for_map(map_tar, thr_array)
                V0_s, V1_s, V2_s = compute_MFs_for_map(map_smp, thr_array)

                # Accumulate sums and sums-of-squares
                for idx in range(n_thresholds):
                    MF['target']['V0_sum'][c_i, idx]  += V0_t[idx]
                    MF['target']['V0_sum2'][c_i, idx] += V0_t[idx]**2
                    MF['target']['V1_sum'][c_i, idx]  += V1_t[idx]
                    MF['target']['V1_sum2'][c_i, idx] += V1_t[idx]**2
                    MF['target']['V2_sum'][c_i, idx]  += V2_t[idx]
                    MF['target']['V2_sum2'][c_i, idx] += V2_t[idx]**2

                    MF['sample']['V0_sum'][c_i, idx]  += V0_s[idx]
                    MF['sample']['V0_sum2'][c_i, idx] += V0_s[idx]**2
                    MF['sample']['V1_sum'][c_i, idx]  += V1_s[idx]
                    MF['sample']['V1_sum2'][c_i, idx] += V1_s[idx]**2
                    MF['sample']['V2_sum'][c_i, idx]  += V2_s[idx]
                    MF['sample']['V2_sum2'][c_i, idx] += V2_s[idx]**2

    ##########################################################################
    # 3) Compute means, errors, and produce plots
    ##########################################################################
    N = float(sample_count)
    mf_names = ['V0', 'V1', 'V2']
    xlabels  = ['Threshold'] * 3
    ylabels  = [r'$V_0(\nu)$', r'$V_1(\nu)$', r'$V_2(\nu)$']

    for c_i in range(n_channels):
        thr_array = thresholds[c_i]

        # Compute means and errors
        final_res = {}
        for map_type in ['target', 'sample']:
            final_res[map_type] = {}
            for mf in mf_names:
                sum_  = MF[map_type][mf+'_sum'][c_i]
                sum2_ = MF[map_type][mf+'_sum2'][c_i]
                mean_ = sum_ / N
                var_  = (sum2_ / N) - mean_**2
                err_  = np.sqrt(var_) / np.sqrt(N)
                final_res[map_type][mf] = (mean_, err_)

        # --- Figure 1: Minkowski functionals for T vs S ---
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
        for ax_i, mf in enumerate(mf_names):
            ax = axes1[ax_i]
            (t_mean, t_err) = final_res['target'][mf]
            (s_mean, s_err) = final_res['sample'][mf]

            ax.errorbar(thr_array, t_mean, yerr=t_err, label='Target', fmt='.-')
            ax.errorbar(thr_array, s_mean, yerr=s_err, label='Sample', fmt='.-')
            ax.set_xlabel(xlabels[ax_i])
            ax.set_ylabel(ylabels[ax_i])
            ax.set_title(f"Channel={c_i}, {mf}")
            ax.legend()
        fig1.tight_layout()
        os.makedirs(cf.plotSaveDir, exist_ok=True)
        out_path1 = os.path.join(cf.plotSaveDir, f"Minkowski_Channel_{c_i}_Absolute.png")
        fig1.savefig(out_path1)
        plt.close(fig1)

        # --- Figure 2: Fractional differences (S - T)/T ---
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        for ax_i, mf in enumerate(mf_names):
            ax = axes2[ax_i]
            (t_mean, t_err) = final_res['target'][mf]
            (s_mean, s_err) = final_res['sample'][mf]

            # (S - T)/T
            denom = np.where(t_mean != 0, t_mean, 1e-14)
            fd_s = (s_mean - t_mean) / denom

            # approximate error
            s_err_term = s_err / denom
            t_err_term = -(s_mean - t_mean)/(denom**2)*t_err
            fd_s_err = np.sqrt(s_err_term**2 + t_err_term**2)

            ax.errorbar(thr_array, fd_s, yerr=fd_s_err, label='(Sample-Target)/Target', fmt='.-')
            ax.axhline(0.0, color='k', linestyle='--')
            ax.set_xlabel(xlabels[ax_i])
            ax.set_ylim(-0.1, 0.1)
            ax.set_ylabel(f"Frac. Diff. of {ylabels[ax_i]}")
            ax.set_title(f"Channel={c_i}, FracDiff {mf}")
            ax.legend()

        fig2.tight_layout()
        out_path2 = os.path.join(cf.plotSaveDir, f"Minkowski_Channel_{c_i}_FracDiff.png")
        fig2.savefig(out_path2)
        plt.close(fig2)

    print("Minkowski functionals computed and plots saved in " + cf.plotSaveDir + " folder.")


###############################################################################
# Helper function to retrieve final maps (Target, Sample) from the model
###############################################################################
def get_final_maps_from_model(model, target, mean_stds_all_levels, cf, cond_on_target=True):
    tar_map = target

    # Now get the final sample from the model:
    # samples = model.sample(...)[-1]
    samples = model.sample(mean_stds_all_levels, 
                           target=target, 
                           n_batch=cf.sample_batch_size, 
                           cond_on_target=cond_on_target, 
                           comp='low')[-1]
    return tar_map, samples
   

def compute_MFs_for_map(map_2d, thresholds):
    """
    Compute (V0, V1, V2) across all thresholds for a single 2D numpy array.
    Requires quantimpy.minkowski.Minkowski2D or a similar tool.
    """
    Minkowski2D = minkowski.functionals

    V0 = np.zeros_like(thresholds, dtype=float)
    V1 = np.zeros_like(thresholds, dtype=float)
    V2 = np.zeros_like(thresholds, dtype=float)

    for i, thr in enumerate(thresholds):
        # Use a boolean mask:
        mask = (map_2d >= thr)  # mask is now bool

        mk = Minkowski2D(mask)
        
        V0[i] = mk[0]
        V1[i] = mk[1]
        V2[i] = mk[2]

    return V0, V1, V2




def compute_and_plot_all_power_spectra(model, iter_loader, cf, mean_stds_all_levels, device, plotSaveDir, comps_to_get, nLevels=6,
                                    get_train_modes=False, cond_on_target=False, max_iterations=3000):
    """
    Compute power spectra statistics over many batches and produce two figures:
    1. Fractional difference plots (one figure)
    2. Absolute spectra plots (one figure)

    Each figure has a grid of subplots:
    - Rows: levels
    - Columns: component pairs (j,k)
    """

    def freq_type(level):
        return 'low' if (level == cf.baseLevel or not get_train_modes) else 'high'

    def get_component_names(components, freq='high'):
        """
        Generate component names based on the input list of component strings.

        :param components: List of component strings, e.g., ['kap', 'cib', 'tsz'].
        :param freq: Either 'high' or 'low'. Determines the naming convention.
        :return: List of component names.
        """
        if len(components) == 0:
            components = ['kappa', 'ksz', 'tsz', 'cib', 'rad']
        if freq == 'low':
            return [f"l_{comp}" for comp in components]
        elif freq == 'high':
            suffixes = ['hori', 'ver', 'dia']
            return [f"h_{suffix}_{comp}" for comp in components for suffix in suffixes]
        else:
            raise ValueError("Invalid value for 'freq'. Must be 'low' or 'high'.")
    timer = 0
    bin_sizes_log = [1, 1, 2, 4, 5, 10, 13, 15, 17, 20]
    util_obj = util()

    accumulators = {}
    total_count = 0

    if cond_on_target:
        print('NOTICE: conditioning on target, remove conditioning for real results')

    # Accumulate statistics across many batches
    for i in tqdm.tqdm(range(max_iterations)):
        try:
            target = next(iter_loader).to(device)
        except StopIteration:
            break
        #noising data
        # target = apply_noise_torch_vectorized(target, cf.channels_to_get, cf.noise_dict)
        # target = scale_data_pt(target, cf.channels_to_get)

        
        # if i % 10 == 0:
        #     print(f"Processed {i} iterations... in {timer} seconds")

        latents = model.sample_latents(n_batch=cf.sample_batch_size)
        
        if get_train_modes:
            start = time.time()
            samples = model.sample(mean_stds_all_levels, target=target, n_batch=cf.sample_batch_size, cond_on_target=cond_on_target, comp='high')
            timer += (time.time() - start)
            data = get_training_modes(target, model, cf.nLevels, cf.baseLevel)
        else:
            start = time.time()
            samples = model.sample(mean_stds_all_levels, target=target, n_batch=cf.sample_batch_size, cond_on_target=cond_on_target, comp='low')
            timer += (time.time() - start)
            data = get_sample_modes(target, model, cf.nLevels, cf.baseLevel)
            latents[-(cf.nLevels - cf.baseLevel):] = data[-(cf.nLevels - cf.baseLevel):]
        
        B = target.shape[0]
        total_count += B

        for lvl_idx, s_map in enumerate(samples):
            level = cf.baseLevel + lvl_idx
            d_map = data[lvl_idx]
            p_map = latents[level] if level < len(latents) else None
            if p_map is None:
                continue

            d_map = d_map.detach().cpu().numpy()
            s_map = s_map.detach().cpu().numpy()
            p_map = p_map.detach().cpu().numpy()
            # print(d_map.shape, s_map.shape, p_map.shape)
            B, C, H, W = d_map.shape
            freq = freq_type(level)
            comps = get_component_names(comps_to_get, freq)
            size_idx = level if level < len(bin_sizes_log) else -1
            size = bin_sizes_log[size_idx]
            dx = (0.5/60. * np.pi/180.) * 2**(nLevels - level)
            nx = W

            if level not in accumulators:
                accumulators[level] = {'target': {}, 'sample': {}, 'prior': {}, 'freq': freq, 'comps': comps}

            for j in range(C):
                for k in range(C):
                    if j == k:  # Add (j - k) == 3 condition if needed
                        bt_list, bs_list, bp_list = [], [], []
                        for b_i in range(B):
                            ell, bt = util_obj.get_2d_power(nx, dx, d_map[b_i, j], d_map[b_i, k], size+1, scale='log')
                            _,   bs = util_obj.get_2d_power(nx, dx, s_map[b_i, j], s_map[b_i, k], size+1, scale='log')
                            _,   bp = util_obj.get_2d_power(nx, dx, p_map[b_i, j], p_map[b_i, k], size+1, scale='log')
                            bt_list.append(np.real(bt))
                            bs_list.append(np.real(bs))
                            bp_list.append(np.real(bp))

                        bt_arr = np.array(bt_list)
                        bs_arr = np.array(bs_list)
                        bp_arr = np.array(bp_list)

                        if (j, k) not in accumulators[level]['target']:
                            shape = bt_arr.shape[1]
                            accumulators[level]['target'][(j,k)] = {'sum': np.zeros(shape), 'sum2': np.zeros(shape)}
                            accumulators[level]['sample'][(j,k)] = {'sum': np.zeros(shape), 'sum2': np.zeros(shape)}
                            accumulators[level]['prior'][(j,k)]  = {'sum': np.zeros(shape), 'sum2': np.zeros(shape)}
                            if 'ell' not in accumulators[level]:
                                accumulators[level]['ell'] = ell

                        accumulators[level]['target'][(j,k)]['sum']  += bt_arr.sum(axis=0)
                        accumulators[level]['target'][(j,k)]['sum2'] += (bt_arr**2).sum(axis=0)
                        accumulators[level]['sample'][(j,k)]['sum']  += bs_arr.sum(axis=0)
                        accumulators[level]['sample'][(j,k)]['sum2'] += (bs_arr**2).sum(axis=0)
                        accumulators[level]['prior'][(j,k)]['sum']   += bp_arr.sum(axis=0)
                        accumulators[level]['prior'][(j,k)]['sum2']  += (bp_arr**2).sum(axis=0)

    # Compute final statistics
    results = {}
    max_pairs = 0
    levels_sorted = sorted(accumulators.keys())

    for level in levels_sorted:
        lvl_data = accumulators[level]
        ell = lvl_data['ell']
        freq = lvl_data['freq']
        comps = lvl_data['comps']
        N = total_count

        results[level] = {'freq': freq, 'comps': comps, 'ell': ell, 'pairs': {}}
        pairs = list(lvl_data['target'].keys())
        max_pairs = max(max_pairs, len(pairs))

        for (j,k) in pairs:
            tgt_sum, tgt_sum2 = lvl_data['target'][(j,k)]['sum'], lvl_data['target'][(j,k)]['sum2']
            smp_sum, smp_sum2 = lvl_data['sample'][(j,k)]['sum'], lvl_data['sample'][(j,k)]['sum2']
            pr_sum,  pr_sum2  = lvl_data['prior'][(j,k)]['sum'], lvl_data['prior'][(j,k)]['sum2']

            tgt_mean = tgt_sum / N
            smp_mean = smp_sum / N
            pr_mean  = pr_sum / N

            tgt_var = (tgt_sum2 / N) - (tgt_mean**2)
            smp_var = (smp_sum2 / N) - (smp_mean**2)
            pr_var  = (pr_sum2 / N) - (pr_mean**2)

            tgt_err = np.sqrt(tgt_var) / np.sqrt(N)
            smp_err = np.sqrt(smp_var) / np.sqrt(N)
            pr_err  = np.sqrt(pr_var) / np.sqrt(N)

            y = (smp_mean - tgt_mean) / tgt_mean
            pX1 = 1.0 / tgt_mean
            pX2 = -(smp_mean - tgt_mean) / (tgt_mean**2)
            y_err = np.sqrt((pX1 * smp_err)**2 + (pX2 * tgt_err)**2)

            results[level]['pairs'][(j,k)] = {
                'tgt_mean': tgt_mean, 'tgt_err': tgt_err,
                'smp_mean': smp_mean, 'smp_err': smp_err,
                'pr_mean': pr_mean,   'pr_err': pr_err,
                'y': y, 'y_err': y_err
            }

    # Create two figures
    nrows, ncols = len(levels_sorted), max_pairs
    fig_diff, axes_diff = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    fig_spec, axes_spec = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))

    if nrows == 1:
        axes_diff = [axes_diff]
        axes_spec = [axes_spec]
    if ncols == 1:
        for r in range(nrows):
            axes_diff[r] = [axes_diff[r]]
            axes_spec[r] = [axes_spec[r]]

    for r, level in enumerate(levels_sorted):
        ell = results[level]['ell']
        freq = results[level]['freq']
        comps = results[level]['comps']
        pairs_list = sorted(results[level]['pairs'].keys(), key=lambda x: (x[0], x[1]))

        for c in range(ncols):
            axd = axes_diff[r][c]
            axs = axes_spec[r][c]

            if c < len(pairs_list):
                (j,k) = pairs_list[c]
                res = results[level]['pairs'][(j,k)]

                # Fractional difference plot
                axd.errorbar(ell[1:], res['y'][:], yerr=res['y_err'][:], fmt='.-', ecolor='red',
                             label='(Sample - Target)/Target')
                
                
                axd.semilogx(ell[1:], np.zeros(len(ell[1:])), 'k-')
                if level == cf.nLevels:
                    axd.semilogx(np.ones(len(ell[1:]))*21600, np.linspace(-0.08, 0.08, len(ell[1:])))
                axd.set_ylim(-0.1, 0.1)
                axd.set_ylabel('$C_{\\ell}$')
                axd.set_xlabel('$\\ell$')
                axd.set_title(f"L={level}, {freq}, {comps[j]}-{comps[k]}", fontsize=10)
                axd.legend(fontsize=8)
                # print(res['tgt_mean'], res['smp_mean'], res['pr_mean'])
                # Absolute spectra plot
                axs.errorbar(ell[1:], res['tgt_mean'][:], yerr=res['tgt_err'][:], fmt='.-', label='Target')
                axs.errorbar(ell[1:], res['smp_mean'][:], yerr=res['smp_err'][:], fmt='.-', label='Sample')
                axs.errorbar(ell[1:], res['pr_mean'][:],  yerr=res['pr_err'][:],  fmt='.-', label='Prior')
                axs.set_xscale('log')
                axs.set_yscale('log')
                axs.set_ylabel('$C_{\\ell}$')
                axs.set_xlabel('$\\ell$')
                axs.set_title(f"L={level}, {freq}, {comps[j]}-{comps[k]}", fontsize=10)
                axs.legend(fontsize=8)
            else:
                axd.axis('off')
                axs.axis('off')

    fig_diff.tight_layout()
    fig_spec.tight_layout()

    # Add a bit more spacing
    fig_diff.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_spec.subplots_adjust(wspace=0.3, hspace=0.3)
    if get_train_modes:
        diff_path = os.path.join(plotSaveDir, 'Training_modes_fractional_difference_all_levels.png')
        spec_path = os.path.join(plotSaveDir, 'Training_modes_absolute_spectra_all_levels.png')
    else:
        diff_path = os.path.join(plotSaveDir, 'Sample_modes_fractional_difference_all_levels.png')
        spec_path = os.path.join(plotSaveDir, 'Sample_modes_absolute_spectra_all_levels.png')
    fig_diff.savefig(diff_path, dpi=150)
    fig_spec.savefig(spec_path, dpi=150)

    plt.close(fig_diff)
    plt.close(fig_spec)

    print("All plots saved successfully.")
    print(f'plot paths = {diff_path} and {spec_path}')

def compute_and_plot_bispectra(model, iter_loader, cf, mean_stds_all_levels, device, plotSaveDir, comps_to_get,
                             cond_on_target=False, max_iterations=3000):
    """
    Compute bispectra for full maps (not DWT levels) and compare target vs samples
    - One figure with fractional differences
    - One figure with absolute bispectra
    - Rows: different channels/components
    """
    
    # Configuration
    util_obj = util()
    bin_size = 9  # Number of bispectrum bins
    n_channels = len(comps_to_get)
    
    timer = 0
    
    # Initialize accumulators
    accumulators = {
        'target': {c: {'sum': 0, 'sum2': 0} for c in range(n_channels)},
        'sample': {c: {'sum': 0, 'sum2': 0} for c in range(n_channels)},
        'count': 0
    }

    # Batch processing
    for i in range(max_iterations):
        try:
            target = next(iter_loader).to(device)
        except StopIteration:
            break
        print(i)    
        # Preprocess target
        # target = apply_noise_torch_vectorized(target, cf.channels_to_get, cf.noise_dict)
        # target = scale_data_pt(target, cf.channels_to_get)
        
        # Generate samples (full map only)
        with torch.no_grad():
            sample = model.sample(
                mean_stds_all_levels,
                target=target,
                n_batch=cf.sample_batch_size,
                cond_on_target=cond_on_target,
                comp='low'
            )[-1].detach().cpu().numpy()
            
        target = target.detach().cpu().numpy()
        accumulators['count'] += target.shape[0]
        
        # Process batch
        nx = target.shape[-1]
        dx = (0.5/60 * np.pi/180)
        for j in range(target.shape[0]): 
            for c in range(n_channels):
                # Target bispectrum
                ell_t, bisp_t = util_obj.get_2d_bispectrum_monte_carlo(nx, dx, target[j,c], bin_size, scale='log')
                
                # Sample bispectrum
                _, bisp_s = util_obj.get_2d_bispectrum_monte_carlo(nx, dx, sample[j,c], bin_size, scale='log')

                # Accumulate statistics
                accumulators['target'][c]['sum'] += bisp_t
                accumulators['target'][c]['sum2'] += bisp_t**2
                accumulators['sample'][c]['sum'] += bisp_s
                accumulators['sample'][c]['sum2'] += bisp_s**2

    # Final statistics
    n_total = accumulators['count']
    results = {}
    
    for c in range(n_channels):
        t_mean = accumulators['target'][c]['sum'] / n_total
        t_err = np.sqrt(accumulators['target'][c]['sum2']/n_total - t_mean**2) / np.sqrt(n_total)
        
        s_mean = accumulators['sample'][c]['sum'] / n_total
        s_err = np.sqrt(accumulators['sample'][c]['sum2']/n_total - s_mean**2) / np.sqrt(n_total)
        
        # Fractional difference
        frac_diff = (s_mean - t_mean) / t_mean
        frac_err = np.sqrt((s_err/t_mean)**2 + (s_mean*t_err/t_mean**2)**2)
        
        results[c] = {
            'ell': ell_t,
            'target_mean': t_mean,
            'target_err': t_err,
            'sample_mean': s_mean,
            'sample_err': s_err,
            'frac_diff': frac_diff,
            'frac_err': frac_err
        }

    # Plotting
    fig_diff, ax_diff = plt.subplots(n_channels, 1, figsize=(8, 4*n_channels))
    fig_spec, ax_spec = plt.subplots(n_channels, 1, figsize=(8, 4*n_channels))

    for c in range(n_channels):
        res = results[c]
        ell = res['ell']
        
        # Fractional difference plot
        ax_diff[c].errorbar(ell[1:], res['frac_diff'][1:], yerr=res['frac_err'][1:],
                          fmt='o-', capsize=3, label=f'Channel {comps_to_get[c]}')
        ax_diff[c].axhline(0, color='k', linestyle='--')
        ax_diff[c].set(xscale='log', ylim=(-0.2, 0.2),
                     xlabel='$\ell$', ylabel='(Sample - Target)/Target')
        ax_diff[c].legend()
        
        # Absolute spectra plot
        ax_spec[c].errorbar(ell[1:], res['target_mean'][1:], yerr=res['target_err'][1:],
                          fmt='o-', label='Target')
        ax_spec[c].errorbar(ell[1:], res['sample_mean'][1:], yerr=res['sample_err'][1:],
                          fmt='o-', label='Sample')
        ax_spec[c].set(xscale='log', yscale='log',
                     xlabel='$\ell$', ylabel='$B_\ell$')
        ax_spec[c].legend()

    # Save plots
    fig_diff.tight_layout()
    fig_spec.tight_layout()
    
    diff_path = os.path.join(plotSaveDir, 'fullmap_bispectrum_fractional.png')
    spec_path = os.path.join(plotSaveDir, 'fullmap_bispectrum_absolute.png')
    
    fig_diff.savefig(diff_path, dpi=150, bbox_inches='tight')
    fig_spec.savefig(spec_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig_diff)
    plt.close(fig_spec)

    print(f"Bispectrum plots saved to:\n{diff_path}\n{spec_path}")
    return results



#get all the high frequecy modes + the lowest low frequecy mode from x
def get_training_modes(x, model, nLevels, BaseLevel):
    mode_list = []
    for i in range(nLevels - BaseLevel):
        y = model.dwt.forward(x)['high']
        x = model.dwt.forward(x)['low']
        mode_list.append(y)
        if i == ((nLevels - BaseLevel) - 1):
            mode_list.append(x)
    mode_list.reverse()
    return mode_list

#get all the low frequecy modes from x
def get_sample_modes(x, model, nLevels, BaseLevel):
    mode_list = []
    mode_list.append(x)
    for i in range(nLevels - BaseLevel):
        x = model.dwt.forward(x)['low']
        mode_list.append(x)
    mode_list.reverse()
    return mode_list


def main():

    #load configs
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    p = SourceFileLoader('cf', f'{args.config}').load_module()
    directory_path = cf.saveDir
    print('loading models from', directory_path)


    # Read all file names from the directory
    file_list = os.listdir(directory_path)

    #This part of the code will automatically find the latest saved model in directory path
    #it will not work if you change the naming conventions of the saved model
    # Regular expression to extract i and j values
    pattern = re.compile(r"waveletflow-agora-(\d+)(?:-(\d+))?")

    # Dictionary to store file name and the maximum j for each i
    files_by_i = defaultdict(lambda: (None, -1))

    for file_name in file_list:
        if file_name.endswith(".pt"):  # Check if the file is a .pt file
            match = pattern.search(file_name)
            if match:
                i = int(match.group(1))
                j = int(match.group(2)) if match.group(2) is not None else 0
                if j > files_by_i[i][1]:
                    files_by_i[i] = (file_name, j)

    # Extracting the sorted list of files with maximum j for each i
    #selected files contain the file names of the models to be loaded
    selected_files = [info[0] for i, info in sorted(files_by_i.items())]
    [print(f'Using {ml_file} for level {i+1}\n') for i,ml_file in enumerate(selected_files)]
    print('loading normalization factors from ', cf.std_path, '\n')
    #dir to save plots
    if not os.path.exists(cf.plotSaveDir):
            os.makedirs(cf.plotSaveDir)

    prior_type = cf.priorType
    for i in range(p.baseLevel, cf.nLevels+1):
        p_level = i
        
        with open(cf.std_path, 'r') as f:
            mean_stds_all_levels = json.load(f)

        #load powerspectra
        #first determine wither the level contains low or high frequency compoenets
        if p_level == cf.baseLevel or p_level == cf.baseLevel + 1:
            dwt_level_number = cf.nLevels - cf.baseLevel
            if p_level == cf.baseLevel:
                freq='low'
                N = 1
            else:
                freq='high'
                N = 3
        else:
            dwt_level_number = cf.nLevels - p_level + 1
            freq='high'
            N = 3

        #load prior for the level
        nx = int(cf.imShape[-1]//(2**dwt_level_number))
        mean_stds_this_levels = mean_stds_all_levels[str(dwt_level_number-1)]
        print('Normalize data = ', cf.normalize[i])
        print('Norm type = ', cf.norm_type[i])
        if p_level not in (cf.gauss_priors):
            dx = (0.5/60. * np.pi/180.)*(2**(dwt_level_number))
            rfourier_shape = (N*cf.imShape[0], nx, int(nx/2 + 1), 2)
            df = pd.read_csv(cf.ps_path+str(nx)+'x'+str(nx)+'.dat', sep=";")
            print('loading power spectra from ', cf.ps_path+str(nx)+'x'+str(nx)+'.dat\n')
            df.columns = df.columns.str.strip()
            power_spec = df.values  # shape (N_ell, N_columns)

            # 3) Build dictionary {col_name -> col_index}
            colnames = list(df.columns)
            colname_to_index = {name: i for i, name in enumerate(colnames)}
            priortype = prior_type
            print('Normalize prior = ', cf.normalize_prior[i])
            if cf.unnormalize_prior[i] == False:
                norm_std = None
            else:
                print('Unnormalizing prior with precomputed stds')
                norm_std = mean_stds_this_levels
            prior = corr_prior.CorrelatedNormalDWTGeneral(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,cl_theo=power_spec,colname_to_index=colname_to_index,torch_device=device, freq=freq, n_channels=cf.imShape[0], prior_type=prior_type, norm_std=norm_std, normalize=cf.normalize_prior[i])
        else:
            if p_level == 200:
                temperature = torch.tensor([1.05, 1.0, 1.0, 1.05, 1.05, 1.05], dtype=torch.float64).to(device).view(N*cf.imShape[0], 1, 1)
            else:
                temperature = 1.0
            priortype = 'WN'
            shape = (N*cf.imShape[0], nx, nx)
            prior = corr_prior.SimpleNormal(torch.zeros(shape).to(device), temperature*torch.ones(shape).to(device))
        
        p.net_type = p.network[i]
        # print('p_level', p_level, ' nx ', nx, ' dwt_level_number ', dwt_level_number, freq, N, priortype)

        #load the models for all levels
        if i == p.baseLevel:
            model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype, device=device)
            model.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel], weights_only=True, map_location=device))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters for level {p_level}: {total_params} \n")
        else:
            model1 = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype, device=device)
            model1.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel], weights_only=True, map_location=device))
            total_params = sum(p.numel() for p in model1.parameters())
            print(f"Total number of parameters for level {p_level}: {total_params} \n")
            model.sub_flows[i] = model1.sub_flows[i]
            del model1
        
    model = model.to(device)
    # init act norm
    for i in range(p.baseLevel, cf.nLevels + 1):
        model.sub_flows[i].set_actnorm_init()
    model = model.eval()


    bdir = cf.val_dataset_path
    file = "data.mdb"
    print('dataset path = ', bdir)
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


    loader = DataLoader(dataset, batch_size=cf.sample_batch_size, shuffle=True, pin_memory=True, drop_last=True)

    print("len loader = ", len(loader))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters for all levels: {total_params} \n")

    #calculate and plot power spectra and minkowski functionals
    #caution: cond_on_target should be False for proper results. Set to True only for debuging.
    start = time.time()
    iter_loader = iter(loader)
    print('Calculating the power spectra of low-frequency DWT components')
    compute_and_plot_all_power_spectra(model, iter_loader, cf, mean_stds_all_levels, device, cf.plotSaveDir, cf.channels_to_get, nLevels=cf.nLevels, get_train_modes=False, cond_on_target=True, max_iterations=10)

    iter_loader = iter(loader)
    print('Calculating the power spectra of high-frequency DWT components')
    compute_and_plot_all_power_spectra(model, iter_loader, cf, mean_stds_all_levels, device, cf.plotSaveDir, cf.channels_to_get, nLevels=cf.nLevels, get_train_modes=True, cond_on_target=True, max_iterations=10)

#     iter_loader = iter(loader)
#     print('Calculating bispectrum of the final map')
#     compute_and_plot_bispectra(model, iter_loader, cf, mean_stds_all_levels, device, cf.plotSaveDir, cf.channels_to_get, cond_on_target=False, max_iterations=50)

#     print('Calculating Minkowski functionals of the final map')
#     compute_and_plot_minkowski_functionals_no_prior(
#     model, 
#     loader, 
#     cf, 
#     mean_stds_all_levels, 
#     device, 
#     cond_on_target=False,
#     n_thresholds=50,
#     max_iterations=len(loader),
# )

    print(time.time() - start)

    

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/example_config_hcc_prior.py", help='specify config')
    parser.add_argument('--data', type=str, default="agora", help='input data')
    args = parser.parse_args()
    main()

    