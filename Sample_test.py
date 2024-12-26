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
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
device = torch.device("cuda")
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
    iter_loader, 
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
        (Optional) if your code needs config details. Not strictly used here.
    mean_stds_all_levels : ...
        Stats for normalization if needed by your model.sample().
    device : torch.device
        Device for PyTorch computations.
    cond_on_target : bool
        Whether model.sample() uses cond_on_target = True/False.
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

    for i in range(max_iterations):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break

        if i % 10 == 0:
            print(f"first pass Processed {i} iterations...")

        count_batches += 1
        target = batch.to(device)


        # Get final maps (Target, Sample)
        tar_map, smp_map = get_final_maps_from_model(
            model, target, mean_stds_all_levels, cond_on_target
        )

        tar_map = tar_map.detach().cpu().numpy()  # (B, C, H, W)
        smp_map = smp_map.detach().cpu().numpy()

        # Combine them to get a universal min/max
        combined = np.concatenate([tar_map, smp_map], axis=0)  # shape (2B, C, H, W)
        if global_min is None:
            # shape (C,)
            global_min = combined.min(axis=(0,2,3))
            global_max = combined.max(axis=(0,2,3))
        else:
            global_min = np.minimum(global_min, combined.min(axis=(0,2,3)))
            global_max = np.maximum(global_max, combined.max(axis=(0,2,3)))

    if global_min is None or global_max is None:
        print("No data found in iter_loader.")
        return

    n_channels = len(global_min)
    thresholds = []
    for c in range(n_channels):
        thr = np.linspace(global_min[c], global_max[c], n_thresholds)
        thresholds.append(thr)

    # Re-initialize or re-create the DataLoader for second pass, if needed:
    # Example: iter_loader = reset_iter_loader_somehow(cf)
    # If your environment doesn't allow that easily,
    # you'll need to combine both passes into one or store all data in memory.

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

    # Try re-initializing loader again for second pass
    # iter_loader = reset_iter_loader_somehow(cf)
    # If you can't, place the Minkowski code in the same pass where you do min/max.

    for i in range(max_iterations):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
        
        if i % 10 == 0:
            print(f"Second pass Processed {i} iterations...")

        target = batch.to(device)
        tar_map, smp_map = get_final_maps_from_model(
            model, target, mean_stds_all_levels, cond_on_target
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
        os.makedirs("plots_mf", exist_ok=True)
        out_path1 = os.path.join("plots_mf", f"Minkowski_Channel_{c_i}_Absolute.png")
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
            ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel(f"Frac. Diff. of {ylabels[ax_i]}")
            ax.set_title(f"Channel={c_i}, FracDiff {mf}")
            ax.legend()

        fig2.tight_layout()
        out_path2 = os.path.join("plots_mf", f"Minkowski_Channel_{c_i}_FracDiff.png")
        fig2.savefig(out_path2)
        plt.close(fig2)

    print("Minkowski functionals computed and plots saved in 'plots_mf' folder.")


###############################################################################
# Helper function to retrieve final maps (Target, Sample) from the model
###############################################################################
def get_final_maps_from_model(model, target, mean_stds_all_levels, cond_on_target=True):
    """
    Return final (target_map, sample_map) from the model, both shape (B, C, H, W).
    The sample is taken from the last element of model.sample(...).
    Adjust as needed for your actual reconstruction pipeline.
    """
    import torch

    # Target is typically your input, so let's keep it as is:
    tar_map = target

    # Now get the final sample from the model:
    # samples = model.sample(...)[-1]
    samples = model.sample(mean_stds_all_levels, 
                           target=target, 
                           n_batch=64, 
                           cond_on_target=cond_on_target, 
                           comp='low')[-1]

    # Both tar_map, samples are (B, C, H, W)
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




def compute_and_plot_all_power_spectra(model, iter_loader, cf, mean_stds_all_levels, device, plotSaveDir, nLevels=6,
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

    def get_component_names(freq):
        return (['comp1_kap', 'comp1_cib'] if freq == 'low'
                else ['comp1_kap', 'comp2_kap', 'comp3_kap', 'comp1_cib', 'comp2_cib', 'comp3_cib'])

    bin_sizes_log = [2, 2, 3, 5, 5, 10, 13, 15, 17, 20]
    util_obj = util()

    accumulators = {}
    total_count = 0

    # Accumulate statistics across many batches
    for i in range(max_iterations):
        try:
            target = next(iter_loader).to(device)
        except StopIteration:
            break
        if i % 10 == 0:
            print(f"Processed {i} iterations...")

        latents = model.sample_latents()
        # latents = unnormalize_training(latents, mean_stds_all_levels, 2, cf.baseLevel)

        if get_train_modes:
            samples = model.sample(mean_stds_all_levels, target=target, n_batch=64, cond_on_target=cond_on_target, comp='high')
            data = get_training_modes(target, model, cf.nLevels, cf.baseLevel)
        else:
            samples = model.sample(mean_stds_all_levels, target=target, n_batch=64, cond_on_target=cond_on_target, comp='low')
            data = get_sample_modes(target, model, cf.nLevels, cf.baseLevel)
            latents[-(cf.nLevels - cf.baseLevel):] = data[-(cf.nLevels - cf.baseLevel):]

        B = target.shape[0]
        total_count += B

        for lvl_idx, d_map in enumerate(data):
            level = cf.baseLevel + lvl_idx
            s_map = samples[lvl_idx]
            p_map = latents[level] if level < len(latents) else None
            if p_map is None:
                continue

            d_map = d_map.detach().cpu().numpy()
            s_map = s_map.detach().cpu().numpy()
            p_map = p_map.detach().cpu().numpy()
            # print(d_map.shape, s_map.shape, p_map.shape)
            B, C, H, W = d_map.shape
            freq = freq_type(level)
            comps = get_component_names(freq)
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

def normalize_training(data, stds, comps, baselevel, nLevel):
    for i in range(len(data)):
        if data[i].shape[1] == comps:
            freq_type = 'low'
        elif data[i].shape[1]*3 > comps:
            freq_type = 'high'
        log2_x = int(np.log2(data[i].shape[-1]))
        data[i] = normalize_dwt_components(data[i], stds[str(nLevel - 1 - log2_x)], freq_type)
    return data

def unnormalize_training(data, stds, comps, baselevel, nLevel):
    for i in range(baselevel, len(data)):
        if data[i].shape[1] == comps:
            freq_type = 'low'
        elif data[i].shape[1]*3 > comps:
            freq_type = 'high'
        log2_x = int(np.log2(data[i].shape[-1]))
        if data[i] is not None:
            data[i] = unnormalize_dwt_components(data[i], stds[str(nLevel - 1 -log2_x)], freq_type)
        else:
            data[i] = None
    return data

def normalize_samples(data, stds):
    for j in range(0, len(data)):
        data[j] = data[j]/stds[j+1][0]
    return data

def set_weights_to_zero(model):
    # Recursively set all Conv2d layer weights to zero
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.ConvTranspose2d):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

def main():

    #load configs
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    p = SourceFileLoader('cf', f'{args.config}').load_module()
    directory_path = cf.saveDir
    print('loading models from', directory_path)


    # Read all file names from the directory
    file_list = os.listdir(directory_path)

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
    print(selected_files)
   
    #dir to save plots
    if not os.path.exists(cf.plotSaveDir):
            os.makedirs(cf.plotSaveDir)

    prior_type = cf.priorType
    for i in range(p.baseLevel, args.hlevel+1):
        p_level = i
        print('loading stds from ', cf.std_path)
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
        if p_level not in (cf.gauss_priors):
            dx = (0.5/60. * np.pi/180.)*(2**(dwt_level_number))
            rfourier_shape = (N*cf.imShape[0], nx, int(nx/2 + 1), 2)
            print('loading ps ', cf.ps_path+str(nx)+'x'+str(nx)+'.dat')
            df = pd.read_csv(cf.ps_path+str(nx)+'x'+str(nx)+'.dat', sep=";")
            df.columns = df.columns.str.strip()
            power_spec = np.array([df['ell'], df['low_auto_ch0'], df['low_auto_ch1'],
                                    df['high_horizontal_auto_ch0'], df['high_vertical_auto_ch0'], 
                                    df['high_diagonal_auto_ch0'], df['high_horizontal_auto_ch1'], 
                                    df['high_vertical_auto_ch1'], df['high_diagonal_auto_ch1'],
                                    df['low_cross_ch0_ch1'], df['high_horizontal_cross_ch0_ch1'],
                                    df['high_vertical_cross_ch0_ch1'], df['high_diagonal_cross_ch0_ch1']])
            power_spec = np.transpose(power_spec)
            priortype = prior_type
            prior = corr_prior.CorrelatedNormal_dwt(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq=freq, prior_type=prior_type)
        else:
            priortype = 'WN'
            shape = (N*cf.imShape[0], nx, nx)
            prior = corr_prior.SimpleNormal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
        print('p_level', p_level, ' nx ', nx, ' dwt_level_number ', dwt_level_number, freq, N, priortype)

        #load the models for all levels
        if i == p.baseLevel:
            model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype)
            model.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel], weights_only=True))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")
        else:
            model1 = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype)
            model1.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel], weights_only=True))
            total_params = sum(p.numel() for p in model1.parameters())
            print(f"Total number of parameters: {total_params}")
            model.sub_flows[i] = model1.sub_flows[i]
            del model1

    model = model.to(device)
    # init act norm
    for i in range(p.baseLevel, args.hlevel + 1):
        model.sub_flows[i].set_actnorm_init()
    model = model.eval()

    #load data: for new dataset add a new elif statement
    bdir = cf.val_dataset_path
    file = "data.mdb"
    transformer1 = None
    noise_level = 0.0
    if cf.dataset == 'My_lmdb':
        print('loading yuuki sims proper')
        dataset = My_lmdb(bdir, file, transformer1, 1, False, noise_level)
    elif cf.dataset == 'yuuki_256':
        print('loading yuuki 256')
        dataset = yuuki_256(bdir, file, transformer1, 1, False, noise_level)

    loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    iter_loader = iter(loader)

    print(len(loader))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    #calculate and plot power spectra and minkowski functionals
    start = time.time()
    compute_and_plot_all_power_spectra(model, iter_loader, cf, mean_stds_all_levels, device, cf.plotSaveDir, nLevels=cf.nLevels, get_train_modes=False, cond_on_target=False, max_iterations=20)
    compute_and_plot_all_power_spectra(model, iter_loader, cf, mean_stds_all_levels, device, cf.plotSaveDir, nLevels=cf.nLevels, get_train_modes=True, cond_on_target=False, max_iterations=20)


    compute_and_plot_minkowski_functionals_no_prior(
    model, 
    iter_loader, 
    cf, 
    mean_stds_all_levels, 
    device, 
    cond_on_target=False,
    n_thresholds=50,
    max_iterations=20,
)

    print(time.time() - start)

    

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/example_config_hcc_prior.py", help='specify config')
    parser.add_argument('--hlevel', type=int, default=6, help='highest level wavelet to sample')
    parser.add_argument('--data', type=str, default="agora", help='input data')
    parser.add_argument('--savesamples', type=str, default=False, help='save samples')
    parser.add_argument('--plotstats', type=str, default=True, help='plot stats')
    args = parser.parse_args()
    main()

    