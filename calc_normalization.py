import os
from collections import defaultdict
import re
import sys
import torch
import random
import numpy as np
from data_loader import yuuki_256, My_lmdb
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

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
# Assume 'dataset' is your dataset, and 'DataLoader' is imported
# Assume 'dwt' is your DWT transform object
# Assume 'torch_device' is defined (e.g., 'cuda' or 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_file = 'configs/example_config_hcc_prior_tsz.py'
cf = SourceFileLoader('cf', config_file).load_module()

# Number of DWT levels to compute
max_m = 5 # Adjust as needed

# Dictionary to hold mean stds for each DWT level
mean_stats_all_levels = {}

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

wavelet = Haar().to(device)
dwt = Dwt(wavelet=wavelet).to(device)
for m in range(0, max_m):
    loader = DataLoader(dataset, batch_size=8096)
    
    # Initialize dictionaries to hold total std deviations, mins, and maxs
    comp_total_std = {}
    comp_min = {}  # Added to hold mins
    comp_max = {}  # Added to hold maxs
    num_batches = 0    # Number of batches processed
    
    # noise_level = 0.4
    # Define high-frequency component types
    high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
    n_high_types = len(high_types)
    
    for i, x in enumerate(loader):
        if (i % 10) == 0:
            print(f"Processing batch {i}... out of {len(loader)}")
            
        # Apply DWT m+1 times
        for k in range(m + 1):
            dwt_result = dwt.forward(x)
            x1 = dwt_result['low']  # x1 shape: (batch_size, N_channels, H, W)
            x2 = dwt_result['high']  # x2 shape: (batch_size, N_channels * 3, H, W)
            x = x1  # Update x for the next level

        N_channels = x1.shape[1]  # Number of input channels

        # Initialize dictionaries on the first batch
        if i == 0:
            # Initialize for 'low' components
            comp_types = ['low'] + high_types
            for comp_type in comp_types:
                comp_total_std[comp_type] = [0 for _ in range(N_channels)]
                comp_min[comp_type] = [np.inf for _ in range(N_channels)]  # Added
                comp_max[comp_type] = [-np.inf for _ in range(N_channels)]  # Added
        
        # Process 'low' components
        x1_np = x1.cpu().numpy()  # Shape: [batch_size, N_channels, H, W]
        batch_size = x1_np.shape[0]
        
        for ch in range(N_channels):
            comp_ch = x1_np[:, ch, :, :]  # Shape: [batch_size, H, W]
            # Compute standard deviation over the batch and spatial dimensions
            std_batch = np.std(comp_ch)
            comp_total_std['low'][ch] += std_batch

            # Compute min and max over the batch and spatial dimensions
            min_batch = np.min(comp_ch)  # Added
            max_batch = np.max(comp_ch)  # Added
            comp_min['low'][ch] = min(comp_min['low'][ch], min_batch)  # Added
            comp_max['low'][ch] = max(comp_max['low'][ch], max_batch)  # Added

        # Process high-frequency components
        x2_np = x2.cpu().numpy()  # Shape: [batch_size, N_channels * 3, H, W]
        
        for ht_idx, ht in enumerate(high_types):
            for ch in range(N_channels):
                idx = ch * n_high_types + ht_idx  # Index in x2
                comp_ch = x2_np[:, idx, :, :]  # Shape: [batch_size, H, W]
                # Compute standard deviation over the batch and spatial dimensions
                std_batch = np.std(comp_ch)
                comp_total_std[ht][ch] += std_batch

                # Compute min and max over the batch and spatial dimensions
                min_batch = np.min(comp_ch)  # Added
                max_batch = np.max(comp_ch)  # Added
                comp_min[ht][ch] = min(comp_min[ht][ch], min_batch)  # Added
                comp_max[ht][ch] = max(comp_max[ht][ch], max_batch)  # Added

        num_batches += 1  # Increment the number of batches
    
    # After processing all batches, compute mean standard deviations
    comp_stats = {}  # Modified from comp_mean_std
    for comp_type in comp_total_std.keys():
        comp_stats[comp_type] = {'mean_std': [], 'min': [], 'max': []}  # Modified
        for ch in range(N_channels):
            
            mean_std = comp_total_std[comp_type][ch] / num_batches
            print(mean_std)
            comp_stats[comp_type]['mean_std'].append(mean_std)
            comp_stats[comp_type]['min'].append(comp_min[comp_type][ch])  # Added
            comp_stats[comp_type]['max'].append(comp_max[comp_type][ch])  # Added
    
    # Save the stats for this DWT level
    mean_stats_all_levels[m] = comp_stats  # Modified from mean_stds_all_levels

# Save mean stats to a file for later use
import json

# Convert the mean stats data to a serializable format
mean_stats_serializable = {}
for m, comp_stats in mean_stats_all_levels.items():
    mean_stats_serializable[m] = {}
    for comp_type, stats in comp_stats.items():
        mean_stats_serializable[m][comp_type] = stats

# Save to a JSON file
filename = f'norm_stds/64x64_final_mean_stats_all_levels_tsz.json'  # Modified filename
with open(filename, 'w') as f:
    json.dump(mean_stats_serializable, f)

print(f"Mean statistics (std, min, max) saved to {filename}")