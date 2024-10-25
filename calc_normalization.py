import os
from collections import defaultdict
import re
import sys
import torch
import random
import numpy as np
from data_loader import ISIC
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

# Assume 'dataset' is your dataset, and 'DataLoader' is imported
# Assume 'dwt' is your DWT transform object
# Assume 'torch_device' is defined (e.g., 'cuda' or 'cpu')

# Number of DWT levels to compute
max_m = 5  # Adjust as needed

# Dictionary to hold mean stds for each DWT level
mean_stds_all_levels = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bdir = "/mnt/half_yuki_sim_64/"
file = "data.mdb"
transformer1 = None
dataset = ISIC(bdir, file, transformer1, 1, False)
wavelet = Haar().to(device)
dwt = Dwt(wavelet=wavelet).to(device)
for m in range(0, max_m):
    loader = DataLoader(dataset, batch_size=512)
    
    # Initialize dictionaries to hold total std deviations and counts
    comp_total_std = {}
    num_batches = 0    # Number of batches processed
    
    # Define high-frequency component types
    high_types = ['high_horizontal', 'high_vertical', 'high_diagonal']
    n_high_types = len(high_types)
    
    for i, x in enumerate(loader):
        x = x.to(device).to(torch.float32)
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
        
        # Process 'low' components
        x1_np = x1.cpu().numpy()  # Shape: [batch_size, N_channels, H, W]
        batch_size = x1_np.shape[0]
        
        for ch in range(N_channels):
            comp_ch = x1_np[:, ch, :, :]  # Shape: [batch_size, H, W]
            # Compute standard deviation over the batch and spatial dimensions
            std_batch = np.std(comp_ch)
            comp_total_std['low'][ch] += std_batch
        
        # Process high-frequency components
        x2_np = x2.cpu().numpy()  # Shape: [batch_size, N_channels * 3, H, W]
        
        for ht_idx, ht in enumerate(high_types):
            for ch in range(N_channels):
                idx = ch * n_high_types + ht_idx  # Index in x2
                comp_ch = x2_np[:, idx, :, :]  # Shape: [batch_size, H, W]
                # Compute standard deviation over the batch and spatial dimensions
                std_batch = np.std(comp_ch)
                comp_total_std[ht][ch] += std_batch

        num_batches += 1  # Increment the number of batches
    
    # After processing all batches, compute mean standard deviations
    comp_mean_std = {}
    for comp_type in comp_total_std.keys():
        comp_mean_std[comp_type] = []
        for ch in range(N_channels):
            mean_std = comp_total_std[comp_type][ch] / num_batches
            comp_mean_std[comp_type].append(mean_std)
    
    # Save the mean stds for this DWT level
    mean_stds_all_levels[m] = comp_mean_std

# Save mean stds to a file for later use
import json

# Convert the mean stds data to a serializable format
mean_stds_serializable = {}
for m, comp_mean_std in mean_stds_all_levels.items():
    mean_stds_serializable[m] = {}
    for comp_type, std_list in comp_mean_std.items():
        mean_stds_serializable[m][comp_type] = std_list

# Save to a JSON file
filename = 'mean_stds_all_levels.json'
with open(filename, 'w') as f:
    json.dump(mean_stds_serializable, f)

print(f"Mean standard deviations saved to {filename}")

