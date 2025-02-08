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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

torch.set_default_dtype(torch.float64)


def main():

    #load configs
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    p = SourceFileLoader('cf', f'{args.config}').load_module()
    directory_path = cf.saveDir
    print('loading models from', directory_path)

    #If you want to load the trained models manually this part is not needed
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
    [print(f'Using {ml_file} for level {i}\n') for i,ml_file in enumerate(selected_files)]
    # print('selected models ', selected_files)
    print('loading normalization factors from ', cf.std_path)
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
        if p_level not in (cf.gauss_priors):
            dx = (0.5/60. * np.pi/180.)*(2**(dwt_level_number))
            rfourier_shape = (N*cf.imShape[0], nx, int(nx/2 + 1), 2)
            df = pd.read_csv(cf.ps_path+str(nx)+'x'+str(nx)+'.dat', sep=";")
            print('loading power spectra from ', cf.ps_path+str(nx)+'x'+str(nx)+'.dat')
            df.columns = df.columns.str.strip()
            power_spec = df.values  # shape (N_ell, N_columns)

            # 3) Build dictionary {col_name -> col_index}
            colnames = list(df.columns)
            colname_to_index = {name: i for i, name in enumerate(colnames)}
            priortype = prior_type

            if cf.unnormalize_prior == False:
                norm_std = None
            else:
                norm_std = mean_stds_this_levels
            prior = corr_prior.CorrelatedNormalDWTGeneral(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,cl_theo=power_spec,colname_to_index=colname_to_index,torch_device=device, freq=freq, n_channels=cf.imShape[0], prior_type=prior_type, norm_std=norm_std)
        else:
            priortype = 'WN'
            shape = (N*cf.imShape[0], nx, nx)
            prior = corr_prior.SimpleNormal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
        
        p.net_type = p.network[i]

        #load the models for all levels
        if i == p.baseLevel:
            model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype, device=device)
            model.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel], weights_only=True, map_location=device))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters for level {p_level}: {total_params}")
        else:
            model1 = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype, device=device)
            model1.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel], weights_only=True, map_location=device))
            total_params = sum(p.numel() for p in model1.parameters())
            print(f"Total number of parameters for level {p_level}: {total_params}")
            model.sub_flows[i] = model1.sub_flows[i]
            del model1
        
    model = model.to(device)
    # init act norm
    for i in range(p.baseLevel, cf.nLevels + 1):
        model.sub_flows[i].set_actnorm_init()
    model = model.eval()

    #sampling from model
    samples = model.sample(n_batch=cf.sample_batch_size)[-1]
    print(samples.shape)

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/old_configs/kappa_cib_saved_model_config.py", help='specify config')
    parser.add_argument('--data', type=str, default="agora", help='input data')
    args = parser.parse_args()
    main()

    