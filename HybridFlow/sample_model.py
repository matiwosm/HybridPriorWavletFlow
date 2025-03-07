import os
from collections import defaultdict
import re
import torch
import random
import numpy as np
from importlib.machinery import SourceFileLoader
from HybridFlow.waveletflow import WaveletFlow
from HybridFlow.conditioning_network import Conditioning_network
from HybridFlow.utilities import *
from HybridFlow.data_loader import scale_data_pt
import HybridFlow.corr_prior as corr_prior
import pandas as pd
import json

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


class sample_HF():
    def __init__(self, config, sample_number=None, verbose=False):

        #load configs
        # cf = SourceFileLoader('cf', f'{data}.py').load_module()
        cf = SourceFileLoader('cf', f'{config}').load_module()
        directory_path = cf.saveDir
        if verbose:
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
        if verbose:
            [print(f'Using {ml_file} for level {i}\n') for i,ml_file in enumerate(selected_files)]
            print('loading normalization factors from ', cf.std_path)
        #dir to save plots
        if not os.path.exists(cf.plotSaveDir):
                os.makedirs(cf.plotSaveDir)

        prior_type = cf.priorType
        for i in range(cf.baseLevel, cf.nLevels+1):
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
                if verbose:
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
            
            cf.net_type = cf.network[i]

            #load the models for all levels
            if i == cf.baseLevel:
                model = WaveletFlow(cf=cf, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype, device=device)
                model.load_state_dict(torch.load(directory_path + selected_files[i - cf.baseLevel], weights_only=True, map_location=device))
                total_params = sum(p.numel() for p in model.parameters())
                if verbose:
                    print(f"Total number of parameters for level {p_level}: {total_params}")
            else:
                model1 = WaveletFlow(cf=cf, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=mean_stds_this_levels, priortype=priortype, device=device)
                model1.load_state_dict(torch.load(directory_path + selected_files[i - cf.baseLevel], weights_only=True, map_location=device))
                total_params = sum(p.numel() for p in model1.parameters())
                if verbose:
                    print(f"Total number of parameters for level {p_level}: {total_params}")
                model.sub_flows[i] = model1.sub_flows[i]
                del model1
            
        model = model.to(device)
        # init act norm
        for i in range(cf.baseLevel, cf.nLevels + 1):
            model.sub_flows[i].set_actnorm_init()
        model = model.eval()

        self.model = model
        self.batch_size = cf.sample_batch_size
        self.verbose = verbose
        self.channels = cf.channels_to_get
    def sample(self, sample_number, reverse_scaling=True, verbose=False):
        if sample_number is None:
            sample_number = self.batch_size

        num_batch = sample_number//self.batch_size
        left_over_samples = sample_number - num_batch*self.batch_size
        if verbose:
            print('batch number = ', self.batch_size)
            print('number of batches to sample = ', num_batch)
            print('left over samples = ', left_over_samples)
            print('you are sampling ', sample_number, ' samples')

        #sampling from model
        for i in range(num_batch):
            if i == 0:
                samples = self.model.sample(n_batch=self.batch_size)[-1]
            else:
                samples = torch.cat([samples, self.model.sample(n_batch=self.batch_size)[-1]], dim=0)

                
        if left_over_samples > 0 and num_batch >0:
            samples = torch.cat([samples, self.model.sample(n_batch=left_over_samples)[-1]], dim=0)
        elif left_over_samples > 0 and num_batch == 0:
            samples = self.model.sample(n_batch=left_over_samples)[-1]
        if reverse_scaling:
            samples = scale_data_pt(samples, self.channels, reverse=True)

        samples = samples.cpu().numpy()
        return samples