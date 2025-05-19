import os
from collections import defaultdict
import re
import sys
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
import corr_prior
import pandas as pd
import json
from helper import utils
from utilities import *

seed = 786
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.autograd.set_detect_anomaly(True)
random.seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
device = torch.device("cuda")

print("Visible devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print("Using CUDA device index:", torch.cuda.current_device())

torch.cuda.empty_cache()



def main():
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    p = SourceFileLoader('cf', f'{args.config}').load_module()
    if not cf.double_precision[args.level]:
        torch.set_default_dtype(torch.float32)
    
    #set this true to resume training
    resume = False
    directory_path = cf.saveDir
    print("saving models to ", directory_path)
    if resume:
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
        selected_files = [info[0] for i, info in sorted(files_by_i.items())]
        print(selected_files)
    else:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    # bdir = cf.dataset_path
    # file = "data.mdb"
    # transformer1 = None
    # noise_level = 0.025
    # if cf.dataset == 'My_lmdb':
    #     print('loading yuuki sims proper')
    #     dataset = My_lmdb(bdir, file, transformer1, 1, False, noise_level)
    # elif cf.dataset == 'yuuki_256':
    #     print('loading yuuki 256')
    #     dataset = yuuki_256(bdir, file, transformer1, 1, False, noise_level)

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
        noise_dict={},        # we do the noising outside the dataloader for efficiency
        apply_scaling=False,           # same with scaling
        data_shape=cf.data_shape       
    )

    warmup_loader = DataLoader(dataset, batch_size=cf.batch_size[args.level], shuffle=False, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cf.batch_size[args.level], shuffle=True,
        num_workers=0, drop_last=True)
    
    #gradient accumulation
    if cf.batch_size[args.level] < cf.eff_batch_size:
        accumulation_steps = int(cf.eff_batch_size/cf.batch_size[args.level])
    else:
        accumulation_steps = 1
    
    # print('batch size = ', cf.batch_size[args.level], 'effective batch size = ', cf.eff_batch_size, ' acc = ', accumulation_steps, len(train_loader))
    #normalization factors for the current dwt level
    print('loading stds ', cf.std_path)
    with open(cf.std_path, 'r') as f:
        mean_stds_all_levels = json.load(f)

    prior_type = cf.priorType
    #load powerspectra
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

    mean_stds_all_levels = mean_stds_all_levels[str(dwt_level_number-1)]
    # print(dwt_level_number, mean_stds_all_levels)

    nx = int(cf.imShape[-1]//(2**dwt_level_number))
    if p_level not in (cf.gauss_priors):
        dx = (0.5/60. * np.pi/180.)*(2**(dwt_level_number))
        rfourier_shape = (N*cf.imShape[0], nx, int(nx/2 + 1), 2)
        print('loading ps ', cf.ps_path+str(nx)+'x'+str(nx)+'.dat')
        df = pd.read_csv(cf.ps_path+str(nx)+'x'+str(nx)+'.dat', sep=";")
        df.columns = df.columns.str.strip()
        power_spec = df.values  # shape (N_ell, N_columns)

        # 3) Build dictionary {col_name -> col_index}
        colnames = list(df.columns)
        colname_to_index = {name: i for i, name in enumerate(colnames)}
        print('Normalize prior = ', cf.normalize_prior[p_level])
        if cf.unnormalize_prior[p_level] == False:
            norm_std = None
        else:
            print('Unnormalizing prior with precomputed stds')
            norm_std = mean_stds_all_levels
        prior = corr_prior.CorrelatedNormalDWTGeneral(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,cl_theo=power_spec,colname_to_index=colname_to_index,torch_device=device, freq=freq, n_channels=cf.imShape[0], prior_type=prior_type, norm_std=norm_std, normalize=cf.normalize_prior[p_level])
    else:
        norm_std = None
        prior_type = 'WN'
        shape = (cf.batch_size[args.level], N*cf.imShape[0], nx, nx)
        prior = corr_prior.SimpleNormal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
    p.net_type = cf.network[p_level]    
    model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=p_level, prior=prior, stds=mean_stds_all_levels, priortype=prior_type, device=device).to(device)
    print('Prior type = ', prior_type)
    # print('norm_std = ', norm_std)
    if resume:
        model.load_state_dict(torch.load(directory_path + selected_files[p_level-1]))
        print('loaded ', directory_path + selected_files[p_level-1], selected_files[p_level-1][19:-8])
    optimizer = optim.Adam(model.parameters(), lr=cf.lr)
    
    lowest = 1e7
    patience = 0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    if cf.parallel[args.level]:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    # print(model)
    model.train()
    ep = 1
    if resume:
        ep = int(selected_files[p_level-1][19:-8])
    loader = warmup_loader
    print('Dataset size = ', len(loader))

    elapsed_time = 0
    elapsed_time_without_data_loader = 0
    start_time = time.time()
    
    while True:
        ep_loss = []
        print('starting loop')
        for idx, x in enumerate(loader):
            if ((idx) % 10) == 0:
                print(f"Epoch: {ep} Level: {p_level}  Progress:      {round((idx * 100) / (len(loader)), 4)}% Likelihood:      {np.mean(ep_loss)} Patience:      {round(patience, 5)}   Time:  {elapsed_time, elapsed_time_without_data_loader}" , end="\n")
                sys.stdout.flush()
                # torch.save(model.state_dict(), f'{directory_path}/waveletflow-{args.data}-{args.level}-{ep}-test.pt')


            x = x.to(device)
            #noising data
            x = apply_noise_torch_vectorized(x, cf.channels_to_get, cf.noise_dict)
            x = scale_data_pt(x, cf.channels_to_get)
            if not cf.double_precision[args.level]:
                x = x.type(torch.float32)

            torch.cuda.synchronize()
            elapsed_time += time.time() - start_time

            time_without_data_loader = time.time()

            res_dict = model(x, partial_level=p_level)
            loss = torch.mean(res_dict["likelihood"])
            loss = loss / accumulation_steps
            loss.backward()
            
            # Accumulate the loss for logging
            loss_ = loss.detach().cpu().numpy() * accumulation_steps  # Scale back for logging
            ep_loss.append(loss_)
            
            # Only step the optimizer and zero gradients after accumulation_steps
            if ((idx + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()


            torch.cuda.synchronize()
            elapsed_time_without_data_loader += time.time() - time_without_data_loader
            start_time = time.time()

        avg_loss = np.mean(ep_loss)
        if ((idx + 1) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()

        if lowest >= avg_loss:    
            lowest = avg_loss
            print('saving ' + f'{directory_path}/waveletflow-{args.data}-{args.level}-{ep}-test.pt', '\n')
            sys.stdout.flush()
            if cf.parallel[args.level]:
                torch.save(model.module.state_dict(), f'{directory_path}/waveletflow-{args.data}-{args.level}-{ep}-test.pt')
            else:
                torch.save(model.state_dict(), f'{directory_path}/waveletflow-{args.data}-{args.level}-{ep}-test.pt')
            patience = 0
            loader = train_loader
        else:
            patience += 1
        ep += 1
        if patience == 10:
            break
        
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/example_config_hcc_prior.py", help='specify config')
parser.add_argument('--level', type=int, default=-1, help='training level')
parser.add_argument('--data', type=str, default="agora", help='input data')
args = parser.parse_args()
p_level = args.level
main()