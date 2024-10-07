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
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
device = torch.device("cuda")

print("Visible devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print("Using CUDA device index:", torch.cuda.current_device())

torch.cuda.empty_cache()



def main():
    resume = False
    directory_path = '/mnt/kaponly_corrprior_gaussian_normalized_nopermandsigmoid/'
    if resume:
        

        # Read all file names from the directory
        file_list = os.listdir(directory_path)

        # Regular expression to extract i and j values
        pattern = re.compile(r"waveletflow-isic-(\d+)(?:-(\d+))?")

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
   
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    # dataset = ISIC(cf, benign=True, test=False, gray=cf.grayscale)
    
    bdir = "/mnt/gaussian_kap_trans_cib_train"
    file = "data.mdb"
    transformer1 = None
    dataset = ISIC(bdir, file, transformer1, 1, False)
    warmup_loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=False, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cf.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, generator=torch.Generator(device='cpu'))
    
    stds = torch.tensor([[10.4746, 10.4746, 10.4746, 10.4746],
        [10.4746, 5.9943,  6.0226,  4.4362],
        [7.0993, 3.6189,  3.6250,  2.5660],
        [4.5613, 2.0915,  2.0961,  1.4750],
        [2.8173, 1.1993,  1.2024,  0.8454],
        [1.6983, 0.6829,  0.6875,  0.4227]])

    if args.model == "glow":
        p = SourceFileLoader('cf', 'config_glow.py').load_module()
        model = Glow(p).to(device)
    elif args.model == "waveletflow":
        p = SourceFileLoader('cf', 'config_waveletflow.py').load_module()
        #load powerspectra
        if p_level == cf.baseLevel or p_level == cf.baseLevel + 1:
            power_spec = pd.read_csv('ps/dwtlevel_all'+str(2**cf.baseLevel)+'x'+str(2**cf.baseLevel)+'_gauss.dat', sep=";", header=None)
            power_spec = (np.array(power_spec))[:, [0, 1, 2, 3, 4]]
            nx = int(64/(2**(6-cf.baseLevel)))
            dx = (0.5/60. * np.pi/180.)*(2**(6-cf.baseLevel))

            if p_level == cf.baseLevel:
                rfourier_shape = (1, nx, int(nx/2 + 1), 2)
                prior = corr_prior.CorrelatedNormal_single(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq='low')
            else:
                rfourier_shape = (3, nx, int(nx/2 + 1), 2)
                prior = corr_prior.CorrelatedNormal_dwt(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq='high')
        else:
            power_spec = pd.read_csv('ps/dwtlevel_all'+str(2**(p_level-1))+'x'+str(2**(p_level-1))+'_gauss.dat', sep=";", header=None)
            power_spec = (np.array(power_spec))[:, [0, 1, 2, 3, 4]]
            nx = int(64/(2**(6-p_level+1)))
            dx = (0.5/60. * np.pi/180.)*(2**(6-p_level+1))
            rfourier_shape = (3, nx, int(nx/2 + 1), 2)
            prior = corr_prior.CorrelatedNormal_dwt(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq='high')
        model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=p_level, prior=prior, stds=stds).to(device)
        if resume:
            model.load_state_dict(torch.load(directory_path + selected_files[p_level-1]))
            model.sub_flows[p_level].set_actnorm_init()
            print('loaded ', directory_path + selected_files[p_level-1], selected_files[p_level-1][19:-8])
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    
    lowest = 1e7
    patience = 0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    model.train()
    ep = 1
    if resume:
        ep = int(selected_files[p_level-1][19:-8])
    loader = warmup_loader


    elapsed_time = 0
    start_time = time.time()
    while True:
        ep_loss = []

        for idx, x in enumerate(loader):
            if (idx % 100) == 0:
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print(f"Epoch: {ep} Level: {p_level}  Progress:      {round((idx * 100) / (len(loader)), 4)}% Likelihood:      {lowest} Patience:      {round(patience, 5)}   Time:  {elapsed_time}" , end="\n")
                sys.stdout.flush()
            x = x.to(device)
            optimizer.zero_grad()
            res_dict = model(x, partial_level=p_level)
            loss = torch.mean(res_dict["likelihood"])
            loss.backward()
            optimizer.step()
            loss_ = loss.detach().cpu().numpy()
            ep_loss.append(loss_)
        avg_loss = np.mean(ep_loss)
        if lowest >= avg_loss:    
            lowest = avg_loss
            print('saving ' + f'/mnt/kaponly_corrprior_gaussian_normalized_nopermandsigmoid/{args.model}-{args.data}-{args.level}-{ep}-test.pt', '\n')
            sys.stdout.flush()
            torch.save(model.state_dict(), f'/mnt/kaponly_corrprior_gaussian_normalized_nopermandsigmoid/{args.model}-{args.data}-{args.level}-{ep}-test.pt')
            patience = 0
            loader = train_loader
        else:
            patience += 1
        ep += 1
        if patience == 10:
            break
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="waveletflow", help='train level')
parser.add_argument('--level', type=int, default=-1, help='train level')
parser.add_argument('--data', type=str, default="agora", help='train level')
args = parser.parse_args()
p_level = args.level
main()