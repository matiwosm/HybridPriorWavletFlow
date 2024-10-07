import os
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
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

seed = 786
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

def plot_samples(data, level):
    comps = ['kap', 'cib']
    fig, axs = plt.subplots(data.shape[1], data.shape[0], figsize=(data.shape[0], data.shape[1]), sharex='row', sharey='row', dpi=200)
    # title = ['Prior samples', 'Flow samples', 'Target samples']
    # plt.suptitle("Component correlated prior to gaussian target")   
    fig, axs = plt.subplots(data.shape[0], data.shape[1], figsize=(2*data.shape[1], 2*data.shape[0]), sharex='row', sharey='row', dpi=200)
    min_max = [3, 3, 50, 50, 50]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            axis = np.arange(0,data.shape[2],1)
            # norm = plt.Normalize(vmin=np.amin(data[:, j, :, :]), vmax=np.amax(data[:, j, :, :]))
            min = np.amin(data[:, j, :, :])
            max = np.amax(data[:, j, :, :])
            
            
            min =-1*min_max[j]
            max = min_max[j]
            norm = plt.Normalize(vmin=min, vmax=max)
            print(min, max, "minmax")
            cmap = plt.get_cmap("RdYlBu_r")
            if (j == 4):
                min = np.amin(data[0, j, :, :])
                max = min_max[j]
                norm = plt.Normalize(vmin=min, vmax=max)
                cmap = plt.get_cmap("gray")
                print(min, max)
            elif(j == 0):
                cmap = plt.get_cmap("RdBu_r")
            print(data.shape, i, j)
            cs = axs[i, j].contourf(axis, axis, data[i, j, :, :], cmap = cmap, norm=norm, levels=100, zorder=0)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.0,
                    hspace=0.02)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        # ax.label_outer()
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False) 
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('samples_wavelet'+str(level)+'.png')
    plt.close()

def main():
    fName = time.strftime("%Y%m%d_%H_%M")
    if not os.path.exists("/mnt/saves/"):
        os.makedirs("/mnt/saves/")
   
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    # dataset = ISIC(cf, benign=True, test=False, gray=cf.grayscale)
    
    bdir = "/mnt/half_yuki_sim_64"
    file = "data.mdb"
    transformer1 = None
    dataset = ISIC(bdir, file, transformer1, 1, False)
    loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=False, pin_memory=True)
    p = SourceFileLoader('cf', 'config_waveletflow.py').load_module()
    for i in range(p.baseLevel, args.hlevel+1):
        if i == p.baseLevel:
            model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i)
            model.load_state_dict(torch.load(f'/mnt/saves_1comp/{args.model}-{args.data}-{i}-test.pt'))
        else:
            print(i)
            model1 = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i)
            model1.load_state_dict(torch.load(f'/mnt/saves_1comp/{args.model}-{args.data}-{i}-test.pt'))
            model.sub_flows[i] = model1.sub_flows[i]
            del model1
    model = model.to(device)
    print(model)
    #init act norm
    for i in range(p.baseLevel, args.hlevel + 1):
        model.sub_flows[i].set_actnorm_init()
    model = model.eval()
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    
    lowest = 1e7
    patience = 0
    # print(model)
    samples = model.sample()
    for i in range(len(samples)):
        plot_samples(samples[i][:5, :, :, :].cpu().numpy(), i+1)
if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="waveletflow", help='train level')
    parser.add_argument('--hlevel', type=int, default=-1, help='train level')
    parser.add_argument('--data', type=str, default="isic", help='train level')
    args = parser.parse_args()
    main()

    