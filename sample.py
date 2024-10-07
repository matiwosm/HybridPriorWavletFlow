import os
from collections import defaultdict
import re
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
from helper import utils as util
from utilities import *
import corr_prior
import pandas as pd

seed = 678 #786
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
            # print(min, max, "minmax")
            cmap = plt.get_cmap("RdYlBu_r")
            if (j == 4):
                min = np.amin(data[0, j, :, :])
                max = min_max[j]
                norm = plt.Normalize(vmin=min, vmax=max)
                cmap = plt.get_cmap("gray")
                # print(min, max)
            elif(j == 0):
                cmap = plt.get_cmap("RdBu_r")
            # print(data.shape, i, j)
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
    PATH = os.path.join('plots', 'samples_wavelet'+str(level)+'.png')
    plt.savefig(PATH)
    plt.close()

def plot_power(target, sample, prior, level):
    bin_sizes = [2, 2, 3, 7, 15, 30, 70]
    bin_sizes_log = [2, 2, 3, 5, 5, 10, 20]
    util_obj = util()
    dx =  (0.5/60. * np.pi/180.)*2**(6 - level)
    nx = target.shape[-1]
    size = bin_sizes[level]
    print(level, size, nx, dx)
    comps = ['comp1', 'comp2', 'comp3']
    for j in range(0, target.shape[1]):
        for k in range(0, target.shape[1]):
            if j == k: 
                binned_spectrum = []
                binned_spectrum_net = []
                binned_spectrum_prior = []
                for i in range(target.shape[0]):
                    ell, bin_spectrum = util_obj.get_2d_power(nx, dx, target[i, j, :, :], target[i, k, :, :], size+1, scale='linear')
                    binned_spectrum.append(np.real(bin_spectrum))
                for i in range(sample.shape[0]):
                    ell, bin_spectrum_net = util_obj.get_2d_power(nx, dx, sample[i, j, :, :], sample[i, k, :, :], size+1, scale='linear')
                    binned_spectrum_net.append(np.real(bin_spectrum_net))
                for i in range(prior.shape[0]):
                    ell, bin_spectrum_pior = util_obj.get_2d_power(nx, dx, prior[i, j, :, :], prior[i, k, :, :], size+1, scale='linear')
                    binned_spectrum_prior.append(np.real(bin_spectrum_pior))
                binned_spectrum = np.real(np.array(binned_spectrum))
                binned_spectrum_net = np.real(np.array(binned_spectrum_net))
                binned_spectrum_prior = np.real(np.array(binned_spectrum_prior))
                # y = (bined_spectrum_net - binned_spectrum)/binned_spectrum

                binned_spectrum_err = np.std(binned_spectrum, axis=0)/np.sqrt(binned_spectrum.shape[0])
                binned_spectrum_net_err = np.std(binned_spectrum_net, axis=0)/np.sqrt(binned_spectrum_net.shape[0])
                # yerr = np.std(y, axis=0)/np.sqrt(y.shape[0])

                binned_spectrum = np.mean(binned_spectrum, axis=0)
                binned_spectrum_net = np.mean(binned_spectrum_net, axis=0)
                binned_spectrum_prior = np.mean(binned_spectrum_prior, axis=0)

                y = (binned_spectrum_net - binned_spectrum)/binned_spectrum


                partial_X1 = 1 / binned_spectrum
                partial_X2 = -(binned_spectrum_net - binned_spectrum) / binned_spectrum**2

                y_err = np.sqrt((partial_X1 * binned_spectrum_net_err)**2 + (partial_X2 * binned_spectrum_err)**2)

                plt.errorbar(ell[1:], y, fmt='.-', yerr=y_err, ecolor='red', label='(Wavelet Flow map - Target map)/Target map')
                plt.semilogx(ell[1:], np.zeros(ell[1:].size))
                plt.legend()
                plt.ylim(-0.1, 0.1)
                plt.ylabel('$C_{\ell}$')
                plt.xlabel('$\ell$')
                plt.title(comps[j]+' '+comps[k]+" power spectrum")
                # PATH = '/plots/nvp_power_diff='+comps[j]+comps[k]+str(level)+'.png'
                PATH = os.path.join('plots', 'nvp_power_diff='+comps[j]+comps[k]+str(level)+'.png')
                print(PATH)
                plt.savefig(PATH)
                plt.close()

                # print(binned_spectrum, binned_spectrum_net)
                plt.errorbar(ell[1:], binned_spectrum, fmt='.-', yerr=binned_spectrum_err, label='Target Map')
                plt.errorbar(ell[1:], binned_spectrum_net, fmt='.-', yerr=binned_spectrum_net_err, label='Wavelet Flow Map')
                plt.plot(ell[1:], binned_spectrum_prior, label='Prior Map')
                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel('$C_{\ell}$')
                plt.xlabel('$\ell$')
                plt.legend()
                plt.title(comps[j]+' '+comps[k]+" power spectrum")
                PATH = os.path.join('plots', 'nvp_power='+comps[j]+comps[k]+str(level)+'.png')
                # PATH = '/plots/nvp_power='+comps[j]+comps[k]+str(level)+'.png'
                print(PATH)
                plt.savefig(PATH)
                plt.close()
    return ell, binned_spectrum, binned_spectrum_err, binned_spectrum_net, binned_spectrum_net_err

#get all the high frequecy modes + the lowest low frequecy mode from x
def get_training_modes(x, model):
    mode_list = []
    for i in range(5):
        y = model.dwt.forward(x)['high']
        x = model.dwt.forward(x)['low']
        mode_list.append(y)
        if i == 4:
            mode_list.append(x)
            mode_list.append(x)
    mode_list.reverse()
    return mode_list

#get all the low frequecy modes from x
def get_sample_modes(x, model):
    mode_list = []
    mode_list.append(x)
    for i in range(5):
        x = model.dwt.forward(x)['low']
        mode_list.append(x)
    mode_list.reverse()
    return mode_list

def normalize_training(data, stds):
    data[0]= data[0]/stds[0][0]
    for j in range(1, len(data)):
        x = data[j]
        if x.shape[1] == 1:
            x = x/stds[j-1][0]
        else:
            for i in range(3):
                x[:, i, :, :] = x[:, i, :, :]/stds[j-1][i+1]
                print(torch.std(x[:, i, :, :]), i, j)
        data[j] = x
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
    directory_path = '/mnt/kaponly_corrprior_gaussian_normalized_nopermandsigmoid/'

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
   
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    # dataset = ISIC(cf, benign=True, test=False, gray=cf.grayscale)
    
    bdir = "/mnt/gaussian_kap_trans_cib_train"
    file = "data.mdb"
    transformer1 = None
    dataset = ISIC(bdir, file, transformer1, 1, False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    stds = torch.tensor([[10.4746, 10.4746, 10.4746, 10.4746],
        [10.4746, 5.9943,  6.0226,  4.4362],
        [7.0993, 3.6189,  3.6250,  2.5660],
        [4.5613, 2.0915,  2.0961,  1.4750],
        [2.8173, 1.1993,  1.2024,  0.8454],
        [1.6983, 0.6829,  0.6875,  0.4227],
        [1.0, 1.0,  1.0,  1.0]])

    p = SourceFileLoader('cf', 'config_waveletflow.py').load_module()
    for i in range(p.baseLevel, args.hlevel+1):
        if i == cf.baseLevel or i == cf.baseLevel + 1:
            power_spec = pd.read_csv('ps/dwtlevel_all'+str(2**cf.baseLevel)+'x'+str(2**cf.baseLevel)+'_gauss.dat', sep=";", header=None)
            power_spec = (np.array(power_spec))[:, [0, 1, 2, 3, 4]]
            nx = int(64/(2**(6-cf.baseLevel)))
            dx = (0.5/60. * np.pi/180.)*(2**(6-cf.baseLevel))

            if i == cf.baseLevel:
                rfourier_shape = (1, nx, int(nx/2 + 1), 2)
                prior = corr_prior.CorrelatedNormal_single(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq='low')
            else:
                rfourier_shape = (3, nx, int(nx/2 + 1), 2)
                prior = corr_prior.CorrelatedNormal_dwt(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq='high')
        else:
            power_spec = pd.read_csv('ps/dwtlevel_all'+str(2**(i-1))+'x'+str(2**(i-1))+'_gauss.dat', sep=";", header=None)
            power_spec = (np.array(power_spec))[:, [0, 1, 2, 3, 4]]
            nx = int(64/(2**(6-i+1)))
            dx = (0.5/60. * np.pi/180.)*(2**(6-i+1))
            rfourier_shape = (3, nx, int(nx/2 + 1), 2)
            prior = corr_prior.CorrelatedNormal_dwt(torch.zeros(rfourier_shape), torch.ones(rfourier_shape),nx,dx,power_spec,device, freq='high')
        print(i, cf.baseLevel, nx, dx, rfourier_shape, len(power_spec[:, 0]))
        if i == p.baseLevel:
            model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=stds)
            model.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel]))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")
        else:
            # print(i)
            model1 = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=i, prior=prior, stds=stds)
            model1.load_state_dict(torch.load(directory_path + selected_files[i - p.baseLevel]))
            total_params = sum(p.numel() for p in model1.parameters())
            print(f"Total number of parameters: {total_params}")
            model.sub_flows[i] = model1.sub_flows[i]
            del model1
    model = model.to(device)
    iter_loader = iter(loader)
    print(model)
    # init act norm
    for i in range(p.baseLevel, args.hlevel + 1):
        model.sub_flows[i].set_actnorm_init()
    model = model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    for i in range(2):
        if i % 100 == 0:
            print(i, '==================================================', i)
        if i == 0:
            target = next(iter_loader).to(device)
            data = normalize_samples(get_sample_modes(target, model), stds)
            latents = model.sample_latents()
            # latents = normalize_training(get_training_modes(target, model), stds)
            print([latents[i].shape for i in range(len(latents))])
            # print([(data[i].shape, torch.std(data[i])) for i in range(len(data))])
            samples = model.sample(target=data, latents=latents, comp = 'low')
        else:
            target1 = next(iter_loader).to(device)
            data = normalize_samples(get_sample_modes(target1, model), stds)
            target = torch.cat((target, target1), dim=0)
            latents_temp = model.sample_latents()
            # latents_temp = normalize_training(get_training_modes(target1, model), stds)
            samples_temp = model.sample(target=data, latents=latents_temp, comp = 'low')
            for i in range(len(samples)):
                samples[i] = torch.cat((samples[i], samples_temp[i]), dim=0)

            for i in range(len(latents)):
                latents[i] = torch.cat((latents[i], latents_temp[i]), dim=0)

    print('Data Loaded Sucessfully', target.shape)
    print('Model Sampled Sucessfully', len(samples))
    print('Prior Sampled Sucessfully', len(latents))

    x = target
    data = get_training_modes(x, model)
    data = normalize_training(data, stds)
    data_std = torch.std(data[1])
    latent_std = torch.std(latents[1])
    # for i in range(1, len(data)):
    #     print('testing ', data[i].shape, latents[i].shape, samples[i-1].shape, data_std, latent_std, data[0].shape)
    #     plot_power(data[i].cpu().numpy(), samples[i-1].cpu().numpy(), latents[i].cpu().numpy(), i-1)

    print(x.cpu().numpy().shape, samples[-1].cpu().numpy().shape, x.cpu().numpy().shape)
    plot_power(x.cpu().numpy(), samples[-1].cpu().numpy(), x.cpu().numpy(), 6)

    # #plot samples
    # for i in range(1, args.hlevel+1-p.baseLevel):
    #     samples_i = samples[i][:3, :, :, :].cpu().numpy()
    #     data_i = data[i][:3, :, :, :]
    #     samples1 = np.concatenate((data_i, samples_i), axis=1)
    #     samples1 = np.transpose(samples1, axes=(1, 0, 2, 3))
    #     plot_samples(samples1, i+1)

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="waveletflow", help='train level')
    parser.add_argument('--hlevel', type=int, default=-1, help='train level')
    parser.add_argument('--data', type=str, default="isic", help='train level')
    args = parser.parse_args()
    main()

    