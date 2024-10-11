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
def get_2d_power(nx, dx, r1, r2=None, num_bins=100):
        if (np.any(np.isnan(r1)) or np.any(np.isinf(r1))):
            print("whyyyyyy")
            
        if (r2 is None):
            r2 = r1
        lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,
                                np.fft.fftfreq( nx, dx )*2.*np.pi )
        ell = np.sqrt(lx**2 + ly**2)
        ell = ell[~np.isnan(ell)]
        lbins = ell.flatten()
        FMap1 = np.fft.rfft2(r1)

       
        FMap2 = np.fft.rfft2(r2)
        # FMap1 = np.fft.ifft2(np.fft.fftshift(r1))
        cvec = (np.conj(FMap1) * (FMap2)).flatten()
        
        return lbins, cvec

for m in range(0, 5):
    loader = DataLoader(ISIC, batch_size=512)
    util_obj = util()
    comp_low_spectrum = []
    comp1_spectrum = []
    comp2_spectrum = []
    comp3_spectrum = []
    for i, x in enumerate(loader):
        for k in range(m+1):
            x1 = dwt.forward(x)['low']
            x2 = dwt.forward(x)['high']
            x = x1
        nx = x.shape[-1]
        dx = (0.5/60. * np.pi/180.)*2**(m+1)
        print(m, x1.shape, x2.shape)
        comp_low = x1.cpu().numpy()
        comp1 = x2[:, :1, :, :].cpu().numpy()
        comp2 = x2[:, 1:2, :, :].cpu().numpy()
        comp3 = x2[:, 2:3, :, :].cpu().numpy()
        for j in range(comp1.shape[0]):
            ell, bin_spectrum = get_2d_power(nx, dx, comp_low[j, 0, :, :], comp_low[j, 0, :, :])
            comp_low_spectrum.append(np.real(bin_spectrum))
            ell, bin_spectrum = get_2d_power(nx, dx, comp1[j, 0, :, :], comp1[j, 0, :, :])
            comp1_spectrum.append(np.real(bin_spectrum))

            ell, bin_spectrum = get_2d_power(nx, dx, comp2[j, 0, :, :], comp2[j, 0, :, :])
            comp2_spectrum.append(np.real(bin_spectrum))

            ell, bin_spectrum = get_2d_power(nx, dx, comp3[j, 0, :, :], comp3[j, 0, :, :])
            comp3_spectrum.append(np.real(bin_spectrum))

    comp_low_spectrum = np.real(np.array(comp_low_spectrum))
    comp_low_spectrum = np.mean(comp_low_spectrum, axis=0).real

    comp1_spectrum = np.real(np.array(comp1_spectrum))
    comp1_spectrum = np.mean(comp1_spectrum, axis=0).real

    comp2_spectrum = np.real(np.array(comp2_spectrum))
    comp2_spectrum = np.mean(comp2_spectrum, axis=0).real

    comp3_spectrum = np.real(np.array(comp3_spectrum))
    comp3_spectrum = np.mean(comp3_spectrum, axis=0).real
    print(ell.shape, comp1_spectrum.shape, comp2_spectrum.shape, comp3_spectrum.shape)
    with open('spec_dwtlevel_all'+str(nx)+'x'+str(nx)+'.dat', 'w') as file:
        print('SAVING')
        # Write each index from both arrays to the file
        for i in range(len(ell)):
            file.write(f"{ell[i]}; {comp_low_spectrum[i]}; {comp1_spectrum[i]}; {comp2_spectrum[i]}; {comp3_spectrum[i]}\n")