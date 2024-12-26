# Hybrid-Prior WaveletFlow

This repository is an extension on the WaveletFlowPyTorch project: https://github.com/A-Vzer/WaveletFlowPytorch.

## Installation

To install the required dependencies, run: `pip install -r requirements.txt`


## Overview

This project includes several key scripts for handling Discrete Wavelet Transforms (DWT) and training a WaveletFlow model with hybrid priors.

### Scripts

1. **Power Spectrum Calculation**

   Use `calc_prior_spec.py` to generate power spectra for the DWT-transformed input data. The results will be saved as `.dat` files in the `ps` directory. Each file contains five columns:

   - Column 1: Wave numbers (ℓ)
   - Column 2: DWT low-frequency component
   - Column 3-5: DWT high-frequency components (1, 2, and 3)

   For example, `spec_dwtlevel_2x2.dat` contains power spectra for one low-frequency component and three high-frequency components of the 2x2 map.

2. **Normalization**

   Run `calc_normalization.py` to compute normalization constants for the DWT-transformed training data. The normalization constants will be saved as `.jason` files in the `norm_stds` folder. It also calculates the golbal min and max of the data.

3. **Model Training**

   Train the WaveletFlow model by running `train.py`. You can train each DWT level independently using the following command:

`python train.py --level n --config configs/example_config_hcc_prior.py`

where `n` specifies the DWT level to train and `configs/example_config_hcc_prior.py` is a config file. See `configs/example_config_hcc_prior.py` for example.

4. **Sampling**

Use `Sample_test.py` for sampling from the trained model. The `--hlevel` argument allows you to specify the highest DWT level to include in the sampling process. The `--config` is used to specify the config file. Training and sampling should be done using the same configuration. For a full map, set `hlevel` to the highest training level.

### Contributions

This project has been modified by Matiwos Mebratu to support hybrid priors and implement inverse transformations for sampling.


