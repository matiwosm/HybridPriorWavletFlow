# Hybrid-Prior WaveletFlow

This repository extends the WaveletFlowPyTorch project, originally found here: [WaveletFlowPyTorch](https://github.com/A-Vzer/WaveletFlowPytorch).

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Overview

This project introduces several key scripts for handling Discrete Wavelet Transforms (DWT) and training a WaveletFlow model with hybrid priors. Below is a summary of the main functionalities:

### Scripts

#### 1. **Power Spectrum Calculation**

The script `calc_prior_spec.py` generates power spectra for DWT-transformed input data. The results are saved as `.dat` files in the `ps` directory, containing the following columns:

- **Column 1:** Wave numbers (ℓ)
- **Column 2:** DWT low-frequency component
- **Columns 3-5:** DWT high-frequency components (1, 2, and 3)

For example, the file `*2x2.dat` contains power spectra for one low-frequency component and three high-frequency components of the 2x2 map.

#### 2. **Normalization**

The script `calc_normalization.py` computes normalization constants for the DWT-transformed training data. These constants are saved as `.json` files in the `norm_stds` folder. The script also calculates the global minimum and maximum values of the data.

#### 3. **Model Training**

To train the WaveletFlow model, use the script `train.py`. You can train each DWT level independently with the following command:

```bash
python train.py --level n --config configs/example_config_hcc_prior.py
```

- `n` specifies the DWT level to train.
- `configs/example_config_hcc_prior.py` is the configuration file. See this file for an example configuration.

#### 4. **Sampling**

Use the script `Sample_test.py` to generate samples from the trained model. The `--hlevel` argument specifies the highest DWT level to include during sampling, while the `--config` argument specifies the configuration file. Training and sampling must use the same configuration. To generate a full map, set `hlevel` to the highest training level.

## Contributions

This project has been enhanced by Matiwos Mebratu to support hybrid priors and implement inverse transformations for sampling.

