# Hybrid-Prior WaveletFlow

This repository extends the WaveletFlowPyTorch project, originally found here: [WaveletFlowPyTorch](https://github.com/A-Vzer/WaveletFlowPytorch). We added inverse sampling and other functionalites needed for our work. 

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

The code has been tested with Python 3.10.

## Overview

This project includes scripts for handling the training and sampling of Wavelet Flow models with hybrid priors. Below is a summary of the main functionalities:

## Key Features

### 1. Power Spectrum Calculation

To calculate the power spectra for DWT-transformed input data, run:

```bash
python calc_prior_spec.py --config config/config_file.py --output_dir ps/output_path
```

This generates power spectra as `dwtlevelZxZ.dat` files in the `ps/output_path/` directory, where `Z` is the size of the map at that DWT level. For instance, the file `*2x2.dat` contains power spectra and cross-spectra for one low-frequency and three high-frequency components of a 2x2 map. These spectra are used to generate correlated priors. For inputs with two channels (e.g., training kappa and CIB together), the columns are as follows:

- **Column 1:** Wave numbers (ℓ).
- **Columns 2-3:** DWT low-frequency component of the two channels.
- **Columns 4-5:** DWT high-horizontal component of the two channels.
- **Columns 6-7:** DWT high-vertical component of the two channels.
- **Columns 8-9:** DWT high-diagonal component of the two channels.
- **Additional Columns:** Cross-spectra for the components of channel 1 and channel 2.

### 2. Normalization

To compute normalization constants (standard deviation, minimum, and maximum) for the DWT-transformed training data, run:

```bash
python calc_normalization.py --config config/config_file.py --output norm_std/normalization_file_path.json
```

The constants are saved as a `.json` file in the `norm_std` folder. These values are used for data normalization during training. Access the normalization constants as follows:

```python
with open("norm_std/normalization_file_path.json", 'r') as f:
    mean_stds_all_levels = json.load(f)

# Example: Get the standard deviation of the low-frequency component of the 2nd channel at the 3rd DWT level
std = mean_stds_all_levels[3]['low']['mean_std'][2]
```

Normalization constants for high-frequency components can be accessed using the keys `high_horizontal`, `high_vertical`, and `high_diagonal`.

### 3. Model Training

To train the WaveletFlow model, use the script `train.py`. You can train each DWT level independently using:

```bash
python train.py --level n --config configs/config_file.py
```

- `n`: Specifies the DWT level to train.
- `configs/config_file.py`: Configuration file for training. Refer to `configs/old_configs/HCC_prior_best_model_256x256_all_levels_kap_noise_0.01.py` for detailed instructions on configuring the file.

### 4. Sampling

To generate power spectra and Minkowski functional plots for the trained models, use the `Sample_test.py` script:

```bash
python Sample_test.py --config config/config_file.py
```

Ensure that the training and sampling scripts use the same configuration file.

## Workflow Summary

1. Compute normalization constants:

   ```bash
   python calc_normalization.py --config config/config_file.py --output norm_std/output_path.json
   ```

2. Calculate power spectra:

   ```bash
   python calc_prior_spec.py --config config/config_file.py --output_dir ps/output_path
   ```

3. Update the configuration file:
   - Set `std_path` to `norm_std/output_path.json`.
   - Set `ps_path` to `ps/output_path`.

4. Train the model:

   ```bash
   python train.py --config config/config_file.py --level X
   ```
   Repeat for all levels `X` from `base_level` to `nLevels` as specified in the configuration file.

5. Generate samples and plot summary statistics:

   ```bash
   python Sample_test.py --config config/config_file.py
   ```


The repo https://github.com/matiwosm/SampleHybridPriorWavletFlow.git includes our trained models and the scripts used to generate the plots in our associated paper. 
