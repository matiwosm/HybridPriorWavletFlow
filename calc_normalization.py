import torch
import numpy as np

# Define the tensor
stds = torch.tensor([[10.4746, 10.4746, 10.4746, 10.4746],
                     [10.4746, 5.9943,  6.0226,  4.4362],
                     [7.0993, 3.6189,  3.6250,  2.5660],
                     [4.5613, 2.0915,  2.0961,  1.4750],
                     [2.8173, 1.1993,  1.2024,  0.8454],
                     [1.6983, 0.6829,  0.6875,  0.4227]])

# Convert the tensor to a numpy array
stds_np = stds.numpy()

# Save as a .npz file
np.savez('norm/kappa_stds_tensor.npz', stds=stds_np)
