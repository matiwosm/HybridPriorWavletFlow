from src.dwt.dwt import Dwt, Dwt_custom
from src.dwt.wavelets import Haar
import torch
import matplotlib.pyplot as plt
from data_loader import ISIC
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bdir = "../../../../../pscratch/sd/m/mati/half_yuki_sim_64"
file = "data.mdb"
transformer1 = None
dataset = ISIC(bdir, file, transformer1, 1, False)
loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
dataloader_iterator = iter(loader)
x = dataloader_iterator.next().to(device)

wavelet = Haar().to(device)
dwt = Dwt(wavelet=wavelet).to(device)
dwt_custom = Dwt_custom(wavelet=wavelet).to(device)


y = dwt.forward(x)

y2 = dwt_custom.forward(x)

x_reconstructed = dwt.inverse(y2)

# Calculate reconstruction error
error = torch.mean((x - x_reconstructed) ** 2).item()
exact_match = torch.allclose(x, x_reconstructed, atol=1e-6)

print("Reconstruction error:", error)
print("Exact match:", exact_match)


error = torch.mean((y['low'] - y2['low']) ** 2).item()
exact_match = torch.allclose(y['low'], y2['low'], atol=1e-6)

print("Reconstruction error low:", error)
print("Exact match low:", exact_match)    

error = torch.mean((y['high'] - y2['high']) ** 2).item()
exact_match = torch.allclose(y['high'], y2['high'], atol=1e-6)

print("Reconstruction error high:", error)
print("Exact match low:", exact_match)  