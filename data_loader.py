from torch.utils.data import TensorDataset, DataLoader, Dataset
from os import listdir
from PIL import Image
import random
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
import numpy as np
import lmdb
import os
import torch
import os
import lmdb
import numpy as np
from torch.utils.data import Dataset

# Channel name -> index mapping
CHANNEL_MAP = {
    'kappa': 0,
    'ksz'  : 1,
    'tsz'  : 2,
    'cib'  : 3,
    'rad'  : 4
}

def scale_data_pt(
    data: torch.Tensor, 
    channels: list[str], 
    reverse: bool = False
) -> torch.Tensor:
    """
    Unified PyTorch function for forward-scaling or reverse-scaling of a subset of channels.
    
    Parameters
    ----------
    data : torch.Tensor
        Shape (B, N, H, W). 
        B = batch size, N = number of channels actually in this tensor, H, W = spatial dims.
    channels : list of str
        Length N list specifying the channels in 'data' order.
        Valid options: ['kappa', 'ksz', 'tsz', 'cib', 'rad'].
    reverse : bool, default=False
        If False, apply *forward* scaling (like your _apply_scaling).
        If True,  apply *reverse* scaling (like your reverse_scale).

    Returns
    -------
    torch.Tensor
        Scaled (or reverse-scaled) data, shape (B, N, H, W), same as input.
    """

    # ---- Constants for all channels ----
    # kappa
    kap_mean  =  0.0024131738313520608
    kap_std   =  0.11190232474340092
    # ksz
    ksz_mean  =  0.5759176599953563
    ksz_std   =  2.0870242684435416
    # tsz
    tsz_std   =  3.2046874257710276
    ttsz_mean = -0.9992715262205959    # "trans_tsz_mean"
    ttsz_std  =  0.23378351341581394   # "trans_tsz_std"
    
    # cib
    cib_std   =  16.5341785469026
    tcib_mean =  0.7042645521815042
    tcib_std  =  0.3754746350117235
    # rad
    rad_std   =  0.0004017594060247909
    trad_mean =  0.6288525847415318
    trad_std  =  2.1106109860689175

    # ---- Define per-channel forward transforms ----
    def forward_kappa(x: torch.Tensor) -> torch.Tensor:
        return (x - kap_mean) / kap_std
    
    def forward_ksz(x: torch.Tensor) -> torch.Tensor:
        return (x - ksz_mean) / ksz_std
    
    def forward_tsz(x: torch.Tensor) -> torch.Tensor:
        # sign & log transform => sign(x)*log(abs(x)/tsz_std + 1)
        x_out = torch.sign(x) * torch.log(torch.abs(x)/tsz_std + 1.0)
        # shift & scale
        x_out = (x_out - ttsz_mean) / ttsz_std
        return x_out

    def forward_cib(x: torch.Tensor) -> torch.Tensor:
        x_out = torch.sign(x) * torch.log(torch.abs(x)/cib_std + 1.0)
        x_out = (x_out - tcib_mean) / tcib_std
        return x_out

    def forward_rad(x: torch.Tensor) -> torch.Tensor:
        x_out = torch.sign(x) * torch.log(torch.abs(x)/rad_std + 1.0)
        x_out = (x_out - trad_mean) / trad_std
        return x_out

    # ---- Define per-channel reverse transforms ----
    def reverse_kappa(x: torch.Tensor) -> torch.Tensor:
        return x * kap_std + kap_mean

    def reverse_ksz(x: torch.Tensor) -> torch.Tensor:
        return x * ksz_std + ksz_mean

    def reverse_tsz(x: torch.Tensor) -> torch.Tensor:
        # undo shift & scale
        x_out = x * ttsz_std + ttsz_mean
        # undo sign & log => sign(...) * (exp(abs(...)) - 1)*tsz_std
        x_out = torch.sign(x_out) * (torch.exp(torch.abs(x_out)) - 1.0) * tsz_std
        return x_out

    def reverse_cib(x: torch.Tensor) -> torch.Tensor:
        x_out = x * tcib_std + tcib_mean
        x_out = torch.sign(x_out) * (torch.exp(torch.abs(x_out)) - 1.0) * cib_std
        return x_out

    def reverse_rad(x: torch.Tensor) -> torch.Tensor:
        x_out = x * trad_std + trad_mean
        x_out = torch.sign(x_out) * (torch.exp(torch.abs(x_out)) - 1.0) * rad_std
        return x_out

    # ---- Map channel name -> (forward_func, reverse_func) ----
    channel_transforms = {
        'kappa': (forward_kappa, reverse_kappa),
        'ksz':   (forward_ksz,   reverse_ksz),
        'tsz':   (forward_tsz,   reverse_tsz),
        'cib':   (forward_cib,   reverse_cib),
        'rad':   (forward_rad,   reverse_rad),
    }

    # Clone input so we don't modify it in-place
    out = data.clone()

    # Loop over the channels we actually have in `out`
    for i, ch_name in enumerate(channels):
        if ch_name not in channel_transforms:
            raise ValueError(f"Unknown channel name '{ch_name}' - must be one of {list(channel_transforms.keys())}")
        
        fwd_func, rev_func = channel_transforms[ch_name]
        
        if reverse:
            out[:, i, :, :] = rev_func(out[:, i, :, :])
        else:
            out[:, i, :, :] = fwd_func(out[:, i, :, :])

    return out


def apply_noise_torch_vectorized(
    data: torch.Tensor,
    channel_names: list[str],
    noise_dict: dict[str, tuple[float, float]],
    dx: float = 0.5
) -> torch.Tensor:
    """
    Vectorized noise injection in frequency space for a subset of channels.
    
    Parameters
    ----------
    data : torch.Tensor
        Shape (B, N, H, W). 
          - B: batch size
          - N: number of channels actually in this tensor
          - H, W: spatial dimensions
    channel_names : list[str]
        A list of length N giving the name of each channel in 'data' order.
        Valid channel names might be ['kappa', 'ksz', 'tsz', 'cib', 'rad'], or any subset.
    noise_dict : dict
        Mapping channel_name -> [noise_std, threshold_ell].
        Example:
            {
              'tsz': [5e-4, 2000.0],
              'cib': [1e-3, 4000.0],
              ...
            }
        If threshold_ell is None, we add *uniform* (white) noise over all frequencies.
    dx : float
        Pixel size in arcminutes (converted to radians internally).

    Returns
    -------
    torch.Tensor
        The noised data, same shape as 'data' (B, N, H, W). Real-valued.
    """
    # Convert dx from arcmin -> radians
    dx_rad = dx / 60.0 * np.pi / 180.0
    
    B, N, H, W = data.shape
    device = data.device

    # 1) Forward FFT for all channels/batch items at once
    data_fft = torch.fft.fft2(data, dim=(-2, -1))  # shape (B, N, H, W), complex

    # 2) Build frequency grid for (H, W) only once
    freq_y = torch.fft.fftfreq(H, d=dx_rad, device=device) * 2.0 * np.pi  # shape (H,)
    freq_x = torch.fft.fftfreq(W, d=dx_rad, device=device) * 2.0 * np.pi  # shape (W,)
    freq2d_y, freq2d_x = torch.meshgrid(freq_y, freq_x, indexing='ij')    # (H, W)
    freq_magnitude = torch.sqrt(freq2d_x**2 + freq2d_y**2)               # (H, W)

    # 3) For each channel in 'channel_names', add noise if specified in noise_dict
    for i, ch_name in enumerate(channel_names):
        if ch_name not in noise_dict:
            # If the user didn't specify noise for this channel, skip
            continue
        
        noise_std, threshold_ell = noise_dict[ch_name]

        # Create mask for frequencies above threshold, or uniform if threshold_ell is None
        if threshold_ell is None:
            # White noise over all freq
            mask_2d = torch.ones_like(freq_magnitude, dtype=torch.bool)
        else:
            mask_2d = (freq_magnitude > threshold_ell)

        # Expand mask to (1, 1, H, W) so it can broadcast over (B, 1, H, W)
        mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)

        # Generate complex Gaussian noise in shape (B, 1, H, W)
        noise_real = torch.randn((B, 1, H, W), device=device, dtype=data_fft.dtype) * noise_std
        noise_imag = torch.randn((B, 1, H, W), device=device, dtype=data_fft.dtype) * noise_std
        noise_complex = noise_real + 1j * noise_imag

        # Apply mask in frequency domain
        noise_complex = noise_complex * mask_4d

        # Add to the FFT of this channel
        data_fft[:, i : i+1, :, :] += noise_complex

    # 4) Inverse FFT to get noised data in spatial domain
    data_noisy = torch.fft.ifft2(data_fft, dim=(-2, -1))

    # 5) Take real part only
    data_noisy = data_noisy.real

    return data_noisy


class My_lmdb(Dataset):    

    def __init__(
        self, 
        db_path, 
        file_path, 
        transformer, 
        num_classes, 
        class_cond,
        # New parameters
        channels_to_use=None,       # list of strings, e.g. ['kappa','tsz']
        noise_dict=None,            # dict of {channel_name: noise_std}, e.g. {'kappa':0.025, 'tsz':0.05}
        apply_scaling=True,         # boolean - whether to scale the data
        data_shape=(5, 64, 64)      # specify the shape to reshape the array from LMDB
    ):
        """
        :param db_path: Path to the LMDB folder
        :param file_path: Not used here, but kept for compatibility
        :param transformer: Some transformer (if you have one)
        :param num_classes: Number of classes (if relevant for your usage)
        :param class_cond: Some class conditioning (bool)
        :param channels_to_use: list of channel names (strings). Only these channels will be returned.
        :param noise_dict: dictionary specifying noise levels per channel, e.g. {'kappa': 0.025, 'tsz': 0.1}.
        :param apply_scaling: whether to apply the scaling logic (True/False)
        :param data_shape: tuple specifying how to reshape the data read from LMDB
        """
        super().__init__()
        self.db_path = db_path
        self.file_path = file_path
        self.transformer = transformer
        self.num_classes = num_classes
        self.class_cond = class_cond
        
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object" error
        self.env = lmdb.open(
            self.db_path, 
            subdir=os.path.isdir(self.db_path),
            readonly=True, 
            lock=False
        )
        
        self.length = self.env.stat()['entries']

        # -- New fields --
        # If channels_to_use is None or empty, default to all available channels
        if channels_to_use is None or len(channels_to_use) == 0:
            channels_to_use = list(CHANNEL_MAP.keys())  # all: ['kappa','ksz','tsz','cib','rad']
        self.channels_to_use = channels_to_use
        
        # Convert channel names to indices
        self.channel_indices = [CHANNEL_MAP[ch] for ch in self.channels_to_use]
        
        # Noise dictionary (channel_name -> float). If None, no noise is applied.
        self.noise_dict = noise_dict if noise_dict is not None else {}
        
        # Whether to apply the scaling transforms or not
        self.apply_scaling = apply_scaling
        
        # The shape of the data in the LMDB
        self.data_shape = data_shape
        
        # (Optional) just to keep track of indexes if you need it
        self.list = []
    
    def _init_db(self):
        """Helper to re-open LMDB if needed."""
        self.env = lmdb.open(
            self.db_path, 
            subdir=os.path.isdir(self.db_path),
            readonly=True, 
            lock=False
        )
        self.txn = self.env.begin()
        self.length = self.env.stat()['entries']
    
    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()
        
        self.list.append(index)
        
        # Read raw bytes
        with self.env.begin() as txn:
            raw_data = txn.get('{:08}'.format(index).encode('ascii'))
        
        # Convert bytes -> np array, then reshape
        lmdb_data = np.frombuffer(raw_data, dtype=np.float64).reshape(self.data_shape)
        
        #remove this bit (the functions have moved out of the class)
        # Optionally apply noise & scaling
        if self.apply_scaling:
            lmdb_data = self._apply_noise(lmdb_data)
            lmdb_data = self._apply_scaling(lmdb_data)
        else:
            # If not scaling, we can still apply noise if desired
            lmdb_data = self._apply_noise(lmdb_data, do_scaling=False)
        
        # Select only the channels user asked for
        # final_data will have shape = (len(self.channel_indices), H, W)
        final_data = lmdb_data[self.channel_indices, :, :]
        
        return final_data

    import numpy as np

    def _apply_noise(self, data, dx=0.5, do_scaling=True):
        """
        Apply noise to the specified channels only after a certain frequency threshold (ell).
        - threshold_ell: The frequency (in cycles/pixel or similar units) after which
        noise should be added.
        """
        data = np.copy(data)  # So we don't modify in-place accidentally
        dx = (dx/60. * np.pi/180.) #arc_min to rad
        # Loop over each channel where noise needs to be added
        for ch_name, noise_params in self.noise_dict.items():
            noise_std = noise_params[0]
            threshold_ell = noise_params[1]
            ch_idx = CHANNEL_MAP[ch_name]
            
            if threshold_ell is not None:
                if ch_idx < data.shape[0]:
                    # Extract the 2D slice for this channel
                    channel_data = data[ch_idx, :, :]
                    
                    # 1) Forward FFT: convert spatial domain -> frequency domain
                    channel_fft = np.fft.fft2(channel_data)
                    
                    # 2) Build frequency grid
                    ny, nx = channel_data.shape
                    freq_y = np.fft.fftfreq(ny, dx)*2.*np.pi   # Frequencies along y-dimension
                    freq_x = np.fft.fftfreq(nx, dx)*2.*np.pi   # Frequencies along x-dimension
                    
                    # Create 2D frequency arrays using meshgrid
                    freq2d_x, freq2d_y = np.meshgrid(freq_x, freq_y)
                    # Magnitude of frequency
                    freq_magnitude = np.sqrt(freq2d_x**2 + freq2d_y**2)
                    # print('in apply noise ', np.max(freq_y), np.max(freq_x), np.max(freq_magnitude), threshold_ell)
                    # 3) Create a mask where freq_magnitude > threshold_ell
                    mask = (freq_magnitude > threshold_ell)
                    # print(mask, np.sum(mask))
                    # 4) Generate noise in the frequency domain
                    #    - Because the FFT of real data is symmetric in frequency space,
                    #      normally you'd want to ensure your noise is Hermitian-symmetric
                    #      if you want the inverse transform to remain real. However, a simple
                    #      approach is to generate complex noise for the entire frequency plane
                    #      and then rely on the inverse transform. (This may slightly break
                    #      real-valued symmetry, but often is acceptable if you're just injecting
                    #      “randomness” for simulation.)
                    
                    # Generate complex Gaussian noise
                    noise_real = np.random.normal(loc=0.0, scale=noise_std, size=channel_fft.shape)
                    noise_imag = np.random.normal(loc=0.0, scale=noise_std, size=channel_fft.shape)
                    noise_complex = noise_real + 1j * noise_imag
                    
                    # Apply mask so that noise is only added above threshold_ell
                    noise_fft = noise_complex * mask
                    
                    # 5) Add noise to the channel in the frequency domain
                    channel_fft_noisy = channel_fft + noise_fft
                    
                    # 6) Inverse FFT: frequency domain -> spatial domain
                    channel_noisy = np.fft.ifft2(channel_fft_noisy)
                    
                    # 7) Because the original data are presumably real,
                    #    take the real part (imaginary parts should be small if done carefully)
                    data[ch_idx, :, :] = np.real(channel_noisy)
            else:
                if ch_idx < data.shape[0]:
                    data[ch_idx, :, :] = self._apply_white_noise(data[ch_idx, :, :], noise_std)

        
        return data

    
    def _apply_white_noise(self, data, noise_std, do_scaling=True):
        data = np.copy(data)  # so we don't modify in-place accidentally
        
        noise = np.random.normal(loc=0.0, scale=noise_std, size=data.shape)
        data += noise
        return data

    def _apply_scaling(self, images):
        """
        This uses your original scaling logic for 
        (kappa, ksz, tsz, cib, rad) => indices (0,1,2,3,4).
        Only applies if those channels exist in 'images'.
        """
        data = np.copy(images)
        
        # Means, stds, etc.
        kap_mean =  0.0024131738313520608 
        ksz_mean =  0.5759176599953563 
        kap_std  =  0.11190232474340092
        ksz_std  =  2.0870242684435416

        tsz_std  =  3.2046874257710276 
        trans_tsz_mean = -0.9992715262205959 
        trans_tsz_std  =  0.23378351341581394

        cib_std  =  16.5341785469026 
        trans_cib_mean =  0.7042645521815042 
        trans_cib_std  =  0.3754746350117235

        rad_std  =  0.0004017594060247909 
        trans_rad_mean =  0.6288525847415318 
        trans_rad_std  =  2.1106109860689175

        # 0: kappa
        if 0 < data.shape[0]:  # i.e. data has at least 1 channel
            data[0, :, :] = (data[0, :, :] - kap_mean) / kap_std

        # 1: ksz
        if 1 < data.shape[0]:
            data[1, :, :] = (data[1, :, :] - ksz_mean) / ksz_std

        # 2: tsz
        if 2 < data.shape[0]:
            data[2, :, :] = np.sign(data[2, :, :]) * np.log(np.abs(data[2, :, :]) / tsz_std + 1)
            data[2, :, :] = (data[2, :, :] - trans_tsz_mean) / trans_tsz_std

        # 3: cib
        if 3 < data.shape[0]:
            data[3, :, :] = np.sign(data[3, :, :]) * np.log(np.abs(data[3, :, :]) / cib_std + 1)
            data[3, :, :] = (data[3, :, :] - trans_cib_mean) / trans_cib_std

        # 4: rad
        if 4 < data.shape[0]:
            data[4, :, :] = np.sign(data[4, :, :]) * np.log(np.abs(data[4, :, :]) / rad_std + 1)
            data[4, :, :] = (data[4, :, :] - trans_rad_mean) / trans_rad_std

        return data

    def reverse_scale(self, images):
        """
        Reverse the scaling. 
        If you only used certain channels, you need to pass them in the same order they were returned.
        For example, if you used ['kappa','cib'], then images will have shape (B,2,H,W) in that order.
        
        This method tries to reverse scale in the order [kappa, ksz, tsz, cib, rad].
        Make sure that aligns with how you're passing `images`.
        """
        # shape of 'images' expected: (batch, channels_used, H, W)

        kap_mean =  0.0024131738313520608 
        ksz_mean =  0.5759176599953563 
        kap_std  =  0.11190232474340092
        ksz_std  =  2.0870242684435416

        tsz_std  =  3.2046874257710276 
        trans_tsz_mean = -0.9992715262205959 
        trans_tsz_std  =  0.23378351341581394

        cib_std  =  16.5341785469026 
        trans_cib_mean =  0.7042645521815042 
        trans_cib_std  =  0.3754746350117235

        rad_std  =  0.0004017594060247909 
        trans_rad_mean =  0.6288525847415318 
        trans_rad_std  =  2.1106109860689175

        data = np.copy(images)

        # We need a careful approach: identify which channels in data correspond to which transformations
        # One way: for i in range(data.shape[1]), check which channel_index it corresponds to.

        # Example: if self.channels_to_use = ['kappa','tsz','cib'],
        #   then self.channel_indices = [0,2,3].
        #   The data in images[:, 0, ...] => channel 0 => 'kappa'
        #   The data in images[:, 1, ...] => channel 2 => 'tsz'
        #   The data in images[:, 2, ...] => channel 3 => 'cib'

        for i, ch_idx in enumerate(self.channel_indices):
            if ch_idx == 0:  # kappa
                data[:, i, :, :] = data[:, i, :, :] * kap_std + kap_mean

            elif ch_idx == 1:  # ksz
                data[:, i, :, :] = data[:, i, :, :] * ksz_std + ksz_mean

            elif ch_idx == 2:  # tsz
                data[:, i, :, :] = data[:, i, :, :] * trans_tsz_std + trans_tsz_mean
                # undo sign/log
                data[:, i, :, :] = np.sign(data[:, i, :, :]) * \
                                   (np.exp(np.abs(data[:, i, :, :])) - 1) * tsz_std

            elif ch_idx == 3:  # cib
                data[:, i, :, :] = data[:, i, :, :] * trans_cib_std + trans_cib_mean
                data[:, i, :, :] = np.sign(data[:, i, :, :]) * \
                                   (np.exp(np.abs(data[:, i, :, :])) - 1) * cib_std

            elif ch_idx == 4:  # rad
                data[:, i, :, :] = data[:, i, :, :] * trans_rad_std + trans_rad_mean
                data[:, i, :, :] = np.sign(data[:, i, :, :]) * \
                                   (np.exp(np.abs(data[:, i, :, :])) - 1) * rad_std

        return data

    def get_stat(self):
        return self.env.stat()

    def __len__(self):
        return self.get_stat()['entries']

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
   

class yuuki_256(Dataset):    
    def __init__(self, db_path, file_path, transformer, num_classes, class_cond, noise_level=0.0):
        self.db_path = db_path
        self.file_path = file_path
        self.transformer = transformer
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False)
        self.length = self.env.stat()['entries']
        self.num_classes = num_classes
        self.class_cond = class_cond
        self.noise_level = noise_level
        self.list = []
    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False)
        self.txn = self.env.begin()
        self.length = self.env.stat()['entries']
            
    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()
        self.list.append(index)
        with self.env.begin() as txn:
            lmdb_data = np.frombuffer(txn.get('{:08}'.format(index).encode('ascii')), dtype=np.float64).astype(np.float64).reshape((2, 256, 256)).copy() #.astype(np.float32)
            lmdb_data = self.rescale(lmdb_data)
            # lmdb_data = lmdb_data[0:1, :, :]
        return lmdb_data

    def rescale(self, images, noise_kappa=True):
        kap_mean = 0.014792284511963825
        kap_std =  0.11103206430738521
        
        cib_std =  16.1974079238
        trans_cib_mean =  0.7104694640995108
        trans_cib_std =  0.3748629431148323
        
        lmdb_data = np.copy(images)
        
        if noise_kappa:
            noise_level = self.noise_level
            noise = np.random.normal(loc=0, scale=noise_level, size=lmdb_data[0, :, :].shape)
            lmdb_data[0, :, :] = (lmdb_data[0, :, :] + noise)
            
        lmdb_data[0, :, :] = (lmdb_data[0, :, :] - kap_mean)/kap_std
        
        lmdb_data[1, :, :] = np.sign(lmdb_data[1, :, :])*(np.log(np.abs(lmdb_data[1, :, :])/cib_std + 1))
        lmdb_data[1, :, :] = (lmdb_data[1, :, :] - trans_cib_mean)/trans_cib_std
        
        
        return lmdb_data
    
    def reverse_scale(self, images):
        kap_mean = 0.014792284511963825
        kap_std =  0.11103206430738521
        
        cib_std =  16.1974079238
        trans_cib_mean =  0.7104694640995108
        trans_cib_std =  0.3748629431148323

        data = np.copy(images)
        data[:, 0, :, :] = data[:, 0, :, :]*kap_std + kap_mean
        data[:, 3, :, :] = data[:, 3, :, :]*trans_cib_std + trans_cib_mean
        
        data[:, 3, :, :] = np.sign(data[:, 3, :, :])*(np.exp(data[:, 3, :, :]*np.sign(data[:, 3, :, :])) - 1)*cib_std

        return data
    

    
    def get_stat(self):
        return self.env.stat()

    def __len__(self):
        return self.get_stat()['entries']

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    
    