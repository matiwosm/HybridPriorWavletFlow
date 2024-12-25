from torch.utils.data import TensorDataset, DataLoader, Dataset
from os import listdir
from PIL import Image
import random
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
import numpy as np
import lmdb
import os

class My_lmdb(Dataset):    
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
            lmdb_data = np.frombuffer(txn.get('{:08}'.format(index).encode('ascii')), dtype=np.float64).astype(np.float64).reshape((5, 64, 64)).copy() #.astype(np.float32)
            lmdb_data = self.scale_data(lmdb_data)
            # lmdb_data = np.flip(lmdb_data, axis=0)
        return lmdb_data

    def scale_data(self, images, noise_kappa=True):
        kap_mean =  0.0024131738313520608 
        ksz_mean =  0.5759176599953563 

        kap_std =  0.11190232474340092
        ksz_std =  2.0870242684435416

        tsz_std =  3.2046874257710276 
        trans_tsz_mean =  -0.9992715262205959 
        trans_tsz_std =  0.23378351341581394

        cib_std =  16.5341785469026 
        trans_cib_mean =  0.7042645521815042 
        trans_cib_std =  0.3754746350117235

        rad_std =  0.0004017594060247909 
        trans_rad_mean =  0.6288525847415318 
        trans_rad_std =  2.1106109860689175

        lmdb_data = np.copy(images)

        if noise_kappa:
            noise_level = self.noise_level
            noise = np.random.normal(loc=0, scale=noise_level, size=lmdb_data[0, :, :].shape)
            lmdb_data[0, :, :] = (lmdb_data[0, :, :] + noise)

        lmdb_data[0, :, :] = (lmdb_data[0, :, :] - kap_mean)/kap_std
        lmdb_data[1, :, :] = (lmdb_data[1, :, :] - ksz_mean)/ksz_std

        lmdb_data[2, :, :] = np.sign(lmdb_data[2, :, :])*(np.log(np.abs(lmdb_data[2, :, :])/tsz_std + 1))
        lmdb_data[2, :, :] = (lmdb_data[2, :, :] - trans_tsz_mean)/trans_tsz_std

        lmdb_data[3, :, :] = np.sign(lmdb_data[3, :, :])*(np.log(np.abs(lmdb_data[3, :, :])/cib_std + 1))
        lmdb_data[3, :, :] = (lmdb_data[3, :, :] - trans_cib_mean)/trans_cib_std

        lmdb_data[4, :, :] = np.sign(lmdb_data[4, :, :])*(np.log(np.abs(lmdb_data[4, :, :])/rad_std + 1))
        lmdb_data[4, :, :] = (lmdb_data[4, :, :] - trans_rad_mean)/trans_rad_std

        return lmdb_data[[0, 3], :, :]
    
    def reverse_scale(self, images):
        kap_mean =  0.0024131738313520608 
        ksz_mean =  0.5759176599953563 

        kap_std =  0.11190232474340092
        ksz_std =  2.0870242684435416

        tsz_std =  3.2046874257710276 
        trans_tsz_mean =  -0.9992715262205959 
        trans_tsz_std =  0.23378351341581394

        cib_std =  16.5341785469026 
        trans_cib_mean =  0.7042645521815042 
        trans_cib_std =  0.3754746350117235

        rad_std =  0.0004017594060247909 
        trans_rad_mean =  0.6288525847415318 
        trans_rad_std =  2.1106109860689175

        data = np.copy(images)
        data[:, 0, :, :] = data[:, 0, :, :]*kap_std + kap_mean
        data[:, 1, :, :] = data[:, 1, :, :]*ksz_std + ksz_mean
        data[:, 2, :, :] = data[:, 2, :, :]*trans_tsz_std + trans_tsz_mean
        data[:, 3, :, :] = data[:, 3, :, :]*trans_cib_std + trans_cib_mean
        data[:, 4, :, :] = data[:, 4, :, :]*trans_rad_std + trans_rad_mean
        
        data[:, 2, :, :] = np.sign(data[:, 2, :, :])*(np.exp(data[:, 2, :, :]*np.sign(data[:, 2, :, :])) - 1)*tsz_std
        data[:, 3, :, :] = np.sign(data[:, 3, :, :])*(np.exp(data[:, 3, :, :]*np.sign(data[:, 3, :, :])) - 1)*cib_std
        data[:, 4, :, :] = np.sign(data[:, 4, :, :])*(np.exp(data[:, 4, :, :]*np.sign(data[:, 4, :, :])) - 1)*rad_std

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

    
    