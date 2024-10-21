import os
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset,DataLoader
from utils.utils import *
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class Cardiac_Map_DataSet(Dataset):

    def __init__(self, config, mode):
        super(Cardiac_Map_DataSet, self).__init__()
        self.config = config
        self.minv_values_list = []

        if mode == 'training':
            # self.kspace_dir = '/data2/yuanyuan/data/data_T1rho_T1/h5_data_train/'
            self.kspace_dir = '/data0/shucong/code/SSJDM_contrast_code/self_diffusion_t1rho_clean/data/train/'
        elif mode == 'test':
            self.kspace_dir = '/data0/shucong/code/SSJDM_contrast_code/self_diffusion_t1rho_clean/data/test/'
        else:
            raise NotImplementedError

        self.mode = mode
        self.file_list = get_all_files(self.kspace_dir)
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(self.kspace_dir, os.path.basename(file)))

            with h5py.File(os.path.join(self.kspace_dir, file), 'r') as data:
                if self.mode == 'training':
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 50)
                elif self.mode =='test':
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] )
                else:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 50)
        
        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()#return list or number

        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0]) #get the 1st array and the 1th element
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] +self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        raw_file = os.path.join(self.kspace_dir,os.path.basename(self.file_list[scan_idx]))

        if self.mode != 'test':
           slice_idx = slice_idx + 30 
        else:
           slice_idx = slice_idx
        with h5py.File(raw_file, 'r') as data:
            ksp_idx =data['kspace'][slice_idx]
            ksp_idx = np.expand_dims(ksp_idx, 0)

            minv = np.std(ksp_idx)
            ksp_idx = ksp_idx /  minv

            self.minv_values_list.append(minv)

            result_scale_dir='./results/recon/scale/'

            if not os.path.exists(result_scale_dir):
                os.makedirs(result_scale_dir)

            scio.savemat(os.path.join(result_scale_dir, 'minv_scale_SSJDM.mat'), {'minv_scale': self.minv_values_list})

            ksp_idx = np.squeeze(ksp_idx, 0)
            kspace = np.asarray(ksp_idx)

            maps_idx = data['csm'][slice_idx]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = np.squeeze(maps_idx, 0)
            maps_idx = np.conjugate(maps_idx)
            maps = np.asarray(maps_idx)

        return kspace, maps

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))

def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == 'UIH_Cardiac' :
        dataset = Cardiac_Map_DataSet(config,mode)

    if mode == 'training':
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True)

    print(mode, "data loaded")
    
    return data

def get_dataset1(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == 'UIH_Cardiac' :
        dataset = Cardiac_Map_DataSet(config,mode)

    if mode == 'training':
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True)

    print(mode, "data loaded")
    
    return data

