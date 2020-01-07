import h5py
import numpy as np
import torch
from netCDF4 import Dataset
import pandas as pd
import pickle as pkl
import torchvision.transforms as t

def h5py_sandeep():

    hpfile = '/Users/mostafa/Dropbox/progs/datas/P_train_pc.h5'
    hf = h5py.File(hpfile, 'r')
    # print(hf.keys())
    hf0 = hf['dataset'][:]
    hf0 = hf0[:, 0, :, :]

    # hf0 = torch.tensor(np.array(hf0))
    hf0 = torch.Tensor(hf0)

    print('hf0_shape_data', hf0.shape)
    return hf0


def my_file():
    hpfile = '/Users/mostafa/Dropbox/progs/datas/myfile.h5'
    hf = h5py.File(hpfile, 'r')
    # print(hf.keys())
    hf0 = hf['DS3'][:]

    hf0 = np.array(hf0).astype('float')

    # # removing nan values
    # df_shape = hf0.shape
    # hf0 = hf0.reshape(df_shape[0] * df_shape[1], df_shape[2])
    # hf0 = pd.DataFrame(hf0)
    #
    # hf0 = hf0.fillna(hf_mean)
    # # hf0 = (hf0 - hf_min) / (hf_max - hf_min)
    # hf0 = hf0.to_numpy().reshape(*df_shape)

    for i in range(hf0.shape[2]):
        hf0_mean = np.nanmean(hf0[:, :, i])
        hf0[np.isnan(hf0[:, :, i])] = hf0_mean

        hf_max = np.nanmax(hf0[:, :, i])
        hf_min = np.nanmin(hf0[:, :, i])

        hf0[:, :, i] = (hf0[:, :, i] - hf_min) / (hf_max - hf_min)

    # hf0 = torch.tensor(np.array(hf0))
    hf0 = torch.Tensor(hf0)
    hf0 = hf0.permute(2, 0, 1)

    print('hf0_shape_data', hf0.shape)
    return hf0


def nc_file():

    ncfile = '/Users/mostafa/Dropbox/progs/datas/nnx2.nc'
    nf = Dataset(ncfile, mode='r')
    nf0 = nf.variables['thetao'][:]

    nf0 = np.array(nf0)

    # nf0 = torch.tensor(np.array(nf0))
    nf0 = torch.Tensor(nf0)
    nf0 = torch.squeeze(nf0)

    print('nf0_shape_data', nf0.shape)
    return nf0
    

# data = h5py_sandeep()
data = my_file()
# data = nc_file()

output = open('/Users/mostafa/Dropbox/progs/datas/train/data_1.pkl', 'wb')
pkl.dump(data, output)
output.close()