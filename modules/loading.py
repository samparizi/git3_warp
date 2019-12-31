import h5py
import numpy as np
import torch





def h5py_sandeep():
    ######################## h5 file ####################

    hpfile = '/Users/mostafa/Desktop/datas/P_train_pc.h5'
    # hpfile = '/Users/mostafa/Desktop/datas/myfile.h5'

    hf = h5py.File(hpfile, 'r')
    # hf0 = hf['DS3'][:]
    hf0 = hf['dataset'][:, 0, :, :]

    # print(hf.keys())

    print('\n\nhf0', hf0.shape)

    hf0 = np.array(hf0).astype('float')

    # df_shape = hf0.shape
    # hf0 = hf0.reshape(df_shape[0]*df_shape[1], df_shape[2])
    #
    #
    # hf0 = pd.DataFrame(hf0)
    # hf0.fillna(method='ffill', axis=1)
    # hf0.fillna(method='ffill', axis=0)
    # hf0 = hf0.to_numpy().reshape(*df_shape)

    # print('\n\nth0_torch_tensor', th0.size())    # for this loading file

    hf0 = torch.Tensor(hf0)
    # hf0 = hf0.permute(2,0,1)  #for the new dataset

    print('hf0 after squeeze', hf0.shape)

    return hf0



def nc_file():
    ########################nc file ####################
    ncfile = '/Users/mostafa/Desktop/datas/nnx2.nc'
    fh = Dataset(ncfile, mode='r')
    # th0 = fh.variables['thetao'][:]
    # th0 = fh.variables['thetao'][0,0,:,:]
    th0 = fh.variables['thetao'][:]
    print('th0', th0.shape)
    th0 = np.array(th0)
    print('th0', th0.shape)

    th0 = torch.Tensor(th0)
    th0 = torch.squeeze(th0)
    print('th0 after squeeze', th0.shape)










######################## test file ####################

# ncfile = '/Users/mostafa/Desktop/datas/test/nnx4.nc'
# fh = Dataset(ncfile, mode='r')
# th0 = fh.variables['thetao'][:]
#
# print('th0000', th0.shape)
#
# th0 = np.array(th0)
#
# print('th0000', th0.shape)
#
# th0 = torch.Tensor(th0)
#
# print('th0000', th0.shape)
#
#
# th0 = torch.squeeze(th0)
#
# print('th0000', th0.shape)
# output = open('/Users/mostafa/Desktop/datas/test/data_1.pkl', 'wb')
# pkl.dump(th0, output)
# output.close()
#
# # data = pkl.load(open('/Users/mostafa/Desktop/datas/test/data_1.pkl', 'rb'))
# # print('data', data.shape)

######################## test file ####################