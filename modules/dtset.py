import pickle as pkl
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch


class Dset(torch.utils.data.Dataset):

    def __init__(self, root, seq_len=4, target_seq_len=6,
                 transform=None,
                 zones=None,
                 ):
        if zones is None:  # using all zones
            zones = range(1, 2)

        self.root = root
        self.zones = zones
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        self.transform = transform

        self.data = {}

        for zone in zones:
            self.data[zone] = pkl.load(open('/Users/mostafa/Desktop/datas/train/data_1.pkl', 'rb'))

        self.num_single = self.data[zones[0]].shape[0] - self.seq_len - self.target_seq_len  #- 1
        self.num = self.num_single * len(zones)

    def __getitem__(self, index):

        zone = self.zones[index // (self.num_single - 1)]
        sample_num = index % (self.num_single - 1)
        pdata = self.data[zone]

        # # # self.data = pkl.load(open('/Users/mostafa/Desktop/datas/train/data_1.pkl', 'rb'))
        # # sample_num = index
        # # sample_num = index % (self.num - 1)
        # sample_num = self.num
        #
        print('sample_num', sample_num)
        #
        # pdata = self.data

        input = pdata[sample_num: sample_num + self.seq_len, :, :]
        target = pdata[sample_num + self.seq_len: sample_num + self.seq_len + self.target_seq_len, :, :]

        # return self.transform(input), self.transform(target)
        return input, target

    def __len__(self):
        return self.num - len(self.zones)
