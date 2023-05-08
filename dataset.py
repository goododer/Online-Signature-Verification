import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re

class SVC2004(Dataset):
    def __init__(self, data_dir) -> None:
        print('initializing')
        self.data_dir = data_dir
        self.list_files = os.listdir(self.data_dir)

        self.list_files.remove('.DS_Store')
        self.list_files = sorted(self.list_files, key = lambda x:(int(re.split(r'(\d+)',x.split('.')[0])[1]),int(re.split(r'(\d+)',x.split('.')[0])[3])) )

    
    def __len__(self):
        # count how many files are in the dir.
        # no matter training/testing data.
        return len(self.list_files)
    
    def __getitem__(self, idx):

        file_name = self.list_files[idx]
        f = open(self.data_dir+file_name, 'rb')
        data = np.load(f) # npy. format
        f.close()

        return torch.from_numpy(data)


if __name__ =='__main__':
    data_dir = './datasets/SVC2004/task1/training/'

    train_data = SVC2004(data_dir)

    print(len(train_data))
    print(train_data[10])
    print(train_data[10].shape)
