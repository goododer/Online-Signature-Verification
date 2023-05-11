import torch
from torch.utils.data import Dataset
import numpy as np
import os
import utils

class SVC2004(Dataset):
    def __init__(self, data_dir, max_length=793, window_size = 10) -> None:
        print('initializing')
        self.data_dir = data_dir
        self.list_files = os.listdir(self.data_dir)
        self.max_length = max_length
        self.window_size = window_size

        self.list_files.remove('.DS_Store') 
        self.list_files = sorted(self.list_files, key = lambda x: utils.fname_sorting_key(x))

    
    def __len__(self):
        # count how many files are in the dir.
        # no matter training/testing data.
        return len(self.list_files)
    
    def __getitem__(self, idx):

        file_name = self.list_files[idx]
        f = open(self.data_dir+file_name, 'rb')
        data = np.load(f) # npy. format
        data = data.astype(np.float32)
        f.close()
        data = torch.from_numpy(data)

        # padding the data to the max_length
        data = utils.padding(data, self.max_length)
        # slicing window
        data = utils.slicing_window(data, self.window_size)
        # stack data
        data = utils.stack(data)
        return data


if __name__ =='__main__':
    data_dir = './datasets/SVC2004/task1/training/'

    train_data = SVC2004(data_dir, 793, 10)

    print(train_data[10].shape)