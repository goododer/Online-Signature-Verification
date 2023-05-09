import torch.nn.functional as functional
import re

def normalize(data):
    # not now
    return data

def padding(data, max_length):
    # Pad the last dimension To the man_length with 0's.
    # input: data.shape: (channels, T) as torch.Tensor.
    # output: data.shape: (channels, max_length) as torch.Tensor with 0's in the tail.
    data = functional.pad(data, pad=(0,max_length-data.shape[1]), value=0.0)
    return data

def slicing_window(data, W_s):
    data = functional.pad(data, pad=(0,W_s-1), value=0.0)
    return data.unfold(1, W_s, 1)

def fname_sorting_key(file_name):
    # file_name: str in the format of 'UXSY.TXT', 
    # where X and Y are integers.
    # return (X,Y) tuple as key in sorted() function.

    file_name = file_name.split('.')[0] # remove the postfix '.TXT'
    splited_filename = re.split(r'(\d+)',file_name)
    return int(splited_filename[1]), int(splited_filename[3])




if __name__ =='__main__':
    print(fname_sorting_key('U20S1.txt'))
