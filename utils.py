import torch.nn.functional as functional

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




if __name__ =='__main__':
    import torch
    test_t = torch.tensor([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    padded_t = padding(test_t, 10)
    print(padded_t)

    sliced_windows = slicing_window(padded_t, 10)
    print(sliced_windows)
    print(sliced_windows.shape)
