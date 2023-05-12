from collections import OrderedDict
import torch.nn as nn

params = OrderedDict(
    max_length = [793],
    window_size = [10],
    channel = [4],
    f_c = [5],
    depth = [2],
    activation = [nn.ReLU()],
    lr = [.01, .001],
    batch_size = [32, 64],
    epoch = [20]
)