from collections import OrderedDict
import torch.nn as nn

params = OrderedDict(
    max_length = [793],
    window_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    channel = [4],
    f_c = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    depth = [1,2,3,4,5,6,7,8],
    activation = [nn.ReLU(), nn.Sigmoid()],
    lr = [.01, .001],
    batch_size = [32, 64, 128],
    epoch = [20, 50, 100, 200, 500, 700, 100]
)