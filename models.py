import torch
import torch.nn as nn

class BigModel(nn.Module):
    def __init__(self, small_model):
        super().__init__()
        self.small_model = small_model
        print('inside id:', id(self.small_model))

    def forward(self):
        print('This is the output of BigModel.')
        return


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        print('fucking small models')
        return


small_model = SmallModel()
big_model = BigModel(small_model=small_model)

print('outside id:', id(small_model))

big_model.small_model()
