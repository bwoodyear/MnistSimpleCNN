import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, base_model):
        super(Mask, self).__init__()

        self.base_model = base_model

        # Get the number of inputs and outputs of the final linear layer
        in_features = base_model.linear.in_features
        out_features = base_model.linear.out_features
        self.mask_size = in_features * out_features

        self.net = nn.Sequential(nn.Linear(1, 10),
                                 nn.Linear(10, 100),
                                 nn.Linear(100, self.mask_size),
                                 nn.Sigmoid())

    def forward(self, x, binary=True):
        x = self.net(x)
        if binary:
            return torch.round(x)
        else:
            return x







