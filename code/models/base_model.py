import torch
import torch.nn as nn
import numpy as np
from itertools import tee


# Define pairwise since not using python 3.10
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# def init(module, weight_init, bias_init, gain=1):
#     weight_init(module.weight.data, gain=gain)
#     bias_init(module.bias.data)
#     return module
#
#
# init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                        constant_(x, 0))
#
#
# init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                             constant_(x, 0), nn.init.calculate_gain('relu'))
#
#
# init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                             constant_(x, 0), np.sqrt(2))


class Model(nn.Module):
    def __init__(self, kernel_size: int = 3, label_layer: int = None):
        super(Model, self).__init__()

        self.label_layer = label_layer
        self.net = self.setup_net(kernel_size)

    def setup_net(self, kernel_size: int):

        layers = nn.ModuleList()

        # Find the layer parameters corresponding to each possible kernel size
        if kernel_size == 3:
            layer_params = [1, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176]
            fc_value = 8 * 8 * layer_params[-1]
        elif kernel_size == 5:
            layer_params = [1, 32, 64, 96, 128, 160]
            fc_value = 8 * 8 * layer_params[-1]
        elif kernel_size == 7:
            layer_params = [1, 48, 96, 144, 192]
            fc_value = 4 * 4 * layer_params[-1]
        else:
            raise NotImplementedError(f'This kernel size: {kernel_size} is not implemented.')

        for n, in_out in enumerate(pairwise(layer_params)):
            i, o = in_out

            # Will have one additional input layer if labels are being included
            if n+1 == self.label_layer:
                i += 1

            # Add each conv layer with the given input and output size
            layers.append(nn.Conv2d(i, o, kernel_size, bias=False))
            # Follow each conv with a batchnorm and a relu
            layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU())

        # Add the final linear layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(fc_value, 10, bias=False))
        layers.append(nn.BatchNorm1d(10))

        return layers

    def forward(self, x, label_value: int = None, device: str = None):
        # Iterate through the layers, injecting labels if specified
        for n, l in enumerate(self.net):
            if n+1 == self.label_layer:
                label_tensor = torch.full(x.shape[:3], label_value).to(device)
                print(x.shape)
                x = torch.cat((x, label_tensor), dim=-1)
                print(x.shape)
            x = l(x)
        return nn.functional.log_softmax(x, dim=1)
