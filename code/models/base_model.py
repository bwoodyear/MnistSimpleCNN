import torch
import torch.nn as nn
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
    def __init__(self, kernel_size: int = 3, label_level: int = None):
        super(Model, self).__init__()

        self.label_level = label_level
        self.linear = None
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

        # Ensure that labels are being added at a valid level
        if self.label_level:
            assert self.label_level in set(range(1, len(layer_params)))

        for n, (i, o) in enumerate(pairwise(layer_params)):

            # Will have one additional input layer if labels are being included
            if n+1 == self.label_level:
                i += 1

            if n:
                # Follow each conv after first with a batch norm and a relu
                bn = nn.BatchNorm2d(i)
                # setattr(self, f'bn{n+1}', bn)
                layers.append(bn)

                relu = nn.ReLU()
                # setattr(self, f'relu{n+1}', bn)
                layers.append(relu)

            # Add each conv layer with the given input and output size
            conv = nn.Conv2d(i, o, kernel_size, bias=False)
            # setattr(self, f'conv{n+1}', conv)
            layers.append(conv)

        # Add the final linear layer
        layers.append(nn.Flatten())
        
        self.linear = nn.Linear(fc_value, 10, bias=False)
        layers.append(self.linear)
        layers.append(nn.BatchNorm1d(10))

        return layers

    def forward(self, x, labels: torch.tensor = None):
        x = (x - 0.5) * 2.0
        # Iterate through the layers, injecting labels if specified
        for n, l in enumerate(self.net):

            # If at the level where labels are being injected project these to the correct dimension then insert
            if n+1 == self.label_level:
                label_tensor = labels[:, None, None].expand(-1, x.shape[-2], x.shape[-1])
                x = torch.cat((x, label_tensor.unsqueeze(1)), dim=1)
            x = l(x)
        return nn.functional.log_softmax(x, dim=1)
