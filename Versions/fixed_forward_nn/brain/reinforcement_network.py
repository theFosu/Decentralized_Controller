import torch
import torch.nn as nn

import numpy as np
import numpy.typing as npt


class ReinforcementLearner(nn.Module):
    """
    Neural network that does not use backpropagation.
    TODO: (later) Architecture is now fixed, make architecture evolve
    """
    def __init__(self, ninput, noutput):
        super(ReinforcementLearner, self).__init__()

        # 3 linear layers, tanh activation
        self.net = nn.Sequential()

        self.net.add_module(name='Lin1', module=nn.Linear(ninput, 25))
        self.net.add_module(name='Act1', module=nn.ReLU())

        self.net.add_module(name='Lin2', module=nn.Linear(25, 50))
        self.net.add_module(name='Act2', module=nn.ReLU())

        self.net.add_module(name='Lin3', module=nn.Linear(50, 25))
        self.net.add_module(name='Act3', module=nn.ReLU())

        self.net.add_module(name='Lin4', module=nn.Linear(25, noutput))

        self.ninput = ninput
        self.noutput = noutput

    def forward(self, input_data):
        return self.net(input_data)

    def make_weights(self, weight_vector) -> None:
        """
        Take in a 1d-array of randomly generated weights and biases and assigns them to the nn.

        Use this function only if backprop is not used, and the best weights are approximated with GAs.
        """

        with torch.no_grad():

            total_length = 0
            weight_tensor = nn.Parameter(torch.tensor(weight_vector), requires_grad=False)

            for _, layer in self.net.named_parameters():

                if hasattr(layer, 'weight'):
                    total_length += len(layer.weight)
                    layer.weight = weight_tensor[:total_length]

                    total_length += len(layer.bias)
                    layer.bias = weight_tensor[:total_length]

    def get_weights(self) -> npt.NDArray[np.float_]:
        with torch.no_grad():

            weight_vector = torch.tensor([])

            for _, layer in self.net.named_modules():
                if hasattr(layer, 'weight'):
                    weight_vector = torch.cat((weight_vector, layer.weight))
                    weight_vector = torch.cat((weight_vector, layer.bias))

        return weight_vector.numpy()
