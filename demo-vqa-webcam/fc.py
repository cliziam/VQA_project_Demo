from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from utils import plot_variance


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network"""

    def __init__(
        self, dims, activation=nn.ReLU, relu_init=False, var_analysis=False, name=""
    ):
        super(FCNet, self).__init__()

        self.name = name
        self.var_analysis = var_analysis
        if var_analysis:
            dims += [dims[-1]] * 4
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim)
                    if var_analysis
                    else weight_norm(nn.Linear(in_dim, out_dim), dim=None),
                    activation(),
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(dims[-2], dims[-1])
                if var_analysis
                else weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None),
                activation(),
            )
        )

        self.main = nn.ModuleList(layers)
        if relu_init:
            self.init_weights()

    def init_weights(self):
        for name, p in self.main.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(p.data, nonlinearity="relu")
        return

    def forward(self, x):
        for idx, layer in enumerate(self.main):
            x = layer(x)
            if self.var_analysis and self.training:
                plot_variance(x.cpu(), self.name + " layer " + str(idx))
        return x


if __name__ == "__main__":
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print("============")
    fc2 = FCNet([10, 20])
    print(fc2)
