import torch
import torch.nn as nn
import math


class Discriminator(nn.Module):
    def __init__(self, img_size, ngpu, nc=3, n_filter=64):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.nc = nc
        self.n_filter = n_filter
        self.ngpu = ngpu
        modules = self.get_modules()
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        return self.model(input)

    def get_modules(self):
        base = int(math.log2(self.img_size)) - 3
        modules = [nn.Conv2d(self.nc, self.n_filter, 4, 2, 1, bias=False),
                   nn.LeakyReLU(0.2, inplace=True)]

        for i in range(1, base +1):
            modules.append(nn.Conv2d(self.n_filter * 2**(i-1), self.n_filter * 2**i, 4, 2, 1, bias=False))
            modules.append(nn.BatchNorm2d(self.n_filter * 2**i))
            modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.Conv2d(self.n_filter * 2**base, 1, 4, 1, 0, bias=False))

        return modules
