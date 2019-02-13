import torch
import torch.nn as nn
import math


class Generator(nn.Module):
    def __init__(self, img_size, noise_size, ngpu, nc=3, n_filter=64):
        super(Generator, self).__init__()
        assert math.log2(img_size).is_integer()
        self.img_size = img_size
        self.noise_size = noise_size
        self.nc = nc
        self.n_filter = n_filter
        self.ngpu = ngpu
        modules = self.get_modules()
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        return self.model(input)

    def get_modules(self):
        base = int(math.log2(self.img_size)) - 3
        modules = [nn.ConvTranspose2d( self.noise_size, self.n_filter * 2**base, 4, 1, 0, bias=False),
                   nn.BatchNorm2d(self.n_filter * 2**base),
                   nn.ReLU(True)]
        for i in range(1, base + 1):
            base_new = base - i
            modules.append(nn.ConvTranspose2d( self.n_filter * 2**(base_new+1),
                                               self.n_filter * 2**base_new, 4, 2, 1, bias=False))
            modules.append(nn.BatchNorm2d(self.n_filter * 2**base_new))
            modules.append(nn.ReLU(True))

        modules.append(nn.ConvTranspose2d( self.n_filter, self.nc, 4, 2, 1, bias=False))
        modules.append(nn.Tanh())

        return modules
