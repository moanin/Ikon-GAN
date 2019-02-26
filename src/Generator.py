import torch
import torch.nn as nn
import math


class Generator(nn.Module):
    def __init__(self, img_size, noise_size, nc=3, n_filter=64):
        '''
        Generator part of GAN, series of transposed convolutions converting noise vector to requested size
        :param img_size: size of output images
        :param noise_size: len of noise vector
        :param nc: number of output channels
        :param n_filter: number of filters in first layer
        '''
        super(Generator, self).__init__()
        assert math.log2(img_size).is_integer()
        self.img_size = img_size
        self.noise_size = noise_size
        self.nc = nc
        self.n_filter = n_filter
        modules = self.get_modules()
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        return self.model(input)

    def get_modules(self):
        base = int(math.log2(self.img_size)) - 3
        modules = [nn.ConvTranspose2d( self.noise_size, self.n_filter * 2**base, 4, 1, 0, bias=False),
                   nn.BatchNorm2d(self.n_filter * 2**base),
                   nn.LeakyReLU(0.2, inplace=True)]
        for i in range(1, base + 1):
            base_new = base - i
            modules.append(nn.ConvTranspose2d( self.n_filter * 2**(base_new+1),
                                               self.n_filter * 2**base_new, 4, 2, 1, bias=False))
            modules.append(nn.BatchNorm2d(self.n_filter * 2**base_new))
            modules.append(nn.Dropout2d(0.5, inplace=True))
            modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.ConvTranspose2d( self.n_filter, self.nc, 4, 2, 1, bias=False))
        modules.append(nn.Tanh())

        return modules
