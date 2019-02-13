import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from .utils import *
from .Discriminator import Discriminator
from  .Generator import Generator


class GANTrainer:

    def __init__(self, data_dir, img_size, nc,
                 n_filter_G=64,
                 n_filter_D=64,
                 noise_size=100,
                 ngpu=1):

        self.noise_size = noise_size
        self.dataset = dset.ImageFolder(root=data_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.CenterCrop(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5)),
                                            ]))

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.modelG = Generator(img_size, noise_size, ngpu, nc, n_filter_G).to(self.device)
        self.modelD = Discriminator(img_size, ngpu, nc, n_filter_D).to(self.device)

        if (self.device.type == 'cuda') and (ngpu > 1):
            self.modelG = nn.DataParallel(self.modelG, list(range(ngpu)))
            self.modelD = nn.DataParallel(self.modelD, list(range(ngpu)))

        self.modelG.apply(gan_weights_init)
        self.modelD.apply(gan_weights_init)

    def train(self, n_epoch, lr, beta1, batch_size=128, workers=2):

        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 num_workers=workers)

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(64, self.noise_size, 1, 1, device=self.device)

        fake_label = 0
        real_label = 1

        optimizerD = optim.Adam(self.modelD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.modelG.parameters(), lr=lr, betas=(beta1, 0.999))

        # training loop

        last_imgs = None
        G_losses = []
        D_losses = []

        print('Starting Training Loop')
        for epoch in range(n_epoch):
            for i, batch in enumerate(dataloader, 0):
                # (1) Update D network: maximize log(D(data)) + log(1 - D(G(noise)))
                self.modelD.zero_grad()
                real = batch[0].to(self.device)
                b_size = real.size(0)
                label = torch.full((b_size,), real_label, device=self.device)

                output = self.modelD(real).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_data = output.mean().item()

                noise = torch.randn(b_size, self.noise_size, 1, 1, device=self.device)

                fake = self.modelG(noise)
                label.fill_(fake_label)
                output = self.modelD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_noise = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # (2) Update G network: maximize log(D(G(noise)))


