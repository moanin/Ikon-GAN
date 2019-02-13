import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from src.Discriminator import Discriminator
from src.Generator import Generator


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

        with torch.no_grad():
            fake = self.modelG(fixed_noise).detach().cpu()
        last_imgs = vutils.make_grid(fake, padding=2, normalize=True)
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
                # D_G_noise1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # (2) Update G network: maximize log(D(G(noise)))
                self.modelG.zero_grad()
                label.fill_(real_label)

                output = self.modelD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_noise2 = output.mean().item()
                optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if i == len(dataloader)-1:
                    print(f"epoch: {epoch}/{n_epoch}; lossD: {errD.item()}; lossG: {errG.item()} \n "
                          f"Scores: D(real): {D_data}; D(G(noise)): {D_G_noise2}")

                    if epoch % 10 == 0:
                        with torch.no_grad():
                            fake = self.modelG(fixed_noise).detach().cpu()
                        last_imgs = vutils.make_grid(fake, padding=2, normalize=True)

                    visualize_training(G_losses, D_losses, last_imgs)


def gan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def visualize_training(lossG, lossD, imgs):
    plt.clf()
    plt.subplot(211)
    plt.plot(lossG, label="G")
    plt.plot(lossD, label="D")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(212)
    plt.axis("off")
    plt.title('current generations')
    plt.imshow(np.transpose(imgs,(1,2,0)))

    display.clear_output(wait=True)
    display.display(plt.gcf())


# if __name__ == '__main__':
#     T = GANTrainer('../data/', 64, 3)
#     T.train(5, 0.0002, 0.5)
