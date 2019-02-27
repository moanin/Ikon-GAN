import json
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

from src.Discriminator import Discriminator
from src.Generator import Generator
from src.utils import gan_weights_init

MANUAL_SEED = 999


class GANTrainer:

    def __init__(self, data_dir=None,
                 img_size=64,
                 nc=3,
                 n_filter_G=64,
                 n_filter_D=64,
                 noise_size=100,
                 load_path=None,
                 ngpu=0):

        '''
        Class to manage all the processes regarding training and testing GANs. It initializes with new networks
        with given parameters or loads previously trained models from given loading path
        :param data_dir: directory with training data; structure data_dir/dataset_folder/*.jpg
        :param img_size: net param, size of generated image
        :param nc: net param, number of channels in inputs (and outputs for that mater)
        :param n_filter_G: net param, number of filters ini Generator
        :param n_filter_D: net param, number of filters ini Discriminator
        :param noise_size: net param, len of input noise vector for Generator
        :param load_path: path to previously saved experiment
        :param ngpu: number of gpu to use; cpu is used if 0
        '''

        random.seed(MANUAL_SEED)
        torch.manual_seed(MANUAL_SEED)

        self.data_dir = data_dir
        self.img_size = img_size
        self.nc = nc
        self.n_filter_G = n_filter_G
        self.n_filter_D = n_filter_D
        self.noise_size = noise_size
        self.load_path = load_path

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        print(f'using: {self.device}')

        if load_path:
            self.load_models()
        else:
            self.modelG = Generator(img_size, noise_size, nc, n_filter_G).to(self.device)
            self.modelD = Discriminator(img_size, nc, n_filter_D).to(self.device)
            self.modelG.apply(gan_weights_init)
            self.modelD.apply(gan_weights_init)

        if (self.device.type == 'cuda') and (ngpu > 1):
            self.modelG = nn.DataParallel(self.modelG, list(range(ngpu)))
            self.modelD = nn.DataParallel(self.modelD, list(range(ngpu)))

    def load_models(self):

        with open(os.path.join(self.load_path, 'model_params.json'), 'r') as f:
            params = json.load(f)
            f.close()

        self.img_size = params['img_size']
        self.nc = params['nc']
        self.n_filter_G = params['n_filter_G']
        self.n_filter_D = params['n_filter_D']
        self.noise_size = params['noise_size']

        self.modelG = torch.load(os.path.join(self.load_path, 'generator.pth')).to(self.device)
        self.modelD = torch.load(os.path.join(self.load_path, 'discriminator.pth')).to(self.device)

    def train(self, n_epoch, lr, beta1, batch_size=128, workers=2,
              output_path=None, soft_labels=False):

        self.dataset = dset.ImageFolder(root=self.data_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(self.img_size),
                                            transforms.CenterCrop(self.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5)),
                                        ]))

        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 num_workers=workers)

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(64, self.noise_size, 1, 1, device=self.device)

        fake_label = 0
        real_label = 1

        optimizerD = optim.Adam(self.modelD.parameters(), lr=lr, betas=(beta1, 0.999))
        # optimizerD = optim.SGD(self.modelD.parameters(), lr=lr)
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
                if soft_labels:
                    label -= -0.3 * torch.rand((b_size,), device=self.device) + 0.3

                output = self.modelD(real).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_data = output.mean().item()

                noise = torch.randn(b_size, self.noise_size, 1, 1, device=self.device)

                fake = self.modelG(noise)
                label.fill_(fake_label)
                if soft_labels:
                    label += -0.3 * torch.rand((b_size,), device=self.device) + 0.3
                output = self.modelD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                # D_G_noise1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # (2) Update G network: maximize log(D(G(noise)))
                self.modelG.zero_grad()
                label.fill_(real_label)
                if soft_labels:
                    label -= -0.3 * torch.rand((b_size,), device=self.device) + 0.3

                output = self.modelD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_noise2 = output.mean().item()
                optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if i == len(dataloader) - 1:
                    print(f"epoch: {epoch}/{n_epoch}; lossD: {round(errD.item(), 3)}; lossG: {round(errG.item(), 3)} "
                          f"Scores: D(real): {round(D_data, 3)}; D(G(noise)): {round(D_G_noise2, 3)}")

                    if output_path:
                        if epoch % 10 == 0:
                            with torch.no_grad():
                                fake = self.modelG(fixed_noise).detach().cpu()
                            last_imgs = vutils.make_grid(fake, padding=2, normalize=True).numpy()
                            last_imgs = np.moveaxis(last_imgs, 0, -1)
                            last_imgs = (last_imgs - last_imgs.min()) * (1 / (last_imgs.max() - last_imgs.min()) * 1)
                            plt.imsave(os.path.join(output_path, f'{epoch}_{i}'), last_imgs)

                    # visualize_training(G_losses, D_losses, last_imgs)
        print('Finished training')

    def save_models(self, models_dir):
        params = {'img_size': self.img_size,
                  'nc': self.nc,
                  'n_filter_G': self.n_filter_G,
                  'n_filter_D': self.n_filter_D,
                  'noise_size': self.noise_size}
        with open(os.path.join(models_dir, 'model_params.json'), 'w') as f:
            json.dump(params, f)
            f.close()
        torch.save(self.modelG, os.path.join(models_dir, 'generator.pth'))
        torch.save(self.modelD, os.path.join(models_dir, 'discriminator.pth'))
        print(f'saved  models in : {models_dir}')

    def generate(self, n_img, output_dir):
        self.modelG.eval()
        for i in range(n_img):
            noise = torch.randn(1, self.noise_size, 1, 1, device=self.device)
            with torch.no_grad():
                img = self.modelG(noise)[0].numpy()
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) * (1 / (img.max() - img.min()) * 1)
            plt.imsave(os.path.join(output_dir, '{:03}'.format(i)), img)

