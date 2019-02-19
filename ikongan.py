import os
import random

import fire

from src.GANTrainer import GANTrainer


def train(data_dir=None,
          img_size=64,
          nc=3,
          n_filter_G=64,
          n_filter_D=64,
          noise_size=100,
          load_path=None,
          ngpu=0,
          n_epoch=5,
          lr=0.0002,
          beta1=0.5,
          batch_size=128,
          workers=2,
          output_path=None,
          save_path=None,
          soft_labels=False):

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    T = GANTrainer(data_dir, img_size, nc, n_filter_G,
                   n_filter_D, noise_size, load_path, ngpu)
    T.train(n_epoch, lr, beta1, batch_size, workers,
            output_path, soft_labels)

    if save_path:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        T.save_models(save_path)


def generate(model_dir, n_img, output_dir):

    T = GANTrainer(load_path=model_dir)
    T.generate(n_img, output_dir)


if __name__ == "__main__":
    fire.Fire()
