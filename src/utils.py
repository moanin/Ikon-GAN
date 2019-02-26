import os
import re

import imageio
import torch.nn as nn


def gan_weights_init(m):
    '''
    initializing weights with normal distribution (Goodfellow 2014)
    :param m: torch model
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def animate(imgs_dir, output_path):
    '''
    create animated gif from all pngs in given directory
    :param imgs_dir: directory with images
    :param output_path: path created file
    '''
    imgs_names = natural_sort([a for a in os.listdir(imgs_dir) if a.endswith('.png')])
    with imageio.get_writer(output_path, mode='I', ) as writer:
        for name in imgs_names:
            img = imageio.imread(os.path.join(imgs_dir, name))
            writer.append_data(img)


def natural_sort(l):
    '''
    natural sorting of given iterable
    :param l: given iterable
    :return: sorted list
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)
