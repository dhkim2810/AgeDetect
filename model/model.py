import logging

from .resnet import ResNet18
from .spinalresnet import SpinalResNet18
from .densenet import *
from .random_bin import random_bin
from .dldlv2 import dldlv2

def create_model(config):
    logger = logging.getLogger()

    model = None
    if config.arch == 'resnet18':
        model = ResNet18(num_classes=1)
    elif config.arch == 'spinalresnet18':
        model = SpinalResNet18(num_classes=1)
    elif config.arch == 'densenet':
        model = densenet(num_class=1)
    elif config.arch == 'random_bin':
        model = random_bin(M=config.M, N=config.N)
    elif config.arch == 'dldlv2':
        model = dldlv2(config.img_size)

    return model