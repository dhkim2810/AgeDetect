import logging

from .resnet import ResNet18
from .spinalresnet import SpinalResNet18
from .densenet import *
from .randbin import *

def create_model(config):
    logger = logging.getLogger()

    model = None
    if config.random_bin:
        if config.arch == 'resnet18':
            model = ResNet18(num_classes=config.M)
        elif config.arch == 'spinalresnet18':
            model = SpinalResNet18(num_classes=config.M)
        elif config.arch == 'densenet':
            model = densenet(num_class=config.M)
    else:
        if config.arch == 'resnet18':
            model = ResNet18(num_classes=1)
        elif config.arch == 'spinalresnet18':
            model = SpinalResNet18(num_classes=1)
        elif config.arch == 'densenet':
            model = densenet(num_class=1)

    return model