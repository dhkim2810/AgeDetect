import logging

from .resnet import *
from .spinalresnet import *


def create_model(args):
    logger = logging.getLogger()

    model = None
    if args == 'resnet18':
        model = ResNet18(num_classes=1)
    elif args == 'spinalresnet18':
        model = SpinalResNet18(num_classes=1)

    return model