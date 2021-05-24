import logging

from .resnet import *
from .spinalresnet import *
from .densenet import *
from .mymodel import *


def create_model(args):
    logger = logging.getLogger()

    model = None
    if args == 'resnet18':
        model = ResNet18(num_classes=1)
    elif args == 'spinalresnet18':
        model = SpinalResNet18(num_classes=1)
    elif args == 'densenet':
        model = densenet(num_class=1)
    elif args == 'mymodel':
        model = mymodel(num_class=1)

    return model