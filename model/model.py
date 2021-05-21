import logging

from .resnet import *


def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.dataloader.dataset == 'faceage':
        # ResNet
        if args.arch == 'resnet18':
            model = ResNet18(num_classes=1)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    logger.info(msg)

    return model