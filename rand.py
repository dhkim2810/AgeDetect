# %%
import os
import sys
import time
import torch

import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.optim.lr_scheduler as schedule
from tqdm import tqdm as tqdm

from util import *
from model import create_model
import data_loader
from collections import OrderedDict


torch.cuda.manual_seed(3334)
train_loader, val_loader = data_loader.

model = create_model(config).cuda()

# Check number of parameters your model
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {pytorch_total_params}")
if int(pytorch_total_params) > 2000000:
    print('Your model has the number of parameters more than 2 millions..')
    sys.exit()

# %%
# Optimizer
optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=config.learning_rate,
                momentum=config.momentum,
                nesterov=config.nesterov,
                weight_decay=config.wd)

# criterion
criterion = nn.CrossEntropyLoss().cuda()


# %%
for i, (input, target) in enumerate(train_loader):
    print(input.shape)
    print(target.shape)
    break
