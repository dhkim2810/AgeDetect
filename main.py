import os
import sys
import time
import torch
import numpy as np
from glob import glob
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm

from util import *
import data_loader

model = 'resnet18' # resnet18, resnet50, resnet101
batch_size = 128  # Input batch size for training (default: 128)
epochs = 100 # Number of epochs to train (default: 100)
learning_rate = 1e-4 # Learning rate
data_augmentation = True # Traditional data augmentation such as augmantation by flipping and cropping?
cutout = True # Apply Cutout?
n_holes = 1 # Number of holes to cut out from image
length = 16 # Length of the holes
seed = 0 # Random seed (default: 0)
num_classes = 1 ##regression
print_freq = 30
num_workers = 1
cuda = torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models
train_val_ratio = 0.9

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

train_loader, val_loader = data_loader.get_data_loader('/content/dataset/train',batch_size,num_workers,train_val_ratio)

def train(train_loader, epoch, model, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    l2 = AverageMeter('L2 Loss', ':.4e')
    l1 = AverageMeter('L1 Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, l2,l1, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    l1_criterion = nn.L1Loss().cuda()
    l2_criterion = nn.MSELoss().cuda()
    end = time.time()
    for i, (input, label) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda()
        label = label.float().flatten().cuda()
        # compute output
        output = model(input).flatten()
        l2_ = l2_criterion(output, label)
        l1_ = l1_criterion(output,label)
        l2.update(l2_.item(), input.size(0))
        l1.update(l1_.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        l2_.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)
    # print('==> Train Accuracy: Loss {losses:.3f} || scores {r2_score:.4e}'.format(losses=losses, r2_score=r2))

def validation(val_loader,epoch, model):
    model.eval()
    l2 = torch.nn.MSELoss().cuda()
    for i,(input,label) in enumerate(val_loader):
        input = input.cuda()
        label = label.float().flatten().cuda()
        output = model(input).flatten()
        l2_ = l2(label,output)
        RMSE = torch.sqrt(l2_)
    print('==> Validate Accuracy:  L2 distance {:.3f} || RMSE {:.3f}'.format(l2_,RMSE))
    return RMSE


###########################################################
model = ResNet18(num_classes=num_classes).cuda()

# Check number of parameters your model
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {pytorch_total_params}")
if int(pytorch_total_params) > 2000000:
    print('Your model has the number of parameters more than 2 millions..')
    sys.exit()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)

best_acc = 1e5
for epoch in range(epochs):
    print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))

    # train for one epoch
    start_time = time.time()
    train(train_loader, epoch, model, optimizer)
    val_acc = validation(val_loader,epoch,model)

    elapsed_time = time.time() - start_time
    print('==> {:.2f} seconds to train this epoch\n'.format(elapsed_time))
    # learning rate scheduling
    scheduler.step()
    
    # Save model for best accuracy
    if best_acc > val_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'model_best.pt')

    torch.save(model.state_dict(),'model_latest.pt')
print(f"Best RMSE Accuracy: {best_acc}")