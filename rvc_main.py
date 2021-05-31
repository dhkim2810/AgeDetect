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
import torch.optim.lr_scheduler as schedule
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm

from rb_util import *
from model import create_model
import data_loader
from collections import OrderedDict

def train(config, train_loader, epoch, model, optimizer, l1_criterion, l2_criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    L1Loss = AverageMeter('L1_Loss', ':.4e')
    L2Loss = AverageMeter('L2_Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, L1Loss,L2Loss, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    model.zero_grad()
    end = time.time()
    for i, (input, label) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda()
        label = label.float().flatten().cuda()
        # compute output
        output = model(input).flatten()
        l1_ = l1_criterion(output, label)
        l2_ = l2_criterion(output, label)
        L1Loss.update(l1_.item(), input.size(0))
        L2Loss.update(l2_.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if config.use_l1_loss:
            l1_.backward()
        else:
            l2_.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)
    return [L1Loss.avg, L2Loss.avg]
    # print('==> Train Accuracy: Loss {losses:.3f} || scores {r2_score:.4e}'.format(losses=losses, r2_score=r2))

def validation(val_loader,epoch, model, criterion):
    model.eval()
    with torch.set_grad_enabled(False):
        loss_, acc, num_ex = 0,0,0
        for i,(input,label) in enumerate(val_loader):
            input = input.cuda()
            label = label.float().flatten().cuda()
            output = model(input).flatten()
            loss = criterion(label,output)
            acc += torch.sqrt(loss)
    acc /= len(val_loader)
    print('==> Validate Accuracy:  RMSE  {:.3f}'.format(acc))
    return acc


def main():
    global config
    config = create_params()
    print(config)

    torch.manual_seed(config.seed)
    if config.cudnn:
        cudnn.benchmark = True
    if config.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    train_loader, val_loader = data_loader.get_data_loader(
                                config,
                                config.data_dir+'/train',
                                config.batch_size,
                                config.workers,
                                config.train_val_ratio)

    start_epoch = 0
    training_loss = []
    validation_loss = []
    if config.resume:
        print("Starting from checkpoint")
        checkpoint = torch.load(config.resume_dir)
        start_epoch = checkpoint['epoch']
        training_loss = checkpoint['train_loss']
        validation_loss = checkpoint['val_loss']
        model = create_model(config)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Model loaded from checkpoint. Starting from epoch ",start_epoch)
    else:
        model = create_model(config)
    available_gpu = torch.cuda.device_count()
    if available_gpu > 0:
        model = torch.nn.DataParallel(model, device_ids=range(available_gpu))

    # Check if checkpoint folder exist
    if not os.path.exists(os.path.join(config.output_dir, config.arch)):
        os.makedirs(os.path.join(config.output_dir, config.arch, 'best'))
        os.makedirs(os.path.join(config.output_dir, config.arch, 'latest'))
        print("Created directory ",str(os.path.join(config.output_dir, config.arch)))

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 2000000:
        print('Your model has the number of parameters more than 2 millions..')
        sys.exit()

    # Optimizer
    optimizer = None
    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(
                        model.parameters(), 
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        nesterov=config.nesterov,
                        weight_decay=config.wd)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=config.learning_rate,
                        betas=(config.beta1, config.beta2),
                        eps=config.eps,
                        weight_decay=config.wd)
    # criterion
    criterion = nn.CrossEntropyLoss()
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()

    # Scheduler
    scheduler = None
    if config.scheduler == 'step':
        scheduler = schedule.StepLR(optimizer, step_size=config.step_size,gamma=config.gamma)
    elif config.scheduler == 'multi_step':
        scheduler = schedule.MultiStepLR(optimizer, milestones=config.milestone, gamma=config.gamma)
    elif config.scheduler == 'exp':
        scheduler = schedule.ExponentialLR(optimizer, gamma=config.gamma)
    elif config.scheduler == 'cos':
        scheduler = schedule.CosineAnnealingLR(optimizer, config.cycle, eta_min=0)
    
    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        l1_criterion = l1_criterion.cuda()
        l2_criterion = l2_criterion.cuda()
        print("Using cuda..")

    best_acc = 1e5
    checkpoint = {}
    for epoch in range(start_epoch, config.epochs):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        train_loss = train(config, train_loader, epoch, model, optimizer,criterion)
        val_acc = validation(val_loader,epoch,model,criterion)
        training_loss.append(train_loss)
        validation_loss.append(val_acc)

        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(elapsed_time))
        # learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        checkpoint = {'epoch':epoch,
                      'arch':config.arch,
                      'config':config,
                      'state_dict':model.state_dict(),
                      'train_loss':training_loss,
                      'val_loss':validation_loss}
        
        # Save model for best accuracy
        if best_acc > val_acc:
            best_acc = val_acc
            torch.save(checkpoint, os.path.join(config.output_dir, config.arch, 'best/model_{}.pt'.format(config.trial)))

        torch.save(checkpoint, os.path.join(config.output_dir, config.arch, 'latest/model_{}.pt'.format(config.trial)))
    print(f"Least Loss : {best_acc}")


if __name__ == '__main__':
    main()