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

from util import *
from model import create_model
import data_loader
from collections import OrderedDict

def train(config, train_loader, epoch, model, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    # l2 = AverageMeter('L2 Loss', ':.4e')
    # l1 = AverageMeter('L1 Loss', ':.4e')
    loss = AverageMeter('CE Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, loss, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()    
    end = time.time()
    for i, (input, label, age) in enumerate(train_loader):
        # measure data loading time
        age = age.float().cuda()
        input = input.cuda()
        label = label.cuda().squeeze()
        # compute output
        output = model(input)

        Loss = (label*-torch.log(output)).sum()
        loss.update(Loss.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)
    return loss.avg
    # print('==> Train Accuracy: Loss {losses:.3f} || scores {r2_score:.4e}'.format(losses=losses, r2_score=r2))

def validation(config, val_loader, model, centroid):
    model.eval()
    with torch.no_grad():
        mse, num_samples = 0, 0
        for i,(input,label,age) in enumerate(val_loader):
            input = input.cuda()
            label = label.squeeze().cuda()

            # compute output
            output = model(input) # BS * (N*M)
            est = (output.detach().cpu() * centroid.view(1,-1)).view(-1, config.M, config.N)
            y_hat = est.sum(dim=2)
            y_bar = y_hat.mean(dim=1)

            mse += cal_loss(age, y_hat, y_bar)
            num_samples += 1

        mse = mse / num_samples
        RMSE = torch.sqrt(mse)
    print('==> Validate Accuracy:  RMSE {:.3f}'.format(RMSE))
    return RMSE


def main():
    global config
    config = create_params()
    print(config)

    if config.cudnn:
        cudnn.benchmark = True
    if config.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    
    start_epoch = 0
    training_loss = []
    validation_loss = []
    centroid, _ = torch.sort(torch.randint(1,120,(config.M,config.N)),dim=1)
    if config.resume:
        print("Starting from checkpoint")
        checkpoint = torch.load(config.resume_dir)
        start_epoch = checkpoint['epoch']
        training_loss = checkpoint['train_loss']
        validation_loss = checkpoint['val_loss']
        centroid = checkpoint['centroid']
        model = create_model(config)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Model loaded from checkpoint. Starting from epoch ",start_epoch)
    else:
        model = create_model(config)

    # Load Data
    train_loader, val_loader = data_loader.get_data_loader(
                                config,
                                config.data_dir+'/train',
                                config.batch_size,
                                config.workers,
                                config.train_val_ratio,
                                centroid)

    # Utilize GPU
    available_gpu = torch.cuda.device_count()
    if available_gpu > 1:
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
        print("Using cuda..")

    best_acc = 1e5
    checkpoint = {}
    for epoch in range(start_epoch, config.epochs):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        train_loss = train(config, train_loader, epoch, model, optimizer)
        val_acc = validation(config, val_loader,model,centroid)
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
                      'val_loss':validation_loss,
                      'centroid':centroid}
        
        # Save model for best accuracy
        if best_acc > val_acc:
            best_acc = val_acc
            torch.save(checkpoint, os.path.join(config.output_dir, config.arch, 'best/model_{}.pt'.format(config.trial)))

        torch.save(checkpoint, os.path.join(config.output_dir, config.arch, 'latest/model_{}.pt'.format(config.trial)))
    print(f"Least Loss : {best_acc}")


if __name__ == '__main__':
    main()