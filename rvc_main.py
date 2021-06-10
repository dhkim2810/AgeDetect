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

def train(config, train_loader, epoch, model, optimizer, centroid):
    batch_time = AverageMeter('Time', ':6.3f')
    l1 = AverageMeter('L2 Loss', ':.4e')
    kl = AverageMeter('KL Loss', ':.4e')
    # l1 = AverageMeter('L1 Loss', ':.4e')
    # loss = AverageMeter('CE Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, l1, kl, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    kl_criterion = kl_loss
    l1_criterion = L1_loss
    end = time.time()
    for i, (input, label, age) in enumerate(train_loader):
        # measure data loading time
        age = age.float().cuda()
        input = input.cuda()
        label = label.cuda()
        # compute output
        output = model(input)
        
        if config.arch == 'random_bin':
            est = (output * centroid.view(1,-1)).view(-1, config.M, config.N)
            y_hat = est.sum(dim=2)
            ages = y_hat.mean(dim=1)
        elif config.arch == 'dldlv2':
            ages = torch.sum(output*centroid, dim=1)
        
        l1_ = l1_criterion(ages, age)
        kl_ = kl_criterion(output, label)
        
        l1.update(l1_.item(), input.size(0))
        kl.update(kl_.item(), input.size(0))

        total_loss = l1_ + kl_

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)
    return l1.avg, kl.avg
    # print('==> Train Accuracy: Loss {losses:.3f} || scores {r2_score:.4e}'.format(losses=losses, r2_score=r2))

def validation(config, val_loader, model, centroid):
    model.eval()
    with torch.no_grad():
        l2 = nn.MSELoss().cuda()
        mse, num_samples = 0, 0
        for i,(input,label,age) in enumerate(val_loader):
            flipped = flip(input).cuda()
            input = input.cuda()
            age = age.cuda()
            label = label.squeeze().cuda()

            # compute output
            output = model(input)
            output_flipped = model(flipped)

            if config.arch == 'random_bin':
                est = (output * centroid.view(1,-1)).view(-1, config.M, config.N)
                y_hat = est.sum(dim=2)
                ages = y_hat.mean(dim=1)
                est_flip = (output_flipped * centroid.view(1,-1)).view(-1, config.M, config.N)
                y_hat_flip = est_flip.sum(dim=2)
                ages_flip = y_hat_flip.mean(dim=1)
            elif config.arch == 'dldlv2':
                ages = torch.sum(output*centroid, dim=1)
                ages_flip = torch.sum(output_flipped*centroid, dim=1)

            ages = ages/2 + ages_flip/2
            l2_ = l2(age, ages)
            mse += torch.sum((ages-age)**2)
            num_samples += label.size(0)
        mse = mse.float() / num_samples
        RMSE = torch.sqrt(mse)
    print('==> Validate Accuracy:  L2 distance {:.3f} || RMSE {:.3f}'.format(l2_,RMSE))
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
    centroid = None
    if config.arch == 'random_bin':
        centroid, _ = torch.sort(torch.randint(1,120,(config.M,config.N)),dim=1)
    elif config.arch == 'dldlv2':
        centroid = torch.Tensor([i for i in range(120)]).cuda()
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
        centroid = centroid.cuda()
        print("Using cuda..")

    best_acc = 1e5
    checkpoint = {}
    for epoch in range(start_epoch, config.epochs):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        train_loss = train(config, train_loader, epoch, model, optimizer, centroid)
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