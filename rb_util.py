import math
import torch
import random
import numpy as np
import argparse

def create_params():
    parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
    # Environmnet
    parser.add_argument('--name', default='AgeDetect', type=str,
                        help='Name of the project')
    parser.add_argument('--output_dir', default='/root/volume/AgeDetect/Result', type=str,
                        help='Output directory')
    parser.add_argument('--use_gpu', action='store_true',
                        help='toggle gpu')
    parser.add_argument('--cudnn', action='store_true',
                        help='toggle cudnn.benchmark')
    parser.add_argument('--seed', default=0,type=int)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini-batch size(default: 128)')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes(default: 1)')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train_val_ratio', default=0.9, type=float,
                        help='train data split ratio for validation')
    parser.add_argument('--save_dir', default='/root/volume/AgeDetect/Eval', type=str,
                        help='Saving directory')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir',  default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', default=30, type=int,
                        help='print frequency (default: 30)')
    parser.add_argument('--trial', default=1,type=int)
    # Model
    parser.add_argument('--arch', default='resent18',type=str,
                        choices=['resnet18','spinalresnet18','densenet','mymodel'])
    parser.add_argument('--random_bin', action='store_true')
    parser.add_argument('--m', default=30,type=int)
    parser.add_argument('--n', default=10,type=int)
    # Data Augmentation
    parser.add_argument('--data_dir',default='/root/volume/AgeDetect/dataset', type=str)
    parser.add_argument('--img_size',default=100,type=int)
    parser.add_argument('--da', action='store_true',
                        help='Traditional data augmentation such as flipping')
    # Training/Evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epochs', default=150, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--optim', default='sgd',type=str,
                        choices=['sgd','adam'])
    parser.add_argument('--learning_rate',default=2e-4,type=float)
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--nesterov',action='store_true')
    parser.add_argument('--wd',default=5e-4,type=float)
    parser.add_argument('--eps',default=1e-8,type=float)
    parser.add_argument('--beta1',default=0.9,type=float)
    parser.add_argument('--beta2',default=0.999,type=float)

    parser.add_argument('--scheduler',default='fixed',type=str,
                        choices=['fixed','step','multi_step','exp','cos'])
    parser.add_argument('--step_size',default=30,type=int)
    parser.add_argument('--milestone',default=[60,90,120],nargs='+',type=int)
    parser.add_argument('--gamma',default=0.2,type=float)
    parser.add_argument('--cycle',default=0.95,type=float)
    
    return parser.parse_args()


class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        # faster topk (ref: https://github.com/pytorch/pytorch/issues/22812)
        _, idx = output.sort(descending=True)
        pred = idx[:,:maxk]
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def random_discretization(labels, c, N, M, val_labels=None):
    centroids = torch.zeros(N, M)
    class_labels = torch.zeros(1, 1, N*M, len(labels))
    val_class_labels = None

    for m in range(M):
        cm = torch.randint(1,c,(N,))
        cm = cm.sort().values
        centroids[:,m] = cm
        for i in range(len(labels)):
            d = abs(labels[i]-cm)
            argmin = min(d)
            class_labels[:,:,m*N+argmin,i] = i

    if val_labels is not None:
        val_class_labels = torch.zeros(1,1,N*M,len(val_labels))
        for m in range(M):
            cm = centroids[:,m]
            for i in range(len(val_labels)):
                d = abs(val_labels[i]-cm)
                argmin = min(d)
                val_class_labels[1,1,m*N+argmin,i] = 1
    
    return centroids, class_labels, val_class_labels

#smoothedCrossEntropyLayer

