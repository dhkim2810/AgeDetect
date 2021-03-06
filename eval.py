from PIL import Image
import os
from glob import glob
import torch
import pandas as pd
import argparse
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from util import create_params, flip
from model import create_model

from collections import OrderedDict
from tqdm import tqdm
normalize = transforms.Normalize(mean=[0.5754, 0.4529, 0.3986],
                                    std=[0.2715, 0.2423, 0.2354])

class FacialDataset_test(Dataset):
    def __init__(self, data_path, img_size):
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        self.imgs = []
        self.transform = transforms.Compose([                            
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            normalize
        ])
        self.age_path = sorted(glob(os.path.join(data_path, "*.*")))
        for pth in self.age_path:
          img = pth
          self.imgs.append(img)
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.imgs[idx]))
        return image

    def __len__(self):
        return len(self.age_path)

def eval():
    config = create_params()
    test_dataset = FacialDataset_test(config.data_dir+'/test', config.img_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

    if not os.path.exists(os.path.join(config.save_dir, config.arch)):
        os.makedirs(os.path.join(config.save_dir, config.arch, 'best'))
        os.makedirs(os.path.join(config.save_dir, config.arch, 'latest'))
        print("Created directory ",str(os.path.join(config.save_dir, config.arch)))
    
    # load checkpoint
    checkpoint = None
    centroid = None
    if config.eval:
        checkpoint = torch.load(os.path.join(config.output_dir, config.arch, 'best','model_{}.pt'.format(config.trial)))
        if 'centroid' in checkpoint.keys():
            centroid = checkpoint['centroid']
        model = create_model(config)
        if torch.cuda.device_count() > 1:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
    
    print('Make an evaluation csv file(best) for submission...')
    Category = []
    for input in tqdm(test_loader):
        input = input.cuda()
        output = model(input)
        if config.arch == 'random_bin':
            est = (output * centroid.view(1,-1)).view(-1, config.M, config.N)
            y_hat = est.sum(dim=2)
            y_bar = y_hat.mean(dim=1)
            output = [y_bar.item()]
        elif config.arch == 'dldlv2':
            flipped = flip(input).cuda()
            output_flipped = model(flipped)
            output = [torch.sum(output*centroid, dim=1).item()/2 + torch.sum(output_flipped*centroid, dim=1).item()/2]
        else:
            output = [output.item()]
        Category = Category + output

    Id = list(range(0, len(Category)))
    samples = {
       'Id': Id,
       'Category': Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Category'])

    df.to_csv(os.path.join(config.save_dir, config.arch, 'best','submission_best_{}.csv'.format(config.trial)), index=False)
    print('Done!!')

    del model
    checkpoint = None
    centroid = None
    if config.eval:
        checkpoint = torch.load(os.path.join(config.output_dir, config.arch, 'latest','model_{}.pt'.format(config.trial)))
        if 'centroid' in checkpoint.keys():
            centroid = checkpoint['centroid']
        model = create_model(config)
        if torch.cuda.device_count() > 1:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
    
    print('Make an evaluation csv file(latest) for submission...')
    Category = []
    for input in tqdm(test_loader):
        input = input.cuda()
        output = model(input)
        if config.arch == 'random_bin':
            est = (output * centroid.view(1,-1)).view(-1, config.M, config.N)
            y_hat = est.sum(dim=2)
            y_bar = y_hat.mean(dim=1)
            output = [y_bar.item()]
        elif config.arch == 'dldlv2':
            flipped = flip(input).cuda()
            output_flipped = model(flipped)
            output = [torch.sum(output*centroid, dim=1).item()/2 + torch.sum(output_flipped*centroid, dim=1).item()/2]
        else:
            output = [output.item()]
        Category = Category + output

    Id = list(range(0, len(Category)))
    samples = {
       'Id': Id,
       'Category': Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Category'])

    df.to_csv(os.path.join(config.save_dir, config.arch, 'latest','submission_latest_{}.csv'.format(config.trial)), index=False)
    print('Done!!')

if __name__ == "__main__":
    eval()