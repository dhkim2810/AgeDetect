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

from util import create_params
from model import create_model

from collections import OrderedDict

class FacialDataset_test(Dataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        self.imgs = []
        self.transform = transforms.Compose([                      
            transforms.Resize((64,64)),
            transforms.ToTensor()    
        ])
        self.age_path = sorted(glob(os.path.join(data_path, "*.*")))
        for pth in self.age_path:
          img = Image.open(pth)
          self.imgs.append(self.transform(img))
    def __getitem__(self, idx):
        image = self.imgs[idx]
        return image

    def __len__(self):
        return len(self.age_path)

def eval():
    config = create_params()
    test_dataset = FacialDataset_test(config.data_dir+'/test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)

    if not os.path.exists(os.path.join(config.save_dir, config.arch)):
        os.makedirs(os.path.join(config.save_dir, config.arch, 'best'))
        os.makedirs(os.path.join(config.save_dir, config.arch, 'latest'))
        print("Created directory ",str(os.path.join(config.save_dir, config.arch)))
    
    # load checkpoint
    checkpoint = None
    if config.eval:
        checkpoint = torch.load(os.path.join(config.output_dir, config.arch, 'best','model_{}.pt'.format(config.trial)))
        model = create_model(checkpoint['arch'])
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()

    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
    
    print('Make an evaluation csv file for submission...')
    Category = []
    for input in test_loader:
        input = input.cuda()
        output = [elem[0] for elem in model(input).detach().cpu().numpy().tolist()]
        # output = torch.argmax(output, dim=1)
        Category.extend(output)

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
    if config.eval:
        checkpoint = torch.load(os.path.join(config.output_dir, config.arch, 'latest','model_{}.pt'.format(config.trial)))
        model = create_model(checkpoint['arch'])
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()

    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
    
    print('Make an evaluation csv file for submission...')
    Category = []
    for input in test_loader:
        input = input.cuda()
        output = [elem[0] for elem in model(input).detach().cpu().numpy().tolist()]
        # output = torch.argmax(output, dim=1)
        Category.extend(output)

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