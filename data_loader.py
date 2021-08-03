import os
import torch
import numpy as np
from glob import glob
from PIL import Image, ImageFile
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm
from util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Thumbnail(object):
    def __init__(self, thumb_size, participation_rate):
        self.thumb_size = thumb_size
        self.participation_rate = participation_rate

    def __call__(self, img):
        
        if np.random.rand(1) > self.participation_rate:
            return img
        
        img = img.clone()
        
        h = img.size(1)
        w = img.size(2)
        
        thumbnail = F.interpolate(img, size=self.thumb_size)
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.thumb_size // 2, 0, h)
        y2 = np.clip(y + self.thumb_size // 2, 0, h)
        x1 = np.clip(x - self.thumb_size // 2, 0, w)
        x2 = np.clip(x + self.thumb_size // 2, 0, w)
        
        centre = self.thumb_size//2
        img[:, y1:y2, x1:x2] = thumbnail[:, centre - min(y, centre) : centre + min(h - y, centre), centre - min(x, centre) : centre + min(w - x, centre)]
        return img

class FacialDataset(Dataset):
    def __init__(self, data_path, transform, model, centroid):
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        self.imgs = []
        self.labels = []
        self.transform = transform
        self.model = model
        self.centroid = centroid
        if self.model == 'random_bin':
            self.M = centroid.shape[0]
            self.N = centroid.shape[1]
        self.age_path = sorted(glob(os.path.join(data_path, "*.*")))
        for pth in self.age_path:
            img = pth
            label = int(pth.split('_')[0].split('/')[-1])
            self.labels.append(label)
            self.imgs.append(img)
    def __getitem__(self, idx):
        # Transform image here to boost training speed
        image = self.transform(Image.open(self.imgs[idx]))
        label = None
        age = self.labels[idx]
        if self.model == 'random_bin':
            _, index = torch.min(abs(age-self.centroid),dim=1)
            label = torch.zeros(self.M*self.N, dtype=torch.long)
            for i,item in enumerate(index):
                label[self.N*i+item] = 1
        elif self.model == 'dldlv2':
            label = [normal_sampling(int(age), i) for i in range(120)]
            label = [i if i > 1e-15 else 1e-15 for i in label]
            label = torch.Tensor(label)
        return image, label, age

    def __len__(self):
        return len(self.age_path)


def get_data_loader(config, data_path, batch_size, num_workers,train_val_ratio, centroid=None):
    normalize = transforms.Normalize(mean=[0.5754, 0.4529, 0.3986],
                                    std=[0.2715, 0.2423, 0.2354])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize((config.img_size,config.img_size)))
    if config.da:
        train_transform.transforms.append(transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.RandomGrayscale())
        train_transform.transforms.append(transforms.RandomRotation(20))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    if config.thumbnail:
        train_transform.transforms.append(Thumbnail(48, 0.8))


    val_transform = transforms.Compose([
        transforms.Resize((config.img_size,config.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    full_dataset = FacialDataset(data_path, val_transform, config.arch, centroid)
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset = FacialDataset(data_path, train_transform, config.arch, centroid)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader