import os
import torch
import numpy as np
from glob import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FacialDataset(Dataset):
    def __init__(self, data_path, transform, centroid):
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        self.imgs = []
        self.age = []
        self.transform = transform
        self.centroid = centroid
        if centroid is not None:
            self.M = centroid.shape[0]
            self.N = centroid.shape[1]
        self.age_path = sorted(glob(os.path.join(data_path, "*.*")))
        for pth in self.age_path:
            img = pth
            label = int(pth.split('_')[0].split('/')[-1])
            self.age.append(label)
            self.imgs.append(img)
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.imgs[idx]))
        age = self.age[idx]
        if self.centroid is not None:
            _, index = torch.min(abs(age-self.centroid),dim=1)
            label = torch.zeros(self.M*self.N, 1, dtype=torch.long)
            for i,item in enumerate(index):
                label[self.N*i+item] = 1
            return image, label, age
        return image, age

    def __len__(self):
        return len(self.age_path)


def get_data_loader(config, data_path, batch_size, num_workers,train_val_ratio, centroid=None):
    normalize = transforms.Normalize(mean=[0.5754, 0.4529, 0.3986],
                                    std=[0.2715, 0.2423, 0.2354])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize((config.img_size,config.img_size)))
    if config.da:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.RandomGrayscale())
        train_transform.transforms.append(transforms.RandomRotation(20))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    val_transform = transforms.Compose([
        transforms.Resize((config.img_size,config.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    full_dataset = FacialDataset(data_path, val_transform, centroid)
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset = FacialDataset(data_path, train_transform, centroid)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader