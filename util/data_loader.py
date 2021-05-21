import os
from glob import glob
from PIL import Image
import numpy as np
import torch as t
import torch.utils.data
import torchvision as tv


class FacialDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        self.imgs = []
        self.labels = []
        self.transform = transform
        self.age_path = sorted(glob(os.path.join(data_path, "*.*")))
        for pth in self.age_path:
            img = Image.open(pth)
            label = int(pth.split('_')[0].split('/')[-1])
            self.labels.append(label)
            if self.train:
                self.imgs.append(self.transform(img))
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        return image , label

    def __len__(self):
        return len(self.age_path)


def load_data(cfg_data, cfg_aug):
    if cfg_data.val_split < 0 or cfg_data.val_split >= 1:
        raise ValueError('val_split should be in the range of [0, 1) but got %.3f' % cfg_data.val_split)

    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    transforms = tv.transforms.Compose([])
    if cfg_aug.data_augmentation:
        transforms.transforms.append(tv.transforms.RandomHorizontalFlip())
        transforms.transforms.append(tv.transforms.RandomCrop(64, 4))
    transforms.transforms.append(tv.transforms.ToTensor())
    transforms.transforms.append(tv_normalize)

    """
    if cfg_aug.cutout:
        transforms.transforms.append(Cutout(n_holes=cfg.n_holes, length=cfg.length, random=True))
    """

    full_dataset = FacialDataset(cfg_data.path, transform=transforms)
    train_size = int(cfg_data.val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    torch.manual_seed(3334)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg_data.batch_size, shuffle=True, num_workers=cfg_data.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,batch_size=cfg_data.batch_size, shuffle=False, num_workers=cfg_data.workers, pin_memory=True)

    return train_loader, val_loader