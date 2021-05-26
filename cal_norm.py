import os
import torch
import numpy as np
from glob import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm

class FacialDataset(Dataset):
    def __init__(self, data_path, transforms):
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        self.imgs = []
        self.labels = []
        self.transform = transforms
        self.age_path = sorted(glob(os.path.join(data_path, "*.*")))
        print("Loading Images..")
        for pth in tqdm(self.age_path):
            img = pth
            label = int(pth.split('_')[0].split('/')[-1])
            self.labels.append(label)
            self.imgs.append(img)
        
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.imgs[idx]))
        label = self.labels[idx]
        return image , label

    def __len__(self):
        return len(self.age_path)

def main():
    val_transform = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda")
    dataset = FacialDataset('./dataset/train', val_transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=4, pin_memory=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for i, (data,_) in enumerate(tqdm(loader)):
        data = data[0].squeeze(0)
        if (i == 0): size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= len(loader)
    print(mean)
    mean = mean.unsqueeze(1).unsqueeze(2)

    for i, (data,_) in enumerate(tqdm(loader)):
        data = data[0].squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size

    std /= len(loader)
    std = std.sqrt()
    print(std)

main()