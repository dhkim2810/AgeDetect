import os
import torch
from glob import glob
from PIL import Image
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
          img = Image.open(pth)
          label = int(pth.split('_')[0].split('/')[-1])
          self.labels.append(label)
          self.imgs.append(self.transform(img))
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        return image , label

    def __len__(self):
        return len(self.age_path)

def get_data_loader(config, data_path, batch_size, num_workers,train_val_ratio):
    normalize = transforms.Normalize(mean=[0.5754, 0.4529, 0.3986],
                                    std=[0.2715, 0.2423, 0.2354])
    transform = transforms.Compose([])
    if config.da:
        transform.transforms.append(transforms.RandomHorizontalFlip())
    # if config.cutout:
        # add cutout
    transform.transforms.append(transforms.Resize((64,64)))
    transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(normalize)

    full_dataset = FacialDataset(data_path, transform)
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
