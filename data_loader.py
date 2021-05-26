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


class Cutout(object):

    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            if self.random:
                length = np.random.randint(1,self.length)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_data_loader(config, data_path, batch_size, num_workers,train_val_ratio):
    normalize = transforms.Normalize(mean=[0.5754, 0.4529, 0.3986],
                                    std=[0.2715, 0.2423, 0.2354])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize((config.img_size,config.img_size)))
    if config.da:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.RandomRotation(15))
        train_transform.transforms.append(transforms.RandomCrop(config.img_size, padding=8))
    # if config.cutout:
    #     train_transform.transforms.append(Cutout(n_holes=config.n_holes, length=config.length, random=True))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    val_transform = transforms.Compose([
        transforms.Resize((config.img_size,config.img_size)),
        transforms.ToTensor(),
        normalize
    ])

    full_dataset = FacialDataset(data_path, val_transform)
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset = FacialDataset(data_path, train_transform)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader