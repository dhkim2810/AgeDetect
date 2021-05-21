import numpy as np
import torch
import cv2

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

def saliency_bbox(img, lam, original=True, in_train=False):
    size = img.shape
    W = H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    if in_train:
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
    else:
        temp_img = img
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)
    if not original:
        shift_x = 16 - (bbx2-bbx1)
        if bbx1 == 0:
            x += shift_x
        else:
            x -= shift_x
        shift_y = 16 - (bby2-bby1)
        if bby1 == 0:
            y += shift_y
        else:
            y -= shift_y
        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def saliency_grid(im1, im2, im3, im4, lam=0.75, original=False, in_train=True):
    new_image = np.ones((3, 32, 32))
    for i, img in enumerate([im1, im2, im3, im4]):
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img, lam, original, in_train)
        patch = np.zeros((3, 16, 16))
        patch[:,:bbx2-bbx1,:bby2-bby1] = img[:,bbx1:bbx2,bby1:bby2].cpu()
        if i == 0:
            new_image[:,0:16,0:16] = patch
        elif i == 1:
            new_image[:,16:,0:16] = patch
        elif i == 2:
            new_image[:,0:16,16:] = patch
        elif i == 3:
            new_image[:,16:,16:] = patch
    return new_image