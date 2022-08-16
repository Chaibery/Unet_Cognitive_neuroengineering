import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class HorseDataset(Dataset):
    def __init__(self, images_dir, masks_dir, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale

        self.ids_img = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids_mask = [splitext(file)[0] for file in listdir(masks_dir) if not file.startswith('.')]

    def pre_process(self, org_img, is_mask):
        width, height = org_img.size
        #newW, newH = int(0.5*width), int(0.5*height)
        newW, newH = 256, 256
        if is_mask:
            org_img = org_img.resize((newW, newH), resample=Image.NEAREST)
        else:
            org_img = org_img.resize((newW, newH), resample=Image.BICUBIC)
        img_preit = np.asarray(org_img)
        if not is_mask:
            if img_preit.ndim == 2:
                img_preit = img_preit[np.newaxis, ...]
            else:
                img_preit = img_preit.transpose((2, 0, 1))
            img_preit = img_preit / 255
        return img_preit

    def __getitem__(self, idx):
        img_name = self.ids_img[idx]
        mask_name = self.ids_mask[idx]
        mask_file = list(self.masks_dir.glob(mask_name + '.png'))
        img_file = list(self.images_dir.glob(img_name + '.png'))

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        img = self.pre_process(img, is_mask=False)
        mask = self.pre_process(mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.ids_img)
