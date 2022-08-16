from cProfile import run
from pathlib import Path

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
from dataload import HorseDataset
from Unet import Unet

#仅供测试模型处理图片效果使用
#未作批量处理
#修改pic_dir来选择图片进行处理
#修改model_dir来选择模型
#修改result_dir更改处理后图像存储路径

if __name__ == '__main__':
    img_dir = "./trainimage/images"
    mask_dir = "./trainimage/masks"
    model_dir = "./output/checkpoint_epoch30.pth"
    pic_dir = "./test_img/001.png"
    result_dir = "./test_result/001.png"
    temp = HorseDataset(img_dir,mask_dir)
    net = Unet(channels=3, classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model_dir, map_location=device))

    img1 = Image.open(pic_dir)
    net.eval()
    img = torch.from_numpy(HorseDataset.pre_process(temp, img1, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        theprob = F.softmax(output, dim=1)[0]
        thetf = transforms.Compose([transforms.ToPILImage(),transforms.Resize((img1.size[1], img1.size[0])),transforms.ToTensor()])
        output_mask = thetf(theprob.cpu()).squeeze()
    pred = F.one_hot(output_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    if pred.ndim == 2:
        aim_pic = Image.fromarray((pred * 255).astype(np.uint8))
    elif pred.ndim == 3:
        aim_pic = Image.fromarray((np.argmax(pred, axis=0) * 255 / pred.shape[0]).astype(np.uint8))
    aim_pic.save(result_dir)

