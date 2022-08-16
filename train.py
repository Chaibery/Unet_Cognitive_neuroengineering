import argparse
from cProfile import label, run
import logging
import sys
from pathlib import Path

import os
from turtle import width
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataload import HorseDataset
from Unet import Unet

def creatdataset(img_dir,mask_dir,val_per,batchsize):#创建训练验证集
    dataset = HorseDataset(img_dir,mask_dir)
    n_val = int(len(dataset) * val_per)
    n_train = len(dataset) - n_val
    #进行训练验证集的划分
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size = batchsize, num_workers = 0, pin_memory = True)
    train_loader = DataLoader(train_set, shuffle = True, **loader_args)
    val_loader = DataLoader(val_set, shuffle = False, **loader_args)
    return train_loader,val_loader

def train(net,device,epoches,lr,train_loader,val_loader,dir_output):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    #记录各项指标用于画图
    TRAIN_ALL_MIOU = []
    TRAIN_ALL_BIOU = []
    TRAIN_ALL_LOSS = []
    VAL_ALL_MIOU = []
    VAL_ALL_BIOU = []
    for epoch in range(1,epoches+1):
        net.train()
        TRAIN_MIOU = []
        TRAIN_BIOU = []
        TRAIN_LOSS = []
        print("epoch",epoch,":")
        print("training......")
        for batch in train_loader:
            images = batch['image']
            real_masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            real_masks = real_masks.to(device=device, dtype=torch.long)
            pre_masks = net(images)
            train_loss = criterion(pre_masks, real_masks)
            train_miou = get_miou(pre_masks, real_masks)
            train_biou = get_biou(pre_masks, real_masks)
            #print("iou:", train_miou, "boundary_iou:", train_biou ,"loss:", train_loss.data)
            TRAIN_LOSS.append(train_loss)
            TRAIN_MIOU.append(train_miou)
            TRAIN_BIOU.append(train_biou)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        TRAIN_ALL_MIOU.append(sum(TRAIN_MIOU) / len(TRAIN_MIOU))
        TRAIN_ALL_BIOU.append(sum(TRAIN_BIOU) / len(TRAIN_BIOU))
        TRAIN_ALL_LOSS.append(sum(TRAIN_LOSS) / len(TRAIN_LOSS))

        net.eval()
        VAL_MIOU = []
        VAL_BIOU = []
        print("epoch",epoch,": testing......")
        with torch.no_grad():
            for batch in val_loader:
                val_images = batch['image']
                val_masks = batch['mask']
                val_images = val_images.to(device=device, dtype=torch.float32)
                val_masks = val_masks.to(device=device, dtype=torch.long)
                val_pres = net(val_images)
                val_miou = get_miou(val_pres, val_masks)
                val_biou = get_biou(val_pres, val_masks)

                VAL_MIOU.append(val_miou)
                VAL_BIOU.append(val_biou)
        scheduler.step(sum(VAL_MIOU) / len(VAL_MIOU))
        VAL_ALL_MIOU.append(sum(VAL_MIOU) / len(VAL_MIOU))
        VAL_ALL_BIOU.append(sum(VAL_BIOU) / len(VAL_BIOU))
        #torch.cuda.empty_cache()
        torch.save(net.state_dict(), os.path.join(dir_output,'epoch{}.pth'.format(epoch)))
    
    return TRAIN_ALL_MIOU,TRAIN_ALL_BIOU,TRAIN_ALL_LOSS,VAL_ALL_MIOU,VAL_ALL_BIOU

def get_miou(pre_pic, real_pic): #计算miou
    miou = 0
    pre_pic = torch.argmax(pre_pic,1)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        union = torch.logical_or(predict,mask).sum()
        inter = ((predict + mask)==2).sum()
        if union < 1e-5:
            return 0
        miou += inter / union
    return miou/batch

def get_boundary(pic,is_mask): #进行图像边界的获取
    if not is_mask:
        pic = torch.argmax(pic,1).cpu().numpy().astype('float64')
    else:
        pic = pic.cpu().numpy()
    batch, width, height = pic.shape
    new_pic = np.zeros([batch, width + 2, height + 2])
    mask_erode = np.zeros([batch, width, height])
    dil = int(round(0.02*np.sqrt(width ** 2 + height ** 2)))
    if dil < 1:
        dil = 1
    for i in range(batch):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(batch):
        pic_erode = cv2.erode(new_pic[j],kernel,iterations=dil)
        mask_erode[j] = pic_erode[1: width + 1, 1: height + 1]
    return torch.from_numpy(pic-mask_erode)

def get_biou(pre_pic ,real_pic): #计算biou指标
    inter = 0
    union = 0
    pre_pic = get_boundary(pre_pic, is_mask=False)
    real_pic = get_boundary(real_pic, is_mask=True)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        inter += ((predict * mask) > 0).sum()
        union += ((predict + mask) > 0).sum()
    if union < 1:
        return 0
    biou = (inter/union)
    return biou

if __name__ == '__main__':
    img_dir = "./trainimage/images"
    mask_dir = "./trainimage/masks"
    output_dir = "./output/"
    #model_dir = "./pre_model/checkpoint_epoch3.pth"
    val_per = 0.15
    batchsize = 5
    lr = 1e-5
    epochs = 40
    trainloader,valloader = creatdataset(img_dir,mask_dir,val_per,batchsize)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(channels=3, classes=2)
    net.to(device=device)
    TRAIN_ALL_MIOU,TRAIN_ALL_BIOU,TRAIN_ALL_LOSS,VAL_ALL_MIOU,VAL_ALL_BIOU = train(net,device,epochs,lr,trainloader,valloader,output_dir)

    plt.figure(dpi=400)
    plt.title("train_M&B_IOU")
    plt.xlabel("epochs")
    plt.ylabel("train_M&B_iou")
    plt.plot(TRAIN_ALL_MIOU,label="train_Miou")
    plt.plot(TRAIN_ALL_BIOU,label="train_Biou")
    plt.legend()
    plt.savefig("./output/train_M&B_IOU")

    plt.figure(dpi=400)
    plt.title("train_loss")
    plt.xlabel("epochs")
    plt.ylabel("train_loss")
    plt.plot(TRAIN_ALL_LOSS,label="train_loss")
    plt.legend()
    plt.savefig("./output/train_loss")

    plt.figure(dpi=400)
    plt.title("val_M&B_IOU")
    plt.xlabel("epochs")
    plt.ylabel("val_M&B_iou")
    plt.plot(VAL_ALL_MIOU,label="val_miou")
    plt.plot(VAL_ALL_BIOU,label="val_biou")
    plt.legend()
    plt.savefig("./output/val_M&B_IOU")


    
