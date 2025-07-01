# -*- coding: utf-8 -*-

import os
import torch
import cv2
from utils import mkdir, get_lr, adjust_lr
from test import test_first_stage
import numpy as np
from PIL import Image

def train_first_stage(viz, writer, dataloader, net_1,optimizer_1,base_lr, thin_criterion, thick_criterion,fusion_criterion, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    #set=dataloader.dataset.deep_lst
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        thin_gt = sample[1].to(device)
        thick_gt = sample[1].to(device)
        # zero the parameter gradients
        optimizer_1.zero_grad()
       # optimizer_2.zero_grad()
        #optimizer_1.zero_grad()
        # forward
        thick_pred, thin_pred, fushion = net_1(img)
        #thin_pred=net_2(img)
        #img = cv2.imread(img, 1)
        #dst=abs(255-thin_pred)
        #ret2, thin_pred = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
       # thin_pred=net_2(img)
        viz.img(name="images", img_=img[0, :, :, :])
        viz.img(name="thin labels", img_=thin_gt[0, :, :, :])
        viz.img(name="thick labels", img_=thick_gt[0, :, :, :])
        # viz.img(name="thin prediction", img_=thin_pred[0, :, :, :])
        # viz.img(name="thick prediction", img_=thick_pred[0, :, :, :])
        viz.img(name="fusion prediction", img_=fushion[0, :, :, :])
       # viz.img(name="test", img_=dst[0, :, :, :])
        #loss= thin_criterion(thin_pred, thin_gt) + thick_criterion(thick_pred, thick_gt)  # 可加权
        #loss_1 = thin_criterion(thin_pred, thin_gt) +thick_criterion(thick_pred, thick_gt)+fusion_criterion(fushion,gt)
        loss_1=fusion_criterion(fushion, gt)
        #loss_1 = thin_criterion(fushion, thin_gt)
        #loss_1=thick_criterion(thick_pred, thick_gt)
        #loss_1 =  thick_criterion(thick_pred, thick_gt)+thin_criterion(thin_pred,thin_gt)
        #loss_2= thin_criterion(thin_pred.squeeze(1)[:], thin_gt.squeeze(1)[:])
       #   loss_2 = thin_criterion(dst.squeeze(1)[:], thin_gt.squeeze(1)[:])
        #print(torch.max(thin_pred.squeeze(1)[:]))
        #print(torch.max(thin_gt.squeeze(1)[:]))
        #loss_2 = thin_criterion(thin_pred, thin_gt)
        loss_1.backward()
        #loss_2.backward()
        optimizer_1.step()
        #optimizer_2.step()

        epoch_loss += loss_1.item()
        
        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss_1.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss_1.item()))
        viz.plot("train loss", loss_1.item())
        
        # 写入当前lr
        current_lr = get_lr(optimizer_1)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)
    
    adjust_lr(optimizer_1, base_lr, epoch, num_epochs, power=power)
    #adjust_lr(optimizer_1, base_lr, epoch, num_epochs, power=power)
    
    return net_1


def train_second_stage(viz, writer, dataloader, front_net_thick, front_net_thin, fusion_net, optimizer, base_lr, thick_criterion,criterion, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        thick=sample[2].to(device)
        thin=sample[3].to(device)
        with torch.no_grad():
            x,thick_pred,SVC_pred = front_net_thick(thick)
            x,thin_pred,DVC_pred= front_net_thin(thin)
        # zero the parameter gradients
        optimizer.zero_grad()
        # thick_pred_1=thick_pred.squeeze(1)
        # thin_pred_1=thin_pred.squeeze(1)
        # forward
        # temp_1=[item.cpu().detach()for item in thick_pred]
        # thick_pred_1=torch.stack(temp_1,2).squeeze(1).to(device)
        # temp_2=[item.cpu().detach()for item in thin_pred]
        # thin_pred_1=torch.tensor(torch.stack(temp_2,2)).squeeze(1).to(device)


        #a,b,fusion_pred = fusion_net(img, thick_pred_1 [:, :1, :, :], thin_pred_1[:, 1:2, :, :])
       # a, b, fusion_pred = fusion_net(img,SVC_pred, thin_pred)
        a, b, fusion_pred = fusion_net(img, SVC_pred,DVC_pred)

        #a, b, fusion_pred = fusion_net(img, thick, thin)

        viz.img(name="images", img_=img[0, :, :, :])
        viz.img(name="thick", img_=thick[0, :, :, :])
        viz.img(name="thin", img_=thin[0, :, :, :])
        viz.img(name="SVC_pred", img_=SVC_pred[0:, :, :, :])
        viz.img(name="DVC_pred", img_=DVC_pred[0:, :, :, :])
        #viz.img(name="thick_pred", img_=thick_pred[0:, :, :, :])
        viz.img(name="labels", img_=gt[0, :, :, :])
        viz.img(name="prediction", img_=fusion_pred[0, :, :, :])
        #loss = criterion(fusion_pred, gt)
        loss= criterion(a, gt) +thick_criterion(b, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        viz.plot("train loss", loss.item())
        
        # 写入当前lr
        current_lr = get_lr(optimizer)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)
    
    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)
    
    return fusion_net
