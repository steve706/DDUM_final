# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fundus_dataset import recompone_overlap
from utils import mkdir
from evaluation import *


def get_results(loss_lst, auc_lst, acc_lst, sen_lst, fdr_lst, spe_lst, kappa_lst, gmean_lst, iou_lst, dice_lst,bacc_lst, dataloader, results_dir, criterion, pred, gt, mask_arr=None, isSave=True):
    if criterion is not None:
        loss_lst.append(criterion(pred, gt).item())
    
    pred_arr = pred.squeeze().cpu().numpy()
    gt_arr = gt.squeeze().cpu().numpy()
    #auc_lst.append(calc_auc(pred_arr, gt_arr, mask_arr=mask_arr))
    #pred_arr[pred_arr > 0] = 1
    pred_img = np.array(pred_arr * 255, np.uint8)
    gt_img = np.array(gt_arr * 255, np.uint8)

    #thresh_pred_img=pred_img

    #thresh_pred_img=pred_img
    thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY )


    print("shape of prediction", thresh_pred_img.shape)
    print("shape of groundtruth", gt_img.shape)
    auc_lst.append(calc_auc(pred_arr, gt_arr, mask_arr=mask_arr))
    acc_lst.append(calc_acc(thresh_pred_img / 255.0, gt_img / 255.0))
    bacc_lst.append(calc_bacc(thresh_pred_img / 255.0, gt_img / 255.0))
    sen_lst.append(calc_sen(thresh_pred_img / 255.0, gt_img / 255.0))
    fdr_lst.append(calc_fdr(thresh_pred_img / 255.0, gt_img / 255.0))
    spe_lst.append(calc_spe(thresh_pred_img / 255.0, gt_img / 255.0))
    kappa_lst.append(calc_kappa(thresh_pred_img / 255.0, gt_img / 255.0))
    gmean_lst.append(calc_gmean(thresh_pred_img / 255.0, gt_img / 255.0))
    iou_lst.append(calc_iou(thresh_pred_img / 255.0, gt_img / 255.0))
    dice_lst.append(calc_dice(thresh_pred_img / 255.0, gt_img / 255.0))
    
    # Save Results
    imgName = dataloader.dataset.getFileName()
    if isSave:
        mkdir(results_dir + "/Thresh")
        mkdir(results_dir + "/noThresh")
        cv2.imwrite(results_dir + "/noThresh//" + imgName, pred_img)
        cv2.imwrite(results_dir + "/Thresh//" + imgName, thresh_pred_img)


def print_results(loss_lst, auc_lst, acc_lst, sen_lst, fdr_lst, spe_lst, kappa_lst, gmean_lst, iou_lst, dice_lst,bacc_lst):
    loss_arr = np.array(loss_lst)
    auc_arr = np.array(auc_lst)
    acc_arr = np.array(acc_lst)
    #bacc_arr = np.array(bacc_lst)
    sen_arr = np.array(sen_lst)
    fdr_arr = np.array(fdr_lst)
    spe_arr = np.array(spe_lst)
    kappa_arr = np.array(kappa_lst)
    gmean_arr = np.array(gmean_lst)
    iou_arr = np.array(iou_lst)
    dice_arr = np.array(dice_lst)
    bacc_arr = np.array(bacc_lst)
    
    print("Loss - mean: " + str(loss_arr.mean()) + "\tstd: " + str(loss_arr.std()))
    print("AUC - mean: " + str(auc_arr.mean()) + "\tstd: " + str(auc_arr.std()))
    print("ACC - mean: " + str(acc_arr.mean()) + "\tstd: " + str(acc_arr.std()))
    print("BACC - mean: " + str(bacc_arr.mean()) + "\tstd: " + str(bacc_arr.std()))
    print("SEN - mean: " + str(sen_arr.mean()) + "\tstd: " + str(sen_arr.std()))
    print("FDR - mean: " + str(fdr_arr.mean()) + "\tstd: " + str(fdr_arr.std()))
    print("SPE - mean: " + str(spe_arr.mean()) + "\tstd: " + str(spe_arr.std()))
    print("Kappa - mean: " + str(kappa_arr.mean()) + "\tstd: " + str(kappa_arr.std()))
    print("G-mean - mean: " + str(gmean_arr.mean()) + "\tstd: " + str(gmean_arr.std()))
    print("IOU - mean: " + str(iou_arr.mean()) + "\tstd: " + str(iou_arr.std()))
    print("Dice - mean: " + str(dice_arr.mean()) + "\tstd: " + str(dice_arr.std()))
    print("bacc - mean: " + str(bacc_arr.mean()) + "\tstd: " + str(bacc_arr.std()))
    
    return loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, kappa_arr, gmean_arr, iou_arr, dice_arr,bacc_arr


def test_first_stage(dataloader, net_1,device, results_dir,
                     thin_criterion=None, thick_criterion=None, fusion_criterion=None, isSave=True):
    loss_dct = {"thin": [], "thick": [], "fusion": []}  # Loss
    auc_dct = {"thin": [], "thick": [], "fusion": []}  # AUC
    acc_dct = {"thin": [], "thick": [], "fusion": []}  # Accuracy
    bacc_dct = {"thin": [], "thick": [], "fusion": []}  # Accuracy
    sen_dct = {"thin": [], "thick": [], "fusion": []}  # Sensitivity (Recall)
    fdr_dct = {"thin": [], "thick": [], "fusion": []}  # FDR
    spe_dct = {"thin": [], "thick": [], "fusion": []}  # Specificity
    kappa_dct = {"thin": [], "thick": [], "fusion": []}  # Kappa
    gmean_dct = {"thin": [], "thick": [], "fusion": []}  # G-mean
    iou_dct = {"thin": [], "thick": [], "fusion": []}  # IOU
    dice_dct = {"thin": [], "thick": [], "fusion": []}# Dice Coefficient (F1-score)
    bacc_dct = {"thin": [], "thick": [], "fusion": []}
    
    i = 1
    with torch.no_grad():
        for sample in dataloader:
            if len(sample) != 5 and len(sample) != 4 and len(sample) != 2:
                print("Error occured in sample %03d, skip" % i)
                continue
            
            print("Evaluate %03d..." % i)
            i += 1
            
            img = sample[0].to(device)
            gt = sample[1].to(device)
            thin_gt = sample[1].to(device)
            thick_gt = sample[1].to(device)
            thick_pred, thin_pred, pred = net_1(img)
            #thin_pred=net_2(img)
            if len(sample) == 5:
                w, h = sample[4]
                
                pred = pred.cpu().squeeze(0)
                pred = transforms.ToTensor()(transforms.ToPILImage()(pred))
                pred = pred.unsqueeze(0).to(device)
                
                mask = sample[2].to(device)
                mask_arr = mask.squeeze().cpu().numpy()
            else:
                mask_arr = None
            
            get_results(loss_dct["thin"], auc_dct["thin"], acc_dct["thin"], sen_dct["thin"],
                        fdr_dct["thin"], spe_dct["thin"], kappa_dct["thin"], gmean_dct["thin"],
                        iou_dct["thin"], dice_dct["thin"], bacc_dct["thin"],dataloader, results_dir + "/thin",
                        thin_criterion, thin_pred, thin_gt, mask_arr=mask_arr,isSave=isSave)
            get_results(loss_dct["thick"], auc_dct["thick"], acc_dct["thick"], sen_dct["thick"],
                        fdr_dct["thick"], spe_dct["thick"], kappa_dct["thick"], gmean_dct["thick"],
                        iou_dct["thick"], dice_dct["thick"], bacc_dct["thick"],dataloader, results_dir + "/thick",
                        thick_criterion, thick_pred, thick_gt, mask_arr=mask_arr,isSave=isSave)
            get_results(loss_dct["fusion"], auc_dct["fusion"], acc_dct["fusion"], sen_dct["fusion"],
                        fdr_dct["fusion"], spe_dct["fusion"], kappa_dct["fusion"],
                        gmean_dct["fusion"], iou_dct["fusion"], dice_dct["fusion"],bacc_dct["fusion"],
                        dataloader, results_dir + "/fusion", fusion_criterion, pred, gt, mask_arr=mask_arr,isSave=isSave)
    
    
    loss_dct["thin"], auc_dct["thin"], acc_dct["thin"], sen_dct["thin"], \
    fdr_dct["thin"], spe_dct["thin"], kappa_dct["thin"], gmean_dct["thin"], \
    iou_dct["thin"], dice_dct["thin"],bacc_dct["thin"] = print_results(loss_dct["thin"], auc_dct["thin"], acc_dct["thin"], sen_dct["thin"], fdr_dct["thin"], spe_dct["thin"], kappa_dct["thin"], gmean_dct["thin"], iou_dct["thin"], dice_dct["thin"],bacc_dct["thin"])
    
    loss_dct["thick"], auc_dct["thick"], acc_dct["thick"], sen_dct["thick"], \
    fdr_dct["thick"], spe_dct["thick"], kappa_dct["thick"], gmean_dct["thick"], \
    iou_dct["thick"], dice_dct["thick"] ,bacc_dct["thick"]= print_results(loss_dct["thick"], auc_dct["thick"], acc_dct["thick"], sen_dct["thick"], fdr_dct["thick"], spe_dct["thick"], kappa_dct["thick"], gmean_dct["thick"], iou_dct["thick"], dice_dct["thick"],bacc_dct["thick"])
    
    loss_dct["fusion"], auc_dct["fusion"], acc_dct["fusion"], sen_dct["fusion"], \
    fdr_dct["fusion"], spe_dct["fusion"], kappa_dct["fusion"], gmean_dct["fusion"], \
    iou_dct["fusion"], dice_dct["fusion"] ,bacc_dct["fusion"]= print_results(loss_dct["fusion"], auc_dct["fusion"], acc_dct["fusion"], sen_dct["fusion"], fdr_dct["fusion"], spe_dct["fusion"], kappa_dct["fusion"], gmean_dct["fusion"], iou_dct["fusion"], dice_dct["fusion"],bacc_dct["fusion"])
    
    return loss_dct, auc_dct, acc_dct, sen_dct, fdr_dct, spe_dct, kappa_dct, gmean_dct, iou_dct, dice_dct,bacc_dct


def test_second_stage(dataloader,first_net_thick, first_net_thin, fusion_net, device, results_dir, criterion=None, isSave=True):
    loss_lst = []  # Loss
    auc_lst = []  # AUC
    acc_lst = []  # Accuracy
    sen_lst = []  # Sensitivity (Recall)
    fdr_lst = []  # FDR
    spe_lst = []  # Specificity
    kappa_lst = []  # Kappa
    gmean_lst = []  # G-mean
    iou_lst = []  # IOU
    dice_lst = []  # Dice Coefficient (F1-score)
    bacc_lst=[]
    
    i = 1
    with torch.no_grad():
        for sample in dataloader:
            if len(sample) != 5 and len(sample) != 4 and len(sample) != 2:
                print("Error occured in sample %03d, skip" % i)
                continue
            
            print("Evaluate %03d..." % i)
            i += 1
            
            img = sample[0].to(device)
            gt = sample[1].to(device)
            thin_gt = sample[2].to(device)
            thick_gt = sample[3].to(device)
            x,thick_pred,SVC_pred= first_net_thick(thin_gt)
            c,thin_pred,DVC_pred= first_net_thin( thick_gt)
            # temp_1 = [item.cpu().detach() for item in thick_pred]
            # thick_pred_1 = torch.stack(temp_1, 2).squeeze(1).to(device)
            # temp_2 = [item.cpu().detach() for item in thin_pred]
            # thin_pred_1 = torch.tensor(torch.stack(temp_2, 2)).squeeze(1).to(device)
            # img_1 = img[:, :1, :, :]
            # thick = thick_pred_1[:, :1, :, :]
            # thin = thin_pred_1[:, 1:2, :, :]
            #a,b,pred = fusion_net(img, thick_gt, thin_gt)
            #a,b,pred = fusion_net(img, thick_pred, thin_pred)
            a, b, pred = fusion_net(img, SVC_pred,DVC_pred)


            #pred = fusion_net(img_1)
            if len(sample) == 5:
                w, h = sample[4]
                
                pred = pred.cpu().squeeze(0)
                pred = transforms.ToTensor()(transforms.ToPILImage()(pred))
                pred = pred.unsqueeze(0).to(device)
                
                mask = sample[2].to(device)
                mask_arr = mask.squeeze().cpu().numpy()
            else:
                mask_arr = None
            
            get_results(loss_lst, auc_lst, acc_lst, sen_lst, fdr_lst, spe_lst, kappa_lst, gmean_lst, iou_lst, dice_lst, bacc_lst,dataloader, results_dir, criterion, pred, gt, mask_arr=mask_arr)
    
    loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, kappa_arr, gmean_arr, iou_arr, dice_arr ,bacc_arr= print_results(loss_lst, auc_lst, acc_lst, sen_lst, fdr_lst, spe_lst, kappa_lst, gmean_lst, iou_lst, dice_lst,bacc_lst)
    
    return loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, kappa_arr, gmean_arr, iou_arr, dice_arr,bacc_arr
