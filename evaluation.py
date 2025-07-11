# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
from sklearn import metrics
import torch

def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()
    
    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])
        
        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)
    
    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = metrics.roc_auc_score(gt_vec, pred_vec)
    
    return roc_auc


def numeric_score(pred_arr, gt_arr, kernel_size=(1, 1), threshold=0.5,predict_b=None):  # DCC & ROSE-2: kernel_size=(3, 3)
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)
    #dilated_gt_arr=gt_arr


    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))
    #predict = torch.sigmoid(pred_arr).cpu().detach().numpy().flatten()
    # if predict_b is not None:
    #     predict_b = predict_b.flatten()
    # else:
    #     predict_b = np.where(pred_arr >= threshold, 1, 0)
    #     predict_b=predict_b.flatten()
    # if torch.is_tensor(gt_arr):
    #     target = gt_arr.cpu().detach().numpy().flatten()
    # else:
    #     target = gt_arr.flatten()
    # TP = (predict_b * target).sum()
    # TN = ((1 - predict_b) * (1 - target)).sum()
    # FP = ((1 - target) * predict_b).sum()
    # FN = ((1 - predict_b) * target).sum()
    # target = gt_arr
    # pred_arr[pred_arr > 0.5] = 1
    # pred_arr[pred_arr <= 0.5] = 0
    #
    # TP = (pred_arr * target).sum().item()
    # TN = ((1 - pred_arr) * (1 - target)).sum().item()
    # FP = ((1 - target) * pred_arr).sum().item()
    # FN = ((1 - pred_arr) * target).sum().item()
    
    return FP, FN, TP, TN


def calc_acc(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)
    
    return acc


def calc_bacc(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    bacc = (TP/(TP+FN))+(TN/(TN+FP))/2
    target = gt_arr
    pred_arr[pred_arr > 0.5] = 1
    pred_arr[pred_arr <= 0.5] = 0

    tp = (pred_arr * target).sum().item()
    tn = ((1 - pred_arr) * (1 - target)).sum().item()
    fp = ((1 - target) * pred_arr).sum().item()
    fn = ((1 - pred_arr) * target).sum().item()
    #temp=((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    bacc=((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    temp=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    #MCC=(TN*TP)-(FN*FP)/math.sqrt(temp)
    #MCC = ((TN*TP) - (FN* FP))/ math.sqrt(temp)
    return bacc


def calc_sen(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    sen = TP / (FN + TP + 1e-12)
    
    return sen


def calc_fdr(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    fdr = FP / (FP + TP + 1e-12)
    
    return fdr


def calc_spe(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    spe = TN / (FP + TN + 1e-12)
    
    return spe


def calc_gmean(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    sen = calc_sen(pred_arr, gt_arr, kernel_size=kernel_size)
    spe = calc_spe(pred_arr, gt_arr, kernel_size=kernel_size)


    return math.sqrt(sen * spe)

def calc_kappa(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size=kernel_size)
    matrix = np.array([[TP, FP],
                       [FN, TN]])
    n = np.sum(matrix)
    
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    
    return (po - pe) / (1 - pe)


def calc_iou(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)& svc kernel_size=(1,1)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    iou = TP / (FP + FN + TP + 1e-12)
    
    return iou


def calc_dice(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)
    
    return dice
