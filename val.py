# -*- coding: utf-8 -*-

import os
import torch

from utils import mkdir
from test import test_first_stage, test_second_stage


def visual_results(loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, kappa_arr, gmean_arr, iou_arr, dice_arr, bacc_arr,viz, writer, epoch, flag=""):
    loss_mean = loss_arr.mean()
    auc_mean = auc_arr.mean()
    acc_mean = acc_arr.mean()
    sen_mean = sen_arr.mean()
    fdr_mean = fdr_arr.mean()
    spe_mean = spe_arr.mean()
    kappa_mean = kappa_arr.mean()
    gmean_mean = gmean_arr.mean()
    iou_mean = iou_arr.mean()
    dice_mean = dice_arr.mean()
    bacc_mean = bacc_arr.mean()
    
    viz.plot("val" + flag + " loss", loss_mean)
    viz.plot("val" + flag + " auc", auc_mean)
    viz.plot("val" + flag + " acc", acc_mean)
    viz.plot("val" + flag + " sen", sen_mean)
    viz.plot("val" + flag + " fdr", fdr_mean)
    viz.plot("val" + flag + " spe", spe_mean)
    viz.plot("val" + flag + " kappa", kappa_mean)
    viz.plot("val" + flag + " gmean", gmean_mean)
    viz.plot("val" + flag + " iou", iou_mean)
    viz.plot("val" + flag + " dice", dice_mean)
    viz.plot("val" + flag + " bacc", bacc_mean)
    
    writer.add_scalars("val" + flag + "_loss", {"val" + flag + "_loss": loss_mean}, epoch)
    writer.add_scalars("val" + flag + "_auc", {"val" + flag + "_auc": auc_mean}, epoch)
    writer.add_scalars("val" + flag + "_acc", {"val" + flag + "_acc": acc_mean}, epoch)
    writer.add_scalars("val" + flag + "_sen", {"val" + flag + "_sen": sen_mean}, epoch)
    writer.add_scalars("val" + flag + "_fdr", {"val" + flag + "_fdr": fdr_mean}, epoch)
    writer.add_scalars("val" + flag + "_spe", {"val" + flag + "_spe": spe_mean}, epoch)
    writer.add_scalars("val" + flag + "_kappa", {"val" + flag + "_kappa": kappa_mean}, epoch)
    writer.add_scalars("val" + flag + "_gmean", {"val" + flag + "_gmean": gmean_mean}, epoch)
    writer.add_scalars("val" + flag + "_iou", {"val" + flag + "_iou": iou_mean}, epoch)
    writer.add_scalars("val" + flag + "_dice", {"val" + flag + "_dice": dice_mean}, epoch)
    writer.add_scalars("val" + flag + "_bacc", {"val" + flag + "_bacc": bacc_mean}, epoch)
    
    return auc_mean,acc_mean,dice_mean


def val_first_stage(best_thin, best_thick, best_fusion, viz, writer, dataloader, net,
                    thin_criterion, thick_criterion, fusion_criterion, device, save_epoch_freq,
                    models_dir, results_dir, epoch, num_epochs=100):
    net.eval()
    #net_1.eval()
    loss_dct, auc_dct, acc_dct, sen_dct, fdr_dct, spe_dct, kappa_dct, gmean_dct, iou_dct, dice_dct,bacc_dct\
    = test_first_stage(dataloader, net,device, results_dir,
                       thin_criterion, thick_criterion, fusion_criterion, isSave=True)

    #thin_auc = round(visual_results(loss_dct["thin"], auc_dct["thin"], acc_dct["thin"], sen_dct["thin"], fdr_dct["thin"], spe_dct["thin"], kappa_dct["thin"], gmean_dct["thin"], iou_dct["thin"], dice_dct["thin"], bacc_dct["thin"],viz, writer, epoch, flag="_thin") + 1e-12, 4)
    thin_auc,thin_acc,_ = visual_results(loss_dct["thin"], auc_dct["thin"], acc_dct["thin"], sen_dct["thin"], fdr_dct["thin"],
                       spe_dct["thin"], kappa_dct["thin"], gmean_dct["thin"], iou_dct["thin"], dice_dct["thin"],
                       bacc_dct["thin"], viz, writer, epoch, flag="_thin")
    thin_auc=round(thin_auc+ 1e-12, 4)
    thin_acc = round(thin_acc + 1e-12, 4)
    # thick_auc,thick_acc = round(visual_results(loss_dct["thick"], auc_dct["thick"], acc_dct["thick"],
    #                   sen_dct["thick"], fdr_dct["thick"], spe_dct["thick"], kappa_dct["thick"],
    #                   gmean_dct["thick"], iou_dct["thick"], dice_dct["thick"], bacc_dct["thick"],viz, writer, epoch,
    #                   flag="_thick") + 1e-12, 4)
    thick_auc, thick_acc,_ = visual_results(loss_dct["thick"], auc_dct["thick"], acc_dct["thick"],
                                                sen_dct["thick"], fdr_dct["thick"], spe_dct["thick"],
                                                kappa_dct["thick"],
                                                gmean_dct["thick"], iou_dct["thick"], dice_dct["thick"],
                                                bacc_dct["thick"], viz, writer, epoch,
                                                flag="_thick")
    thick_auc=round(thick_auc+ 1e-12, 4)
    thick_acc = round(thick_acc + 1e-12, 4)
    # fusion_auc,fusion_acc = round(visual_results(loss_dct["fusion"], auc_dct["fusion"], acc_dct["fusion"],
    #                    sen_dct["fusion"], fdr_dct["fusion"], spe_dct["fusion"], kappa_dct["fusion"],
    #                    gmean_dct["fusion"], iou_dct["fusion"], dice_dct["fusion"], bacc_dct["fusion"],viz, writer,
    #                    epoch, flag="_fusion") + 1e-12, 4)
    fusion_auc, fusion_acc,fusion_dice= visual_results(loss_dct["fusion"], auc_dct["fusion"], acc_dct["fusion"],
                                                  sen_dct["fusion"], fdr_dct["fusion"], spe_dct["fusion"],
                                                  kappa_dct["fusion"],
                                                  gmean_dct["fusion"], iou_dct["fusion"], dice_dct["fusion"],
                                                  bacc_dct["fusion"], viz, writer,
                                                  epoch, flag="_fusion")
    fusion_auc=round( fusion_auc+ 1e-12, 4)
    fusion_acc = round(fusion_acc + 1e-12, 4)
    
    # 保存模型
    mkdir(models_dir)
    if thin_auc >= best_thin["auc"]:
        best_thin["epoch"] = epoch + 1
        best_thin["auc"] = thin_auc
        torch.save(net,os.path.join(models_dir, "front_model-best_thin.pth"))
    if thick_auc >= best_thick["auc"]:
        best_thick["epoch"] = epoch + 1
        best_thick["auc"] = thick_auc
        torch.save(net, os.path.join(models_dir, "front_model-best_thick.pth"))
    if fusion_auc >= best_fusion["auc"]:
        best_fusion["epoch"] = epoch + 1
        best_fusion["auc"] = fusion_auc
        torch.save(net, os.path.join(models_dir, "front_model-best_fusion.pth"))
    if thin_acc >= best_thin["acc"]:
        best_thin["epoch"] = epoch + 1
        best_thin["acc"] = thin_acc
        torch.save(net, os.path.join(models_dir, "front_model-best_acc_thin.pth"))
    if thick_acc >= best_thick["acc"]:
        best_thick["epoch"] = epoch + 1
        best_thick["acc"] = thick_acc
        torch.save(net, os.path.join(models_dir, "front_model-best_acc_thick.pth"))
    if fusion_acc >= best_fusion["acc"]:
        best_fusion["epoch"] = epoch + 1
        best_fusion["acc"] = fusion_acc
        torch.save(net, os.path.join(models_dir, "front_model-best_acc_fusion.pth"))
    if fusion_acc >= best_fusion["dice"]:
        best_fusion["epoch"] = epoch + 1
        best_fusion["dice"] = fusion_dice
        torch.save(net, os.path.join(models_dir, "front_model-best_dice_fusion.pth"))

    print("best thin: epoch %d\tauc %.4f" % (best_thin["epoch"], best_thin["auc"]))
    print("best thick: epoch %d\tauc %.4f" % (best_thick["epoch"], best_thick["auc"]))
    print("best fusion: epoch %d\tauc %.4f" % (best_fusion["epoch"], best_fusion["auc"]))
    
    checkpoint_path = os.path.join(models_dir, "{net}-{epoch}-{thin_AUC}-{thick_AUC}-{fusion_AUC}.pth")
    if (epoch + 1) % save_epoch_freq == 0:
        torch.save(net, checkpoint_path.format(net="front_model", epoch=epoch+1,
                   thin_AUC=thin_auc, thick_AUC=thick_auc, fusion_AUC=fusion_auc))
    if epoch == num_epochs - 1:
        torch.save(net, os.path.join(models_dir, "front_model-latest.pth"))
    net.train(mode=True)
    
    return net,best_thin, best_thick, best_fusion


def val_second_stage(best_fusion, viz, writer, dataloader, first_net_thick, first_net_thin, fusion_net, criterion, device, save_epoch_freq, models_dir, results_dir, epoch, num_epochs=100):
    fusion_net.eval()
    loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, kappa_arr, gmean_arr, iou_arr, dice_arr ,bacc_arr= test_second_stage(dataloader, first_net_thick, first_net_thin, fusion_net, device, results_dir, criterion, isSave=True)
    fusion_auc,fusion_acc ,fusion_dice= visual_results(loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, kappa_arr, gmean_arr, iou_arr, dice_arr, bacc_arr,viz, writer, epoch)

    fusion_auc = round(fusion_auc + 1e-12, 4)
    fusion_acc = round(fusion_acc + 1e-12, 4)
    fusion_dice = round(fusion_dice + 1e-12, 4)

    # 保存模型
    mkdir(models_dir)
    if fusion_auc >= best_fusion["auc"]:
        best_fusion["epoch"] = epoch + 1
        best_fusion["auc"] = fusion_auc
        torch.save(fusion_net, os.path.join(models_dir, "fusion_model-best_fusion.pth"))
    if  fusion_dice >=best_fusion["dice"]:
        best_fusion["epoch"] = epoch + 1
        best_fusion["dice"] = fusion_dice
        torch.save(fusion_net, os.path.join(models_dir, "fusion_model-best_fusion_dice.pth"))
    if fusion_acc >= best_fusion["acc"]:
        best_fusion["epoch"] = epoch + 1
        best_fusion["acc"] = fusion_acc
        torch.save(fusion_net, os.path.join(models_dir, "fusion_model-best_acc_fusion.pth"))
    print("best fusion: epoch %d\tauc %.4f" % (best_fusion["epoch"], best_fusion["auc"]))


    checkpoint_path = os.path.join(models_dir, "{net}-{epoch}-{AUC}.pth")
    if (epoch + 1) % save_epoch_freq == 0:
        torch.save(fusion_net, checkpoint_path.format(net="fusion_model", epoch=epoch+1, AUC=fusion_auc))
    if epoch == num_epochs - 1:
        torch.save(fusion_net, os.path.join(models_dir, "fusion_model-latest.pth"))
    fusion_net.train(mode=True)
    
    return fusion_net, best_fusion
