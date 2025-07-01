# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
#from networks.vit_seg_modeling import SRF_UNet

import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from thop import profile
from thop import clever_format
from options import args
from utils import mkdir, build_dataset, Visualizer  # build_model,
from first_stage import SRF_UNet
# from second_stage import fusion
# from losses import build_loss
from OCT2Former import OCT2Former
from train import train_first_stage
from val import val_first_stage
from test import test_first_stage
from torchsummary import summary
from losses import build_loss
from other_models import U_Net
if __name__=='__main__':
    a=0.9
    b=0.1
    # 是否使用cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        isTraining = True
    else:
        isTraining = False

    database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining,
                             crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
    sub_dir = args.dataset + "/first_stage"  # + args.model + "/" + args.loss
    # config_vit = CONFIGS_ViT_seg[args.vit_name]
    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    # if args.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # net_t= ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net_t.load_from(weights=np.load(config_vit.pretrained_path))

    if isTraining:  # train
        NAME = args.dataset + "_first_stage-2nd"  # + args.model + "_" + args.loss
        viz = Visualizer(env=NAME)
        writer = SummaryWriter(args.logs_dir + "/" + sub_dir)
        mkdir(args.models_dir + "/" + sub_dir)  # two stage时可以创建first_stage和second_stage这两个子文件夹

        # 加载数据集
        train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        val_database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=False,
                                     crop_size=(args.crop_size, args.crop_size),
                                     scale_size=(args.scale_size, args.scale_size))
        val_dataloader = DataLoader(val_database, batch_size=1)

        # 构建模型
        first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1,attention=True,num_heads=3, window_size=7,input_size=(304,304)).to(device)
        #first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1, attention=True, num_heads=3, window_size=7,input_size=(400, 400)).to(device)
        #first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1, attention=True, num_heads=3, window_size=7, input_size=(91, 91)).to(device)
        #first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1, attention=True, num_heads=3, window_size=7, input_size=(512, 512)).to(device)
        #first_net= U_Net()
        first_net = torch.nn.DataParallel(first_net)
        #net=torch.nn.DataParallel(net_t)
        first_optim = optim.Adam(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        #first_optim = optim.SGD(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, momentum=0.9)
        #first_optim_1 = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=0.0001)



        # first_net_1 = OCT2Former(img_ch=args.input_nc, output_ch=1).to(device)
        # first_net_1= torch.nn.DataParallel(first_net_1)
        # first_optim_1 = optim.Adam(first_net_1.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

        print(summary(first_net, input_size=(3,304,304)))

       # #flops, params = profile(first_net, inputs=(3,304,304)).to(device)
       #  # print(flops, params)
       #  macs, params = clever_format([flops, params], "%.3f")
       #  print(macs, params)


        # second_net = fusion(channels=args.base_channels, pn_size=args.pn_size, kernel_size=3, avg=0.0, std=0.1).to(device)
        # second_net = torch.nn.DataParallel(second_net)
        # second_optim = optim.Adam(second_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

        thick_criterion =build_loss("dice_thick")#build_loss("dice_thick")#torch.nn.BCELoss()#build_loss("dice_thick") # 可更改torch.nn.BCELoss()
        thin_criterion =build_loss("dice_thin")#build_loss("dice_thin")#torch.nn.BCELoss()#build_loss("dice_thin")# 可更改
        fusion_criterion =build_loss("dice_fusion")#torch.nn.BCELoss()#torch.nn.BCELoss() #build_loss("dice_fusion") # 可更改
        best_thin = {"epoch": 0, "dice": 0,"auc":0,"acc":0}
        best_thick = {"epoch": 0, "dice": 0,"auc":0,"acc":0}
        best_fusion = {"epoch": 0, "dice": 0,"auc":0,"acc":0}
        # start training
        print("Start training...")
        for epoch in range(args.first_epochs):
            print('Epoch %d / %d' % (epoch + 1, args.first_epochs))
            print('-' * 10)
            # first_net_1 ,net_1= train_first_stage(viz, writer, train_dataloader, first_net, net,first_optim, first_optim_1,args.init_lr,
            #                           thin_criterion, thick_criterion, device, args.power, epoch, args.first_epochs)
            first_net= train_first_stage(viz, writer, train_dataloader, first_net,
                                                         first_optim, args.init_lr,
                                                         thin_criterion, thick_criterion,fusion_criterion ,device, args.power, epoch,
                                                         args.first_epochs)
            if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.first_epochs - 1:
                first_net,best_thin, best_thick, best_fusion = val_first_stage(best_thin, best_thick, best_fusion,
                                                                                viz, writer, val_dataloader, first_net,
                                                                                thin_criterion, thick_criterion,
                                                                                fusion_criterion, device,
                                                                                args.save_epoch_freq,
                                                                                args.models_dir + "/" + sub_dir,
                                                                                args.results_dir + "/" + sub_dir, epoch,
                                                                                args.first_epochs)
        print("Training finished.")
    else:  # test
        # 加载数据集和模型
        test_dataloader = DataLoader(database, batch_size=1)


        net = torch.load(args.models_dir + "/" + sub_dir + "/front_model-" + args.first_suffix).to(
            device)  # two stage时可以加载first_stage和second_stage的模型
        net.eval()
        #print(summary(net, input_size=(3, 304, 304)))

        # start testing
        print("Start testing...")
        test_first_stage(test_dataloader, net,device, args.results_dir + "/" + sub_dir, thin_criterion=None,
                         thick_criterion=None, fusion_criterion=None, isSave=True)
        print("Testing finished.")

# # 是否使用cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# if args.mode == "train":
#     isTraining = True
# else:
#     isTraining = False
#
# database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining,
#                          crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
# sub_dir = args.dataset + "/first_stage"  # + args.model + "/" + args.loss
#
# if isTraining:  # train
#     NAME = args.dataset + "_first_stage-2nd"  # + args.model + "_" + args.loss
#     viz = Visualizer(env=NAME)
#     writer = SummaryWriter(args.logs_dir + "/" + sub_dir)
#     mkdir(args.models_dir + "/" + sub_dir)  # two stage时可以创建first_stage和second_stage这两个子文件夹
#
#     # 加载数据集
#     train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
#     val_database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=False,
#                                  crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
#     val_dataloader = DataLoader(val_database, batch_size=1)
#
#     # 构建模型
#     first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1).to(device)
#     first_net = torch.nn.DataParallel(first_net)
#     first_optim = optim.Adam(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
#
#     # second_net = fusion(channels=args.base_channels, pn_size=args.pn_size, kernel_size=3, avg=0.0, std=0.1).to(device)
#     # second_net = torch.nn.DataParallel(second_net)
#     # second_optim = optim.Adam(second_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
#
#     thick_criterion = torch.nn.MSELoss()  # 可更改
#     thin_criterion = torch.nn.MSELoss()  # 可更改
#     fusion_criterion = torch.nn.MSELoss()  # 可更改
#
#     best_thin = {"epoch": 0, "dice": 0}
#     best_thick = {"epoch": 0, "dice": 0}
#     best_fusion = {"epoch": 0, "dice": 0}
#     # start training
#     print("Start training...")
#     for epoch in range(args.first_epochs):
#         print('Epoch %d / %d' % (epoch + 1, args.first_epochs))
#         print('-'*10)
#         first_net = train_first_stage(viz, writer, train_dataloader, first_net, first_optim, args.init_lr, thin_criterion, thick_criterion, device, args.power, epoch, args.first_epochs)
#         if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.first_epochs - 1:
#             first_net, best_thin, best_thick, best_fusion = val_first_stage(best_thin, best_thick, best_fusion,
#                                                             viz, writer, val_dataloader, first_net,
#                                                             thin_criterion, thick_criterion, fusion_criterion, device,
#                                                             args.save_epoch_freq, args.models_dir + "/" + sub_dir,
#                                                             args.results_dir + "/" + sub_dir, epoch, args.first_epochs)
#     print("Training finished.")
# else:  # test
#     # 加载数据集和模型
#     test_dataloader = DataLoader(database, batch_size=1)
#     net = torch.load(args.models_dir + "/" + sub_dir + "/front_model-" + args.first_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
#     net.eval()
#
#     # start testing
#     print("Start testing...")
#     test_first_stage(test_dataloader, net, device, args.results_dir + "/" + sub_dir, thin_criterion=None, thick_criterion=None, fusion_criterion=None,  isSave=True)
#     print("Testing finished.")
