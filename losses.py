# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: clipped tensor
    """
    t = t.float()
    
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    
    return result


class dice_loss_thick(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_loss_thick, self).__init__()
        self.eps = eps
    
    def forward(self, pred, gt):
        a=1
        b=0
        assert pred.size() == gt.size() and pred.size()[1] == 1
        criterion=nn.BCELoss()
        loss_1=criterion(pred,gt)
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        intersection = pred_flat * gt_flat
        dice = (2.0 * intersection.sum(1) + self.eps) / (pred_flat.sum(1) + gt_flat.sum(1) + self.eps)
        loss= 1.0 - dice.mean()
        
        return a*loss_1+b*loss


class dice_loss_thin(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_loss_thin, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        c = 1
        d = 0
        assert pred.size() == gt.size() and pred.size()[1] == 1
        criterion = nn.BCELoss()
        loss_1 = criterion(pred, gt)
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        intersection = pred_flat * gt_flat
        dice = (2.0 * intersection.sum(1) + self.eps) / (pred_flat.sum(1) + gt_flat.sum(1) + self.eps)
        loss = 1.0 - dice.mean()

        return c * loss_1 + d * loss
class dice_loss_fusion(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_loss_fusion, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        c = 0
        d = 1
        assert pred.size() == gt.size() and pred.size()[1] == 1
       # criterion = nn.BCELoss()
        #loss_1 = criterion(pred, gt)
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        intersection = pred_flat * gt_flat
        dice = (2.0 * intersection.sum(1) + self.eps) / (pred_flat.sum(1) + gt_flat.sum(1) + self.eps)
        loss = 1.0 - dice.mean()
        return d * loss#c * loss_1 + d * loss
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        
        pred_oh = torch.cat((pred, 1.0 - pred), dim=1)  # [b, 2, h, w]
        gt_oh = torch.cat((gt, 1.0 - gt), dim=1)  # [b, 2, h, w]
        pt = (gt_oh * pred_oh).sum(1)  # [b, h, w]
        focal_map = - self.alpha * torch.pow(1.0 - pt, self.gamma) * torch.log2(clip_by_tensor(pt, 1e-12, 1.0))  # [b, h, w]
        
        if self.size_average:
            loss = focal_map.mean()
        else:
            loss = focal_map.sum()
        
        return loss
class Adaptive_Region_Specific_TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-5, num_region_per_axis=(16, 16, 16), do_bg=True, batch_dice=True, A=0.3, B=0.4):
        """
        num_region_per_axis: the number of boxes of each axis in (z, x, y)
        3D num_region_per_axis's axis in (z, x, y)
        2D num_region_per_axis's axis in (x, y)
        """
        super(Adaptive_Region_Specific_TverskyLoss, self).__init__()
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.dim = len(num_region_per_axis)
        assert self.dim in [2, 3], "The num of dim must be 2 or 3."
        if self.dim == 3:
            self.pool = nn.AdaptiveAvgPool3d(num_region_per_axis)
        elif self.dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(num_region_per_axis)

        self.A = A
        self.B = B

    def forward(self, x, y):
        # 默认x是未经过softmax的。2D/3D: [batchsize, c, (z,) x, y]
        x = torch.softmax(x, dim=1)

        shp_x, shp_y = x.shape, y.shape
        assert self.dim == (len(shp_x) - 2), "The region size must match the data's size."

        if not self.do_bg:
            x = x[:, 1:]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # the three in [batchsize, class_num, (z,) x, y]
        tp = x * y_onehot
        fp = x * (1 - y_onehot)
        fn = (1 - x) * y_onehot

        # the three in [batchsize, class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        if self.batch_dice:
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        # [(batchsize,) class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        # [(batchsize,) class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)
        region_tversky = 1 - region_tversky

        # [(batchsize,) class_num]
        if self.batch_dice:
            region_tversky = region_tversky.sum(list(range(1, len(shp_x)-1)))
        else:
            region_tversky = region_tversky.sum(list(range(2, len(shp_x))))

        region_tversky = region_tversky.mean()

        return region_tversky


# 构建损失函数，可扩展
def build_loss(loss):
    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "l1":
        criterion = nn.L1Loss()
    elif loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss == "bce":
        criterion = focal_loss(alpha=1.0, gamma=0.0)
    elif loss == "focal":
        criterion = focal_loss(alpha=0.25, gamma=2.0)
    elif loss == "dice_thick":
        criterion = dice_loss_thick()
    elif loss == "dice_thin":
        criterion = dice_loss_thin()
    elif loss == "dice_fusion":
        criterion = dice_loss_fusion()
    elif loss== "ARSTLoss":
        criterion=Adaptive_Region_Specific_TverskyLoss(num_region_per_axis=(8, 8))
    else:
        raise NotImplementedError('loss [%s] is not implemented' % loss)
    
    return criterion
