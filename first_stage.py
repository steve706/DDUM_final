# -*- coding: utf-8 -*-
from swinunet import KanSwinTransformerBlock
import numpy as np
#from  second_stage import base
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import resnest
from splat import SplAtConv2d
from model.utils.dca import DCA
from swinunet import  SwinTransformerBlock
from torchvision.ops import deform_conv2d
# #########--------- Components ---------#########
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),#3
            nn.BatchNorm2d(ch_out),
            #nn.GELU(),
            #nn.ReLU(inplace=True)
            SplAtConv2d(ch_out, ch_out, kernel_size=3,padding=1,groups=2,radix=2,norm_layer=nn.BatchNorm2d),
            nn.GELU(),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1,bias=False),#1
            nn.BatchNorm2d(ch_out),
        )
        self.gelu = nn.GELU()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        
        return self.gelu(out + residual)
class spatial_attention(nn.Module):
    def __init__(self,ch_in=2,ch_out=1):
        super(spatial_attention, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,2,kernel_size=7,stride=1,padding='same'),
            nn.BatchNorm2d(2),
            nn.Conv2d(2,384, kernel_size=1),
            nn.Conv2d(384, ch_out, kernel_size=1 ),

            nn.Sigmoid()
        )
    def forward(self, x):
        x1,_=torch.max(x,axis=1, keepdims=True)
        _,x2=torch.var_mean(x,axis=1,keepdims=True)
        x3=torch.concat((x1,x2),dim=1)
        x4=self.conv(x3)
        out=torch.mul(x,x4)
        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
           # nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride)
        )
    
    def forward(self, x):
        x = self.up(x)
        
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 空间注意力的卷积核大小（默认为7x7）
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            out: 空间注意力加权后的特征图 [B, C, H, W]
        """
        # 计算空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均 [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大 [B, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # 拼接 [B, 2, H, W]
        spatial_attn = self.sigmoid(self.conv(concat))  # 空间注意力图 [B, 1, H, W]

        # 应用空间注意力
        out = x * spatial_attn  # 逐元素相乘 [B, C, H, W]
        return out


class DepthAwareGatingWithSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        """
        Args:
            in_channels: 输入特征图的通道数（浅层和深层特征需通道数相同）
        """
        super().__init__()
        # 通道注意力模块
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力模块
        self.spatial_attn_s = SpatialAttention()  # 浅层特征的空间注意力
        self.spatial_attn_d = SpatialAttention()  # 深层特征的空间注意力

        #   1self.conv1_1-=nn.Conv2d(32,2,kernel_size=1)
        self.Conv_1x1 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, F_s, F_d):
        """
        Args:
            F_s: 浅层特征图 [B, C, H, W]
            F_d: 深层特征图 [B, C, H, W]（需与F_s分辨率一致）
        Returns:
            P_final: 融合后的概率图 [B, 1, H, W]
        """
        # 应用空间注意力
        F_s_attn = self.spatial_attn_s(F_s)  # 浅层特征的空间注意力加权
        F_d_attn = self.spatial_attn_d(F_d)  # 深层特征的空间注意力加权

        # 特征拼接与门控权重生成
        concat_feat = torch.cat([F_s_attn, F_d_attn], dim=1)  # [B, 2C, H, W]
        gate = self.sigmoid(self.conv(concat_feat))  # [B, C, H, W]

        # 深度感知加权融合
        F_fused = (1 - gate) * F_s_attn + gate * F_d_attn
        F_fused=self.Conv_1x1(F_fused)

        # 概率映射
        P_final = self.sigmoid(F_fused)  # 可替换为1x1卷积+Softmax（多分类）
        return P_final


# #########--------- Networks ---------#########
class DDU_Net(nn.Module):
    def __init__(self, num_heads, window_size,n=1,
                 img_ch=3,
                 output_ch=1,
                 attention=True,
                 patch_size=16,
                 #patch_size=24,
                 spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1],input_size=(304, 304),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(DDU_Net, self).__init__()
        filters = [64, 128, 256, 512]
        self.h,self.w=input_size

#         # self.block_4=nn.ModuleList([SwinTransformerBlock(dim=1024, input_resolution=(self.h//16,self.w//16),
#         #                          num_heads=32, window_size=4,
#         #                          shift_size=0 if (i % 2 == 0) else window_size // 2,
#         #                          mlp_ratio=mlp_ratio,
#         #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #                          drop=drop, attn_drop=attn_drop,
#         #                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#         #                          norm_layer=norm_layer) for i in range(2)])
#         #
#         #
#         # self.block_3 = nn.ModuleList([SwinTransformerBlock(dim=512, input_resolution=(self.h//8,self.w//8),
#         #                                             num_heads=16, window_size=4,
#         #                                             shift_size=0 if (i % 2 == 0) else window_size // 2,
#         #                                             mlp_ratio=mlp_ratio,
#         #                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #                                             drop=drop, attn_drop=attn_drop,
#         #                                             drop_path=drop_path[i] if isinstance(drop_path,
#         #                                                                                  list) else drop_path,
#         #                                             norm_layer=norm_layer)for i in range(2)])
#         # self.block_3 = nn.ModuleList([SwinTransformerBlock(dim=512, input_resolution=(self.h//8,self.w//8),
#         #                                             num_heads=16, window_size=4,
#         #                                             shift_size=0 if (i % 2 == 0) else window_size // 2,
#         #                                             mlp_ratio=mlp_ratio,
#         #                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #                                             drop=drop, attn_drop=attn_drop,
#         #                                             drop_path=drop_path[i] if isinstance(drop_path,
#         #                                                                                  list) else drop_path,
#         #                                             norm_layer=norm_layer)for i in range(2)])
#         # self.block_2 = nn.ModuleList([KanSwinTransformerBlock(dim=256, input_resolution=(self.h//4,self.w//4),
#         #                                             num_heads=8, window_size=4,
#         #                                             shift_size=0 if (i % 2 == 0) else window_size // 2,
#         #                                             mlp_ratio=mlp_ratio,
#         #                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #                                             drop=drop, attn_drop=attn_drop,
#         #                                             drop_path=drop_path[i] if isinstance(drop_path,
#         #                                                                                  list) else drop_path,
#         #                                             norm_layer=norm_layer)for i in range(2)])
#         # self.block_1 =nn.ModuleList([KanSwinTransformerBlock(dim=64, input_resolution=(self.h//4,self.w//4),
#         #                                             num_heads=2 ,window_size=4,
#         #                                             shift_size=0  if (i % 2 == 0) else window_size // 2,
#         #                                             mlp_ratio=mlp_ratio,
#         #                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #                                             drop=drop, attn_drop=attn_drop,
#         #                                             drop_path=drop_path[i] if isinstance(drop_path,
#         #                                                                                  list) else drop_path,
#         #                                             norm_layer=norm_layer)for i in range(2)])
#
 #############ROSE##########################################
        self.block_4 = nn.ModuleList(
            [KanSwinTransformerBlock(dim=1024, input_resolution=(self.h // 16 + 1, self.w // 16 + 1),
                                  num_heads=32, window_size=4,
                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer) for i in range(2)])
        self.block_3 = nn.ModuleList([KanSwinTransformerBlock(dim=512, input_resolution=(self.h//8,self.w//8),
                                                    num_heads=16, window_size=19,
                                                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop, attn_drop=attn_drop,
                                                    drop_path=drop_path[i] if isinstance(drop_path,
                                                                                         list) else drop_path,
                                                    norm_layer=norm_layer)for i in range(2)])

        self.block_2 = nn.ModuleList([KanSwinTransformerBlock(dim=256, input_resolution=(self.h//4,self.w//4),
                                                    num_heads=8, window_size=4,
                                                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop, attn_drop=attn_drop,
                                                    drop_path=drop_path[i] if isinstance(drop_path,
                                                                                         list) else drop_path,
                                                    norm_layer=norm_layer)for i in range(2)])
        self.block_1 =nn.ModuleList([KanSwinTransformerBlock(dim=64, input_resolution=(self.h//4,self.w//4),
                                                    num_heads=2 ,window_size=4,
                                                    shift_size=0  if (i % 2 == 0) else window_size // 2,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop, attn_drop=attn_drop,
                                                    drop_path=drop_path[i] if isinstance(drop_path,
                                                                                         list) else drop_path,
                                                    norm_layer=norm_layer)for i in range(2)])

        # self.block_5 =nn.ModuleList([SwinTransformerBlock(dim=64, input_resolution=(self.h//2,self.w//2),
        #                                             num_heads=2 ,window_size=4,
        #                                             shift_size=0  if (i % 2 == 0) else window_size // 2,
        #                                             mlp_ratio=mlp_ratio,
        #                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                             drop=drop, attn_drop=attn_drop,
        #                                             drop_path=drop_path[i] if isinstance(drop_path,
        #                                                                                  list) else drop_path,
        #                                             norm_layer=norm_layer)for i in range(2)])
        # self.block_6 =nn.ModuleList([SwinTransformerBlock(dim=32, input_resolution=(self.h,self.w),
        #                                             num_heads=2 ,window_size=4,
        #                                             shift_size=0  if (i % 2 == 0) else window_size // 2,
        #                                             mlp_ratio=mlp_ratio,
        #                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                             drop=drop, attn_drop=attn_drop,
        #                                             drop_path=drop_path[i] if isinstance(drop_path,
        #                                                                                  list) else drop_path,
        #                                             norm_layer=norm_layer)for i in range(2)])
#
# ######################################Octa_500_6M###########################################################
#         self.block_4 = nn.ModuleList(
#             [SwinTransformerBlock(dim=1024, input_resolution=(self.h // 16+1 , self.w // 16+1 ),
#                                   num_heads=32, window_size=13,
#                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                   mlp_ratio=mlp_ratio,
#                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                   drop=drop, attn_drop=attn_drop,
#                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                   norm_layer=norm_layer) for i in range(2)])
#         self.block_3 = nn.ModuleList([SwinTransformerBlock(dim=512, input_resolution=(self.h//8,self.w//8),
#                                                     num_heads=16, window_size=5,
#                                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                     mlp_ratio=mlp_ratio,
#                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                     drop=drop, attn_drop=attn_drop,
#                                                     drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                          list) else drop_path,
#                                                     norm_layer=norm_layer)for i in range(2)])
#
#         self.block_2 = nn.ModuleList([KanSwinTransformerBlock(dim=256, input_resolution=(self.h//4,self.w//4),
#                                                     num_heads=8, window_size=4,
#                                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                     mlp_ratio=mlp_ratio,
#                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                     drop=drop, attn_drop=attn_drop,
#                                                     drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                          list) else drop_path,
#                                                     norm_layer=norm_layer)for i in range(2)])
#         self.block_1 =nn.ModuleList([KanSwinTransformerBlock(dim=64, input_resolution=(self.h//4,self.w//4),
#                                                     num_heads=2 ,window_size=4,
#                                                     shift_size=0  if (i % 2 == 0) else window_size // 2,
#                                                     mlp_ratio=mlp_ratio,
#                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                     drop=drop, attn_drop=attn_drop,
#                                                     drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                          list) else drop_path,
#                                                     norm_layer=norm_layer)for i in range(2)])
# #         ################################prevent#######################################################################################3
#         self.block_4 = nn.ModuleList(
#             [SwinTransformerBlock(dim=1024, input_resolution=(self.h // 16+1, self.w // 16+1 ),
#                                   num_heads=32, window_size=6,
#                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                   mlp_ratio=mlp_ratio,
#                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                   drop=drop, attn_drop=attn_drop,
#                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                   norm_layer=norm_layer) for i in range(2)])
#         self.block_3 = nn.ModuleList([SwinTransformerBlock(dim=512, input_resolution=(self.h//8+1,self.w//8+1),
#                                                     num_heads=16, window_size=4,
#                                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                     mlp_ratio=mlp_ratio,
#                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                     drop=drop, attn_drop=attn_drop,
#                                                     drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                          list) else drop_path,
#                                                     norm_layer=norm_layer)for i in range(2)])
#
#         self.block_2 = nn.ModuleList([KanSwinTransformerBlock(dim=256, input_resolution=(self.h//4+1,self.w//4+1),
#                                                     num_heads=8, window_size=23,
#                                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                     mlp_ratio=mlp_ratio,
#                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                     drop=drop, attn_drop=attn_drop,
#                                                     drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                          list) else drop_path,
#                                                     norm_layer=norm_layer)for i in range(2)])
#         self.block_1 =nn.ModuleList([KanSwinTransformerBlock(dim=64, input_resolution=(self.h//4+1,self.w//4+1),
#                                                     num_heads=2 ,window_size=23,
#                                                     shift_size=0  if (i % 2 == 0) else window_size // 2,
#                                                     mlp_ratio=mlp_ratio,
#                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                     drop=drop, attn_drop=attn_drop,
#                                                     drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                          list) else drop_path,
#                                                     norm_layer=norm_layer)for i in range(2)])
#
# #         ########################################roes-2###############################################
#         self.block_4 = nn.ModuleList(
#             [SwinTransformerBlock(dim=1024, input_resolution=(self.h // 16 , self.w // 16 ),
#                                   num_heads=32, window_size=4,
#                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                   mlp_ratio=mlp_ratio,
#                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                   drop=drop, attn_drop=attn_drop,
#                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                   norm_layer=norm_layer) for i in range(2)])
#         self.block_3 = nn.ModuleList([SwinTransformerBlock(dim=512, input_resolution=(self.h // 8, self.w // 8),
#                                                            num_heads=16, window_size=4,
#                                                            shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                            mlp_ratio=mlp_ratio,
#                                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                            drop=drop, attn_drop=attn_drop,
#                                                            drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                                 list) else drop_path,
#                                                            norm_layer=norm_layer) for i in range(2)])
#
#         self.block_2 = nn.ModuleList([KanSwinTransformerBlock(dim=256, input_resolution=(self.h // 4, self.w // 4),
#                                                               num_heads=8, window_size=4,
#                                                               shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                               mlp_ratio=mlp_ratio,
#                                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                               drop=drop, attn_drop=attn_drop,
#                                                               drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                                    list) else drop_path,
#                                                               norm_layer=norm_layer) for i in range(2)])
#         self.block_1 = nn.ModuleList([KanSwinTransformerBlock(dim=64, input_resolution=(self.h // 4, self.w // 4),
#                                                               num_heads=2, window_size=4,
#                                                               shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                                               mlp_ratio=mlp_ratio,
#                                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                                               drop=drop, attn_drop=attn_drop,
#                                                               drop_path=drop_path[i] if isinstance(drop_path,
#                                                                                                    list) else drop_path,
#                                                               norm_layer=norm_layer) for i in range(2)])



        #######################################################################
        resnet = resnest.resnest50(pretrained=True)
        self.attention=attention
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.secondconv = resnet.conv1
        # self.secondbn = resnet.bn1
        # self.secondrelu = resnet.relu
        # self.secondmaxpool = resnet.maxpool


       # self.encoder5 = resnet.layer1
       # self.encoder6 = resnet.layer2

        
        #self.Up5_thick = up_conv(ch_in=2048, ch_out=1024)
        self.Up5_thick_sameple=up_conv_sameple(ch_in=2048, ch_out=1024)
        self.Up_conv5_thick = res_conv_block(ch_in=2048, ch_out=1024)
        
       # self.Up4_thick = up_conv(ch_in=1024, ch_out=64)
        self.Up4_thick_sameple = up_conv_sameple(ch_in=1024, ch_out=512)
        self.Up_conv4_thick = res_conv_block(ch_in=1024, ch_out=512)
        
        #self.Up3_thick = up_conv(ch_in=512, ch_out=64)
        self.Up3_thick_sameple = up_conv_sameple(ch_in=512, ch_out=256)
        self.Up_conv3_thick = res_conv_block(ch_in=512, ch_out=256)
        
        #self.Up2_thick = up_conv(ch_in=256, ch_out=64)
        self.Up2_thick_sameple = up_conv_sameple(ch_in=256, ch_out=64)
        self.Up_conv2_thick = res_conv_block(ch_in=128, ch_out=64)
        
        #self.Up1_thick = up_conv(ch_in=64, ch_out=64)
        self.Up1_thick_sameple = up_conv_sameple(ch_in=64, ch_out=64)
        self.Up_conv1_thick = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thick = nn.Conv2d(32, output_ch, kernel_size=1)
        
        # ##
        # self.Up5_thin = up_conv(ch_in=2048, ch_out=1024)
        # self.Up5_thin_sameple = up_conv_sameple(ch_in=2048, ch_out=1024)
        # self.Up_conv5_thin = res_conv_block(ch_in=2048, ch_out=1024)
        #
        # self.Up4_thin = up_conv(ch_in=1024, ch_out=512)
        # self.Up4_thin_sameple = up_conv_sameple(ch_in=1024, ch_out=512)
        # self.Up_conv4_thin = res_conv_block(ch_in=1024, ch_out=512)
        #
        # self.Up3_thin = up_conv(ch_in=512, ch_out=256)
        # self.Up3_thin_sameple = up_conv_sameple(ch_in=512, ch_out=256)
        # self.Up_conv3_thin = res_conv_block(ch_in=512, ch_out=256)
        # ##
        
        #self.Up2_thin = up_conv(ch_in=256, ch_out=64)
        self.Up2_thin_sameple = up_conv_sameple(ch_in=256, ch_out=64)
        self.Up_conv2_thin = res_conv_block(ch_in=128, ch_out=64)
        
        #self.Up1_thin = up_conv(ch_in=64, ch_out=64)
        self.Up1_thin_sameple = up_conv_sameple(ch_in=64, ch_out=64)
        self.Up_conv1_thin = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thin = nn.Conv2d(32, output_ch, kernel_size=1)
        
        # ##
        self.Up1_sameple = up_conv_sameple(ch_in=64, ch_out=64)
        self.Up_conv1 = res_conv_block(ch_in=64,ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1)
        # ##
        #self.spatial=spatial_attention(ch_in=2,ch_out=1)
        #self.attention = attention
        self.patch_size = patch_size
        patch = input_size[0] // self.patch_size
        self.DCA = DCA(n=n,
                       features=[64,256, 512,1024],
                       strides=[self.patch_size// 2, self.patch_size // 4,self.patch_size//8,self.patch_size//16],
                       patch=patch,
                       spatial_att=spatial_att,
                       channel_att=channel_att,
                       spatial_head=spatial_head_dim,
                       channel_head=channel_head_dim,
                       )
        # self.DCA = DCA(n=n,
        #                features=[256, 512,1024],
        #                strides=[self.patch_size// 3,self.patch_size // 6,self.patch_size//12],
        #                patch=patch,
        #                spatial_att=spatial_att,
        #                channel_att=channel_att,
        #                spatial_head=spatial_head_dim,
        #                channel_head=channel_head_dim,
        #                )
        #self.fuse=feature_fuse(3)
        self.maxpool_1 = nn.MaxPool2d(2)
       # self.Conv1_1= nn.Conv2d(256, 512, kernel_size=3,dilation=2,padding="same")
        #self.Conv1_1 = EdgeConv(256, 512)
        self.maxpool_2 = nn.MaxPool2d(2)
        #self.Conv1_2 = nn.Conv2d(256,1024 , kernel_size=1)
        #self.Conv1_2 = EdgeConv(256, 1024)
        self.maxpool_3 = nn.MaxPool2d(2)
        self.maxpool_4 = nn.MaxPool2d(2)

        #self.Conv1_3 = nn.Conv2d(256, 2048, kernel_size=1)
        #self.Conv1_3 = EdgeConv(256, 2048)
        # model_path =
        # #self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        # checkpoint = torch.load(model_path, map_location=torch.device)['model']
        # unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
        #               "patch_embed.norm.bias",
        #               "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
        #               "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
        #               "layers.1.downsample.norm.bias",
        #               "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
        #               "layers.2.downsample.norm.bias",
        #               "layers.2.downsample.reduction.weight", "layers.3.downsample.norm.weight",
        #               "layers.3.downsample.norm.bias",
        #               "layers.3.downsample.reduction.weight", "norm.weight", "norm.bias"]
        #
        #self.p1_ch = EdgeConv(64, 64)
        self.p2_ch = EdgeConv(64, 256,1)
        self.p3_ch =EdgeConv(256, 512,2)
        self.p4_ch = EdgeConv(256, 1024,3)
        self.p5_ch = EdgeConv(256, 2048,4)
        #self.dep=DEP(3)
        # self.dep_1 = DEP(64)
        # self.dep_2=DEP(256)
        # self.dep_3 = DEP(512)
        # self.dep_4 = DEP(1024)
        #self.gate_fusion = DepthAwareGating(in_channels=32)
        self.gate_fusion = DepthAwareGatingWithSpatialAttention(in_channels=32)



    def forward(self, x):
        #noise_features = self.noise_estimator(x)  # (B,3,H,W)
        #mask = self.mask_generator(noise_features)  # (B,1,H,W)

        # # Step3: 噪声抑制
        #x = x * mask
        #x=self.dep(x)
        # encoding path
        down_pad_1 = False
        right_pad_1 = False
        if x.size()[2] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
            down_pad_1 = True
        if x.size()[3] % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))
            right_pad_1 = True
        # edge_1 = self.maxpool_1(x)
        # # for i in range(4):
        # #     edge_1= self.pidinet_layers[i](edge_1)
        # edge_1_1=self.p1_ch(edge_1)
        #edge_1 = self.maxpool_1(x)
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        #x0=torch.add(x0,edge_1_1)
        #x0=n,64,152,152]

        x1 = self.firstmaxpool(x0)
        ##edge_1=self.avgpool_1(x0)

        for blk in self.block_1:
            x1=blk(x1)
        #x1=x1+x1
        x2 = self.encoder1(x1)
        #x2=[n,256,76,76]
        #x2_t=x2
        for blk in self.block_2:
            x2=blk(x2)
        #x2=x2+x2_t
        #x2=torch.add(edge_3_3,x2)
        # edge_2=self.maxpool_2(x2)
        # # for i in range(4, 8):
        # #     edge_2 = self.pidinet_layers[i](edge_2)
        # edge_2_2=self.p2_ch(edge_2)

        #x2=torch.add(edge_2_2,x2)
        down_pad_2 = False
        right_pad_2 = False
        if x2.size()[2] % 2 == 1:
            x2 = F.pad(x2, (0, 0, 0, 1))
            down_pad_2 = True
        if x2.size()[3] % 2 == 1:
            x2 = F.pad(x2, (0, 1, 0, 0))
            right_pad_2 = True
        #edge_1 = self.maxpool_1(x)
        ##edge_1_1=self.Conv1_1(edge_1)

        #edge_1_1=edge_1
        x3 = self.encoder2(x2)
        #x3=[n,512,38,38]
        #x3_t = x3

        edge_3 =self.maxpool_3(x2)
        # for i in range(8, 12):
        #     edge_3 = self.pidinet_layers[i](edge_3)
        edge_3_3 = self.p3_ch(edge_3)

        for blk in self.block_3:
            x3=blk(x3)
        #x3=x3+x3_t
        x3=torch.add(edge_3_3,x3)
        #edge_2=self.maxpool_2(edge_1)
        #edge_2_2=edge_2
        #edge_2_2=self.Conv1_2(edge_2)
        x4 = self.encoder3(x3)
        edge_4=self.maxpool_4(edge_3)
        # for i in range(12, 16):
        #     edge_4 = self.pidinet_layers[i](edge_4)


        if self.attention:
            x0_,x2_,x3_,x4_=self.DCA([x0,x2,x3,x4])
        # x0_=self.dep_1(x0)
        # x2_=self.dep_2(x2)
        # x3_=self.dep_3(x3)
        #x4_=self.dep_4(x4)
            #x0_, x2_, x3_, x4_=x0,x2,x3,x4,
          #x2_, x3_, x4_ = self.DCA([ x2, x3, x4])#prevent
           #x0_, x2_= self.DCA([x0, x2])
        #x4 = torch.add(edge_2_2, x4)
        down_pad = False
        right_pad = False
        if x4.size()[2] % 2 == 1:
            x4 = F.pad(x4, (0, 0, 0, 1))
            x4_= F.pad(x4_, (0, 0, 0, 1))
            edge_4 = F.pad(edge_4, (0, 0, 0, 1))
            down_pad = True
        if x4.size()[3] % 2 == 1:
            x4 = F.pad(x4, (0, 1, 0, 0))
            x4_ = F.pad(x4_, (0, 1, 0, 0))
            edge_4 = F.pad(edge_4, (0, 1, 0, 0))
            right_pad = True
        # x4_ = self.dep_4(x4)
        #x4_t=x4
        for blk in self.block_4:
            x4=blk(x4)
        #x4=x4_t+x4
        edge_4_4 = self.p4_ch(edge_4)
        x4=torch.add(edge_4_4,x4)
        # #x4=[n,1024,19,19]

        edge_5 = self.maxpool_1(edge_4)
        edge_5_5 = self.p5_ch(edge_5)





        #edge_3_3=self.Conv1_3(edge_3)
        x5 = self.encoder4(x4)
        # #x5=[2,2048,10,10]
        # #x4=[2,1024,20,20]
        x5=torch.add(edge_5_5,x5)

        # decoding + concat path
        #d5_thick = self.Up5_thick(x5)
        d5_thick = self.Up5_thick_sameple(x5)
##x4=[n,512,]
        d5_thick = torch.cat((x4_, d5_thick), dim=1)

        # Decoder
        if down_pad and (not right_pad):
            d5_thick = d5_thick[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5_thick = d5_thick[:, :, :, :-1]
        if down_pad and right_pad:
            d5_thick = d5_thick[:, :, :-1, :-1]

        d5_thick = self.Up_conv5_thick(d5_thick)

        #d4_thick = self.Up4_thick(d5_thick)
        d4_thick = self.Up4_thick_sameple(d5_thick)
        #x3=[n,]
        d4_thick = torch.cat((x3_, d4_thick), dim=1)
        d4_thick_= self.Up_conv4_thick(d4_thick)

        #d3_thick = self.Up3_thick(d4_thick)
        d3_thick = self.Up3_thick_sameple(d4_thick_)
        #d3_thick = self.Up3_thick_sameple(x3)
        d3_thick = torch.cat((x2_, d3_thick), dim=1)

        if down_pad_2 and (not right_pad_2):
            d3_thick = d3_thick[:, :, :-1, :]
        if (not down_pad_2) and right_pad_2:
            d3_thick = d3_thick[:, :, :, :-1]
        if down_pad_2 and right_pad_2:
            d3_thick = d3_thick[:, :, :-1, :-1]
        d3_thick_ = self.Up_conv3_thick(d3_thick)

        #d2_thick = self.Up2_thick(d3_thick)
        d2_thick = self.Up2_thick_sameple(d3_thick_)
        #d2_thick = self.Up2_thick_sameple(x2)
        #d2_thick = self.Up2_thick_sameple(d4_thick)
        d2_thick = torch.cat((x0_, d2_thick), dim=1)   #rose
        #d2_thick = torch.cat((x0, d2_thick), dim=1)#prevent
        d2_thick_ = self.Up_conv2_thick(d2_thick)

        #d1_thick = self.Up1_thick(d2_thick)
        d1_thick = self.Up1_thick_sameple(d2_thick_)
        #d1_thick = torch.cat((x, d1_thick), dim=1)
        if down_pad_1 and (not right_pad_1):
            d1_thick = d1_thick[:, :, :-1, :]
        if (not down_pad_1) and right_pad_1:
            d1_thick = d1_thick[:, :, :, :-1]
        if down_pad_1 and right_pad_1:
            d1_thick = d1_thick[:, :, :-1, :-1]
        d1_thick_1 = self.Up_conv1_thick(d1_thick)

        d1_thick = self.Conv_1x1_thick(d1_thick_1)
        out_thick = nn.Sigmoid()(d1_thick)
        # out_thick_1=self.fuse(d2_thick_,d3_thick_,d4_thick_)
        # assert out_thick.size() == out_thick_1.size()
        # out_thick=torch.max(out_thick_1,out_thick)

        #
        # x0_thin = self.firstconv(x)
        # x0_thin = self.firstbn(x0_thin)
        # x0_thin = self.firstrelu(x0_thin)
        # x1_thin = self.firstmaxpool(x0_thin)
        #
        #
        # #x1=self.block_1(x1)
        # #x1n,64,152,152]
        # x2_thin = self.encoder1(x1_thin)
        # #x2=[n,256,76,76]
        # #x2=self.block_2(x2)
        # x3_thin = self.encoder2(x2_thin)
        # #x3=[n,512,38,38]
        # #x3=self.block_3(x3)
        # x4_thin = self.encoder3(x3_thin)
        # #x4=[n,1024,19,19]
        # if self.attention:
        #    x0_thin,x2_thin,x3_thin,x4_thin=self.DCA([x0_thin,x2_thin,x3_thin,x4_thin])
        # down_pad = False
        # right_pad = False
        # if x4_thin.size()[2] % 2 == 1:
        #     x4_thin = F.pad(x4_thin, (0, 0, 0, 1))
        #     down_pad = True
        # if x4_thin.size()[3] % 2 == 1:
        #     x4_thin= F.pad(x4_thin, (0, 1, 0, 0))
        #     right_pad = True
        # #x4 = self.block_4(x4)
        # x5_thin= self.encoder4(x4_thin)
        # #
        # # ###########################
        # d5_thin = self.Up5_thin(x5_thin)
        # d5_thin = torch.cat((x4_thin, d5_thin), dim=1)
        # if down_pad and (not right_pad):
        #     d5_thin = d5_thin[:, :, :-1, :]
        # if (not down_pad) and right_pad:
        #     d5_thin= d5_thin[:, :, :, :-1]
        # if down_pad and right_pad:
        #     d5_thin = d5_thin[:, :, :-1, :-1]
        #
        # d5_thin = self.Up_conv5_thin(d5_thin)
        #
        # d4_thin = self.Up4_thin(d5_thin)
        # d4_thin = torch.cat((x3_thin, d4_thin), dim=1)
        # d4_thin = self.Up_conv4_thin(d4_thin)
        #
        # d3_thin = self.Up3_thin(x3)  # x3
        # d3_thin = torch.cat((x2_,d3_thin), dim=1)
        # d3_thin = self.Up_conv3_thin(d3_thin)
        # x_second=self.fuse(x,out_thick)
        # down_pad_1 = False
        # right_pad_1 = False
        # if  x_second.size()[2] % 2 == 1:
        #     x_second = F.pad( x_second, (0, 0, 0, 1))
        #     down_pad_1 = True
        # if  x_second.size()[3] % 2 == 1:
        #     x_second = F.pad( x_second, (0, 1, 0, 0))
        #     right_pad_1 = True
        # #x0=n,64,152,152]
        # x0_2 = self.secondconv(x_second)
        # x0_2 = self.secondbn(x0_2)
        # x0_2 = self.secondrelu(x0_2)
        # x1_2= self.secondmaxpool(x0_2)
        #
        # x2_1 = self.encoder5(x1_2)
        #x2=[n,256,76,76]



        if down_pad_2 and (not right_pad_2):
            x2 = x2[:, :, :-1, :]
        if (not down_pad_2) and right_pad_2:
            x2 = x2[:, :, :, :-1]
        if down_pad_2 and right_pad_2:
            x2 = x2[:, :, :-1, :-1]


        #d2_thin = self.Up2_thin(x2)  # d3_thin
        d2_thin = self.Up2_thin_sameple(x2)
        #d2_thin = self.Up2_thin_sameple(x2_1)
        #d2_thin = torch.cat((x0_2, d2_thin), dim=1)
        d2_thin = torch.cat((x0, d2_thin), dim=1)
        #d2_thin = torch.cat((x0, d2_thin), dim=1)   #prevent

        d2_thin = self.Up_conv2_thin(d2_thin)


        #d1_thin = self.Up1_thin(d2_thin)
        d1_thin = self.Up1_thin_sameple(d2_thin)
        #d1_thin = torch.cat((x, d1_thin), dim=1)
        if down_pad_1 and (not right_pad_1):
            d1_thin = d1_thin[:, :, :-1, :]
        if (not down_pad_1) and right_pad_1:
            d1_thin = d1_thin[:, :, :, :-1]
        if down_pad_1 and right_pad_1:
            d1_thin= d1_thin[:, :, :-1, :-1]
        d1_thin_1 = self.Up_conv1_thin(d1_thin)

        d1_thin = self.Conv_1x1_thin(d1_thin_1)
        out_thin = nn.Sigmoid()(d1_thin)



        # d1=self.Up1_sameple(x0)
        # d1=self.Up_conv1(d1)
        # d1=self.Conv_1x1(d1)
        # out_temp=nn.Sigmoid()(d1)
        #out_thin=out_thick

        # assert out_thick.size() == out_thin.size()
        # out = torch.max(out_thick, out_thin)
        assert d1_thin_1.size() == d1_thick_1.size()
        out = self.gate_fusion(d1_thin_1, d1_thick_1)
        # P_final=self.Conv_1x1(P_final)
        # out = nn.Sigmoid()(P_final)

        #out_1=torch.max(out_temp,out_1)
        #out=self.fuse(d2_thin,d2_thick_,out_temp)
        #out = self.fuse(d1_thin, d1_thick, out_temp)
        #out=torch.max(out_1,out_2)
        #out=torch.cat((d1_thin,d1_thick),dim=1)
        #out=self.fuse(d1_thin,d1_thick)
        #out = self.fuse(d1_thin, d1_thick)
        #out=nn.Sigmoid()(out)
       # out=out_thick

        """
        d1 = torch.cat([out_thick, out_thin], dim=1)  # d2_thick, d2_thin
        d1 = self.Up_conv1(d1)
        out = self.Conv_1x1(d1)
        out = nn.Sigmoid()(out)
        """


        return out_thick, out_thin,out

class DEP(nn.Module):
    def __init__(self,in_c):
        super(DEP, self).__init__()
        # self.kernel_size=3
        # self.conv1_kernel = create_conv_kernel(in_channels=64,out_channels=64,
        #                                        kernel_size=3, avg=0, std=0.1)  # ##
        # # self.conv2_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        # #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # # self.conv3_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        # #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv4_kernel = create_conv_kernel(in_channels=64, out_channels=32,
        #                                        kernel_size=3, avg=0, std=0.1)
        self.conv5_1 = nn.Conv2d(
            64, 32, kernel_size=(5,1), padding="same", bias=True)
        self.conv1_5 = nn.Conv2d(
            64, 32, kernel_size=(1,5),padding='same', bias=True)
        self.Up5_1= up_conv(ch_in=in_c, ch_out=64)
        self.down5_1 = nn.MaxPool2d(2)
        self.Up1_5= up_conv(ch_in=in_c, ch_out=64)
        self.down1_5 = nn.MaxPool2d(2)
        self.conv7_7 = nn.Conv2d(
            in_c, 32, kernel_size=(5,5), padding='same', bias=True)
        self.conv1 = nn.Conv2d(
            96, 96, kernel_size=(1,1), padding='same', bias=True)
        self.conv3 = nn.Conv2d(
            96, 96, kernel_size=(3,3), padding='same', bias=True)

        self.conv3_3 = nn.Conv2d(
            96, 32, kernel_size=(3,3), padding='same', bias=True)

        #self.conv3_3=

        self.conv3_1 = nn.Conv2d(
            64, 48, kernel_size=(3,1), padding="same", bias=True)
        self.conv1_3 = nn.Conv2d(
            64, 48, kernel_size=(1,3), padding='same', bias=True)
        self.Up3_1= up_conv(ch_in=48, ch_out=16)
        self.down3_1 = nn.MaxPool2d(2)
        self.Up1_3= up_conv(ch_in=48, ch_out=16)
        self.down1_3 = nn.MaxPool2d(2)

        self.conv1_1=nn.Conv2d(
            32, in_c, kernel_size=(1,1), padding="same", bias=True)



        self.relu4 = nn.ReLU(inplace=True)
        self.conv77_2 = nn.Conv2d(
            8, 16, kernel_size=7, padding='same', bias=False, dilation=2)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            80, 1, kernel_size=1, padding=0, bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.bn1=nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)


    def forward(self, x):
        #x1=self.conv1(x1)
        x_5_1=self.Up5_1(x)
        x_5_1=self.conv5_1(x_5_1)#F.conv2d( x_5_1, self.conv1_kernel, padding=self.kernel_size // 2)
        x_5_1=self.down5_1(x_5_1)

        x_1_5=self.Up1_5(x)
        x_1_5=self.conv1_5(x_1_5)
        x_1_5=self.down1_5(x_1_5)


        #x=self.conv7_7(x)
        #x_temp=torch.cat((x_5_1,x),dim=1)
        #x = torch.cat((x_temp, x_1_5), dim=1)#[3,96]
        x=torch.cat((x_5_1,x_1_5),dim=1)
        # x=self.bn(x_5_1)
        # x=self.relu(x)


        # x=self.conv1(x)
        # x=self.conv3(x)


        x_3_1=self.down3_1(x)
        x_3_1=self.conv3_1(x_3_1)#F.conv2d( x_3_1, self.conv4_kernel, padding=self.kernel_size // 2)
        x_3_1=self.Up3_1(x_3_1)

        x_1_3=self.down1_3(x)
        x_1_3=self.conv1_3(x_1_3)
        x_1_3=self.Up1_3(x_1_3)

       # x=self.conv3_3(x)

        x=torch.cat((x_3_1,x_1_3),dim=1)
        #x = torch.cat((x_temp, x_1_3), dim=1)
        # x=self.bn1(x_3_1)
        # x=self.relu1(x)


        #x = torch.cat((x_3_1, x_1_3), dim=1)


        x=self.conv1_1(x)


        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='pl', groups=8, dyscope=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
            #print(out_channels)
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
        self.conv=nn.Conv2d(in_channels, in_channels//2, 1)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        #print(x_.shape)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        #print(offset.shape)
        return self.sample(x, offset)

    def forward(self, x):

        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class up_conv_sameple(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_sameple, self).__init__()
        self.up = DySample(ch_in)
        #self.cov=nn.Conv2d(ch_in, ch_out, 1)
        self.up_2 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding="same", bias=True,dilation=2),
            #nn.Dropout(0.2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            #nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.up(x)
        x=self.up_2(x)

        return x

class EdgeConv(nn.Module):
    def __init__(self, ch_in, ch_out,d):
        super(EdgeConv, self).__init__()
        #self.up = DySample(ch_in)
        #self.cov=nn.Conv2d(ch_in, ch_out, 1)
        self.conv = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            ##nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding="same", bias=True,dilation=d),
            #nn.Dropout(0.2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            #nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv_sameple_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_sameple_1, self).__init__()
        self.up = DySample(ch_in)
        self.cov=nn.Conv2d(ch_in, ch_out, 1)
        # self.up_2 = nn.Sequential(
        #     # nn.Upsample(scale_factor=2),
        #     nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True)
        #     #nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        # )
    def forward(self, x):
        x = self.up(x)
        x=self.cov(x)

        return x

class feature_fuse(nn.Module):
    def __init__(self, in_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, 64, kernel_size=1, padding=0, bias=False,dilation=1)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            3, 1, kernel_size=1, padding=0, bias=False)

        self.conv33 = nn.Conv2d(
            in_c, 32, kernel_size=3, padding=1, bias=False,dilation=1)
        #self.relu2 = nn.ReLU(inplace=True)
        self.conv33_di2 = nn.Conv2d(
            32,16 , kernel_size=3, padding='same', bias=False, dilation=2)
        self.conv33_di3 = nn.Conv2d(
            32, 16, kernel_size=3, padding='same', bias=False, dilation=3)
        self.conv33_di4 = nn.Conv2d(
            32, 32, kernel_size=3, padding='same', bias=False, dilation=4)
        #self.relu3 = nn.ReLU(inplace=True)
        #self.norm = nn.BatchNorm2d(out_c)
        # self.up_1=nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        # self.up_2 = nn.ConvTranspose2d(256, 1, kernel_size=4, stride=4)
        # self.up_3 = nn.ConvTranspose2d(512, 1, kernel_size=8, stride=8)
        self.conv55_1 = nn.Conv2d(
            in_c, 32, kernel_size=5, padding='same', bias=False,dilation=1)
        #self.relu2 = nn.ReLU(inplace=True)
        self.conv77_1 = nn.Conv2d(
            in_c, 32, kernel_size=7, padding='same', bias=False, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.Up1 = up_conv(ch_in=64, ch_out=64)
        #self.Up1_thin_sameple = up_conv_sameple(ch_in=64, ch_out=64)
        self.Up_conv1 = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, 1, kernel_size=1)
        # self.conv55_2 = nn.Conv2d(
        #     8, 16, kernel_size=5, padding='same', bias=False,dilation=2)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.conv77_2 = nn.Conv2d(
        #     8, 16, kernel_size=7, padding='same', bias=False, dilation=2)
        # self.relu5 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     80, 1, kernel_size=1, padding=0, bias=False)


    def forward(self, x1,x2,x3):
        #x1=self.conv1(x1)
        x=torch.cat((x1,x2,x3),dim=1)
        x1 = self.conv11(x)
        #x1=self.relu1(x1)

        x2_1 = self.conv33(x)
        #x2_1 = self.relu2(x2)
        x2_2=self.conv55_1(x)
        #x2_2=self.relu2(x2_2)
        x2_3=self.conv77_1(x)
        #x2_3=self.relu3(x2_3)
        x3_1=self.conv33_di2(x2_1)
        x3_2=self.conv33_di3(x2_2)
        x3_3=self.conv33_di4(x2_3)
        x4=torch.cat((x3_1,x3_2,x3_3),dim=1)
        x_fusion=torch.add(x4,x1)
        #out=self.Up1(x_fusion)
        out=self.Up_conv1(x_fusion)
        out=self.Conv_1x1(out)
        out=nn.Sigmoid()(out)
        # x3_1= self.conv33_di(x)
        # x3_2=self.conv55_2(x3_1)
        # x3_2 = self.relu4(x3_2)
        # x3_3=self.conv77_2(x3_1)
        # x3_3 = self.relu5(x3_3)
       # out = self.norm(x1+x2+x3)
        #out=nn.Sigmoid()(out)
        return out


class DeformAlign(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 偏移量预测网络（两层卷积）
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * 3 * 3, 3, padding=1)  # 输出2*K*K偏移量（K=3）
        )
        # 可变形卷积参数
        self.weight = nn.Parameter(torch.randn(64, in_channels, 3, 3))  # 假设输出通道64

    def forward(self, x, ref_feature):
        # 输入x: 待对齐特征 (B,C,H,W)
        # ref_feature: 参考特征（如IVC图像）
        # 预测偏移量 (B,2*K*K,H,W)
        offsets = self.offset_conv(ref_feature)
        # 执行可变形卷积
        aligned_feature = deform_conv2d(
            x, offsets, self.weight, padding=1
        )
        return aligned_feature


# -------------------------- 多维度噪声估计 --------------------------
class NoiseEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel算子计算梯度
        self.sobel_x = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.sobel_y = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = sobel_kernel_x.T
        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3)

    def forward(self, ivc_img):
        # 输入ivc_img: (B,1,H,W)
        # 梯度幅值计算
        grad_x = self.sobel_x(ivc_img)
        grad_y = self.sobel_y(ivc_img)
        G = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (B,1,H,W)

        # 梯度方向一致性
        theta = torch.atan2(grad_y, grad_x)  # (B,1,H,W)
        theta_pad = F.pad(theta, (1, 1, 1, 1), mode='replicate')  # 边界填充
        cos_sum = 0
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1: continue  # 跳过中心
                theta_neighbor = theta_pad[:, :, i:i + theta.size(2), j:j + theta.size(3)]
                cos_sum += torch.cos(theta - theta_neighbor)
        C = cos_sum / 8  # (B,1,H,W)

        # 局部方差计算
        unfold = nn.Unfold(kernel_size=5, padding=2)
        patches = unfold(ivc_img).view(ivc_img.size(0), 25, ivc_img.size(2), ivc_img.size(3))
        #patches = unfold(ivc_img)  # (B,5*5,H,W)
        #patches = patches.view(patches.size(0), 25, -1)  # (B,25,H*W)
        mu = patches.mean(dim=1, keepdim=True)  # (B,1,H*W)
       # V = ((patches - mu) ** 2).mean(dim=1).view_as(ivc_img)  # (B,1,H,W)
        V = ((patches - mu) ** 2).mean(dim=1, keepdim=True)

        return torch.cat([G, C, V], dim=1)  # (B,3,H,W)


# -------------------------- 噪声感知FFA --------------------------
class FAM(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 可变形对齐模块
        self.deform_align_svc = DeformAlign(in_channels)
        self.deform_align_dvc = DeformAlign(in_channels)

        # 噪声估计模块
        self.noise_estimator = NoiseEstimation()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 方向敏感卷积
        self.orient_convs = nn.ModuleList([
            self._make_orient_conv(angle) for angle in [0, 45, 90, 135]
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 * 4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def _make_orient_conv(self, angle):
        # 创建旋转后的卷积核（示例实现，需根据实际调整）
        conv = nn.Conv2d(128, 64, 3, padding=1)
        if angle != 0:  # 实际应通过旋转核权重实现
            conv.weight.data = torch.rot90(conv.weight.data, k=angle // 90, dims=[2, 3])
        return conv

    def forward(self, svc_feat, dvc_feat, ivc_img):
        # Step1: 可变形对齐
        svc_aligned = self.deform_align_svc(svc_feat, ivc_img)
        dvc_aligned = self.deform_align_dvc(dvc_feat, ivc_img)

        # # Step2: 噪声估计
        # noise_features = self.noise_estimator(ivc_img)  # (B,3,H,W)
        # mask = self.mask_generator(noise_features)  # (B,1,H,W)

        # # Step3: 噪声抑制
        # svc_clean = svc_feat * mask
        # dvc_clean = dvc_feat * mask
        # Step4: 方向敏感卷积
        concat_feat = torch.cat([svc_aligned, dvc_aligned], dim=1)  # (B,2,H,W)
        orient_outputs = []
        for conv in self.orient_convs:
            orient_outputs.append(conv(concat_feat))
        fused_feat = torch.cat(orient_outputs, dim=1)  # (B,64*4,H,W)
        #fused_feat=torch.cat((), dim=1)

        # Step5: 残差细化
        output = self.final_conv(fused_feat) + concat_feat.sum(dim=1, keepdim=True)
        #output = concat_feat.sum(dim=1, keepdim=True)
        output=torch.sigmoid(output)
        return  torch.cat((output, output,output), dim=1)
        #return torch.cat((output, output,output), dim=1)


#######################################################################
class DDU_Net_final(nn.Module):
    def __init__(self, num_heads, window_size, n=1,
                 img_ch=3,
                 output_ch=1,
                 attention=False,
                 patch_size=16,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=[4, 4, 4, 4],
                 channel_head_dim=[1, 1, 1, 1], input_size=(304, 304),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(DDU_Net_final, self).__init__()
        filters = [64, 128, 256, 512]

        self.block_4 = nn.ModuleList([KanSwinTransformerBlock(dim=1024, input_resolution=(20, 20),
                                                           num_heads=32, window_size=4,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                           drop=drop, attn_drop=attn_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                           norm_layer=norm_layer) for i in range(2)])
        self.block_3 = nn.ModuleList([KanSwinTransformerBlock(dim=512, input_resolution=(38, 38),
                                                           num_heads=16, window_size=19,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                           drop=drop, attn_drop=attn_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                           norm_layer=norm_layer) for i in range(2)])
        self.block_2 = nn.ModuleList([KanSwinTransformerBlock(dim=256, input_resolution=(76, 76),
                                                           num_heads=8, window_size=4,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                           drop=drop, attn_drop=attn_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                           norm_layer=norm_layer) for i in range(2)])
        self.block_1 = nn.ModuleList([KanSwinTransformerBlock(dim=64, input_resolution=(76, 76),
                                                           num_heads=2, window_size=4,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                           drop=drop, attn_drop=attn_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                           norm_layer=norm_layer) for i in range(2)])

        resnet = resnest.resnest50(pretrained=True)
        self.attention = attention
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Up5_thick = up_conv(ch_in=2048, ch_out=1024)
        self.Up5_thick_sameple = up_conv_sameple_1(ch_in=2048, ch_out=1024)
        self.Up_conv5_thick = res_conv_block(ch_in=2048, ch_out=1024)

        self.Up4_thick = up_conv(ch_in=1024, ch_out=512)
        self.Up4_thick_sameple = up_conv_sameple_1(ch_in=1024, ch_out=512)
        self.Up_conv4_thick = res_conv_block(ch_in=1024, ch_out=512)

        self.Up3_thick = up_conv(ch_in=512, ch_out=256)
        self.Up3_thick_sameple = up_conv_sameple_1(ch_in=512, ch_out=256)
        self.Up_conv3_thick = res_conv_block(ch_in=512, ch_out=256)

        self.Up2_thick = up_conv(ch_in=256, ch_out=64)
        self.Up2_thick_sameple = up_conv_sameple_1(ch_in=256, ch_out=64)
        self.Up_conv2_thick = res_conv_block(ch_in=128, ch_out=64)

        self.Up1_thick = up_conv(ch_in=64, ch_out=64)
        self.Up1_thick_sameple = up_conv_sameple_1(ch_in=64, ch_out=64)
        self.Up_conv1_thick = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thick = nn.Conv2d(32, output_ch, kernel_size=1)

        # ##
        self.Up5_thin = up_conv(ch_in=2048, ch_out=1024)
        self.Up5_thin_sameple = up_conv_sameple_1(ch_in=2048, ch_out=1024)
        self.Up_conv5_thin = res_conv_block(ch_in=2048, ch_out=1024)

        self.Up4_thin = up_conv(ch_in=1024, ch_out=512)
        self.Up4_thin_sameple = up_conv_sameple_1(ch_in=1024, ch_out=512)
        self.Up_conv4_thin = res_conv_block(ch_in=1024, ch_out=512)

        self.Up3_thin = up_conv(ch_in=512, ch_out=256)
        self.Up3_thin_sameple = up_conv_sameple_1(ch_in=512, ch_out=256)
        self.Up_conv3_thin = res_conv_block(ch_in=512, ch_out=256)
        # ##

        self.Up2_thin = up_conv(ch_in=256, ch_out=64)
        self.Up2_thin_sameple = up_conv_sameple_1(ch_in=256, ch_out=64)
        self.Up_conv2_thin = res_conv_block(ch_in=128, ch_out=64)

        self.Up1_thin = up_conv(ch_in=64, ch_out=64)
        self.Up1_thin_sameple = up_conv_sameple_1(ch_in=64, ch_out=64)
        self.Up_conv1_thin = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thin = nn.Conv2d(32, output_ch, kernel_size=1)

        # ##
        self.Up_conv1 = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1)
        # ##
        self.spatial = spatial_attention(ch_in=2, ch_out=1)
        # self.attention = attention
        self.patch_size = patch_size
        patch = input_size[0] // self.patch_size
        self.DCA = DCA(n=n,
                       features=[64, 256, 512, 1024],
                       strides=[self.patch_size // 2, self.patch_size // 4, self.patch_size // 8,
                                self.patch_size // 16],
                       patch=patch,
                       spatial_att=spatial_att,
                       channel_att=channel_att,
                       spatial_head=spatial_head_dim,
                       channel_head=channel_head_dim,
                       )
        self.agg=base(channels=256,pn_size=5,kernel_size=3, avg=0.0, std=0.1,out_channels=1)
        # self.DF=DF_Module(1,1,False)

        #self.fuse=feature_fuse(2,1)
        #self.agg_1= base(channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1,out_channels=1)
       # self.eff=EFF(2,1)
        self.p2_ch = EdgeConv(64, 256, 1)
        self.p3_ch = EdgeConv(256, 512, 2)
        self.p4_ch = EdgeConv(256, 1024, 3)
        self.p5_ch = EdgeConv(256, 2048, 4)
        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.maxpool_3 = nn.MaxPool2d(2)
        self.maxpool_4 = nn.MaxPool2d(2)
        self.FAM=FAM(in_channels=1)
        self.gate_fusion = DepthAwareGatingWithSpatialAttention(in_channels=32)

    def forward(self, x,thick,DVC):
        x_=self.FAM(thick,DVC,x)
        # encoding path
        # x = self.dep(x)
        # x_=self.agg(x,thick,DVC)
        #input_all = torch.cat((x_, thick,DVC), dim=1)  # [b, 3, h, w] ##
        #assert input_all.size()[1] == 3  # ##
        #x_=self.dep(input_all)
        #x_=self.DF(x,thin)
        #x= torch.cat((x,thin,thick), dim=1)
        #x_=self.fuse(x)
        # x_=self.eff(x,x)

        x0 = self.firstconv(x_)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        # x0=torch.add(x0,edge_1_1)
        # x0=n,64,152,152]

        x1 = self.firstmaxpool(x0)
        ##edge_1=self.avgpool_1(x0)

        for blk in self.block_1:
            x1 = blk(x1)
        # x1=x1+x1
        x2 = self.encoder1(x1)
        # x2=[n,256,76,76]
        # x2_t=x2
        for blk in self.block_2:
            x2 = blk(x2)
        # x2=x2+x2_t
        # x2=torch.add(edge_3_3,x2)
        # edge_2=self.maxpool_2(x2)
        # # for i in range(4, 8):
        # #     edge_2 = self.pidinet_layers[i](edge_2)
        # edge_2_2=self.p2_ch(edge_2)

        # x2=torch.add(edge_2_2,x2)
        down_pad_2 = False
        right_pad_2 = False
        if x2.size()[2] % 2 == 1:
            x2 = F.pad(x2, (0, 0, 0, 1))
            down_pad_2 = True
        if x2.size()[3] % 2 == 1:
            x2 = F.pad(x2, (0, 1, 0, 0))
            right_pad_2 = True
        # edge_1 = self.maxpool_1(x)
        ##edge_1_1=self.Conv1_1(edge_1)

        # edge_1_1=edge_1
        x3 = self.encoder2(x2)
        # x3=[n,512,38,38]
        # x3_t = x3

        edge_3 = self.maxpool_3(x2)
        # for i in range(8, 12):
        #     edge_3 = self.pidinet_layers[i](edge_3)
        edge_3_3 = self.p3_ch(edge_3)

        for blk in self.block_3:
            x3 = blk(x3)
        # x3=x3+x3_t
        x3 = torch.add(edge_3_3, x3)
        # edge_2=self.maxpool_2(edge_1)
        # edge_2_2=edge_2
        # edge_2_2=self.Conv1_2(edge_2)
        x4 = self.encoder3(x3)
        edge_4 = self.maxpool_4(edge_3)
        # for i in range(12, 16):
        #     edge_4 = self.pidinet_layers[i](edge_4)

        if self.attention:
            x0_, x2_, x3_, x4_ = self.DCA([x0, x2, x3, x4])
            # x2_, x3_, x4_ = self.DCA([ x2, x3, x4])#prevent
            # x0_, x2_= self.DCA([x0, x2])
        # x4 = torch.add(edge_2_2, x4)
        down_pad = False
        right_pad = False
        if x4.size()[2] % 2 == 1:
            x4 = F.pad(x4, (0, 0, 0, 1))
            x4_ = F.pad(x4_, (0, 0, 0, 1))
            edge_4 = F.pad(edge_4, (0, 0, 0, 1))
            down_pad = True
        if x4.size()[3] % 2 == 1:
            x4 = F.pad(x4, (0, 1, 0, 0))
            x4_ = F.pad(x4_, (0, 1, 0, 0))
            edge_4 = F.pad(edge_4, (0, 1, 0, 0))
            right_pad = True
        # x4_t=x4
        for blk in self.block_4:
            x4 = blk(x4)
        # x4=x4_t+x4
        edge_4_4 = self.p4_ch(edge_4)
        x4 = torch.add(edge_4_4, x4)
        # #x4=[n,1024,19,19]

        edge_5 = self.maxpool_1(edge_4)
        edge_5_5 = self.p5_ch(edge_5)

        # edge_3_3=self.Conv1_3(edge_3)
        x5 = self.encoder4(x4)
        # #x5=[2,2048,10,10]
        # #x4=[2,1024,20,20]
        x5 = torch.add(edge_5_5, x5)

        # decoding + concat path
        # d5_thick = self.Up5_thick(x5)
        d5_thick = self.Up5_thick_sameple(x5)
        ##x4=[n,512,]
        d5_thick = torch.cat((x4_, d5_thick), dim=1)

        # Decoder
        if down_pad and (not right_pad):
            d5_thick = d5_thick[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5_thick = d5_thick[:, :, :, :-1]
        if down_pad and right_pad:
            d5_thick = d5_thick[:, :, :-1, :-1]

        d5_thick = self.Up_conv5_thick(d5_thick)

        # d4_thick = self.Up4_thick(d5_thick)
        d4_thick = self.Up4_thick_sameple(d5_thick)
        # x3=[n,]
        d4_thick = torch.cat((x3_, d4_thick), dim=1)
        d4_thick_ = self.Up_conv4_thick(d4_thick)

        # d3_thick = self.Up3_thick(d4_thick)
        d3_thick = self.Up3_thick_sameple(d4_thick_)
        # d3_thick = self.Up3_thick_sameple(x3)
        d3_thick = torch.cat((x2_, d3_thick), dim=1)

        if down_pad_2 and (not right_pad_2):
            d3_thick = d3_thick[:, :, :-1, :]
        if (not down_pad_2) and right_pad_2:
            d3_thick = d3_thick[:, :, :, :-1]
        if down_pad_2 and right_pad_2:
            d3_thick = d3_thick[:, :, :-1, :-1]
        d3_thick_ = self.Up_conv3_thick(d3_thick)

        # d2_thick = self.Up2_thick(d3_thick)
        d2_thick = self.Up2_thick_sameple(d3_thick_)
        # d2_thick = self.Up2_thick_sameple(x2)
        # d2_thick = self.Up2_thick_sameple(d4_thick)
        d2_thick = torch.cat((x0_, d2_thick), dim=1)  # rose
        # d2_thick = torch.cat((x0, d2_thick), dim=1)#prevent
        d2_thick_ = self.Up_conv2_thick(d2_thick)

        # d1_thick = self.Up1_thick(d2_thick)
        d1_thick = self.Up1_thick_sameple(d2_thick_)
        # d1_thick = torch.cat((x, d1_thick), dim=1)

        d1_thick_1 = self.Up_conv1_thick(d1_thick)

        d1_thick = self.Conv_1x1_thick(d1_thick_1)
        out_thick = nn.Sigmoid()(d1_thick)
        # out_thick_1=self.fuse(d2_thick_,d3_thick_,d4_thick_)
        # assert out_thick.size() == out_thick_1.size()
        # out_thick=torch.max(out_thick_1,out_thick)

        #
        # x0_thin = self.firstconv(x)
        # x0_thin = self.firstbn(x0_thin)
        # x0_thin = self.firstrelu(x0_thin)
        # x1_thin = self.firstmaxpool(x0_thin)
        #
        #
        # #x1=self.block_1(x1)
        # #x1n,64,152,152]
        # x2_thin = self.encoder1(x1_thin)
        # #x2=[n,256,76,76]
        # #x2=self.block_2(x2)
        # x3_thin = self.encoder2(x2_thin)
        # #x3=[n,512,38,38]
        # #x3=self.block_3(x3)
        # x4_thin = self.encoder3(x3_thin)
        # #x4=[n,1024,19,19]
        # if self.attention:
        #    x0_thin,x2_thin,x3_thin,x4_thin=self.DCA([x0_thin,x2_thin,x3_thin,x4_thin])
        # down_pad = False
        # right_pad = False
        # if x4_thin.size()[2] % 2 == 1:
        #     x4_thin = F.pad(x4_thin, (0, 0, 0, 1))
        #     down_pad = True
        # if x4_thin.size()[3] % 2 == 1:
        #     x4_thin= F.pad(x4_thin, (0, 1, 0, 0))
        #     right_pad = True
        # #x4 = self.block_4(x4)
        # x5_thin= self.encoder4(x4_thin)
        # #
        # # ###########################
        # d5_thin = self.Up5_thin(x5_thin)
        # d5_thin = torch.cat((x4_thin, d5_thin), dim=1)
        # if down_pad and (not right_pad):
        #     d5_thin = d5_thin[:, :, :-1, :]
        # if (not down_pad) and right_pad:
        #     d5_thin= d5_thin[:, :, :, :-1]
        # if down_pad and right_pad:
        #     d5_thin = d5_thin[:, :, :-1, :-1]
        #
        # d5_thin = self.Up_conv5_thin(d5_thin)
        #
        # d4_thin = self.Up4_thin(d5_thin)
        # d4_thin = torch.cat((x3_thin, d4_thin), dim=1)
        # d4_thin = self.Up_conv4_thin(d4_thin)
        #
        # d3_thin = self.Up3_thin(x3)  # x3
        # d3_thin = torch.cat((x2_,d3_thin), dim=1)
        # d3_thin = self.Up_conv3_thin(d3_thin)
        # x_second=self.fuse(x,out_thick)
        # down_pad_1 = False
        # right_pad_1 = False
        # if  x_second.size()[2] % 2 == 1:
        #     x_second = F.pad( x_second, (0, 0, 0, 1))
        #     down_pad_1 = True
        # if  x_second.size()[3] % 2 == 1:
        #     x_second = F.pad( x_second, (0, 1, 0, 0))
        #     right_pad_1 = True
        # #x0=n,64,152,152]
        # x0_2 = self.secondconv(x_second)
        # x0_2 = self.secondbn(x0_2)
        # x0_2 = self.secondrelu(x0_2)
        # x1_2= self.secondmaxpool(x0_2)
        #
        # x2_1 = self.encoder5(x1_2)
        # x2=[n,256,76,76]

        if down_pad_2 and (not right_pad_2):
            x2 = x2[:, :, :-1, :]
        if (not down_pad_2) and right_pad_2:
            x2 = x2[:, :, :, :-1]
        if down_pad_2 and right_pad_2:
            x2 = x2[:, :, :-1, :-1]

        # d2_thin = self.Up2_thin(x2)  # d3_thin
        d2_thin = self.Up2_thin_sameple(x2)
        # d2_thin = self.Up2_thin_sameple(x2_1)
        # d2_thin = torch.cat((x0_2, d2_thin), dim=1)
        d2_thin = torch.cat((x0, d2_thin), dim=1)
        # d2_thin = torch.cat((x0, d2_thin), dim=1)   #prevent

        d2_thin = self.Up_conv2_thin(d2_thin)

        # d1_thin = self.Up1_thin(d2_thin)
        d1_thin = self.Up1_thin_sameple(d2_thin)
        # d1_thin = torch.cat((x, d1_thin), dim=1)

        d1_thin_1= self.Up_conv1_thin(d1_thin)

        d1_thin = self.Conv_1x1_thin(d1_thin_1)
        out_thin = nn.Sigmoid()(d1_thin)

        # d1=self.Up1_sameple(x0)
        # d1=self.Up_conv1(d1)
        # d1=self.Conv_1x1(d1)
        # out_temp=nn.Sigmoid()(d1)
        # out_thin=out_thick
        #
        # assert out_thick.size() == out_thin.size()
        # out = torch.max(out_thick, out_thin)
        assert d1_thin_1.size() == d1_thick_1.size()
        out = self.gate_fusion(d1_thin_1, d1_thick_1)
        #out=torch.cat((out,thin), dim=1)
        #out=torch.max(out,thick)
        #out=torch.max(out,thin)
        #out=torch.max(out,thick)
        """
        d1 = torch.cat([out_thick, out_thin], dim=1)  # d2_thick, d2_thin
        d1 = self.Up_conv1(d1)
        out = self.Conv_1x1(d1)
        out = nn.Sigmoid()(out)
        """
        return out_thick, out_thin, out


def create_conv_kernel(in_channels, out_channels, kernel_size=3, avg=0.0, std=0.1):
    # [out_channels, in_channels, kernel_size, kernel_size]
    kernel_arr = np.random.normal(loc=avg, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
    kernel_arr = kernel_arr.astype(np.float32)
    kernel_tensor = torch.from_numpy(kernel_arr)
    kernel_params = nn.Parameter(data=kernel_tensor.contiguous(), requires_grad=True)
    print(kernel_params.type())
    return kernel_params


def create_conv_bias(channels):
    # [channels, ]
    bias_arr = np.zeros(channels, np.float32)
    assert bias_arr.shape[0] % 2 == 1

    bias_arr[bias_arr.shape[0] // 2] = 1.0
    bias_tensor = torch.from_numpy(bias_arr)
    bias_params = nn.Parameter(data=bias_tensor.contiguous(), requires_grad=True)

    return bias_params


class base(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.1, std=0.1,out_channels=3):
        """
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        """
        super(base, self).__init__()
        self.kernel_size = kernel_size

        self.conv1_kernel = create_conv_kernel(in_channels=3,out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)  # ##
        # self.conv2_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv3_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2 * channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv5_kernel = create_conv_kernel(in_channels=2*channels, out_channels=2*channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv6_kernel = create_conv_kernel(in_channels=2*channels, out_channels=2*channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_kernel = create_conv_kernel(in_channels=2 * channels, out_channels=3,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(3)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.bn2 = nn.BatchNorm2d(channels)
        # self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2 * channels)
        # self.bn5 = nn.BatchNorm2d(2*channels)
        # self.bn6 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=False)
        self.conv9_kernel = create_conv_kernel(in_channels=3, out_channels=pn_size * pn_size,
                                               kernel_size=1, avg=avg, std=std)

    def forward(self, input_src, input_thick,DVC):
        input_all = torch.cat((input_src, input_thick,DVC), dim=1)  # [b, 3, h, w] ##
        assert input_all.size()[1] == 3  # ##

        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size // 2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        # fm_2 = F.conv2d(fm_1, self.conv2_kernel, padding=self.kernel_size//2)
        # fm_2 = self.bn2(fm_2)
        # fm_2 = self.relu(fm_2)
        # fm_3 = F.conv2d(fm_2, self.conv3_kernel, padding=self.kernel_size//2)
        # fm_3 = self.bn3(fm_3)
        # fm_3 = self.relu(fm_3)
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size // 2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        # fm_5 = F.conv2d(fm_4, self.conv5_kernel, padding=self.kernel_size//2)
        # fm_5 = self.bn5(fm_5)
        # fm_5 = self.relu(fm_5)
        # fm_6 = F.conv2d(fm_5, self.conv6_kernel, padding=self.kernel_size//2)
        # fm_6 = self.bn6(fm_6)
        # fm_6 = self.relu(fm_6)
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size // 2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)
        # input_all=F.conv2d(input_all,self.conv9_kernel)
        # fm_7=fm_7+input_all
        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]

