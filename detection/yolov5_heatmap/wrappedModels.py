import os
import random
import time
from pathlib import Path
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm

from models.yolo import Model
from torch import nn
from utils.torch_utils import intersect_dicts, scale_img


def resize_like(x, reference, mode='nearest'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=='bilinear':
            x = F.interpolate(x, size=reference.shape[2:],mode='bilinear',align_corners=False)
        if mode=='nearest':
            x = F.interpolate(x, size=reference.shape[2:],mode='nearest')
    return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()

        #channel squeeze excite
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, 1),
            nn.Sigmoid(),
        )

        #spatial squeeze excite
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel(x) + x * self.spatial(x)
        return x


# Batch Normalization with Enhanced Linear Transformation
class EnBatchNorm2d(nn.Module):
    def __init__(self, in_channel, k=3, eps=1e-5):
        super(EnBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel, eps=1e-5,affine=True)
        self.conv = nn.Conv2d(in_channel, in_channel,
                              kernel_size=k,
                              padding=(k - 1) // 2,
                              groups=in_channel,
                              bias=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class ResDecode(nn.Module):
    def __init__( self, in_channel, out_channel ):
        super().__init__()
        self.attent1 = SqueezeExcite(in_channel)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,bias=False),
            EnBatchNorm2d(out_channel), #nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attent2 = SqueezeExcite(out_channel)

    def forward(self, x):

        x = torch.cat(x, 1)
        x = self.attent1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attent2(x)
        return x

class SingleV5(nn.Module):
    def __init__(self, cfg, num_classes=4, pretrained=None, device='cpu'):
        super().__init__()
        self.model = Model(cfg, ch=3, nc=num_classes)
        if pretrained:
            # weights = torch.load(pretrained)
            # self.model.load_state_dict(weights)
            exclude = []  # exclude keys
            ckpt = torch.load(pretrained, map_location=device)  # load checkpoint
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            print('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), pretrained))  # report

            del ckpt, state_dict


    def forward(self, x, augment=False):
        y, dt = [], []  # outputs
        for index, m in enumerate(self.model.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.model.save else None)  # save output

        return x

class V5Centernet(nn.Module):
    def __init__(self, cfg, num_classes=14, pretrained=None, device='cpu'):
        super().__init__()
        self.model = Model(cfg, ch=3, nc=num_classes)
        if pretrained:
            # weights = torch.load(pretrained)
            # self.model.load_state_dict(weights)
            exclude = []  # exclude keys
            ckpt = torch.load(pretrained, map_location=device)  # load checkpoint
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            print('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), pretrained))  # report

            del ckpt, state_dict

        #upsampling's head
        self.center = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ).to(device)

        self.decode1 = ResDecode(384 + 512, 256).to(device) #layer11 9
        self.decode2 = ResDecode(192 + 256, 128).to(device) #layer8 6
        self.decode3 = ResDecode(96 + 128, 64).to(device) #layer6 4
        self.decode4 = ResDecode(48 + 64, 32).to(device) #layer3 2
        self.decode5 = ResDecode(32, 16).to(device)  #layer2 0
        self.logit = nn.Conv2d(16, 1, kernel_size=3, padding=1) #segmentation output

        # self.logit = nn.Sequential(
        #             nn.Conv2d(16, 256,
        #                       kernel_size=3, padding=1, bias=True),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(256, 1,
        #                       kernel_size=1, stride=1,
        #                       padding=0, bias=True))
        # self.logit[-1].bias.data.fill_(-2.19)


        #~upsampling's head


    def forward(self, x, augment=False):
        y, dt = [], []  # outputs

        ipt = x.clone()
        skip = []

        for index, m in enumerate(self.model.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.model.save else None)  # save output

            # if not isinstance(x, list):
            #     print(index, x.shape)

            if index in [0,2,4,6,9]:
                skip.append(x)

            if index==9:
                z = self.center(skip[-1])
                # print('z shape: ', z.shape)
                z = self.decode1([skip[-2], resize_like(z, skip[-2])])  # ; print('d1',x.size())
                # print('z shape: ', z.shape)
                z = self.decode2([skip[-3], resize_like(z, skip[-3])])  # ; print('d2',x.size())
                # print('z shape: ', z.shape)
                z = self.decode3([skip[-4], resize_like(z, skip[-4])])  # ; print('d3',x.size())
                # print('z shape: ', z.shape)
                z = self.decode4([skip[-5], resize_like(z, skip[-5])])  # ; print('d4',x.size())
                # print('z shape: ', z.shape)
                z = self.decode5([resize_like(z, ipt)])
                # print('z shape: ', z.shape)

                seg_logit = self.logit(z)
                # print('z shape: ', seg_logit.shape)

        return x, seg_logit


class BlendV5xm(nn.Module):
    def __init__(self, cfg, num_classes=14, pretrained=None, device='cuda:0'):
        super().__init__()
        if pretrained:
            ckpt = torch.load(pretrained, map_location='cpu')  # load checkpoint
            self.model = Model(cfg or ckpt['model'].yaml, ch=3, nc=num_classes) 
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            print('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), pretrained))  # report
        else:
            self.model = Model(cfg, ch=3, nc=num_classes)

        if 1:
            pretrained1 = 'pretrained/yolov5m.pt'
            cfg1 = 'models/yolov5m.yaml'
            ckpt = torch.load(pretrained1, map_location='cpu')  # load checkpoint
            self.model1 = Model(cfg1 or ckpt['model'].yaml, ch=3, nc=num_classes) 
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model1.state_dict(), exclude=[])  # intersect
            self.model1.load_state_dict(state_dict, strict=False)  # load
            print('Transferred %g/%g items from %s' % (len(state_dict), len(self.model1.state_dict()), pretrained1))  # report
        else:
            self.model1 = Model(cfg1, ch=3, nc=num_classes)

        yolov5x_channel_list = [80, 160, 160, 320, 320, 640, 640, 1280, 1280, 1280, 640, 0, 1280, 640, 320, 0, 640, 320, 0, 640, 640, 0, 1280]
        yolov5m_channel_list = [48, 96, 96, 192, 192, 384, 384, 768, 768, 768, 384, 0, 768, 384, 192, 0, 384, 192, 0, 384, 384, 0, 768]
        self.skip = nn.Identity()
        self.trans = []
        for c1, c2 in zip(yolov5x_channel_list, yolov5m_channel_list):
            if c1*c2>0:
                self.trans.append(nn.Conv2d(c1+c2, c1, 1).to(device))
            else:
                self.trans.append(self.skip)

        # for t in self.trans:
        #     t.half()

    def forward(self, x, augment=False):
        # x = self.pre_conv(x)
        feats = []
        y = []
        y1 = []  # outputs
        x1 = x.clone()
        for index, (m, m1) in enumerate(zip(self.model.model,self.model1.model)):
            if index==17 or index==20 or index==23:
                # global_cls_out = x
                # print('x.shape', x.shape)
                feats.append(x1)

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if m1.f != -1:  # if not from previous layer
                x1 = y1[m1.f] if isinstance(m1.f, int) else [x1 if j == -1 else y1[j] for j in m1.f]  # from earlier layers

            if not isinstance(x, list):
                if index>0:
                    x = torch.cat([x, x1], 1)
                    x = self.trans[index-1](x)
                # print('m', index, x.shape)

            # if not isinstance(x1, list):
                # print('m1', index, x1.shape)

            x = m(x)  # run
            x1 = m1(x1)  # run

            y.append(x if m.i in self.model.save else None)  # save output
            y1.append(x1 if m1.i in self.model1.save else None)  # save output

        return x
