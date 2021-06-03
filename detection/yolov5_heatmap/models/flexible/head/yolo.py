import torch
import torch.nn as nn
from utils.autoanchor import check_anchor_order
import torch.nn.functional as F

class YOLOHead(nn.Module):
    stride = None
    export = False

    def __init__(self, nc, anchors=None, ch=(256, 512, 1024), stride=[8., 16., 32.]):  # detection layer
        super(YOLOHead, self).__init__()
        if anchors is None:
            anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        else:
            anchors = anchors
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.stride = torch.tensor(stride)
        self.anchors /= self.stride.view(-1, 1, 1)
        check_anchor_order(self)

    def forward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



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

class DecoderHead(nn.Module):
    def __init__(self, channel_list, out_channel=1):  # detection layer
        super(DecoderHead, self).__init__()
        '''
        channel list [768, 384, 192, 96, 48]
        '''
        self.center = nn.Sequential(
            nn.Conv2d(channel_list[-1], 512, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ) #.to(device)

        self.decode1 = ResDecode(channel_list[-2] + 512, 256) #.to(device) #layer11 9
        self.decode2 = ResDecode(channel_list[-3] + 256, 128) #.to(device) #layer8 6
        self.decode3 = ResDecode(channel_list[-4] + 128, 64) #.to(device) #layer6 4
        self.decode4 = ResDecode(channel_list[-5] + 64, 32) #.to(device) #layer3 2
        self.decode5 = ResDecode(32, 16) #.to(device)  #layer2 0
        self.logit = nn.Conv2d(16, out_channel, kernel_size=3, padding=1) #segmentation output

    def forward(self, x):
        #x [input, down1, down2,..., down5]
        z = self.center(x[-1])
        # print('z shape: ', z.shape)
        z = self.decode1([x[-2], resize_like(z, x[-2])])  # ; print('d1',x.size())
        # print('z shape: ', z.shape)
        z = self.decode2([x[-3], resize_like(z, x[-3])])  # ; print('d2',x.size())
        # print('z shape: ', z.shape)
        z = self.decode3([x[-4], resize_like(z, x[-4])])  # ; print('d3',x.size())
        # print('z shape: ', z.shape)
        z = self.decode4([x[-5], resize_like(z, x[-5])])  # ; print('d4',x.size())
        # print('z shape: ', z.shape)
        z = self.decode5([resize_like(z, x[-6])])
        # print('z shape: ', z.shape)

        seg_logit = self.logit(z)

        return seg_logit