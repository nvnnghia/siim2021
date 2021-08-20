from torch import nn
import torch
import torch.nn.functional as F
import timm
from torch.nn.parameter import Parameter



def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


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
        self.bn = nn.BatchNorm2d(in_channel, eps=1e-5,affine=False)
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


class R200DUnet(nn.Module):
    def __init__(self, n_class=4, out_dim=11, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super(R200DUnet, self).__init__()
        self.backbone = timm.create_model('resnet200d_320', pretrained=True)

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

        n_features = self.backbone.fc.in_features
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)


        # self.center = nn.Sequential(
        #     nn.Conv2d(2048, 2048, kernel_size=11, padding=5, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.decode1 = ResDecode(1024 + 2048, 1024)
        # self.decode2 = ResDecode(512 + 1024, 512)
        # self.decode3 = ResDecode(256 + 512, 256)
        # self.decode4 = ResDecode(64 + 256, 64)
        # self.decode5 = ResDecode(64, 32)


        self.center = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decode1 = ResDecode(1024 + 512, 256)
        self.decode2 = ResDecode(512 + 256, 128)
        self.decode3 = ResDecode(256 + 128, 64)
        self.decode4 = ResDecode(64 + 64, 32)
        self.decode5 = ResDecode(32, 16)



        self.logit = nn.Conv2d(16, n_class, kernel_size=3, padding=1)

    def layer0(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        return x

    def layer1(self, x):
        return self.backbone.layer1(x)

    def layer2(self, x):
        return self.backbone.layer2(x)

    def layer3(self, x):
        return self.backbone.layer3(x)

    def layer4(self, x):
        return self.backbone.layer4(x)

    def forward(self, ipt):
        bs = ipt.size()[0]

        x0 = self.layer0(ipt)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        skip = [x0, x1, x2, x3]
        z = self.center(x4)
        z = self.decode1([skip[-1], resize_like(z, skip[-1])])  # ; print('d1',x.size())
        z = self.decode2([skip[-2], resize_like(z, skip[-2])])  # ; print('d2',x.size())
        z = self.decode3([skip[-3], resize_like(z, skip[-3])])  # ; print('d3',x.size())
        z = self.decode4([skip[-4], resize_like(z, skip[-4])])  # ; print('d4',x.size())
        z = self.decode5([resize_like(z, ipt)])

        seg_logit = self.logit(z)

        pooled_features = self.pooling(x4).view(bs, -1)
        cls_logit = self.fc(self.dropout(pooled_features))

        return seg_logit, cls_logit

    def infer(self, ipt):
        bs = ipt.size()[0]

        x0 = self.layer0(ipt)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        pooled_features = self.pooling(x4).view(bs, -1)
        cls_logit = self.fc(self.dropout(pooled_features))

        return cls_logit



class R200DUnetS(nn.Module):
    def __init__(self, n_class=4, out_dim=11, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super(R200DUnetS, self).__init__()
        self.model = timm.create_model('resnet200d', pretrained=True)

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

        n_features = self.model.fc.in_features
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)


        # self.center = nn.Sequential(
        #     nn.Conv2d(2048, 2048, kernel_size=11, padding=5, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.decode1 = ResDecode(1024 + 2048, 1024)
        # self.decode2 = ResDecode(512 + 1024, 512)
        # self.decode3 = ResDecode(256 + 512, 256)
        # self.decode4 = ResDecode(64 + 256, 64)
        # self.decode5 = ResDecode(64, 32)


        self.center = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decode1 = ResDecode(1024 + 64, 64)
        self.decode2 = ResDecode(512 + 64, 32)
        self.decode3 = ResDecode(256 + 32, 32)
        self.decode4 = ResDecode(64 + 32, 16)
        self.decode5 = ResDecode(16, 16)



        self.logit = nn.Conv2d(16, n_class, kernel_size=3, padding=1)

    def layer0(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        return x

    def layer1(self, x):
        return self.model.layer1(x)

    def layer2(self, x):
        return self.model.layer2(x)

    def layer3(self, x):
        return self.model.layer3(x)

    def layer4(self, x):
        return self.model.layer4(x)

    def forward(self, ipt):
        bs = ipt.size()[0]

        x0 = self.layer0(ipt)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        skip = [x0, x1, x2, x3]
        z = self.center(x4)
        z = self.decode1([skip[-1], resize_like(z, skip[-1])])  # ; print('d1',x.size())
        z = self.decode2([skip[-2], resize_like(z, skip[-2])])  # ; print('d2',x.size())
        z = self.decode3([skip[-3], resize_like(z, skip[-3])])  # ; print('d3',x.size())
        z = self.decode4([skip[-4], resize_like(z, skip[-4])])  # ; print('d4',x.size())
        z = self.decode5([resize_like(z, ipt)])

        seg_logit = self.logit(z)

        pooled_features = self.pooling(x4).view(bs, -1)
        cls_logit = self.fc(self.dropout(pooled_features))

        return seg_logit, cls_logit

    def infer(self, ipt):
        bs = ipt.size()[0]

        x0 = self.layer0(ipt)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        pooled_features = self.pooling(x4).view(bs, -1)
        cls_logit = self.fc(self.dropout(pooled_features))

        return cls_logit



class R50DUnetS(nn.Module):
    def __init__(self, n_class=4, out_dim=11, pretrained=True, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super(R50DUnetS, self).__init__()
        self.model = timm.create_model('resnet50d', pretrained=True)

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

        n_features = self.model.fc.in_features
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)


        # self.center = nn.Sequential(
        #     nn.Conv2d(2048, 2048, kernel_size=11, padding=5, bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.decode1 = ResDecode(1024 + 2048, 1024)
        # self.decode2 = ResDecode(512 + 1024, 512)
        # self.decode3 = ResDecode(256 + 512, 256)
        # self.decode4 = ResDecode(64 + 256, 64)
        # self.decode5 = ResDecode(64, 32)


        self.center = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decode1 = ResDecode(1024 + 64, 64)
        self.decode2 = ResDecode(512 + 64, 32)
        self.decode3 = ResDecode(256 + 32, 32)
        self.decode4 = ResDecode(64 + 32, 16)
        self.decode5 = ResDecode(16, 16)



        self.logit = nn.Conv2d(16, n_class, kernel_size=3, padding=1)

    def layer0(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        return x

    def layer1(self, x):
        return self.model.layer1(x)

    def layer2(self, x):
        return self.model.layer2(x)

    def layer3(self, x):
        return self.model.layer3(x)

    def layer4(self, x):
        return self.model.layer4(x)

    def forward(self, ipt):
        bs = ipt.size()[0]

        x0 = self.layer0(ipt)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        skip = [x0, x1, x2, x3]
        z = self.center(x4)
        z = self.decode1([skip[-1], resize_like(z, skip[-1])])  # ; print('d1',x.size())
        z = self.decode2([skip[-2], resize_like(z, skip[-2])])  # ; print('d2',x.size())
        z = self.decode3([skip[-3], resize_like(z, skip[-3])])  # ; print('d3',x.size())
        z = self.decode4([skip[-4], resize_like(z, skip[-4])])  # ; print('d4',x.size())
        z = self.decode5([resize_like(z, ipt)])

        seg_logit = self.logit(z)

        pooled_features = self.pooling(x4).view(bs, -1)
        cls_logit = self.fc(self.dropout(pooled_features))

        return seg_logit, cls_logit