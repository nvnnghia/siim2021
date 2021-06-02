import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.resnet import Bottleneck

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

class SIIMModel(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=4, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model_name = model_name

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        else:
            raise NotImplementedError(f"pooling type {pool} has not implemented!")

        if 'resne' in model_name: #resnet
            n_features = self.model.fc.in_features
            self.model.global_pool = nn.Identity()
            self.model.fc = nn.Identity()
            feats_list = [n_features, 1024, 512, 256, 64]
        elif "efficientnet" in model_name: 
            self.conv_stem = self.model.conv_stem
            self.bn1 = self.model.bn1
            self.act1 = self.model.act1
            ### Original blocks ###
            for i in range(len((self.model.blocks))):
                setattr(self, "block{}".format(str(i)), self.model.blocks[i])
            self.conv_head = self.model.conv_head
            self.bn2 = self.model.bn2
            self.act2 = self.model.act2
            self.global_pool = SelectAdaptivePool2d(pool_type="avg")
            n_features = self.model.num_features
            self.bottleneck_b4 = Bottleneck(inplanes=self.block4[-1].bn3.num_features,
                                            planes=int(self.block4[-1].bn3.num_features / 4))
            self.bottleneck_b5 = Bottleneck(inplanes=self.block5[-1].bn3.num_features,
                                            planes=int(self.block5[-1].bn3.num_features / 4))
            self.fc_b4 = nn.Linear(self.block4[-1].bn3.num_features, out_dim)
            self.fc_b5 = nn.Linear(self.block5[-1].bn3.num_features, out_dim)

            feats_list = [n_features, self.block4[-1].bn3.num_features, self.block2[-1].bn3.num_features, self.block1[-1].bn3.num_features, self.block0[-1].bn2.num_features]

            del self.model

        

        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.center = nn.Sequential(
            nn.Conv2d(n_features, 512, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decode1 = ResDecode(feats_list[1] + 512, 256)
        self.decode2 = ResDecode(feats_list[2] + 256, 128)
        self.decode3 = ResDecode(feats_list[3] + 128, 64)
        self.decode4 = ResDecode(feats_list[4] + 64, 32)
        self.decode5 = ResDecode(32, 16)

        self.logit = nn.Conv2d(16, 1, kernel_size=3, padding=1)


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

    def _features(self, x):
        if "efficientnet" in self.model_name: 
            x = self.conv_stem(x)
            x = self.bn1(x)
            x = self.act1(x)
            # print('0', x.shape)
            x = self.block0(x); b0 = x
            # print('1', x.shape)
            x = self.block1(x); b1 = x
            # print('2', x.shape)
            x = self.block2(x); b2 = x
            # print('3', x.shape)
            x = self.block3(x); b3 = x
            # print('4', x.shape)
            x = self.block4(x); b4 = x
            # print('5', x.shape)
            x = self.block5(x); b5 = x
            # print('6', x.shape)
            x = self.block6(x)
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.act2(x)
            # print('7', x.shape)
            return b0, b1, b2, b4, x
        elif "resne" in self.model_name: 
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x0, x1, x2, x3, x4

    def forward(self, ipt):
        bs = ipt.size()[0]

        x0, x1, x2, x3, x4 = self._features(ipt)
        print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)

        skip = [x0, x1, x2, x3]
        z = self.center(x4)
        z = self.decode1([skip[-1], resize_like(z, skip[-1])])   #; print('d1',z.size())
        z = self.decode2([skip[-2], resize_like(z, skip[-2])])   #; print('d2',z.size())
        z = self.decode3([skip[-3], resize_like(z, skip[-3])])   #; print('d3',z.size())
        z = self.decode4([skip[-4], resize_like(z, skip[-4])])   #; print('d4',z.size())
        z = self.decode5([resize_like(z, ipt)])

        seg_logit = self.logit(z)
        #print('seg_logit',seg_logit.size())

        pooled_features = self.pooling(x4).view(bs, -1)
        cls_logit = self.fc(self.dropout(pooled_features))

        return cls_logit

    @property
    def net(self):
        return self.model


