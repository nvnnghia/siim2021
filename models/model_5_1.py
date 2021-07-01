import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.resnet import Bottleneck
from utils.config import mixed_precision

def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator

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

class EfficientNet(nn.Module):
    """
    EfficientNet B0-B8.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, model_name):
        super(EfficientNet, self).__init__()
        backbone = timm.create_model(
            model_name=model_name,
            pretrained=True,
            in_chans=3,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        ### Original blocks ###
        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2
        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.num_features = backbone.num_features
        self.bottleneck_b4 = Bottleneck(inplanes=self.block4[-1].bn3.num_features,
                                        planes=int(self.block4[-1].bn3.num_features / 4))
        self.bottleneck_b5 = Bottleneck(inplanes=self.block5[-1].bn3.num_features,
                                        planes=int(self.block5[-1].bn3.num_features / 4))
        self.fc_b4 = nn.Linear(self.block4[-1].bn3.num_features, 5)
        self.fc_b5 = nn.Linear(self.block5[-1].bn3.num_features, 5)

        self.fc = nn.Linear(self.num_features, 5)
        del backbone

        self.mask = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.dropout = nn.Dropout(0.5)
        self.pooling = GeM()


    def _features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x); b2 = x
        x = self.block3(x)
        x = self.block4(x); b4 = x
        x = self.block5(x); b5 = x #
        x = self.block6(x); b6 = x
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return b2,b4,b5,b6,x

    def forward(self, x):
            bs = x.size()[0]
            b2,b4, b5,b6, x = self._features(x)
            pooled_features = self.pooling(x).view(bs, -1)
            cls_logit = self.fc(self.dropout(pooled_features))
#             b4_logits = self.fc_b4(torch.flatten(self.global_pool(self.bottleneck_b4(b4)), 1))
            b5_logits = self.fc_b5(self.dropout(torch.flatten(self.global_pool(self.bottleneck_b5(b5)), 1)))
            #x = torch.flatten(x, 1)
            #logits = self.fc(self.dropout(x))
            #output = (logits + b5_logits) / 2.
            return cls_logit,b5_logits,self.mask(b4)

    @property
    def net(self):
        return self.model