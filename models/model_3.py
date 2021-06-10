import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.resnet import Bottleneck
import typing as tp
# from config import mixed_precision

# def conditional_decorator(dec, condition):
#     def decorator(func):
#         if not condition:
#             # Return the function unchanged, not decorated.
#             return func
#         return dec(func)

#     return 
def get_activation(activ_name: str="relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity()}
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError
        

class Conv2dBNActiv(nn.Module):
    """Conv2d -> (BN ->) -> Activation"""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: int=1, padding: int=0,
        bias: bool=False, use_bn: bool=True, activ: str="relu"
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward"""
        return self.layers(x)
        

class SSEBlock(nn.Module):
    """channel `S`queeze and `s`patial `E`xcitation Block."""

    def __init__(self, in_channels: int):
        """Initialize."""
        super(SSEBlock, self).__init__()
        self.channel_squeeze = nn.Conv2d(
            in_channels=in_channels, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward."""
        # # x: (bs, ch, h, w) => h: (bs, 1, h, w)
        h = self.sigmoid(self.channel_squeeze(x))
        # # x, h => return: (bs, ch, h, w)
        return x * h
    
    
class SpatialAttentionBlock(nn.Module):
    """Spatial Attention for (C, H, W) feature maps"""
    
    def __init__(
        self, in_channels: int,
        out_channels_list: tp.List[int],
    ):
        """Initialize"""
        super(SpatialAttentionBlock, self).__init__()
        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list
        assert self.n_layers > 0
        assert channels_list[-1] == 1
        
        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i: i + 2]
            layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="relu")
            setattr(self, f"conv{i + 1}", layer)
            
        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="sigmoid")
        setattr(self, f"conv{self.n_layers}", layer)
    
    def forward(self, x):
        """Forward"""
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)
            
        h = h * x
        return h

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

        self.global_pool = SelectAdaptivePool2d(pool_type="avg")

        if 'resne' in model_name: #resnet
            n_features = self.model.fc.in_features
            self.model.global_pool = nn.Identity()
            self.model.fc = nn.Identity()

            # print(self.model.conv1[-1].out_channels)
            # print(self.model.layer1[-1].bn3.num_features)
            # print(self.model.layer2[-1].bn3.num_features)
            # print(self.model.layer3[-1].bn3.num_features)
            # print(self.model.layer4[-1].bn3.num_features)

            feats_list = [n_features, self.model.layer3[-1].bn3.num_features, self.model.layer2[-1].bn3.num_features, self.model.layer1[-1].bn3.num_features, self.model.conv1[-1].out_channels]

            self.bottleneck_b5 = nn.Identity() #Bottleneck(inplanes=1024, planes=int(1024 / 4))
            self.fc_b5 = nn.Linear(1024, out_dim)

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
            n_features = self.model.num_features
            self.bottleneck_b4 = Bottleneck(inplanes=self.block4[-1].bn3.num_features,
                                            planes=int(self.block4[-1].bn3.num_features / 4))
            self.bottleneck_b5 = Bottleneck(inplanes=self.block5[-1].bn3.num_features,
                                            planes=int(self.block5[-1].bn3.num_features / 4))
            self.fc_b4 = nn.Linear(self.block4[-1].bn3.num_features, out_dim)
            self.fc_b5 = nn.Linear(self.block5[-1].bn3.num_features, out_dim)

            feats_list = [n_features, self.block4[-1].bn3.num_features, self.block2[-1].bn3.num_features, self.block1[-1].bn3.num_features, self.block0[-1].bn2.num_features]

            del self.model

        elif "nfnet" in model_name: 
            self.model.head = nn.Identity()
            n_features = self.model.final_conv.out_channels
            feats_list = [n_features, self.model.stages[-2][-1].conv3.out_channels, self.model.stages[-3][-1].conv3.out_channels, self.model.stages[-4][-1].conv3.out_channels, self.model.stem[-1].out_channels]
        else:
            raise NotImplementedError(f"model type {model_name} has not implemented!")
        

        # self.fc = nn.Linear(n_features, out_dim)
        # self.dropout = nn.Dropout(dropout)
        self.n_heads = out_dim

        for i, dim in enumerate([1,1,1,1]):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(n_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(n_features, n_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(n_features, dim))
            setattr(self, layer_name, layer)


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
            return b4, b5, x
        elif "resne" in self.model_name: 
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x2, x3, x4
        elif "nfnet" in self.model_name: 
            x = self.model.stem(x)
            # print(x.shape)
            # x = self.model.stages(x)
            feats = [x]
            for m in self.model.stages:
                x = m(x)
                feats.append(x)

            x = self.model.final_conv(x)
            features = self.model.final_act(x)

            return feats[2], feats[3], features

    # @conditional_decorator(autocast(), mixed_precision)
    def forward(self, ipt):
        bs = ipt.size()[0]

        x2, x3, x4 = self._features(ipt)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        # b5_logits = self.fc_b5(torch.flatten(self.global_pool(self.bottleneck_b5(x3)), 1))

        # pooled_features = self.pooling(x4).view(bs, -1)
        # cls_logit = self.fc(self.dropout(pooled_features))

        # cls_logit = (torch.sigmoid(cls_logit) + torch.sigmoid(b5_logits)) / 2.

        hs = [
            getattr(self, f"head_{i}")(x4) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y

        # return cls_logit

    @property
    def net(self):
        return self.model


