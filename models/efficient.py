import torch
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
import timm


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EfficinetNetFN(nn.Module):
    '''
    Version with feature linear!


    '''
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313, dropout=0.5, feature_dim=512):
        super().__init__()
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
                                    pretrained=(pretrained == 'imagenet'))
        print(name)
        self.feature_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=feature_dim)
        self.last_linear = nn.Linear(in_features=feature_dim, out_features=out_features)
        self.pool = GeM()
        self.dropout = dropout

    def forward(self, x, infer=False):
        x = self.model.features(x)
        f = self.feature_linear(nn.Flatten()(self.pool(x)))
        if infer:
            return self.pool(x)
        else:
            f = nn.ReLU()(f)
            if self.dropout:
                return self.last_linear(nn.Dropout(self.dropout)(f))
            else:
                return self.last_linear(f)


class EfficinetNet(nn.Module):
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
                                    pretrained=(pretrained == 'imagenet'))
        print(name)
        self.last_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, infer=False):
        x = self.model.features(x)
        x = nn.Flatten()(self.pooling(x))
        x = self.dropout(x)
        return self.last_linear(x)


class EfficinetNetV2(nn.Module):
    def __init__(self, name='efficientnetv2_s', pretrained='imagenet', out_features=81313, dropout=0.5, pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        print(name)
        self.last_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.dropout = nn.Dropout(dropout)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        # self.model.classifier = nn.Sequential(self.dropout, self.last_linear)
        # self.model.train()
        # # self.model.global_pool = self.pooling
        # for x in self.model.parameter():
        #     print(x.requires_grad)

    def forward(self, x, infer=False):
        x = self.model(x)
        x = nn.Flatten()(self.pooling(x))
        x = self.dropout(x)
        return self.last_linear(x)


class AUXNet(nn.Module):
    def __init__(self, name, dropout=0, pool='AdaptiveAvgPool2d'):
        super(AUXNet, self).__init__()

        print('[ AUX model ] dropout: {}, pool: {}'.format(dropout, pool))
        e = timm.models.__dict__[name](pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        self.model = e
        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]
        self.b8 = nn.Sequential(
            e.conv_head, #384, 1536
            e.bn2,
            e.act2,
        )

        self.logit = nn.Linear(1280, 2)
        self.mask = nn.Sequential(
            nn.Conv2d(176, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.dropout = nn.Dropout(p=dropout)

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()

    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        # x = 2*image-1     # ; print('input ',   x.shape)
        x = image

        x = self.b0(x) #; print (x.shape)  # torch.Size([2, 40, 256, 256])
        x = self.b1(x) #; print (x.shape)  # torch.Size([2, 24, 256, 256])
        x = self.b2(x) #; print (x.shape)  # torch.Size([2, 32, 128, 128])
        x = self.b3(x) #; print (x.shape)  # torch.Size([2, 48, 64, 64])
#         print(x.shape)
        x = self.b4(x) #; print (x.shape)  # torch.Size([2, 96, 32, 32])
#         print(x.shape)
        x = self.b5(x) #; print (x.shape)  # torch.Size([2, 136, 32, 32])
        #------------
        mask = self.mask(x)
        #-------------
        x = self.b6(x) #; print (x.shape)  # torch.Size([2, 232, 16, 16])
        x = self.b7(x) #; print (x.shape)  # torch.Size([2, 384, 16, 16])
        x = self.b8(x) #; print (x.shape)  # torch.Size([2, 1536, 16, 16])
        # x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = nn.Flatten()(self.pooling(x))
        #x = F.dropout(x, 0.5, training=self.training)
        x = self.dropout(x)
        logit = self.logit(x)
        return logit, mask

