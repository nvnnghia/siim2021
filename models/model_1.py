import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from utils.config import mixed_precision
# from torch.cuda.amp import autocast

# print('mixed_precision: ', mixed_precision)
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return 

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
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
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
        elif "efficientnet" in model_name: 
            n_features = self.model.classifier.in_features
            self.model.global_pool = nn.Identity()
            self.model.classifier =  nn.Identity()

        elif "nfnet" in model_name or "regnet" in model_name: 
            self.model.head = nn.Identity()
            n_features = self.model.final_conv.out_channels
        elif 'densenet' in model_name:
            self.model.global_pool = nn.Identity()
            self.model.classifier = nn.Identity()
            n_features = self.model.num_features
        elif 'coat_' in model_name or 'cait_' in model_name or 'swin_' in model_name or 'vit_' in model_name:
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise NotImplementedError(f"model type {model_name} has not implemented!")
        

        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    # @conditional_decorator(autocast(), mixed_precision)
    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        if not 'coat_' in self.model_name and not 'cait_' in self.model_name and not 'swin_' in self.model_name and not 'vit_' in self.model_name:
            features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(features))
        return output

    @property
    def net(self):
        return self.model
