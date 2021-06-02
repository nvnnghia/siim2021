import torch
from torch import nn
import timm

class ECANFNET(nn.Module):
    def __init__(self, model_name='eca_nfnet_l1', out_dim=4, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        else:
            raise NotImplementedError(f"pooling type {pool} has not implemented!")

        self.model = timm.create_model('eca_nfnet_l1', pretrained=False)
        print(self.model.stem)
        # print(self.model.stages)
        # print(dir(self.model.final_conv.out_channels))
        print(self.model.stem[-1].out_channels)
        self.model.head = nn.Identity()

        self.fc = nn.Linear(self.model.final_conv.out_channels, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        # features = self.model(x)
        x = self.model.stem(x)
        print(x.shape)
        # x = self.model.stages(x)
        for m in self.model.stages:
            # print(dir(m))
            x = m(x)
            print(x.shape)
        x = self.model.final_conv(x)
        # print(x.shape)
        features = self.model.final_act(x)
        # print(x.shape)
        # x = self.model.head(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

model = ECANFNET()
ipt = torch.rand(4,3,384,384)

out = model(ipt)

print(out.shape)

