import torch
from torch import nn
import timm
from timm.models.layers.std_conv import  ScaledStdConv2d

class ECANFNET(nn.Module):
    def __init__(self, model_name='nf_regnet_b1', out_dim=4, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()

        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        else:
            raise NotImplementedError(f"pooling type {pool} has not implemented!")

        self.model = timm.create_model('vit_large_patch16_384', pretrained=False) #cait_xs24_384 coat_lite_small swin_base_patch4_window12_384 vit_deit_base_distilled_patch16_384
        # self.model.stem.conv1 = ScaledStdConv2d(in_channels = 6, out_channels=self.model.stem.conv1.out_channels, 
        #     kernel_size=self.model.stem.conv1.kernel_size[0], stride=self.model.stem.conv1.stride[0])

        print(self.model)
        # print(self.model.stem.conv1.in_channels)
        # print(self.model.stem.conv1.out_channels)
        # print(self.model.features.denseblock4)
        # self.conv_stem = self.model.conv_stem
        # self.bn1 = self.model.bn1
        # self.act1 = self.model.act1
        # ### Original blocks ###
        # for i in range(len((self.model.blocks))):
        #     setattr(self, "block{}".format(str(i)), self.model.blocks[i])
        # self.conv_head = self.model.conv_head
        # self.bn2 = self.model.bn2
        # self.act2 = self.model.act2
        # n_features = self.model.num_features
        # self.bottleneck_b4 = Bottleneck(inplanes=self.block4[-1].bn3.num_features,
        #                                 planes=int(self.block4[-1].bn3.num_features / 4))
        # self.bottleneck_b5 = Bottleneck(inplanes=self.block5[-1].bn3.num_features,
        #                                 planes=int(self.block5[-1].bn3.num_features / 4))
        # self.fc_b4 = nn.Linear(self.block4[-1].bn3.num_features, out_dim)
        # self.fc_b5 = nn.Linear(self.block5[-1].bn3.num_features, out_dim)

        # print(self.model.global_pool)
        # print(self.model)

        # self.model.global_pool = nn.Identity()
        # self.model.classifier = nn.Identity()
        # self.model.global_pool = nn.Identity()

        self.fc = nn.Linear(self.model.head.in_features, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.model.head = nn.Identity()
        self.model.head_dist = None

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        print(features[0].shape, features[1].shape)
        # x = self.model.stem(x)
        # print(x.shape)
        # # x = self.model.stages(x)
        # for m in self.model.features:
            # print(dir(m))
            # x = m(x)
            # print(x.shape)
        # x = self.model.final_conv(x)
        # # print(x.shape)
        # features = self.model.final_act(x)
        # print(x.shape)
        # x = self.model.head(x)
        # pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(features))
        return output

model = ECANFNET()
ipt = torch.rand(4,3,384,384)

out = model(ipt)

print(out.shape)

