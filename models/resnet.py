import pretrainedmodels
from torch import nn


class CustomResnetModel(nn.Module):
    def __init__(self, backbone, pretrained='imagenet', out_features=81313):
        super().__init__()
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.net = pretrainedmodels.__dict__[backbone](pretrained=pretrained)
        self.net.last_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        )

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, backbone, pretrained='imagenet', out_features=81313):
        super().__init__()
        if backbone in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            self.net = pretrainedmodels.__dict__[backbone](pretrained=pretrained)
        self.last_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        )

    def forward(self, x):
        x = self.net.features(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        return self.last_linear(x)


class CustomSenet(nn.Module):
    def __init__(self, name='se_resnext50_32x4d', pretrained='imagenet', out_features=11):
        super().__init__()
        self.net = pretrainedmodels.__dict__[name](pretrained=pretrained)
        self.fc = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)

    def forward(self, x):
        x = self.net.features(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc(x)
        return x


class CustomDenseNet(nn.Module):
    def __init__(self, name='densenet161', pretrained='imagenet'):
        super().__init__()
        self.net = pretrainedmodels.__dict__[name](pretrained=pretrained)
        self.fc = nn.Linear(in_features=self.net.last_linear.in_features, out_features=186)

    def forward(self, x):
        x = self.net.features(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc(x)
        return x