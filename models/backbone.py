import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    """
    SimCLR용 ResNet backbone.
    마지막 FC layer를 제거하고 feature vector만 출력.
    """
    ARCH_CONFIGS = {
        'resnet18':  (models.resnet18,  models.ResNet18_Weights.DEFAULT,  512),
        'resnet34':  (models.resnet34,  models.ResNet34_Weights.DEFAULT,  512),
        'resnet50':  (models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048),
    }

    def __init__(self, arch='resnet18', pretrained=False, cifar_stem=True):
        super().__init__()
        if arch not in self.ARCH_CONFIGS:
            raise ValueError(f"arch must be one of {list(self.ARCH_CONFIGS.keys())}")

        model_fn, weights, self.feature_dim = self.ARCH_CONFIGS[arch]
        net = model_fn(weights=weights if pretrained else None)

        # CIFAR-10은 32x32 이미지 → stride=1, kernel=3으로 첫 conv 교체
        if cifar_stem:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        # FC layer 제거 (projection head와 분리)
        self.encoder = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        h = self.encoder(x)        # (B, C, 1, 1)
        return h.flatten(start_dim=1)  # (B, feature_dim)


def get_backbone(arch='resnet18', pretrained=False, cifar_stem=True):
    return ResNetBackbone(arch=arch, pretrained=pretrained, cifar_stem=cifar_stem)
