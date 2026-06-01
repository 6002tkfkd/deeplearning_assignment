import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    SimCLR projection head: backbone feature → contrastive space.
    SimCLR v1: 2-layer MLP (hidden=2048, out=128)
    SimCLR v2: 3-layer MLP + larger hidden dim
    """
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128, num_layers=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class SimCLRModel(nn.Module):
    """backbone + projection head 통합 모델."""
    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x):
        h = self.backbone(x)   # representation
        z = self.projection_head(h)  # projection
        return h, z

    def get_representation(self, x):
        """평가 시 representation만 반환 (projection head 미사용)."""
        with torch.no_grad():
            return self.backbone(x)


def build_simclr_model(arch='resnet18', proj_hidden_dim=2048,
                       proj_output_dim=128, proj_num_layers=2,
                       pretrained=False, cifar_stem=True):
    from models.backbone import get_backbone
    backbone = get_backbone(arch=arch, pretrained=pretrained, cifar_stem=cifar_stem)
    head = ProjectionHead(
        input_dim=backbone.feature_dim,
        hidden_dim=proj_hidden_dim,
        output_dim=proj_output_dim,
        num_layers=proj_num_layers,
    )
    return SimCLRModel(backbone, head)
