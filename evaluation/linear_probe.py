import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.cifar10_loader import get_eval_dataloaders
from models.projection_head import build_simclr_model


class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_features(model, loader, device):
    """backbone feature 전체 추출."""
    model.eval()
    all_features, all_labels = [], []

    for x, y in tqdm(loader, desc='extracting features', leave=False):
        x = x.to(device)
        h = model.backbone(x)
        all_features.append(h.cpu())
        all_labels.append(y)

    return torch.cat(all_features), torch.cat(all_labels)


def run_linear_probe(checkpoint_path, config: dict, save_dir='./results'):
    """
    SimCLR로 학습된 backbone을 freeze하고 선형 분류기만 학습.
    SimCLR 논문의 standard evaluation protocol.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # 모델 로드
    model = build_simclr_model(
        arch=config.get('arch', 'resnet18'),
        proj_num_layers=config.get('proj_num_layers', 2),
        cifar_stem=config.get('cifar_stem', True),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Backbone freeze
    for p in model.backbone.parameters():
        p.requires_grad = False

    train_loader, test_loader = get_eval_dataloaders(
        batch_size=config.get('eval_batch_size', 256),
        num_workers=config.get('num_workers', 4),
        root=config.get('data_root', './data/cifar10'),
    )

    # feature 미리 추출 (빠른 학습)
    print("  Extracting train features...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print("  Extracting test features...")
    test_feats, test_labels = extract_features(model, test_loader, device)

    feature_dim = train_feats.size(1)
    probe = LinearProbe(feature_dim).to(device)

    # feature를 dataset으로 래핑
    from torch.utils.data import TensorDataset, DataLoader
    probe_train = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=256, shuffle=True,
    )
    probe_test = DataLoader(
        TensorDataset(test_feats, test_labels),
        batch_size=256, shuffle=False,
    )

    epochs = config.get('probe_epochs', 100)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, history = 0.0, []

    for epoch in range(1, epochs + 1):
        probe.train()
        for feats, labels in probe_train:
            feats, labels = feats.to(device), labels.to(device)
            loss = criterion(probe(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            probe.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for feats, labels in probe_test:
                    feats, labels = feats.to(device), labels.to(device)
                    correct += probe(feats).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            best_acc = max(best_acc, acc)
            history.append({'epoch': epoch, 'acc': acc})
            print(f"    [LinearProbe] epoch={epoch:3d} acc={acc:.4f}")

    exp_name = config.get('exp_name', 'simclr')
    result = {'best_acc': best_acc, 'history': history, 'exp_name': exp_name}

    result_path = os.path.join(save_dir, f'{exp_name}_linear_probe.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  [LinearProbe] best acc: {best_acc:.4f} -> {result_path}")
    return best_acc
