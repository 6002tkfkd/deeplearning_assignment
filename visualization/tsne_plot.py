import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from tqdm import tqdm

from data.cifar10_loader import get_eval_dataloaders
from models.projection_head import build_simclr_model
from training.supervised_trainer import SupervisedResNet

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


@torch.no_grad()
def _extract(model_type, model, loader, device, max_samples=5000):
    """feature 추출 (최대 max_samples개)."""
    model.eval()
    features, labels = [], []
    count = 0

    for x, y in tqdm(loader, desc='extracting', leave=False):
        if count >= max_samples:
            break
        x = x.to(device)

        if model_type == 'simclr':
            h = model.backbone(x)
        else:  # supervised
            h = model.get_features(x)

        features.append(h.cpu().numpy())
        labels.append(y.numpy())
        count += x.size(0)

    features = np.concatenate(features)[:max_samples]
    labels   = np.concatenate(labels)[:max_samples]
    return features, labels


def plot_tsne(features, labels, title, save_path, perplexity=30, n_iter=1000):
    print(f"  Running t-SNE ({features.shape[0]} samples)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=42, verbose=0)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.tab10(np.linspace(0, 1, 10))

    for cls_idx in range(10):
        mask = labels == cls_idx
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[colors[cls_idx]], label=CIFAR10_CLASSES[cls_idx],
            s=5, alpha=0.6,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', markerscale=3, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {save_path}")
    return emb


def run_tsne_comparison(simclr_ckpt, supervised_ckpt,
                        simclr_config, supervised_config,
                        save_dir='./results', max_samples=5000):
    """SimCLR vs Supervised feature의 t-SNE 비교 플롯 생성."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    _, test_loader = get_eval_dataloaders(
        batch_size=256,
        root=simclr_config.get('data_root', './data/cifar10'),
    )

    # SimCLR feature
    simclr_model = build_simclr_model(
        arch=simclr_config.get('arch', 'resnet18'),
        proj_num_layers=simclr_config.get('proj_num_layers', 2),
        cifar_stem=simclr_config.get('cifar_stem', True),
    ).to(device)
    ckpt = torch.load(simclr_ckpt, map_location=device)
    simclr_model.load_state_dict(ckpt['model_state_dict'])

    print("\n[t-SNE] Extracting SimCLR features...")
    simclr_feats, simclr_labels = _extract(
        'simclr', simclr_model, test_loader, device, max_samples
    )

    # Supervised feature
    sup_model = SupervisedResNet(
        arch=supervised_config.get('arch', 'resnet18'),
        cifar_stem=supervised_config.get('cifar_stem', True),
    ).to(device)
    ckpt = torch.load(supervised_ckpt, map_location=device)
    sup_model.load_state_dict(ckpt['model_state_dict'])

    print("[t-SNE] Extracting Supervised features...")
    sup_feats, sup_labels = _extract(
        'supervised', sup_model, test_loader, device, max_samples
    )

    exp_name = simclr_config.get('exp_name', 'simclr')

    simclr_emb = plot_tsne(
        simclr_feats, simclr_labels,
        title='SimCLR Feature Space (t-SNE)',
        save_path=os.path.join(save_dir, f'{exp_name}_tsne_simclr.png'),
    )
    sup_emb = plot_tsne(
        sup_feats, sup_labels,
        title='Supervised Feature Space (t-SNE)',
        save_path=os.path.join(save_dir, f'{exp_name}_tsne_supervised.png'),
    )

    # 나란히 비교 플롯
    _plot_side_by_side(
        simclr_emb, simclr_labels,
        sup_emb, sup_labels,
        save_path=os.path.join(save_dir, f'{exp_name}_tsne_comparison.png'),
    )


def _plot_side_by_side(emb1, labels1, emb2, labels2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = cm.tab10(np.linspace(0, 1, 10))

    for ax, emb, labels, title in zip(
        axes,
        [emb1, emb2],
        [labels1, labels2],
        ['SimCLR (Self-Supervised)', 'Supervised'],
    ):
        for cls_idx in range(10):
            mask = labels == cls_idx
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[colors[cls_idx]], label=CIFAR10_CLASSES[cls_idx],
                       s=5, alpha=0.6)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', markerscale=3, fontsize=8)
        ax.axis('off')

    plt.suptitle('Feature Space Comparison: SimCLR vs Supervised', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> comparison plot saved: {save_path}")


def run_tsne_single(checkpoint_path, config, model_type='simclr',
                    save_dir='./results', max_samples=5000):
    """단일 모델 t-SNE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    _, test_loader = get_eval_dataloaders(
        batch_size=256,
        root=config.get('data_root', './data/cifar10'),
    )

    if model_type == 'simclr':
        model = build_simclr_model(
            arch=config.get('arch', 'resnet18'),
            proj_num_layers=config.get('proj_num_layers', 2),
            cifar_stem=config.get('cifar_stem', True),
        ).to(device)
    else:
        model = SupervisedResNet(
            arch=config.get('arch', 'resnet18'),
            cifar_stem=config.get('cifar_stem', True),
        ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    feats, labels = _extract(model_type, model, test_loader, device, max_samples)
    exp_name = config.get('exp_name', model_type)

    plot_tsne(
        feats, labels,
        title=f'{exp_name} Feature Space (t-SNE)',
        save_path=os.path.join(save_dir, f'{exp_name}_tsne.png'),
    )
