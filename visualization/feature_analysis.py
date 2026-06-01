"""
Feature representation 분석 모듈.
- Representation collapse 탐지
- Class separation 정량화 (Silhouette Score, Davies-Bouldin Index)
- Feature uniformity / alignment 측정 (Wang & Isola, 2020)
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm

from data.cifar10_loader import get_eval_dataloaders
from models.projection_head import build_simclr_model


@torch.no_grad()
def extract_all_features(model, loader, device, normalize=True):
    model.eval()
    features, labels = [], []
    for x, y in tqdm(loader, desc='extracting', leave=False):
        x = x.to(device)
        h = model.backbone(x)
        if normalize:
            h = F.normalize(h, dim=1)
        features.append(h.cpu())
        labels.append(y)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


# ─── Collapse 분석 ─────────────────────────────────────────────────────────────

def compute_collapse_metrics(features: np.ndarray) -> dict:
    """
    Representation collapse 탐지 지표.

    - feature_std: 각 dimension의 표준편차 평균 (낮으면 collapse 의심)
    - effective_rank: feature matrix의 유효 rank (낮으면 collapse)
    - dead_dims: std < 0.01인 dimension 수
    """
    std_per_dim = features.std(axis=0)
    mean_std = float(std_per_dim.mean())
    dead_dims = int((std_per_dim < 0.01).sum())

    # effective rank (Roy & Vetterli, 2007)
    _, s, _ = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    effective_rank = float(np.exp(entropy))

    return {
        'mean_std':       mean_std,
        'dead_dims':      dead_dims,
        'effective_rank': effective_rank,
        'singular_values': s[:20].tolist(),  # 상위 20개만 저장
    }


def compute_uniformity(features: np.ndarray, t: float = 2.0) -> float:
    """
    Uniformity loss (Wang & Isola, 2020).
    hypersphere 위에서 feature가 얼마나 균일하게 분포하는지.
    낮을수록 좋음 (더 uniform).
    """
    z = torch.tensor(features, dtype=torch.float32)
    z = F.normalize(z, dim=1)
    sq_dists = torch.pdist(z, p=2).pow(2)
    return float(sq_dists.mul(-t).exp().mean().log())


def compute_alignment(features1: np.ndarray, features2: np.ndarray,
                      alpha: float = 2.0) -> float:
    """
    Alignment loss (Wang & Isola, 2020).
    positive pair 간 feature가 얼마나 가까운지.
    낮을수록 좋음.
    """
    z1 = F.normalize(torch.tensor(features1, dtype=torch.float32), dim=1)
    z2 = F.normalize(torch.tensor(features2, dtype=torch.float32), dim=1)
    return float((z1 - z2).norm(dim=1).pow(alpha).mean())


# ─── Class separation 분석 ────────────────────────────────────────────────────

def compute_class_separation(features: np.ndarray, labels: np.ndarray,
                              sample_size: int = 3000) -> dict:
    """
    Silhouette Score, Davies-Bouldin Index로 class separation 정량화.
    """
    if len(features) > sample_size:
        idx = np.random.choice(len(features), sample_size, replace=False)
        features, labels = features[idx], labels[idx]

    sil  = float(silhouette_score(features, labels, metric='cosine'))
    dbi  = float(davies_bouldin_score(features, labels))
    return {'silhouette_score': sil, 'davies_bouldin_index': dbi}


# ─── 시각화 ────────────────────────────────────────────────────────────────────

def plot_singular_values(results_dict: dict, save_path: str):
    """여러 실험의 singular value spectrum 비교."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, metrics in results_dict.items():
        sv = metrics.get('collapse', {}).get('singular_values', [])
        if sv:
            ax.plot(sv, label=name, marker='o', markersize=3)

    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Spectrum (collapse indicator)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {save_path}")


def plot_feature_std(results_dict: dict, save_path: str):
    """실험별 mean feature std 바 차트."""
    names = list(results_dict.keys())
    stds  = [v['collapse']['mean_std'] for v in results_dict.values()]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, stds, color='steelblue', edgecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=9)
    ax.set_ylabel('Mean Feature Std')
    ax.set_title('Feature Standard Deviation (higher = less collapse)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {save_path}")


# ─── 통합 실행 ─────────────────────────────────────────────────────────────────

def run_feature_analysis(checkpoint_path, config, save_dir='./results', model_type='simclr'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    if model_type == 'simclr':
        model = build_simclr_model(
            arch=config.get('arch', 'resnet18'),
            proj_num_layers=config.get('proj_num_layers', 2),
            cifar_stem=config.get('cifar_stem', True),
        ).to(device)
    else:
        from training.supervised_trainer import SupervisedResNet
        model = SupervisedResNet(
            arch=config.get('arch', 'resnet18'),
            cifar_stem=config.get('cifar_stem', True),
        ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    _, test_loader = get_eval_dataloaders(
        batch_size=256,
        root=config.get('data_root', './data/cifar10'),
    )

    print("  Extracting features for analysis...")
    if model_type != 'simclr':
        # supervised 모델은 backbone 속성 없음 — get_features 사용
        model.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                h = model.get_features(x)
                all_feats.append(h.cpu())
                all_labels.append(y)
        feats  = torch.cat(all_feats).numpy()
        labels = torch.cat(all_labels).numpy()
    else:
        feats, labels = extract_all_features(model, test_loader, device, normalize=True)

    collapse  = compute_collapse_metrics(feats)
    separation = compute_class_separation(feats, labels)
    uniformity = compute_uniformity(feats)

    result = {
        'exp_name':   config.get('exp_name', 'simclr'),
        'collapse':   collapse,
        'separation': separation,
        'uniformity': uniformity,
    }

    exp_name = config.get('exp_name', 'simclr')
    result_path = os.path.join(save_dir, f'{exp_name}_feature_analysis.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  [Feature Analysis] {exp_name}")
    print(f"    mean_std={collapse['mean_std']:.4f} | "
          f"effective_rank={collapse['effective_rank']:.1f} | "
          f"dead_dims={collapse['dead_dims']}")
    print(f"    silhouette={separation['silhouette_score']:.4f} | "
          f"DBI={separation['davies_bouldin_index']:.4f}")
    print(f"    uniformity={uniformity:.4f}")
    print(f"  -> saved: {result_path}")

    return result
