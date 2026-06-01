"""
보고서용 시각화 그림 일괄 생성.
- ablation별 accuracy 비교 차트 (6종)
- collapse 분석 비교 차트
- t-SNE: NT-Xent vs SimSiam (collapse 시각화)
- temperature에 따른 silhouette / effective rank 변화
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS = './results'
FIGURES = './results/figures'
os.makedirs(FIGURES, exist_ok=True)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ─── 데이터 로드 헬퍼 ────────────────────────────────────────────────────────

def load_exp(path_prefix):
    """JSON 파일 3종을 읽어 dict 반환."""
    result = {}
    lp = path_prefix + '_linear_probe.json'
    kn = path_prefix + '_knn.json'
    fa = path_prefix + '_feature_analysis.json'
    if os.path.exists(lp):
        result['linear_probe'] = json.load(open(lp))['best_acc']
    if os.path.exists(kn):
        result['knn'] = json.load(open(kn))['results'].get('k=200')
    if os.path.exists(fa):
        d = json.load(open(fa))
        result['silhouette'] = d['separation']['silhouette_score']
        result['dbi'] = d['separation']['davies_bouldin_index']
        result['eff_rank'] = d['collapse']['effective_rank']
        result['dead_dims'] = d['collapse']['dead_dims']
        result['uniformity'] = d['uniformity']
    return result


def collect_ablation(ablation_dir):
    """ablation 디렉토리에서 실험별 결과 수집."""
    exps = {}
    for d in sorted(os.listdir(ablation_dir)):
        full = os.path.join(ablation_dir, d)
        if not os.path.isdir(full):
            continue
        prefix = os.path.join(full, d)
        data = load_exp(prefix)
        if data:
            exps[d] = data
    return exps


# ─── 공통 스타일 ─────────────────────────────────────────────────────────────

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
BLUE = '#4C72B0'
SALMON = '#DD8452'
GREEN = '#55A868'
RED = '#C44E52'
PURPLE = '#8172B2'


# ─── 1. Ablation accuracy 비교 바차트 ────────────────────────────────────────

def plot_ablation_accuracy(exps: dict, title: str, save_path: str,
                           highlight: str = None):
    names = list(exps.keys())
    lp = [exps[n].get('linear_probe', 0) for n in names]
    knn = [exps[n].get('knn', 0) for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 5))
    x = np.arange(len(names))
    w = 0.35

    colors_lp = [RED if n == highlight else BLUE for n in names]
    colors_kn = [RED if n == highlight else SALMON for n in names]

    b1 = ax.bar(x - w/2, lp, w, label='Linear Probe', color=colors_lp, edgecolor='white')
    b2 = ax.bar(x + w/2, knn, w, label='k-NN (k=200)', color=colors_kn, edgecolor='white')

    ax.bar_label(b1, fmt='%.3f', padding=2, fontsize=8)
    ax.bar_label(b2, fmt='%.3f', padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {save_path}')


# ─── 2. Collapse 분석 차트 (dead dims + silhouette 병렬) ─────────────────────

def plot_collapse_analysis(exps_dict: dict, title: str, save_path: str):
    """여러 그룹의 collapse 지표를 한눈에 비교."""
    names = list(exps_dict.keys())
    dead = [exps_dict[n].get('dead_dims', 0) for n in names]
    sil = [exps_dict[n].get('silhouette', 0) for n in names]
    eff = [exps_dict[n].get('eff_rank', 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))

    # Dead Dimensions
    colors = [RED if d > 10 else BLUE for d in dead]
    b = axes[0].bar(x, dead, color=colors, edgecolor='white')
    axes[0].bar_label(b, padding=2, fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=25, ha='right')
    axes[0].set_title('Dead Dimensions (↓ better, red = collapse risk)')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Silhouette Score
    colors = [GREEN if s > 0 else RED for s in sil]
    b = axes[1].bar(x, sil, color=colors, edgecolor='white')
    axes[1].bar_label(b, fmt='%.3f', padding=2, fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=25, ha='right')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_title('Silhouette Score (↑ better, red = no separation)')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    # Effective Rank
    b = axes[2].bar(x, eff, color=PURPLE, edgecolor='white')
    axes[2].bar_label(b, fmt='%.0f', padding=2, fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=25, ha='right')
    axes[2].set_title('Effective Rank (↑ = richer representation)')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {save_path}')


# ─── 3. Temperature 영향 라인 차트 ───────────────────────────────────────────

def plot_temperature_tradeoff(save_path: str):
    temps = [0.05, 0.07, 0.10, 0.50]
    lp    = [0.7879, 0.7997, 0.8187, 0.8286]
    sil   = [0.0365, 0.0403, 0.0661, 0.1368]
    rank  = [300.4, 309.8, 304.1, 254.5]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [str(t) for t in temps]
    x = np.arange(len(temps))

    for ax, vals, ylabel, color, title in zip(
        axes,
        [lp, sil, rank],
        ['Linear Probe Accuracy', 'Silhouette Score', 'Effective Rank'],
        [BLUE, GREEN, PURPLE],
        ['Linear Probe Accuracy', 'Class Separation (Silhouette)', 'Feature Diversity (Effective Rank)'],
    ):
        ax.plot(x, vals, marker='o', color=color, linewidth=2, markersize=8)
        for xi, v in zip(x, vals):
            ax.annotate(f'{v:.4f}' if vals is sil else f'{v:.3f}',
                        (xi, v), textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'τ={t}' for t in temps])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Temperature (τ) Trade-off Analysis', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {save_path}')


# ─── 4. Supervised vs SimCLR feature 비교 (radar/bar) ───────────────────────

def plot_ssl_vs_supervised(save_path: str):
    categories = ['Linear Probe\nAccuracy', 'Silhouette\nScore', 'Effective\nRank (norm)', 'Uniformity\n(−value)']
    supervised = [0.9541, 0.8230, 50.2 / 350, 2.133 / 2.5]
    simclr     = [0.8003, 0.0393, 307.3 / 350, 1.053 / 2.5]

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, supervised, w, label='Supervised ResNet18', color=BLUE, edgecolor='white')
    b2 = ax.bar(x + w/2, simclr,     w, label='SimCLR (Self-Supervised)', color=SALMON, edgecolor='white')

    raw_sup = [0.9541, 0.8230, 50.2, 2.133]
    raw_ssl = [0.8003, 0.0393, 307.3, 1.053]
    for bar, val in zip(b1, raw_sup):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='navy')
    for bar, val in zip(b2, raw_ssl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='darkred')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Normalized Score')
    ax.set_ylim(0, 1.15)
    ax.set_title('Supervised vs Self-Supervised Feature Comparison\n(Effective Rank & Uniformity normalized for scale)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {save_path}')


# ─── 5. t-SNE: NT-Xent vs SimSiam collapse ───────────────────────────────────

def plot_tsne_collapse(save_path: str):
    import torch
    import torch.nn.functional as F
    from sklearn.manifold import TSNE
    from tqdm import tqdm
    from data.cifar10_loader import get_eval_dataloaders
    from models.projection_head import build_simclr_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_eval_dataloaders(batch_size=256, root='./data/cifar10')

    ckpts = {
        'NT-Xent (No Collapse)': './results/ablation_loss/loss_ntxent/loss_ntxent_best.pt',
        'SimSiam (Collapsed)':   './results/ablation_loss/loss_simsiam/loss_simsiam_best.pt',
    }

    embeddings = {}
    all_labels = None

    for name, ckpt_path in ckpts.items():
        print(f'  Loading {name}...')
        model = build_simclr_model(arch='resnet18', proj_num_layers=2, cifar_stem=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        feats, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc='  extracting', leave=False):
                h = model.backbone(x.to(device))
                feats.append(h.cpu().numpy())
                labels.append(y.numpy())
                if sum(len(l) for l in labels) >= 5000:
                    break

        feats = np.concatenate(feats)[:5000]
        labels_arr = np.concatenate(labels)[:5000]

        print(f'  Running t-SNE for {name}...')
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        embeddings[name] = (tsne.fit_transform(feats), labels_arr)
        if all_labels is None:
            all_labels = labels_arr

    colors = cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (name, (emb, lbls)) in zip(axes, embeddings.items()):
        for cls_idx in range(10):
            mask = lbls == cls_idx
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[colors[cls_idx]], label=CIFAR10_CLASSES[cls_idx],
                       s=5, alpha=0.6)
        ax.set_title(name, fontsize=14)
        ax.legend(loc='best', markerscale=3, fontsize=8)
        ax.axis('off')

    plt.suptitle('Representation Collapse Visualization\nNT-Xent vs SimSiam (t-SNE)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {save_path}')


# ─── 6. Augmentation별 t-SNE (color_only vs full) ────────────────────────────

def plot_tsne_augmentation(save_path: str):
    import torch
    from sklearn.manifold import TSNE
    from tqdm import tqdm
    from data.cifar10_loader import get_eval_dataloaders
    from models.projection_head import build_simclr_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_eval_dataloaders(batch_size=256, root='./data/cifar10')

    ckpts = {
        'Color-only Aug\n(Partial Collapse)': './results/ablation_augmentation/aug_color_only/aug_color_only_best.pt',
        'Full Aug\n(Stable)':                 './results/ablation_augmentation/aug_full/aug_full_best.pt',
    }

    embeddings = {}
    for name, ckpt_path in ckpts.items():
        print(f'  Loading {name.strip()}...')
        model = build_simclr_model(arch='resnet18', proj_num_layers=2, cifar_stem=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        feats, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc='  extracting', leave=False):
                h = model.backbone(x.to(device))
                feats.append(h.cpu().numpy())
                labels.append(y.numpy())
                if sum(len(l) for l in labels) >= 5000:
                    break

        feats = np.concatenate(feats)[:5000]
        labels_arr = np.concatenate(labels)[:5000]

        print(f'  Running t-SNE...')
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        embeddings[name] = (tsne.fit_transform(feats), labels_arr)

    colors = cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (name, (emb, lbls)) in zip(axes, embeddings.items()):
        for cls_idx in range(10):
            mask = lbls == cls_idx
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[colors[cls_idx]], label=CIFAR10_CLASSES[cls_idx],
                       s=5, alpha=0.6)
        ax.set_title(name, fontsize=13)
        ax.legend(loc='best', markerscale=3, fontsize=8)
        ax.axis('off')

    plt.suptitle('Augmentation Effect on Feature Space (t-SNE)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {save_path}')


# ─── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Generating report figures ===\n')

    # 1. Ablation accuracy 비교 차트 (6종)
    ablation_groups = {
        'augmentation': ('Augmentation Ablation: Linear Probe & k-NN', None),
        'projection':   ('Projection Head Ablation: Linear Probe & k-NN', None),
        'batchsize':    ('Batch Size Ablation: Linear Probe & k-NN', None),
        'temperature':  ('Temperature Ablation: Linear Probe & k-NN', None),
        'backbone':     ('Backbone Ablation: Linear Probe & k-NN', None),
        'loss':         ('Loss Function Ablation: Linear Probe & k-NN', 'loss_simsiam'),
    }

    for name, (title, highlight) in ablation_groups.items():
        abl_dir = os.path.join(RESULTS, f'ablation_{name}')
        if not os.path.isdir(abl_dir):
            print(f'  [skip] {abl_dir} not found')
            continue
        exps = collect_ablation(abl_dir)
        if exps:
            print(f'[{name}] accuracy chart')
            plot_ablation_accuracy(
                exps, title,
                os.path.join(FIGURES, f'ablation_{name}_accuracy.png'),
                highlight=highlight,
            )

    # 2. Collapse 분석 (loss function)
    print('\n[collapse] loss function collapse analysis')
    loss_exps = collect_ablation(os.path.join(RESULTS, 'ablation_loss'))
    if loss_exps:
        plot_collapse_analysis(
            loss_exps,
            'Loss Function: Collapse Analysis (Dead Dims / Silhouette / Effective Rank)',
            os.path.join(FIGURES, 'collapse_loss_analysis.png'),
        )

    # 3. Collapse 분석 (augmentation)
    print('[collapse] augmentation collapse analysis')
    aug_exps = collect_ablation(os.path.join(RESULTS, 'ablation_augmentation'))
    if aug_exps:
        plot_collapse_analysis(
            aug_exps,
            'Augmentation: Collapse Analysis',
            os.path.join(FIGURES, 'collapse_aug_analysis.png'),
        )

    # 4. Temperature trade-off 라인 차트
    print('\n[temperature] trade-off chart')
    plot_temperature_tradeoff(os.path.join(FIGURES, 'temperature_tradeoff.png'))

    # 5. Supervised vs SimCLR 비교 바차트
    print('\n[ssl vs supervised] comparison chart')
    plot_ssl_vs_supervised(os.path.join(FIGURES, 'ssl_vs_supervised.png'))

    # 6. t-SNE: NT-Xent vs SimSiam collapse
    print('\n[t-SNE] collapse comparison (NT-Xent vs SimSiam)')
    plot_tsne_collapse(os.path.join(FIGURES, 'tsne_collapse_comparison.png'))

    # 7. t-SNE: augmentation 비교
    print('\n[t-SNE] augmentation comparison (color_only vs full)')
    plot_tsne_augmentation(os.path.join(FIGURES, 'tsne_aug_comparison.png'))

    print('\n=== Done. Figures saved to', FIGURES, '===')
