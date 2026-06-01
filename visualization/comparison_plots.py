"""
ablation 실험 결과를 비교하는 차트 생성.
"""
import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def load_all_results(results_dir: str) -> pd.DataFrame:
    """results/ 디렉토리의 JSON 파일들을 읽어 DataFrame으로 통합."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*_linear_probe.json'))):
        with open(path) as f:
            data = json.load(f)
        exp = data['exp_name']
        best_acc = data['best_acc']

        knn_path = path.replace('_linear_probe.json', '_knn.json')
        knn_acc = None
        if os.path.exists(knn_path):
            with open(knn_path) as f:
                knn_data = json.load(f)
            knn_acc = knn_data['results'].get('k=200', None)

        analysis_path = path.replace('_linear_probe.json', '_feature_analysis.json')
        sil, dbi, eff_rank, uniformity = None, None, None, None
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                ana = json.load(f)
            sil       = ana['separation']['silhouette_score']
            dbi       = ana['separation']['davies_bouldin_index']
            eff_rank  = ana['collapse']['effective_rank']
            uniformity = ana['uniformity']

        rows.append({
            'exp_name':       exp,
            'linear_probe':   best_acc,
            'knn_k200':       knn_acc,
            'silhouette':     sil,
            'davies_bouldin': dbi,
            'effective_rank': eff_rank,
            'uniformity':     uniformity,
        })

    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame, save_path: str,
                             group_label: str = 'Experiment'):
    """Linear Probe vs k-NN accuracy 비교 바 차트."""
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.4), 6))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w/2, df['linear_probe'], w, label='Linear Probe', color='steelblue')
    if df['knn_k200'].notna().any():
        bars2 = ax.bar(x + w/2, df['knn_k200'].fillna(0), w, label='k-NN (k=200)', color='salmon')
        ax.bar_label(bars2, fmt='%.3f', padding=2, fontsize=8)
    ax.bar_label(bars1, fmt='%.3f', padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df['exp_name'], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{group_label}: Linear Probe & k-NN Accuracy')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {save_path}")


def plot_class_separation(df: pd.DataFrame, save_path: str):
    """Silhouette Score & Davies-Bouldin Index 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = df['exp_name'].tolist()
    x = np.arange(len(names))

    ax = axes[0]
    bars = ax.bar(x, df['silhouette'].fillna(0), color='mediumseagreen', edgecolor='black')
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_title('Silhouette Score (higher = better separation)')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, df['davies_bouldin'].fillna(0), color='tomato', edgecolor='black')
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_title('Davies-Bouldin Index (lower = better separation)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {save_path}")


def plot_training_curves(results_dir: str, exp_names: list, save_path: str):
    """여러 실험의 학습 손실 곡선 비교."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp in exp_names:
        hist_path = os.path.join(results_dir, f'{exp}_history.json')
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        losses = hist.get('train_loss', [])
        ax.plot(range(1, len(losses)+1), losses, label=exp, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('NT-Xent Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {save_path}")


def generate_summary_report(results_dir: str, save_dir: str):
    """결과 디렉토리 전체를 읽어 비교 차트 일괄 생성."""
    os.makedirs(save_dir, exist_ok=True)
    df = load_all_results(results_dir)

    if df.empty:
        print("  No results found.")
        return

    print(df.to_string(index=False))

    plot_accuracy_comparison(
        df, os.path.join(save_dir, 'accuracy_comparison.png')
    )
    if df['silhouette'].notna().any():
        plot_class_separation(
            df, os.path.join(save_dir, 'class_separation.png')
        )
    plot_training_curves(
        results_dir, df['exp_name'].tolist(),
        os.path.join(save_dir, 'training_curves.png')
    )

    csv_path = os.path.join(save_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  -> summary CSV: {csv_path}")
