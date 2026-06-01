import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.cifar10_loader import get_eval_dataloaders
from models.projection_head import build_simclr_model


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    all_features, all_labels = [], []
    for x, y in tqdm(loader, desc='extracting', leave=False):
        x = x.to(device)
        h = model.backbone(x)
        h = F.normalize(h, dim=1)   # cosine similarity를 위해 정규화
        all_features.append(h.cpu())
        all_labels.append(y)
    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def knn_predict(train_feats, train_labels, test_feats, k=200,
                num_classes=10, temperature=0.07, chunk_size=512):
    """
    cosine similarity 기반 weighted k-NN 예측.
    메모리 절약을 위해 test를 chunk 단위로 처리.
    """
    total, correct = 0, 0
    train_feats = train_feats.cuda()
    train_labels = train_labels.cuda()

    for start in range(0, test_feats.size(0), chunk_size):
        chunk = test_feats[start:start + chunk_size].cuda()

        # (chunk, train_size) cosine similarity
        sim = torch.mm(chunk, train_feats.T)

        # top-k
        top_sim, top_idx = sim.topk(k, dim=1)

        # temperature-scaled softmax → 각 class별 가중합
        top_sim = (top_sim / temperature).exp()
        top_labels = train_labels[top_idx]  # (chunk, k)

        # class-wise score 합산
        scores = torch.zeros(chunk.size(0), num_classes, device='cuda')
        scores.scatter_add_(1, top_labels, top_sim)

        preds = scores.argmax(dim=1)
        test_chunk_labels = train_labels.new_tensor(
            range(start, min(start + chunk_size, test_feats.size(0)))
        )
        # 실제 test label은 별도로 전달
        correct += preds.eq(
            test_feats._knn_labels[start:start + chunk_size].cuda()
        ).sum().item()
        total += chunk.size(0)

    return correct / total


@torch.no_grad()
def run_knn_eval(checkpoint_path, config: dict, save_dir='./results',
                 k_list=(1, 5, 10, 20, 200)):
    """k-NN evaluation: 별도 학습 없이 feature 유사도로 분류."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    model = build_simclr_model(
        arch=config.get('arch', 'resnet18'),
        proj_num_layers=config.get('proj_num_layers', 2),
        cifar_stem=config.get('cifar_stem', True),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    train_loader, test_loader = get_eval_dataloaders(
        batch_size=config.get('eval_batch_size', 256),
        num_workers=config.get('num_workers', 4),
        root=config.get('data_root', './data/cifar10'),
    )

    print("  Extracting train features for k-NN...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print("  Extracting test features for k-NN...")
    test_feats, test_labels = extract_features(model, test_loader, device)

    train_feats  = train_feats.cuda()
    train_labels = train_labels.cuda()
    test_feats   = test_feats.cuda()
    test_labels  = test_labels.cuda()

    results = {}
    for k in k_list:
        k = min(k, train_feats.size(0))
        correct, total = 0, 0
        chunk = 512

        for start in range(0, test_feats.size(0), chunk):
            end = min(start + chunk, test_feats.size(0))
            q = test_feats[start:end]  # (chunk, D)
            sim = torch.mm(q, train_feats.T)  # (chunk, N_train)
            top_sim, top_idx = sim.topk(k, dim=1)

            top_sim_scaled = (top_sim / 0.07).exp()
            top_lbl = train_labels[top_idx]  # (chunk, k)

            scores = torch.zeros(q.size(0), 10, device='cuda')
            scores.scatter_add_(1, top_lbl, top_sim_scaled)

            preds = scores.argmax(dim=1)
            correct += preds.eq(test_labels[start:end]).sum().item()
            total += end - start

        acc = correct / total
        results[f'k={k}'] = acc
        print(f"  [k-NN] k={k:4d}  acc={acc:.4f}")

    exp_name = config.get('exp_name', 'simclr')
    result_path = os.path.join(save_dir, f'{exp_name}_knn.json')
    with open(result_path, 'w') as f:
        json.dump({'results': results, 'exp_name': exp_name}, f, indent=2)

    print(f"  -> k-NN results saved: {result_path}")
    return results
