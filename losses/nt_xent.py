import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    SimCLR 논문의 contrastive loss.

    batch 내 2N개의 sample에서 같은 이미지의 두 view를 positive pair로,
    나머지 2N-2개를 negative pair로 처리.
    """
    def __init__(self, temperature=0.07, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z1, z2):
        """
        z1, z2: (N, D) — L2 정규화되지 않은 projection output
        """
        N = z1.size(0)

        # L2 정규화
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # (2N, D) 로 concat
        z = torch.cat([z1, z2], dim=0)

        # (2N, 2N) cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temperature

        # 자기 자신과의 유사도는 -inf로 마스킹
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, float('-inf'))

        # positive pair: i번째와 (i+N)번째 (또는 그 역)
        labels = torch.cat([
            torch.arange(N, 2*N),
            torch.arange(0, N)
        ]).to(self.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss — redundancy reduction 방식.
    추가 실험용 (representation collapse에 강함).
    """
    def __init__(self, lambda_coeff=5e-3, projection_dim=128):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.projection_dim = projection_dim

    def forward(self, z1, z2):
        N = z1.size(0)

        # batch normalization (statistics across batch)
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)

        # cross-correlation matrix
        c = torch.mm(z1.T, z2) / N  # (D, D)

        # invariance + redundancy reduction
        on_diag  = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_coeff * off_diag
        return loss

    @staticmethod
    def _off_diagonal(x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SimSiamLoss(nn.Module):
    """
    SimSiam loss — predictor 없이 stop-gradient만으로 단순화한 버전.
    z1, z2를 교차로 stop-gradient 적용해 collapse 방지 효과를 유지.
    """
    def forward(self, z1, z2):
        loss = (
            -F.cosine_similarity(z1, z2.detach(), dim=-1).mean() +
            -F.cosine_similarity(z2, z1.detach(), dim=-1).mean()
        ) / 2
        return loss


_LOSS_VALID_KWARGS = {
    'nt_xent':      {'temperature', 'device'},
    'barlow_twins': {'lambda_coeff', 'projection_dim'},
    'simsiam':      set(),
}


def get_loss(loss_type='nt_xent', **kwargs):
    # loss 타입에 맞는 kwarg만 필터링 (base config 잔여값 제거)
    valid = _LOSS_VALID_KWARGS.get(loss_type, set())
    filtered = {k: v for k, v in kwargs.items() if k in valid}

    if loss_type == 'nt_xent':
        return NTXentLoss(**filtered)
    elif loss_type == 'barlow_twins':
        return BarlowTwinsLoss(**filtered)
    elif loss_type == 'simsiam':
        return SimSiamLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
