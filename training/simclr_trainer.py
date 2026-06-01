import os
import json
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.cifar10_loader import get_simclr_dataloader
from models.projection_head import build_simclr_model
from losses.nt_xent import get_loss


class SimCLRTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = config.get('save_dir', './results')
        os.makedirs(self.save_dir, exist_ok=True)

        self._build_model()
        self._build_optimizer()
        self._build_dataloader()
        self._build_loss()

        self.history = {'train_loss': []}
        self.best_loss = float('inf')
        self.best_epoch = 0

        # 로그 파일 초기화
        exp_name = config.get('exp_name', 'simclr')
        self.log_path = os.path.join(self.save_dir, f'{exp_name}_train.log')
        with open(self.log_path, 'w') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started\n")
            f.write(f"config: {json.dumps(config, indent=2, default=str)}\n")
            f.write("-" * 60 + "\n")

    def _build_model(self):
        cfg = self.cfg
        self.model = build_simclr_model(
            arch=cfg.get('arch', 'resnet18'),
            proj_hidden_dim=cfg.get('proj_hidden_dim', 2048),
            proj_output_dim=cfg.get('proj_output_dim', 128),
            proj_num_layers=cfg.get('proj_num_layers', 2),
            pretrained=cfg.get('pretrained', False),
            cifar_stem=cfg.get('cifar_stem', True),
        ).to(self.device)

    def _build_optimizer(self):
        cfg = self.cfg
        lr = cfg.get('lr', 0.03) * cfg.get('batch_size', 256) / 256
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=cfg.get('weight_decay', 1e-4),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.get('epochs', 200),
            eta_min=0,
        )

    def _build_dataloader(self):
        cfg = self.cfg
        aug_type = cfg.get('augmentation', 'full')
        aug_kwargs = cfg.get('aug_kwargs', {})
        self.train_loader = get_simclr_dataloader(
            batch_size=cfg.get('batch_size', 256),
            num_workers=cfg.get('num_workers', 4),
            augmentation=aug_type,
            root=cfg.get('data_root', './data/cifar10'),
            **aug_kwargs,
        )

    def _build_loss(self):
        cfg = self.cfg
        loss_type = cfg.get('loss_type', 'nt_xent')
        loss_kwargs = cfg.get('loss_kwargs', {})
        loss_kwargs['device'] = self.device
        self.criterion = get_loss(loss_type, **loss_kwargs)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for v1, v2, _ in tqdm(self.train_loader, leave=False, desc='train'):
            v1, v2 = v1.to(self.device), v2.to(self.device)

            _, z1 = self.model(v1)
            _, z2 = self.model(v2)

            loss = self.criterion(z1, z2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self):
        epochs = self.cfg.get('epochs', 200)
        save_freq = self.cfg.get('save_freq', 50)
        exp_name = self.cfg.get('exp_name', 'simclr')

        print(f"\n[SimCLR] {exp_name} | device={self.device} | epochs={epochs}")
        print(f"  arch={self.cfg.get('arch','resnet18')} | "
              f"batch={self.cfg.get('batch_size',256)} | "
              f"loss={self.cfg.get('loss_type','nt_xent')} | "
              f"temp={self.cfg.get('loss_kwargs',{}).get('temperature',0.07)}")

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            self.scheduler.step()
            self.history['train_loss'].append(loss)

            # best.pt: train loss 기준
            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss
                self.best_epoch = epoch
                self._save_named(exp_name, 'best')

            # last.pt: 매 epoch 덮어쓰기
            self._save_named(exp_name, 'last')

            # epoch 50/100/150/200 주기 저장
            if epoch % save_freq == 0 or epoch == epochs:
                self._save_named(exp_name, f'epoch{epoch}')

            # 매 epoch 로그 파일에 기록
            self._write_log(epoch, epochs, loss, is_best)

            if epoch % 10 == 0:
                best_mark = f' * (best @ epoch {self.best_epoch})' if is_best else ''
                print(f"  Epoch [{epoch:3d}/{epochs}] loss={loss:.4f} "
                      f"lr={self.scheduler.get_last_lr()[0]:.5f}{best_mark}")

        self._save_history(exp_name)

        # 학습 완료 요약 로그
        with open(self.log_path, 'a') as f:
            f.write("-" * 60 + "\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training finished\n")
            f.write(f"best epoch : {self.best_epoch}\n")
            f.write(f"best loss  : {self.best_loss:.4f}\n")
        print(f"\n  Training complete. best epoch={self.best_epoch}, best loss={self.best_loss:.4f}")
        print(f"  Log saved: {self.log_path}")
        return self.model

    def _write_log(self, epoch, total_epochs, loss, is_best):
        lr = self.scheduler.get_last_lr()[0]
        best_mark = f' <-- BEST (epoch {self.best_epoch}, loss {self.best_loss:.4f})' if is_best else ''
        line = (f"epoch {epoch:4d}/{total_epochs} | "
                f"loss {loss:.4f} | lr {lr:.5f}{best_mark}\n")
        with open(self.log_path, 'a') as f:
            f.write(line)

    def _save_named(self, exp_name, tag):
        """tag: 'best' or 'last' — 항상 덮어씀."""
        path = os.path.join(self.save_dir, f'{exp_name}_{tag}.pt')
        torch.save({
            'epoch':      len(self.history['train_loss']),
            'best_loss':  self.best_loss,
            'best_epoch': self.best_epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config':  self.cfg,
            'history': self.history,
        }, path)

    def _save_history(self, exp_name):
        path = os.path.join(self.save_dir, f'{exp_name}_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history    = ckpt.get('history', {'train_loss': []})
        self.best_loss  = ckpt.get('best_loss', float('inf'))
        self.best_epoch = ckpt.get('best_epoch', 0)
        print(f"  -> loaded checkpoint: {path} (epoch {ckpt['epoch']})")
        return ckpt['epoch']
