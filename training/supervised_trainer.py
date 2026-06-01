import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class SupervisedResNet(nn.Module):
    """비교용 fully supervised ResNet."""
    def __init__(self, arch='resnet18', num_classes=10, cifar_stem=True):
        super().__init__()
        arch_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
        }
        net = arch_map[arch](weights=None)

        if cifar_stem:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        feature_dim = net.fc.in_features
        net.fc = nn.Linear(feature_dim, num_classes)
        self.net = net
        self.feature_dim = feature_dim

    def forward(self, x, return_features=False):
        for name, module in self.net.named_children():
            if name == 'fc':
                h = x.flatten(start_dim=1)
                out = module(h)
                if return_features:
                    return h, out
                return out
            x = module(x)
        return x

    def get_features(self, x):
        """FC layer 이전 feature 반환."""
        modules = list(self.net.children())[:-1]
        for m in modules:
            x = m(x)
        return x.flatten(start_dim=1)


class SupervisedTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = config.get('save_dir', './results')
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = SupervisedResNet(
            arch=config.get('arch', 'resnet18'),
            cifar_stem=config.get('cifar_stem', True),
        ).to(self.device)

        root = config.get('data_root', './data/cifar10')
        batch_size = config.get('batch_size', 256)
        num_workers = config.get('num_workers', 4)

        # 표준 CIFAR-10 supervised augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])
        self.train_loader = DataLoader(
            datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        self.test_loader = DataLoader(
            datasets.CIFAR10(root=root, train=False, transform=eval_transform, download=False),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        lr = config.get('lr', 0.1)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr,
            momentum=0.9, weight_decay=5e-4,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 200),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        self.best_acc = 0.0
        self.best_epoch = 0

        # 로그 파일 초기화
        exp_name = config.get('exp_name', 'supervised')
        self.log_path = os.path.join(self.save_dir, f'{exp_name}_train.log')
        with open(self.log_path, 'w') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started\n")
            f.write(f"config: {json.dumps(config, indent=2, default=str)}\n")
            f.write("-" * 60 + "\n")

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in tqdm(self.train_loader, leave=False, desc='supervised'):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)
        return correct / total

    def _save_named(self, exp_name, tag):
        path = os.path.join(self.save_dir, f'{exp_name}_{tag}.pt')
        torch.save({
            'epoch':      len(self.history['train_loss']),
            'best_acc':   self.best_acc,
            'best_epoch': self.best_epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config':  self.cfg,
            'history': self.history,
        }, path)

    def train(self):
        epochs = self.cfg.get('epochs', 200)
        save_freq = self.cfg.get('save_freq', 50)
        exp_name = self.cfg.get('exp_name', 'supervised')

        print(f"\n[Supervised] {exp_name} | device={self.device} | epochs={epochs}")

        for epoch in range(1, epochs + 1):
            loss, train_acc = self.train_epoch()
            self.scheduler.step()

            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(train_acc)

            # 매 epoch test accuracy 측정
            test_acc = self.evaluate()
            self.history['test_acc'].append({'epoch': epoch, 'acc': test_acc})

            # best.pt: test accuracy 기준
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
                self.best_epoch = epoch
                self._save_named(exp_name, 'best')

            if epoch % 10 == 0:
                best_mark = f' * (best @ epoch {self.best_epoch})' if is_best else ''
                print(f"  Epoch [{epoch:3d}/{epochs}] loss={loss:.4f} "
                      f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}{best_mark}")

            # last.pt: 매 epoch 덮어쓰기
            self._save_named(exp_name, 'last')

            # epoch 50/100/150/200 주기 저장
            if epoch % save_freq == 0 or epoch == epochs:
                self._save_named(exp_name, f'epoch{epoch}')
                print(f"  -> epoch{epoch}.pt saved")

            # 로그 파일 기록
            best_mark = f' <-- BEST (epoch {self.best_epoch}, test_acc {self.best_acc:.4f})' if is_best else ''
            log_line = (f"epoch {epoch:4d}/{epochs} | loss {loss:.4f} | "
                        f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f}"
                        + best_mark + "\n")
            with open(self.log_path, 'a') as f:
                f.write(log_line)

        # 학습 완료 요약
        with open(self.log_path, 'a') as f:
            f.write("-" * 60 + "\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training finished\n")
            f.write(f"best epoch : {self.best_epoch}\n")
            f.write(f"best test acc : {self.best_acc:.4f}\n")

        hist_path = os.path.join(self.save_dir, f'{exp_name}_history.json')
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Training complete. best epoch={self.best_epoch}, best test_acc={self.best_acc:.4f}")
        print(f"  Log saved: {self.log_path}")
        return self.model
