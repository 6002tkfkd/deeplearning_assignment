import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
import numpy as np


class SimCLRTransform:
    """
    두 개의 augmented view를 생성하는 SimCLR augmentation pipeline.
    original paper: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, image_size=32, s=1.0, gaussian_blur=False):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5 if gaussian_blur else 0.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class AblationTransform:
    """ablation 실험용: 특정 augmentation만 선택적으로 적용."""
    def __init__(self, image_size=32, use_crop=True, use_color=True,
                 use_grayscale=True, use_blur=False):
        aug_list = []

        if use_crop:
            aug_list.append(transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)))
        else:
            aug_list.append(transforms.Resize(image_size))

        aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if use_color:
            aug_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8))

        if use_grayscale:
            aug_list.append(transforms.RandomGrayscale(p=0.2))

        if use_blur:
            aug_list.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5))

        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ]
        self.transform = transforms.Compose(aug_list)

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR10Pair(Dataset):
    """SimCLR 학습용: (view1, view2) 쌍 반환."""
    def __init__(self, root='./data/cifar10', train=True, transform=None, download=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        v1, v2 = self.transform(img)
        return v1, v2, label


def get_eval_transforms():
    """linear probe / k-NN 평가용 transform (augmentation 없음)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])


def get_simclr_dataloader(batch_size=256, num_workers=4, augmentation='full',
                          root='./data/cifar10', **aug_kwargs):
    """SimCLR 학습용 dataloader."""
    _full_keys     = {'image_size', 's', 'gaussian_blur'}
    _ablation_keys = {'image_size', 'use_crop', 'use_color', 'use_grayscale', 'use_blur'}

    if augmentation == 'full':
        filtered = {k: v for k, v in aug_kwargs.items() if k in _full_keys}
        transform = SimCLRTransform(**filtered)
    elif augmentation == 'ablation':
        filtered = {k: v for k, v in aug_kwargs.items() if k in _ablation_keys}
        transform = AblationTransform(**filtered)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation}")

    dataset = CIFAR10Pair(root=root, train=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def get_eval_dataloaders(batch_size=256, num_workers=4, root='./data/cifar10'):
    """linear probe / k-NN 평가용 dataloader."""
    transform = get_eval_transforms()

    train_set = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    test_set  = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
