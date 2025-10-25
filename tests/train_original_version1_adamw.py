import sys
sys.path.append('..')

import os
import time
import random
import argparse
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, RandomErasing

from utils.cifar10_loaders import get_cifar10_dataloaders
from utils.cifar100_loaders import get_cifar100_dataloaders
from utils.mnist_loaders import get_mnist_dataloaders
from utils.tinyimagenet_loaders import get_tinyimagenet_dataloaders
from utils.fashionmnist_loaders import get_fashionmnist_dataloaders
from utils.flowers102_loaders import get_flowers102_dataloaders
from utils.oxford_pets_loaders import get_oxford_pets_dataloaders
from utils.stl10_classification_loaders import get_stl10_classification_dataloaders

from utils.accuracy_measures import topk_accuracy
from utils.num_param import count_parameters

from models.vit_original import VisionTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch) / float(max(1, num_warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - num_warmup_epochs) / max(1, (num_training_epochs - num_warmup_epochs))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=0.8):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.size(0)
    index = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, float(lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_train_loader(dataset: str, data_root: str, batch_size: int, image_size: int, train_size: str):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_cifar10_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_cifar100_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        loader, _ = get_mnist_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    if dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_tinyimagenet_dataloaders(data_root, transform_train, transform_val, transform_val, batch_size, image_size, repeat_count=1)[:2]
        return loader
    if dataset == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        loader, _ = get_fashionmnist_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    if dataset == 'flowers102':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_flowers102_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    if dataset == 'oxford_pets':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_oxford_pets_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    if dataset == 'stl10':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_stl10_classification_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
        return loader
    raise ValueError(f"Unknown dataset: {dataset}")


def main():
    p = argparse.ArgumentParser("trainer")
    p.add_argument('--run_id', type=str, default='Run_001')
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--data_root', type=str, default='../datasets')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--save_rate', type=int, default=5)
    
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        set_seed(args.seed)

    result_dir = os.path.join('../results', args.run_id)
    acc_dir = os.path.join(result_dir, 'accuracy_stats')
    model_dir = os.path.join(result_dir, 'model_stats')
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_loader = get_train_loader(args.dataset, args.data_root, args.batch_size, args.image_size, 'default')
    
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=16,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
    ).to(device)

    total_param = count_parameters(model)
    with open(os.path.join(model_dir, 'model_info.txt'), 'a') as f:
        f.write(f'model_type={args.model_type}\n'
                f'num_parameters_total={total_param}\n'
                f'dataset={args.dataset}\n'
                f'seed={args.seed}\n')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    def train_epoch(loader, epoch):
        model.train()
        running_loss = 0.0
        correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        start = time.time()

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, ya, yb, lam = mixup_data(inputs, targets, alpha=0.8)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = mixup_criterion(criterion, logits, ya, yb, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accs = topk_accuracy(logits, targets, topk=(1, 2, 3, 4, 5))
            for k in accs:
                correct[k] += float(accs[k]['correct'])

        elapsed = time.time() - start
        top_values = [correct[k] / len(loader.dataset) for k in (1, 2, 3, 4, 5)]
        avg_loss = running_loss / len(loader.dataset)
        report = (f'{args.model_type} | epoch {epoch} | '
                  f'top1={top_values[0]:.4f} top2={top_values[1]:.4f} top3={top_values[2]:.4f} '
                  f'top4={top_values[3]:.4f} top5={top_values[4]:.4f} '
                  f'loss={avg_loss:.6f} time={elapsed:.2f}s')
        print(report)
        return report, float(top_values[0])

    best_top1 = -1.0
    print(f"Training {args.model_type} for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        report, top1 = train_epoch(train_loader, epoch)
        scheduler.step()

        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), os.path.join(model_dir, 'Best_Train_Model.pth'))

        if epoch % args.save_rate == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'Model_epoch_{epoch}.pth'))

        with open(os.path.join(acc_dir, 'report_train.txt'), 'a') as f:
            f.write(report + '\n')


if __name__ == "__main__":
    main()
