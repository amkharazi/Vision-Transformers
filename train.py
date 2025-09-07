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
from utils.food101_loaders import get_food101_dataloaders

from utils.accuracy_measures import topk_accuracy
from utils.num_param import count_parameters, param_counts


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


def build_model(model_type: str,
                num_classes: int,
                tensor_kwargs: Dict,
                img_shape: Tuple[int, int, int, int]) -> nn.Module:
    if model_type == "pretrained":
        from models.vit_timm import VisionTransformer
        model = VisionTransformer(num_classes=num_classes, pretrained=True, freeze_encoder=False)
    elif model_type == "tensorized":
        from models.vit_tensorized import VisionTransformer
        model = VisionTransformer(
            input_size=img_shape,
            patch_size=tensor_kwargs["patch_size"],
            num_classes=num_classes,
            embed_dim=tuple(tensor_kwargs["embed_dim"]),
            num_heads=tuple(tensor_kwargs["num_heads"]),
            num_layers=tensor_kwargs["num_layers"],
            mlp_dim=tuple(tensor_kwargs["mlp_dim"]),
            dropout=tensor_kwargs["dropout"],
            bias=tensor_kwargs["bias"],
            out_embed=tensor_kwargs["out_embed"],
            drop_path=tensor_kwargs["drop_path"],
            ignore_modes=tuple(tensor_kwargs["ignore_modes"]),
            tensor_method_mlp=tuple(tensor_kwargs["tensor_method_mlp"]),
            tensor_method=tensor_kwargs["tensor_method"],
            tdle_level=tensor_kwargs["tdle_level"],
            rank_patch=tensor_kwargs.get("rank_patch"),
            rank_attn=tensor_kwargs.get("rank_attn"),
            rank_mlp1=tensor_kwargs.get("rank_mlp1"),
            rank_mlp2=tensor_kwargs.get("rank_mlp2"),
            rank_classifier=tensor_kwargs.get("rank_classifier"),
        )
    elif model_type == "original":
        from models.vit_original import VisionTransformer
        model = VisionTransformer(
            input_size=img_shape,
            patch_size=tensor_kwargs["patch_size"],
            num_classes=num_classes,
            embed_dim=tensor_kwargs["embed_dim_original"],
            num_heads=tensor_kwargs["num_heads_original"],
            num_layers=tensor_kwargs["num_layers"],
            mlp_dim=tensor_kwargs["mlp_dim_original"],
            dropout=tensor_kwargs["dropout"],
            bias=tensor_kwargs["bias"],
            out_embed=tensor_kwargs["out_embed"],
            drop_path=tensor_kwargs["drop_path"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


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
        loader, _ = get_tinyimagenet_dataloaders(data_root, transform_train, transform_val, transform_val, batch_size, image_size, repeat_count=5)[:2]
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
    if dataset == 'food101':
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
        loader, _ = get_food101_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
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
    p.add_argument('--model_type', type=str, default='pretrained', choices=['pretrained', 'tensorized', 'original'])
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--data_root', type=str, default='./datasets')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--train_size', type=str, default='default')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--save_rate', type=int, default=5)

    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--bias', action='store_true')
    p.add_argument('--no-bias', dest='bias', action='store_false')
    p.set_defaults(bias=True)
    p.add_argument('--out_embed', action='store_true')
    p.add_argument('--no-out_embed', dest='out_embed', action='store_false')
    p.set_defaults(out_embed=True)
    p.add_argument('--drop_path', type=float, default=0.1)

    p.add_argument('--embed_dim', type=int, nargs=3, default=[3, 4, 4])
    p.add_argument('--num_heads', type=int, nargs=3, default=[1, 2, 2])
    p.add_argument('--mlp_dim', type=int, nargs=3, default=[3, 4, 8])
    p.add_argument('--ignore_modes', type=int, nargs=3, default=[0, 1, 2])
    p.add_argument('--tensor_method', type=str, default='tle', choices=['tle', 'tdle', 'tp'])
    p.add_argument('--tensor_method_mlp', type=str, nargs=2, default=['tle', 'tle'])
    p.add_argument('--tdle_level', type=int, default=3)

    p.add_argument('--rank_patch', type=int, nargs='*', default=None)
    p.add_argument('--rank_attn', type=int, nargs='*', default=None)
    p.add_argument('--rank_mlp1', type=int, nargs='*', default=None)
    p.add_argument('--rank_mlp2', type=int, nargs='*', default=None)
    p.add_argument('--rank_classifier', type=int, nargs='*', default=None)

    p.add_argument('--embed_dim_original', type=int, default=3*4*4)
    p.add_argument('--num_heads_original', type=int, default=1*2*2)
    p.add_argument('--mlp_dim_original', type=int, default=128)

    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        set_seed(args.seed)

    result_dir = os.path.join('./results', args.run_id)
    acc_dir = os.path.join(result_dir, 'accuracy_stats')
    model_dir = os.path.join(result_dir, 'model_stats')
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_loader = get_train_loader(args.dataset, args.data_root, args.batch_size, args.image_size, args.train_size)

    img_shape = (args.batch_size, 3, args.image_size, args.image_size)
    tensor_kwargs = vars(args)
    model = build_model(args.model_type, args.num_classes, tensor_kwargs, img_shape).to(device)

    total_params, trainable_params = param_counts(model)
    with open(os.path.join(model_dir, 'model_info.txt'), 'a') as f:
        f.write(f'model_type={args.model_type}\n'
                f'num_parameters_total={total_params}\n'
                f'num_parameters_trainable={trainable_params}\n'
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
