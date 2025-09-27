import sys

sys.path.append("..")

import os
import time
import math
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, RandomErasing
from torch import nn, optim
import numpy as np

from utils.accuracy_measures import topk_accuracy
from utils.tinyimagenet_loaders import get_tinyimagenet_dataloaders
from utils.mnist_loaders import get_mnist_dataloaders
from utils.cifar10_loaders import get_cifar10_dataloaders
from utils.cifar100_loaders import get_cifar100_dataloaders
from utils.fashionmnist_loaders import get_fashionmnist_dataloaders
from utils.flowers102_loaders import get_flowers102_dataloaders
from utils.oxford_pets_loaders import get_oxford_pets_dataloaders
from utils.stl10_classification_loaders import get_stl10_classification_dataloaders
from utils.num_param import count_parameters
from models.vit_original import VisionTransformer as VIT
from models.vit_tensorized import VisionTransformer as VALTT


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch) / float(max(1, num_warmup_epochs))
        return 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (epoch - num_warmup_epochs)
                / max(1, (num_training_epochs - num_warmup_epochs))
            )
        )

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


def to_tuple_int(vals):
    return tuple(int(v) for v in vals)


def build_transforms(image_size, gray_scale=False):
    aug = [
        RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        RandomErasing(p=0.25),
    ]
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    if gray_scale:
        aug = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            RandomErasing(p=0.25),
        ]
        base = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    return transforms.Compose(aug), transforms.Compose(base), transforms.Compose(base)


def get_loaders_and_meta(
    dataset, data_dir, batch_size, image_size, train_size, repeat_count
):
    if dataset == "tinyimagenet":
        ttr, tv, tt = build_transforms(image_size, gray_scale=False)
        train_loader, test_loader, _ = get_tinyimagenet_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_val=tv,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 200
    if dataset == "cifar10":
        ttr, tv, tt = build_transforms(image_size, gray_scale=False)
        train_loader, test_loader = get_cifar10_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 10
    if dataset == "cifar100":
        ttr, tv, tt = build_transforms(image_size, gray_scale=False)
        train_loader, test_loader = get_cifar100_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 100
    if dataset == "mnist":
        ttr, tv, tt = build_transforms(image_size, gray_scale=True)
        train_loader, test_loader = get_mnist_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 10
    if dataset == "fashionmnist":
        ttr, tv, tt = build_transforms(image_size, gray_scale=True)
        train_loader, test_loader = get_fashionmnist_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 10
    if dataset == "flowers102":
        ttr, tv, tt = build_transforms(image_size, gray_scale=False)
        train_loader, test_loader = get_flowers102_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 102
    if dataset == "oxford_pets":
        ttr, tv, tt = build_transforms(image_size, gray_scale=False)
        train_loader, test_loader = get_oxford_pets_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 37
    if dataset == "stl10":
        ttr, tv, tt = build_transforms(image_size, gray_scale=False)
        train_loader, test_loader = get_stl10_classification_dataloaders(
            data_dir=data_dir,
            transform_train=ttr,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return train_loader, test_loader, 10
    raise ValueError("unsupported dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_id", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "tinyimagenet",
            "cifar10",
            "cifar100",
            "mnist",
            "fashionmnist",
            "flowers102",
            "oxford_pets",
            "stl10",
        ],
    )
    parser.add_argument(
        "--model_type", type=str, choices=["original", "tensorized"], required=True
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--embed_dim", nargs="+", default=["16", "16", "3"])
    parser.add_argument("--num_heads", nargs="+", default=["2", "2", "3"])
    parser.add_argument("--mlp_dim", nargs="+", default=["16", "16", "4"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", type=int, choices=[0, 1], default=1)
    parser.add_argument("--out_embed", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ignore_modes", nargs="*", default=[])
    parser.add_argument("--tensorized_mlp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--tensor_type", nargs="+", default=["tle", "tle"])
    parser.add_argument("--tdle_level", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--save_rate", type=int, default=5)
    parser.add_argument("--repeat_count", type=int, default=5)
    parser.add_argument("--train_size", type=str, default="default")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    args = parser.parse_args()

    device = (
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    train_loader, test_loader, inferred_classes = get_loaders_and_meta(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.image_size,
        args.train_size,
        args.repeat_count,
    )
    num_classes = args.num_classes if args.num_classes is not None else inferred_classes

    expect_tuple = args.model_type == "tensorized"
    embed_dim_t = to_tuple_int(args.embed_dim)
    num_heads_t = to_tuple_int(args.num_heads)
    mlp_dim_t = to_tuple_int(args.mlp_dim)
    tensor_type = tuple(args.tensor_type) if expect_tuple else None
    ignore_modes = (
        tuple(int(m) for m in args.ignore_modes)
        if args.ignore_modes
        else (0, 1, 2) if expect_tuple else None
    )

    if args.model_type == "original":
        embed_dim_s = math.prod(embed_dim_t)
        num_heads_s = math.prod(num_heads_t)
        mlp_dim_s = math.prod(mlp_dim_t)
        model = VIT(
            input_size=(args.batch_size, 3, args.image_size, args.image_size),
            patch_size=args.patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim_s,
            num_heads=num_heads_s,
            num_layers=args.num_layers,
            mlp_dim=mlp_dim_s,
            dropout=args.dropout,
            bias=bool(args.bias),
            out_embed=bool(args.out_embed),
            device=device,
            ignore_modes=None,
            Tensorized_mlp=False,
        ).to(device)
    else:
        model = VALTT(
            input_size=(args.batch_size, 3, args.image_size, args.image_size),
            patch_size=args.patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim_t,
            num_heads=num_heads_t,
            num_layers=args.num_layers,
            mlp_dim=mlp_dim_t,
            dropout=args.dropout,
            bias=bool(args.bias),
            out_embed=bool(args.out_embed),
            device=device,
            ignore_modes=ignore_modes,
            Tensorized_mlp=bool(args.tensorized_mlp),
            tensor_type=tensor_type,
            tdle_level=args.tdle_level,
        ).to(device)

    num_parameters = count_parameters(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs
    )

    result_dir = os.path.join(args.results_dir, args.test_id)
    acc_dir = os.path.join(result_dir, "accuracy_stats")
    model_dir = os.path.join(result_dir, "model_stats")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model_info.txt"), "a") as f:
        f.write(f"parameters={num_parameters}\n")
        f.write(f"dataset={args.dataset}\n")
        f.write(f"model_type={args.model_type}\n")
        f.write(f"embed_dim={embed_dim_t}\n")
        f.write(f"num_heads={num_heads_t}\n")
        f.write(f"mlp_dim={mlp_dim_t}\n")
        f.write(f"num_layers={args.num_layers}\n")

    best_top1 = -1.0

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
                correct[k] += float(accs[k]["correct"])

        elapsed = time.time() - start
        top_values = [correct[k] / len(loader.dataset) for k in (1, 2, 3, 4, 5)]
        avg_loss = running_loss / len(loader.dataset)
        report = (
            f"{args.model_type} | epoch {epoch} | "
            f"top1={top_values[0]:.4f} top2={top_values[1]:.4f} top3={top_values[2]:.4f} "
            f"top4={top_values[3]:.4f} top5={top_values[4]:.4f} "
            f"loss={avg_loss:.6f} time={elapsed:.2f}s"
        )
        print(report)
        return report, float(top_values[0])

    best_top1 = -1.0
    print(f"Training {args.model_type} for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        report, top1 = train_epoch(train_loader, epoch)
        scheduler.step()

        if top1 > best_top1:
            best_top1 = top1
            torch.save(
                model.state_dict(), os.path.join(model_dir, "Best_Train_Model.pth")
            )

        if epoch % args.save_rate == 0:
            torch.save(
                model.state_dict(), os.path.join(model_dir, f"Model_epoch_{epoch}.pth")
            )

        with open(os.path.join(acc_dir, "report_train.txt"), "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
