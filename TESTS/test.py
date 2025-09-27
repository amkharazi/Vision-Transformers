import sys

sys.path.append("..")

import os
import time
import math
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, RandomErasing
from torch import nn
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
        base =  [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]
    return transforms.Compose(aug), transforms.Compose(base), transforms.Compose(base)


def get_loaders_and_meta(dataset, data_dir, batch_size, image_size, train_size, repeat_count):
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
        return test_loader, 200
    if dataset == "cifar10":
        _, _, tt = build_transforms(image_size, gray_scale=False)
        _, test_loader = get_cifar10_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 10
    if dataset == "cifar100":
        _, _, tt = build_transforms(image_size, gray_scale=False)
        _, test_loader = get_cifar100_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 100
    if dataset == "mnist":
        _, _, tt = build_transforms(image_size, gray_scale=True)
        _, test_loader = get_mnist_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 10
    if dataset == "fashionmnist":
        _, _, tt = build_transforms(image_size, gray_scale=True)
        _, test_loader = get_fashionmnist_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 10
    if dataset == "flowers102":
        _, _, tt = build_transforms(image_size, gray_scale=False)
        _, test_loader = get_flowers102_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 102
    if dataset == "oxford_pets":
        _, _, tt = build_transforms(image_size, gray_scale=False)
        _, test_loader = get_oxford_pets_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 37
    if dataset == "stl10":
        _, _, tt = build_transforms(image_size, gray_scale=False)
        _, test_loader = get_stl10_classification_dataloaders(
            data_dir=data_dir,
            transform_train=tt,
            transform_test=tt,
            batch_size=batch_size,
            image_size=image_size,
            repeat_count=repeat_count,
            train_size=train_size,
        )
        return test_loader, 10
    raise ValueError("unsupported dataset")


def build_model(args, num_classes, device):
    embed_dim_t = to_tuple_int(args.embed_dim)
    num_heads_t = to_tuple_int(args.num_heads)
    mlp_dim_t = to_tuple_int(args.mlp_dim)
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
            ignore_modes=(0, 1, 2),
            Tensorized_mlp=bool(args.tensorized_mlp),
            tensor_type=tuple(args.tensor_type),
            tdle_level=args.tdle_level,
        ).to(device)
    return model


def test_epoch(model, loader, device, criterion, epoch):
    model.eval()
    start = time.time()
    running_loss = 0.0
    correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            accs = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
            for k in accs:
                correct[k] += accs[k]["correct"]
    elapsed = time.time() - start
    top_vals = [(correct[k] / len(loader.dataset)) for k in correct]
    avg_loss = running_loss / len(loader.dataset)
    report = f"Test epoch {epoch}: top1={top_vals[0]}%, top2={top_vals[1]}%, top3={top_vals[2]}%, top4={top_vals[3]}%, top5={top_vals[4]}%, loss={avg_loss}, time={elapsed}s"
    return report


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
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--embed_dim", nargs="+", default=["16", "16", "3"])
    parser.add_argument("--num_heads", nargs="+", default=["2", "2", "3"])
    parser.add_argument("--mlp_dim", nargs="+", default=["16", "16", "4"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", type=int, choices=[0, 1], default=1)
    parser.add_argument("--out_embed", type=int, choices=[0, 1], default=1)
    parser.add_argument("--tensorized_mlp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--tensor_type", nargs="+", default=["tle", "tle"])
    parser.add_argument("--tdle_level", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save_rate", type=int, default=5)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--eval_best", action="store_true")
    parser.add_argument("--repeat_count", type=int, default=5)
    parser.add_argument("--train_size", type=str, default="default")
    
    args = parser.parse_args()

    device = (
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    test_loader, inferred_classes = get_loaders_and_meta(
        args.dataset, args.data_dir, args.batch_size, args.image_size, args.train_size, args.repeat_count
    )
    num_classes = args.num_classes if args.num_classes is not None else inferred_classes
    model = build_model(args, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    result_dir = os.path.join(args.results_dir, args.test_id)
    acc_dir = os.path.join(result_dir, "accuracy_stats")
    model_dir = os.path.join(result_dir, "model_stats")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if args.weights_path:
        state = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state)
        report = test_epoch(model, test_loader, device, criterion, epoch="custom")
        print(report)
        with open(os.path.join(acc_dir, "report_test.txt"), "a") as f:
            f.write(report + "\n")
        return

    for epoch in range(1, args.epochs + 1):
        if epoch % args.save_rate == 0:
            weights_path = os.path.join(model_dir, f"Model_epoch_{epoch}.pth")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=device))
                report = test_epoch(model, test_loader, device, criterion, epoch)
                print(report)
                with open(os.path.join(acc_dir, "report_test.txt"), "a") as f:
                    f.write(report + "\n")

    if args.eval_best:
        best_path = None
        for fname in os.listdir(model_dir):
            if fname.startswith("Best_train_epoch_") and fname.endswith(".pth"):
                best_path = os.path.join(model_dir, fname)
                break
        if best_path and os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            report = test_epoch(model, test_loader, device, criterion, epoch="best")
            print(report)
            with open(os.path.join(acc_dir, "report_test.txt"), "a") as f:
                f.write(report + "\n")


if __name__ == "__main__":
    main()
