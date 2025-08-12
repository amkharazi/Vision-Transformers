import sys
sys.path.append('.')

import os
import time
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, RandomErasing

from utils.cifar10_loaders import get_cifar10_dataloaders
from utils.cifar100_loaders import get_cifar100_dataloaders
from utils.mnist_loaders import get_mnist_dataloaders
from utils.tinyimagenet_loaders import get_tinyimagenet_dataloaders
from utils.fashionmnist_loaders import get_fashionmnist_dataloaders
from utils.flowers102_loaders import get_flowers102_dataloaders
from utils.food101_loaders import get_food101_dataloaders
from utils.oxford_pets_loaders import get_oxford_pets_dataloaders
from utils.stl10_classification_loaders import get_stl10_classification_dataloaders

from utils.accuracy_measures import topk_accuracy


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_test_loader(dataset: str, data_root: str, batch_size: int, image_size: int, train_size: str):
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
        return get_cifar10_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_cifar100_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_mnist_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_tinyimagenet_dataloaders(data_root, transform_train, transform_val, transform_val, batch_size, image_size, repeat_count=5)[1]
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
        return get_fashionmnist_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_flowers102_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_food101_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_oxford_pets_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
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
        return get_stl10_classification_dataloaders(data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)[1]
    raise ValueError(f"Unknown dataset: {dataset}")


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: str) -> Tuple[float, float, float, float, float, float]:
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    running_loss = 0.0
    correct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        running_loss += float(loss.item())
        accs = topk_accuracy(logits, targets, topk=(1, 2, 3, 4, 5))
        for k in accs:
            correct[k] += float(accs[k]['correct'])
    elapsed = time.time() - start
    n = len(loader.dataset)
    top1 = correct[1] / n
    top2 = correct[2] / n
    top3 = correct[3] / n
    top4 = correct[4] / n
    top5 = correct[5] / n
    avg_loss = running_loss / n
    return top1, top2, top3, top4, top5, avg_loss, elapsed


def main():
    p = argparse.ArgumentParser("tester")
    p.add_argument('--run_id', type=str, required=True)
    p.add_argument('--weights', type=str, default=None)
    p.add_argument('--model_type', type=str, default='pretrained', choices=['pretrained', 'tensorized', 'original'])
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--data_root', type=str, default='./datasets')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--train_size', type=str, default='default')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=None)

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
    p.add_argument('--mlp_dim', type=int, nargs=3, default=[3, 4, 4])
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

    test_loader = get_test_loader(args.dataset, args.data_root, args.batch_size, args.image_size, args.train_size)

    img_shape = (args.batch_size, 3, args.image_size, args.image_size)
    tensor_kwargs = vars(args)

    result_dir = os.path.join('./results', args.run_id)
    acc_dir = os.path.join(result_dir, 'accuracy_stats')
    model_dir = os.path.join(result_dir, 'model_stats')
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(acc_dir, 'report_test.txt')

    def eval_weights_file(weights_path: str) -> str:
        model = build_model(args.model_type, args.num_classes, tensor_kwargs, img_shape).to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        top1, top2, top3, top4, top5, loss, elapsed = evaluate_model(model, test_loader, device)
        report = (f'{args.model_type} | test | weights={os.path.basename(weights_path)} | '
                  f'top1={top1:.4f} top2={top2:.4f} top3={top3:.4f} top4={top4:.4f} top5={top5:.4f} '
                  f'loss={loss:.6f} time={elapsed:.2f}s')
        print(report)
        with open(log_path, 'a') as f:
            f.write(report + '\n')
        return report

    evaluated = []

    if args.weights is None:
        best_path = os.path.join(model_dir, 'Best_Train_Model.pth')
        epoch_files = []
        for fname in os.listdir(model_dir):
            if fname.startswith('Model_epoch_') and fname.endswith('.pth'):
                try:
                    num = int(fname.replace('Model_epoch_', '').replace('.pth', ''))
                    epoch_files.append((num, os.path.join(model_dir, fname)))
                except ValueError:
                    pass
        epoch_files.sort(key=lambda t: t[0])
        for _, path in epoch_files:
            evaluated.append(eval_weights_file(path))
        if os.path.isfile(best_path):
            evaluated.append(eval_weights_file(best_path))
    else:
        if os.path.isdir(args.weights):
            best_path = os.path.join(args.weights, 'Best_Train_Model.pth')
            epoch_files = []
            for fname in os.listdir(args.weights):
                if fname.startswith('Model_epoch_') and fname.endswith('.pth'):
                    try:
                        num = int(fname.replace('Model_epoch_', '').replace('.pth', ''))
                        epoch_files.append((num, os.path.join(args.weights, fname)))
                    except ValueError:
                        pass
            epoch_files.sort(key=lambda t: t[0])
            for _, path in epoch_files:
                evaluated.append(eval_weights_file(path))
            if os.path.isfile(best_path):
                evaluated.append(eval_weights_file(best_path))
        else:
            evaluated.append(eval_weights_file(args.weights))


if __name__ == "__main__":
    main()
