import sys
sys.path.append('..')

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
from utils.oxford_pets_loaders import get_oxford_pets_dataloaders
from utils.stl10_classification_loaders import get_stl10_classification_dataloaders

from utils.accuracy_measures import topk_accuracy

from models.vit_original import VisionTransformer


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--data_root', type=str, default='../datasets')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--train_size', type=str, default='default')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=None)


    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        set_seed(args.seed)

    test_loader = get_test_loader(args.dataset, args.data_root, args.batch_size, args.image_size, 'default')

    img_shape = (args.batch_size, 3, args.image_size, args.image_size)
    tensor_kwargs = vars(args)

    result_dir = os.path.join('../results', args.run_id)
    acc_dir = os.path.join(result_dir, 'accuracy_stats')
    model_dir = os.path.join(result_dir, 'model_stats')
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(acc_dir, 'report_test.txt')

    def eval_weights_file(weights_path: str) -> str:
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=16,
            in_chans=3,
            num_classes=args.num_classes,
            embed_dim=768,
            depth=4,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            drop_path=0.0,
        ).to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        top1, top2, top3, top4, top5, loss, elapsed = evaluate_model(model, test_loader, device)
        report = (f'original | test | weights={os.path.basename(weights_path)} | '
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
