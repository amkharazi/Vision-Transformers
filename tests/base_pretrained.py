import sys
sys.path.append('..')


from models.vit_timm import VisionTransformer
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
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import time
import torch
import os

import argparse

import numpy as np
import random

from torchvision.transforms import RandAugment, RandomErasing
from torch.optim.lr_scheduler import CosineAnnealingLR



def set_seed(seed: int = 42):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.8):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch) / float(max(1, num_warmup_epochs))
        return 0.5 * (1. + np.cos(np.pi * (epoch - num_warmup_epochs) / (num_training_epochs - num_warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main(dataset = 'cifar10', 
        TEST_ID = 'Test_ID001',
         batch_size = 256,
         n_epoch = 200,
         image_size = 32,
         train_size = 'default',
         num_classes = 10,
         SEED = None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device is set to : {device}')

    if SEED is None:
        print(f'No seed is set!')
    else:
        set_seed(seed=SEED)

    model = VisionTransformer(num_classes=num_classes, pretrained=True).to(device)
    
    # CIFAR-10
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
        train_loader, _ = get_cifar10_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

    # CIFAR-100
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
        train_loader, _ = get_cifar100_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

    # MNIST
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
        train_loader, _ = get_mnist_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

    # TinyImageNet
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
        transform_val = transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_loader, _, _ = get_tinyimagenet_dataloaders('../datasets', transform_train, transform_val, transform_test, batch_size, image_size, repeat_count=5)

    # FashionMNIST
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
        train_loader, _ = get_fashionmnist_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

    # Flowers102
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
        train_loader, _ = get_flowers102_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

    # Oxford Pets
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
        train_loader, _ = get_oxford_pets_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

    # STL10
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
        train_loader, _ = get_stl10_classification_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
    
    num_parameters = count_parameters(model)
    print(f'This Model has {num_parameters} parameters')
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs=10, num_training_epochs=n_epoch)

    
    def train_epoch(loader, epoch):
        model.train()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}

        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.8)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
            for k in accuracies:
                correct[k] += accuracies[k]['correct']

        elapsed_time = time.time() - start_time
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)
    
        report_train = f'Train epoch {epoch}: top1%={top1_acc}, top2%={top2_acc}, top3%={top3_acc}, top4%={top4_acc}, top5%={top5_acc}, loss={avg_loss}, time={elapsed_time}s'
        print(report_train)
        
        scheduler.step()

        return report_train, top1_acc

    
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'model_stats', 'model_info.txt'), 'a') as f:
        f.write(f'total number of parameters:\n{num_parameters}\n dataset is {dataset}\n seed is ${SEED}')

    best_acc = 0.0
    print(f'Training for {len(range(n_epoch))} epochs\n')
    for epoch in range(0+1,n_epoch+1):
        report_train, top1_acc = train_epoch(train_loader, epoch)

        if top1_acc>best_acc:
            best_acc=top1_acc
            model_path = os.path.join(result_dir, 'model_stats', f'Best_Train_Model.pth')
            torch.save(model.state_dict(), model_path)
    
        report = report_train + '\n'
        with open(os.path.join(result_dir, 'accuracy_stats', 'report_train.txt'), 'a') as f:
            f.write(report)     

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Basic Experiment Settings - Test")

    parser.add_argument('--TEST_ID', type=str, default='Test_ID001', help='Experiment test ID')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=32, help='Image size (square only)')
    parser.add_argument('--train_size', type=str, default='default', help='Training set size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in dataset')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None)')

    args = parser.parse_args()

    main(dataset=args.dataset,
         TEST_ID=args.TEST_ID,
         batch_size=args.batch_size,
         n_epoch=args.n_epoch,
         image_size=args.image_size,
         train_size=args.train_size,
         num_classes=args.num_classes,
         SEED=args.seed)



