# Add all .py files to path
import sys
sys.path.append('..')

from models.vit_original import VisionTransformer
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


def set_seed(seed: int = 42):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def main(dataset = 'cifar10', 
        TEST_ID = 'Test_ID001',
        batch_size = 256,
        n_epoch = 200,
        image_size = 32,
        train_size = 'default',
        patch_size = 4,
        num_classes = 10,
        embed_dim = 3*4*4,
        num_layers = 6,
        num_heads = 1*2*2,
        mlp_dim = 128,
        dropout=0.1,  
        bias=True, 
        out_embed=True, 
        drop_path=0.1,
        SEED = None):
    
    # Setup the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device is set to : {device}')

    if SEED is None:
        print(f'No seed is set!')
    else:
        set_seed(seed=SEED)


    # Set up the vit model
    model = VisionTransformer(
                input_size=(batch_size,3,image_size,image_size),
                patch_size=patch_size,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_dim=mlp_dim,
                dropout=dropout,
                bias=bias,
                out_embed=out_embed,
                drop_path =drop_path,).to(device)
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
        _, test_loader = get_cifar10_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

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
        _, test_loader = get_cifar100_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

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
        _, test_loader = get_mnist_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

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
        _, test_loader, _ = get_tinyimagenet_dataloaders('../datasets', transform_train, transform_val, transform_test, batch_size, image_size, repeat_count=5)

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
        _, test_loader = get_fashionmnist_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

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
        _, test_loader = get_flowers102_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

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
        _, test_loader = get_oxford_pets_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)

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
        _, test_loader = get_stl10_classification_dataloaders('../datasets', transform_train, transform_test, batch_size, image_size, train_size, repeat_count=5)
    
    criterion = nn.CrossEntropyLoss()    

    def test_epoch(loader, epoch):
        model.eval()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}

        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
            for k in accuracies:
                correct[k] += accuracies[k]['correct']

        elapsed_time = time.time() - start_time
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)
    
        report_test = f'Test epoch {epoch}: top1%={top1_acc}, top2%={top2_acc}, top3%={top3_acc}, top4%={top4_acc}, top5%={top5_acc}, loss={avg_loss}, time={elapsed_time}s'
        print(report_test)

        return report_test
    
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)
    
    for epoch in range(0+1,n_epoch+1):
        if epoch%5 == 0:
            weights_path = os.path.join('../results',TEST_ID, 'model_stats', f'Model_epoch_{epoch}.pth')
            print(model.load_state_dict(torch.load(weights_path)))
            model = model.to(device)
            report_test = test_epoch(test_loader, epoch)
            report = report_test + '\n'
            with open(os.path.join(result_dir, 'accuracy_stats', 'report_val.txt'), 'a') as f:
                f.write(report)  
        
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Basic Experiment Settings - Test")

    parser.add_argument('--TEST_ID', type=str, help='Experiment test ID')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=32, help='Image size (must be square / only width or height)')
    parser.add_argument('--train_size', type=str, default='default', help='Size of the training set')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for the model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=3*4*4, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=1*2*2, help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=128, help='MLP hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--bias', type=bool, default=True, help='Use bias in linear layers')
    parser.add_argument('--out_embed', type=bool, default=True, help='Use output embedding')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Stochastic depth drop path rate')
    parser.add_argument('--SEED', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    main(dataset=args.dataset,
         TEST_ID=args.TEST_ID,
         batch_size=args.batch_size,
         n_epoch=args.n_epoch,
         image_size=args.image_size,
         train_size=args.train_size,
         patch_size=args.patch_size,
         num_classes=args.num_classes,
         embed_dim=args.embed_dim,
         num_layers=args.num_layers,
         num_heads=args.num_heads,
         mlp_dim=args.mlp_dim,
         dropout=args.dropout,
         bias=args.bias,
         out_embed=args.out_embed,
         drop_path=args.drop_path,
         SEED=args.SEED)



           

