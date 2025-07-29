# Check Test Plan for more details 
# Test vit-tensorized model on CIFAR-10  dataset
# Optimizer Adam
# CIFAR-10 dataset -> (3, 224, 224) 
########################################################

# Add all .py files to path
import sys
sys.path.append('..')

# Import Libraries
from Utils.Accuracy_measures import topk_accuracy
from Utils.Cifar10_loader import get_cifar10_dataloaders
from Models.vit_original import VisionTransformer

import torchvision.transforms as transforms
from torch import nn

import time
import torch
import os

if __name__ == '__main__':
    
    # Setup the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device is set to : {device}')

    # Configs 
    
    TEST_ID = 'Test_ID053'
    batch_size = 32
    n_epoch = 400
    image_size = 224

    # Set up the vit model
    model = VisionTransformer(input_size=(batch_size,3,image_size,image_size),
                patch_size=12,
                num_classes=10,
                embed_dim=12*12*3,
                num_heads=3*3*1,
                num_layers=12,
                mlp_dim=24*24*6,
                dropout=0.1,
                bias=True,
                out_embed=True,
                device=device,
                ignore_modes=None,
                Tensorized_mlp=False).to(device)

    # Set up the transforms and train/test loaders

    cifar10_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)), 
            transforms.RandomCrop(image_size, padding=5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    cifar10_transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    
    train_loader, test_loader = get_cifar10_dataloaders(
                                                        data_dir = '../datasets',
                                                        transform_train=cifar10_transform_train,
                                                        transform_test=cifar10_transform_test,
                                                        batch_size=batch_size,
                                                        image_size=image_size)
   
    
    criterion = nn.CrossEntropyLoss()

    def test_epoch(loader, epoch):
        model.eval()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} # set the initial correct count for top1-to-top5 accuracy

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
    
        report_test = f'Test epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
        print(report_test)

        return report_test
    
    # Set up the directories to save the results
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)
    
    print(f'Testing ... \n')

    for epoch in range(0+1,n_epoch+1):
        if epoch%5 == 0:
            weights_path = os.path.join('../results',TEST_ID, 'model_stats', f'Model_epoch_{epoch}.pth')
            print(model.load_state_dict(torch.load(weights_path)))
            model = model.to(device)
            report_test = test_epoch(test_loader, epoch)
            report = report_test + '\n'
            with open(os.path.join(result_dir, 'accuracy_stats', 'report_val.txt'), 'a') as f:
                f.write(report)       

    
            
    
    
    

